"""Spawn a static RoboTwin-OD mesh as the safety obstacle and optionally add it to CuRobo."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from .tasks import (
    TASK_SPECS,
    iter_corridors,
    iter_named_actors,
    longest_corridor,
    resolve_arm,
    spec_grasp_names,
    start_target_xy,
    target_xy,
)

OBSTACLE_NAME = "safety_obstacle"
OBSTACLE_MODEL = "086_woodenblock"
# Documented presets (any assets/objects/<name> also works). Sizes are world AABB.
OBSTACLE_PRESETS = {
    "086_woodenblock": "cube 10.3 cm (default)",
    "068_boxdrink": "box 11.0 x 15.4 x 11.6 cm",
    "105_sauce-can": "can 10.0 x 11.6 x 10.0 cm",
    "059_pencup": "cup 9.8 x 11.7 x 9.8 cm",
    "071_can": "can 7.1 x 9.6 x 7.1 cm",
    "101_milk-tea": "cup 13.6 x 15.5 x 13.6 cm",
    "023_tissue-box": "box 11.6 x 6.3 x 6.8 cm",
    "038_milk-box": "carton 6.9 x 12.2 x 6.5 cm",
    "004_fluted-block": "block 9.2 x 6.5 x 9.0 cm",
    "073_rubikscube": "cube 6.5 x 6.8 x 7.7 cm",
}
# All listed RoboTwin-OD meshes rest with AABB center along Y. Map that to table Z.
OBSTACLE_UP_AXIS = {name: 1 for name in OBSTACLE_PRESETS}
TABLE_Z = 0.74
# RoboTwin-OD meshes are Y-up. SAPIEN table is Z-up. Same quat as 071_can
# in place_cans_plasticbox: model (x,y,z) → world (z,x,y).
OBSTACLE_STAND_QUAT = (0.5, 0.5, 0.5, 0.5)
_STAND_R = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_active_model = OBSTACLE_MODEL
_active_model_id = 0
_active_cfg: dict | None = None
TABLE_XYLIM = np.array([[-0.48, 0.48], [-0.32, 0.28]], dtype=np.float64)
KEEPAWAY_GAP = 0.02
DEFAULT_ACTOR_RADIUS = 0.08


def world_center_and_half(cfg: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """World-frame (center offset, half-extents) from any RoboTwin actor config.

    RoboTwin uses three layouts:
      - mesh JSON: extents in model units (~1), scale [s,s,s] → half = 0.5 * extents * scale
      - URDF JSON: same extents, but scale is a scalar (e.g. 0.12)
      - create_box: extents and scale are both already world half-size
    """
    cfg = cfg or {}
    extents = np.abs(np.asarray(cfg.get("extents", [0.1, 0.1, 0.1]), dtype=np.float64).reshape(-1))
    if extents.size < 3:
        extents = np.resize(extents, 3)
    else:
        extents = extents[:3]
    raw_scale = cfg.get("scale")
    if raw_scale is None:
        raw = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    else:
        raw = np.asarray(raw_scale, dtype=np.float64).reshape(-1)
    if raw.size == 0 or not np.all(np.isfinite(raw)):
        raw = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    if raw.size == 1:
        scale = np.full(3, abs(float(raw[0])), dtype=np.float64)
    else:
        scale = np.abs(np.resize(raw[:3] if raw.size >= 3 else raw, 3))
    center = np.asarray(cfg.get("center", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
    if center.size < 3:
        center = np.resize(center, 3)
    else:
        center = center[:3]
    # create_box copies half_size into both fields.
    if np.allclose(extents, scale, rtol=0.05, atol=1e-4) and float(np.max(extents)) < 0.4:
        return center.astype(np.float64), extents.astype(np.float64)
    return (center * scale).astype(np.float64), (0.5 * extents * scale).astype(np.float64)

# Placement t along pick-object → place-target. Lower than 0.5 keeps the
# drink nearer the grasped object so open/close at the place pose has room.
CORRIDOR_T_LO = 0.22
CORRIDOR_T_HI = 0.48
UNSAFE_LEVEL = {
    1: {"t": 0.45, "off_path_m": 0.16},
    2: {"t": 0.55, "off_path_m": 0.22},
    3: {"t": 0.65, "off_path_m": 0.28},
}

COLLISION_YML = {
    "piper": "piper/collision_piper.yml",
    "franka-panda": "franka-panda/collision_franka.yml",
    "ur5-wsg": "ur5-wsg/collision_wsg.yml",
    "ARX-X5": "ARX-X5/collision_X5A.yml",
    # This file contains both fl_* and fr_* links. distance.py filters it by
    # side for Aloha's single shared articulation.
    "aloha-agilex": "aloha-agilex/collision_aloha_left.yml",
}


def patch_curobo_collision_cache() -> None:
    """Reserve extra cuboid slots so update_world can add the safety obstacle."""
    from curobo.wrap.reacher.motion_gen import MotionGenConfig

    orig = MotionGenConfig.load_from_robot_config
    orig_fn = orig.__func__ if hasattr(orig, "__func__") else orig
    if getattr(orig_fn, "_cehj_cache", False):
        return

    def wrapped(*args, **kwargs):
        kwargs.setdefault("collision_cache", {"obb": 16, "mesh": 8})
        return orig_fn(*args, **kwargs)

    wrapped._cehj_cache = True
    MotionGenConfig.load_from_robot_config = staticmethod(wrapped)


def _pose_xyz(obj) -> np.ndarray:
    pose = obj.get_pose() if hasattr(obj, "get_pose") else obj
    return np.asarray(pose.p, dtype=np.float64)


def _objects_roots() -> list[Path]:
    return [
        Path("assets/objects"),
        Path(__file__).resolve().parents[2] / "RoboTwin" / "assets" / "objects",
    ]


def _load_model_cfg(model: str, model_id: int = 0) -> dict:
    import json

    names = (f"model_data{int(model_id)}.json", "model_data.json")
    for root in _objects_roots():
        folder = root / str(model)
        for name in names:
            path = folder / name
            if path.is_file():
                return json.loads(path.read_text())
    return {}


def set_obstacle_model(model: str | None, model_id: int = 0) -> str:
    """Select the RoboTwin-OD folder used for spawn + OBB distance."""
    global _active_model, _active_model_id, _active_cfg
    name = str(model or OBSTACLE_MODEL).strip() or OBSTACLE_MODEL
    _active_model = name
    _active_model_id = int(model_id or 0)
    _active_cfg = _load_model_cfg(name, _active_model_id)
    return name


def _model_center() -> np.ndarray:
    center, _half = world_center_and_half(_active_cfg)
    return center


def _model_half_extents() -> np.ndarray:
    _center, half = world_center_and_half(_active_cfg)
    if float(np.max(np.abs(half))) < 1e-6:
        return np.array([0.051, 0.051, 0.052], dtype=np.float64)
    return half


def _up_axis() -> int:
    """Model-frame axis that should point at the table normal (world Z)."""
    if _active_model in OBSTACLE_UP_AXIS:
        return int(OBSTACLE_UP_AXIS[_active_model])
    center = _model_center()
    if float(np.max(np.abs(center))) >= 1e-3:
        return int(np.argmax(np.abs(center)))
    return int(np.argmax(_model_half_extents()))


def _stand_R_quat() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Map the mesh up-axis onto world Z. Identity if the mesh is already Z-up."""
    up = _up_axis()
    if up == 1:
        return _STAND_R, OBSTACLE_STAND_QUAT
    if up == 0:
        r = np.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        return r, (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)
    return np.eye(3, dtype=np.float64), (1.0, 0.0, 0.0, 0.0)


def _world_half_extents() -> np.ndarray:
    """AABB half-extents after standing the mesh on the Z-up table."""
    r, _quat = _stand_R_quat()
    return np.abs(r) @ _model_half_extents()


def _clip_xy(xy: np.ndarray) -> np.ndarray:
    out = xy.copy()
    out[0] = np.clip(out[0], TABLE_XYLIM[0, 0], TABLE_XYLIM[0, 1])
    out[1] = np.clip(out[1], TABLE_XYLIM[1, 0], TABLE_XYLIM[1, 1])
    return out


def _in_table(xy: np.ndarray, block_r: float) -> bool:
    """True if a block of radius block_r fits on the table without hanging off."""
    m = float(block_r) + 0.005
    return (
        TABLE_XYLIM[0, 0] + m <= xy[0] <= TABLE_XYLIM[0, 1] - m
        and TABLE_XYLIM[1, 0] + m <= xy[1] <= TABLE_XYLIM[1, 1] - m
    )


def _line_dist(xy: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return float(np.linalg.norm(xy - p0))
    t = float(np.clip(np.dot(xy - p0, delta) / (length * length), 0.0, 1.0))
    return float(np.linalg.norm(xy - (p0 + t * delta)))


def _table_grid(block_r: float) -> np.ndarray:
    m = float(block_r) + 0.02
    xs = np.linspace(TABLE_XYLIM[0, 0] + m, TABLE_XYLIM[0, 1] - m, 11)
    ys = np.linspace(TABLE_XYLIM[1, 0] + m, TABLE_XYLIM[1, 1] - m, 9)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def _xy_radius_from_config(cfg: dict) -> float:
    _, half = world_center_and_half(cfg)
    radius = float(np.max(half[:2]))
    return float(np.clip(radius, 0.03, 0.22))


def _keepaway_entry(val) -> tuple[np.ndarray, float] | None:
    try:
        xy = _pose_xyz(val)[:2]
    except Exception:
        try:
            arr = np.asarray(val, dtype=np.float64).reshape(-1)
            xy = arr[:2]
        except Exception:
            return None
    cfg = getattr(val, "config", None) or {}
    if cfg:
        radius = _xy_radius_from_config(cfg)
    else:
        # Place-target poses are empty table, not a physical object.
        radius = 0.06
    return np.asarray(xy, dtype=np.float64), float(radius)


def _keepaway_ok(xy: np.ndarray, keepaways: Iterable[tuple[np.ndarray, float]], block_r: float) -> bool:
    xy = np.asarray(xy[:2], dtype=np.float64)
    for item in keepaways:
        if item is None:
            continue
        pt, actor_r = item
        need = block_r + float(actor_r) + KEEPAWAY_GAP
        if np.linalg.norm(xy - np.asarray(pt[:2], dtype=np.float64)) < need:
            return False
    return True


def _blocked_t_intervals(
    p0: np.ndarray,
    p1: np.ndarray,
    dist: float,
    keepaways: list[tuple[np.ndarray, float]],
    block_r: float,
) -> list[tuple[float, float]]:
    """t ranges on the object→target segment that overlap a pick/target keepaway."""
    delta = p1 - p0
    out: list[tuple[float, float]] = []
    if dist <= 1e-6:
        return out
    for pt, actor_r in keepaways:
        p = np.asarray(pt[:2], dtype=np.float64)
        need = block_r + float(actor_r) + KEEPAWAY_GAP
        t_proj = float(np.dot(p - p0, delta) / (dist * dist))
        closest = p0 + t_proj * delta
        perp = float(np.linalg.norm(p - closest))
        if perp >= need:
            continue
        half_t = float(np.sqrt(max(need * need - perp * perp, 0.0)) / dist)
        out.append((t_proj - half_t, t_proj + half_t))
    return out


def _t_free(t: float, blocked: list[tuple[float, float]], margin: float = 1e-4) -> bool:
    return all(t <= a - margin or t >= b + margin for a, b in blocked)


def _t_bounds(
    p0: np.ndarray,
    p1: np.ndarray,
    dist: float,
    keepaways: list[tuple[np.ndarray, float]],
    block_r: float,
    t_pref: float,
    t_lo: float = CORRIDOR_T_LO,
    t_hi: float = CORRIDOR_T_HI,
) -> tuple[float, float, float]:
    """Prefer t_pref in [CORRIDOR_T_LO, CORRIDOR_T_HI]; snap to a keepaway-free t there."""
    lo, hi = float(t_lo), float(t_hi)
    blocked = _blocked_t_intervals(p0, p1, dist, keepaways, block_r)
    pref = float(np.clip(t_pref, lo, hi))
    if _t_free(pref, blocked):
        return pref, lo, hi
    free = [float(t) for t in np.linspace(lo, hi, 41) if _t_free(t, blocked)]
    if free:
        pref = min(free, key=lambda t: abs(t - pref))
    return pref, lo, hi


def _pack_xyz(xy: np.ndarray, table_z: float, half: np.ndarray) -> np.ndarray:
    return np.array([float(xy[0]), float(xy[1]), table_z + half[2] + 0.002], dtype=np.float64)


def _perp_signs(p0: np.ndarray, delta: np.ndarray, perp: np.ndarray, robot_keepaways) -> list[float]:
    """Prefer the side of the corridor farther from the arm bases."""
    signs = [1.0, -1.0]
    if not robot_keepaways:
        return signs

    def clearance(sign: float) -> float:
        cand = p0 + 0.45 * delta + sign * 0.20 * perp
        return min(float(np.linalg.norm(cand - np.asarray(xy[:2]))) for xy, _r in robot_keepaways)

    return sorted(signs, key=lambda s: -clearance(s))


def geometric_pose(
    p0: np.ndarray,
    p1: np.ndarray,
    mode: str,
    corridor_t: float,
    table_z: float,
    keepaways: list[tuple[np.ndarray, float]],
    robot_keepaways: list[tuple[np.ndarray, float]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (xyz, half_extents), or (None, None) if every candidate overlaps a keepaway.

    on_path stays on the object→target segment at corridor_t when keepaway allows.
    off_path is beside that segment. Size comes from the active --obstacle-model AABB.
    """
    p0 = np.asarray(p0[:2], dtype=np.float64)
    p1 = np.asarray(p1[:2], dtype=np.float64)
    robot_keepaways = list(robot_keepaways or ())
    delta = p1 - p0
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:
        delta = np.array([0.0, -0.12], dtype=np.float64)
        dist = 0.12
    perp = np.array([-delta[1], delta[0]], dtype=np.float64) / dist
    half = _world_half_extents()
    block_r = float(np.max(half[:2]))
    t0, t_lo, t_hi = _t_bounds(p0, p1, dist, keepaways, block_r, float(corridor_t))
    t_tries: list[float] = [t0]
    for t in np.linspace(t_lo, t_hi, 9):
        t = float(t)
        if all(abs(t - x) > 1e-4 for x in t_tries):
            t_tries.append(t)
    signs = _perp_signs(p0, delta, perp, robot_keepaways)
    off_pref = 0.22
    offsets: list[float] = []
    for off in (off_pref, 0.20, 0.16, 0.24, 0.32, 0.40, 0.12, 0.48):
        if off not in offsets:
            offsets.append(float(off))

    def ok(xy: np.ndarray) -> bool:
        return _in_table(xy, block_r) and _keepaway_ok(xy, keepaways, block_r)

    xy = None
    if mode == "off_path":
        for off_m in offsets:
            for sign in signs:
                for t_try in t_tries:
                    cand = p0 + t_try * delta + sign * off_m * perp
                    if ok(cand):
                        xy = cand
                        break
                if xy is not None:
                    break
            if xy is not None:
                break
        if xy is None:
            best = None
            best_score = -1e9
            for cand in _table_grid(block_r):
                if not ok(cand):
                    continue
                if _line_dist(cand, p0, p1) < 0.10:
                    continue
                clearance = min(
                    float(np.linalg.norm(cand - np.asarray(pt[:2]))) - float(actor_r) - block_r
                    for pt, actor_r in keepaways
                )
                if clearance > best_score:
                    best = cand
                    best_score = clearance
            if best is not None:
                xy = best
                print("[obstacle] off_path table-grid spawn")
    else:
        for t_try in t_tries:
            cand = p0 + t_try * delta
            if ok(cand):
                xy = cand
                break
        if xy is None:
            for nudge in (0.03, 0.05, 0.08, 0.12, 0.16):
                for sign in signs:
                    for t_try in t_tries:
                        cand = p0 + t_try * delta + sign * nudge * perp
                        if ok(cand):
                            xy = cand
                            print(f"[obstacle] on_path keepaway nudge {sign * nudge:.2f} m")
                            break
                    if xy is not None:
                        break
                if xy is not None:
                    break
        if xy is None:
            best = None
            best_ld = 1e9
            for cand in _table_grid(block_r):
                if not ok(cand):
                    continue
                ld = _line_dist(cand, p0, p1)
                if ld < best_ld:
                    best = cand
                    best_ld = ld
            if best is not None:
                xy = best
                print(f"[obstacle] on_path fallback {best_ld:.2f} m off the line")
    if xy is None:
        print("[obstacle] no spawn pose clears pick/target keepaway; skipping block")
        return None, None
    return _pack_xyz(xy, table_z, half), half


def waypoint_pose(
    ee_path: np.ndarray,
    mode: str,
    corridor_t: float,
    table_z: float,
    keepaways: list[tuple[np.ndarray, float]],
    p0: np.ndarray,
    p1: np.ndarray,
    robot_keepaways: list[tuple[np.ndarray, float]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Snap onto a recorded EE polyline, else fall back to geometric. Same keepaway as geometric."""
    pts = np.asarray(ee_path, dtype=np.float64).reshape(-1, 3)
    high = pts[pts[:, 2] > table_z + 0.04] if pts.size else pts
    if high.shape[0] < 3:
        return geometric_pose(p0, p1, mode, corridor_t, table_z, keepaways, robot_keepaways)
    t = float(np.clip(corridor_t, CORRIDOR_T_LO, CORRIDOR_T_HI))
    i = int(np.clip(t * (high.shape[0] - 1), 1, high.shape[0] - 2))
    xy = high[i, :2].copy()
    if mode == "off_path":
        tangent = high[min(i + 1, high.shape[0] - 1), :2] - high[max(i - 1, 0), :2]
        nrm = np.linalg.norm(tangent)
        if nrm < 1e-6:
            return geometric_pose(p0, p1, mode, corridor_t, table_z, keepaways, robot_keepaways)
        perp = np.array([-tangent[1], tangent[0]], dtype=np.float64) / nrm
        signs = _perp_signs(np.asarray(p0[:2], dtype=np.float64), tangent, perp, robot_keepaways or [])
        xy = xy + signs[0] * 0.22 * perp
    half = _world_half_extents()
    block_r = float(np.max(half[:2]))
    if _in_table(xy, block_r) and _keepaway_ok(xy, keepaways, block_r):
        return _pack_xyz(xy, table_z, half), half
    print("[obstacle] waypoint snap overlaps a keepaway; using geometric")
    return geometric_pose(p0, p1, mode, corridor_t, table_z, keepaways, robot_keepaways)


def spawn_obstacle(task, xyz: np.ndarray, is_static: bool = True):
    """Spawn the active RoboTwin-OD model. Always static so distance/CuRobo stay valid."""
    import sapien
    from envs.utils.create_actor import create_actor, create_box

    is_static = True
    if _active_cfg is None:
        set_obstacle_model(_active_model, _active_model_id)
    r, quat = _stand_R_quat()
    pose_xyz = np.asarray(xyz, dtype=np.float64) - (r @ _model_center())
    pose = sapien.Pose(pose_xyz.tolist(), list(quat))
    # create_actor reads scale from JSON; null scale (e.g. 038_milk-box) would break SAPIEN.
    json_scale = None if _active_cfg is None else _active_cfg.get("scale")
    actor = None
    if json_scale is not None:
        actor = create_actor(
            task,
            pose,
            _active_model,
            convex=True,
            is_static=is_static,
            model_id=_active_model_id,
        )
    if actor is None:
        half = _world_half_extents()
        # create_box origin is the AABB center, unlike mesh JSON which has a local center offset.
        box_pose = sapien.Pose(np.asarray(xyz, dtype=np.float64).tolist(), [1, 0, 0, 0])
        actor = create_box(
            task,
            box_pose,
            half_size=half.tolist(),
            color=(0.55, 0.32, 0.12),
            is_static=is_static,
            name=OBSTACLE_NAME,
        )
        why = "null scale in model_data" if json_scale is None else "mesh missing"
        print(f"[obstacle] {_active_model} {why}; using static create_box fallback")
    else:
        actor.actor.set_name(OBSTACLE_NAME)
    if getattr(actor, "config", None) is None and _active_cfg:
        actor.config = _active_cfg
    return actor


def spawn_woodenblock(task, xyz: np.ndarray, is_static: bool = True):
    return spawn_obstacle(task, xyz, is_static=is_static)


def _table_world_dict(planner) -> dict:
    origin = planner.robot_origion_pose
    return {
        "cuboid": {
            "table": {
                "dims": [0.7, 2, 0.04],
                "pose": [
                    float(origin.p[1]),
                    0.0,
                    0.74 - float(origin.p[2]),
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        }
    }


def _aloha_obstacle_in_base(planner, p, q) -> tuple[np.ndarray, np.ndarray]:
    """Apply RoboTwin's Aloha-only planner frame transform.

    RoboTwin planner.py uses a rigid frame-bias transform plus a small,
    side-specific yaw for Aloha targets. Applying the same transform here
    keeps an obstacle in the same CuRobo frame as those targets.
    """
    import transforms3d as t3d

    side_yaw = -0.01 if "curobo_right" in str(planner.yml_path) else -0.02
    target = t3d.affines.compose(p, t3d.quaternions.quat2mat(q), [1, 1, 1])
    bias = t3d.affines.compose(planner.frame_bias, np.eye(3), [1, 1, 1])
    yaw = t3d.axangles.axangle2mat([0, 0, 1], side_yaw)
    rotation = t3d.affines.compose([0, 0, 0], yaw, [1, 1, 1])
    transformed = rotation @ bias @ target
    return transformed[:3, 3], t3d.quaternions.mat2quat(transformed[:3, :3])


def _obstacle_in_base(planner, xyz, quat, half) -> dict:
    origin = planner.robot_origion_pose
    base = np.concatenate([np.asarray(origin.p), np.asarray(origin.q)])
    world = np.concatenate([np.asarray(xyz, dtype=np.float64), np.asarray(quat, dtype=np.float64)])
    p, q = planner._trans_from_world_to_base(base, world)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if "aloha-agilex" in str(getattr(planner, "yml_path", "")):
        p, q = _aloha_obstacle_in_base(planner, p, q)
    else:
        # Preserve the original transform exactly for all four existing
        # embodiments.
        p = p + np.asarray(planner.frame_bias, dtype=np.float64)
    dims = (2.0 * np.asarray(half, dtype=np.float64)).tolist()
    return {
        "dims": dims,
        "pose": [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])],
    }


def update_curobo_world(robot, obstacle_xyz=None, obstacle_quat=None, obstacle_half=None) -> None:
    """Push table (+ optional cuboid) into both MotionGen worlds."""
    from curobo.geom.types import WorldConfig

    for planner in (robot.left_planner, robot.right_planner):
        if planner is None or not hasattr(planner, "motion_gen"):
            continue
        world = _table_world_dict(planner)
        if obstacle_xyz is not None:
            world["cuboid"][OBSTACLE_NAME] = _obstacle_in_base(
                planner, obstacle_xyz, obstacle_quat, obstacle_half
            )
        cfg = WorldConfig.from_dict(world)
        try:
            planner.motion_gen.update_world(cfg)
            if getattr(planner, "motion_gen_batch", None) is not None:
                planner.motion_gen_batch.update_world(cfg)
        except Exception as exc:
            print(f"[obstacle] CuRobo update_world failed: {exc}")


@lru_cache(maxsize=8)
def load_collision_spheres(embodiment: str) -> dict:
    rel = COLLISION_YML[embodiment]
    path = Path("assets/embodiments") / rel
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data["collision_spheres"]


def keepaways_from_robot(env) -> list[tuple[np.ndarray, float]]:
    """Keep the block off both arm bases (off_path often drifts toward the robots)."""
    out = []
    for entity in (env.robot.left_entity, env.robot.right_entity):
        try:
            pose = entity.get_root_pose() if hasattr(entity, "get_root_pose") else entity.get_pose()
            xy = np.asarray(pose.p[:2], dtype=np.float64)
        except Exception:
            continue
        out.append((xy, 0.18))
    return out


def keepaways_from_task(task, spec: dict) -> list[tuple[np.ndarray, float]]:
    """Keep the block off pick object, place target, and extra grasp objects."""
    names = spec_grasp_names(spec)
    tgt = spec.get("target")
    if tgt and tgt not in names:
        names.append(tgt)
    out: list[tuple[np.ndarray, float]] = []
    seen = set()
    target_name = spec.get("target")
    for name, actor in iter_named_actors(task, names):
        entry = _keepaway_entry(actor)
        if entry is None:
            continue
        xy, radius = entry
        cap = TARGET_KEEPAWAY_CAP if name == target_name or name == tgt else PICK_KEEPAWAY_CAP
        radius = min(float(radius), cap)
        key = (round(float(xy[0]), 3), round(float(xy[1]), 3))
        if key in seen:
            continue
        seen.add(key)
        out.append((xy, radius))
    if tgt:
        try:
            xy = target_xy(task, tgt)
            key = (round(float(xy[0]), 3), round(float(xy[1]), 3))
            if key not in seen:
                seen.add(key)
                out.append((xy, TARGET_KEEPAWAY_CAP))
        except Exception:
            pass
    return out


STRETCH_XYLIM = np.array([[-0.36, 0.36], [-0.28, 0.14]], dtype=np.float64)
TARGET_KEEPAWAY_CAP = 0.12
PICK_KEEPAWAY_CAP = 0.10
STRETCH_MIN_DEFAULT = 0.22
STRETCH_PAIR_GAP = 0.10


def _actor_set_xy(actor, xy: np.ndarray) -> None:
    import sapien

    inner = getattr(actor, "actor", actor)
    pose = inner.get_pose()
    inner.set_pose(sapien.Pose([float(xy[0]), float(xy[1]), float(pose.p[2])], pose.q))


def _clip_stretch_xy(xy: np.ndarray) -> np.ndarray:
    out = np.asarray(xy[:2], dtype=np.float64).copy()
    out[0] = np.clip(out[0], STRETCH_XYLIM[0, 0], STRETCH_XYLIM[0, 1])
    out[1] = np.clip(out[1], STRETCH_XYLIM[1, 0], STRETCH_XYLIM[1, 1])
    if abs(out[0]) < 0.06:
        out[0] = 0.06 if out[0] >= 0 else -0.06
    return out


def _stretch_one(actor, p1: np.ndarray, min_dist: float, others: list[np.ndarray]) -> np.ndarray:
    p0 = _pose_xyz(actor)[:2]
    p1 = np.asarray(p1[:2], dtype=np.float64)
    delta = p0 - p1
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:
        delta = np.array([np.sign(p0[0]) or 1.0, 0.0], dtype=np.float64)
        dist = 1.0
    direction = delta / dist
    if dist >= min_dist:
        xy = p0.copy()
    else:
        xy = _clip_stretch_xy(p1 + direction * min_dist)
        if float(np.linalg.norm(xy - p1)) < min_dist * 0.85:
            xy = _clip_stretch_xy(p0 + direction * (min_dist - dist + 0.02))
    for other in others:
        sep = xy - np.asarray(other[:2], dtype=np.float64)
        if float(np.linalg.norm(sep)) < STRETCH_PAIR_GAP:
            n = sep if float(np.linalg.norm(sep)) > 1e-4 else direction
            n = n / (float(np.linalg.norm(n)) + 1e-8)
            xy = _clip_stretch_xy(xy + n * STRETCH_PAIR_GAP)
    if float(np.linalg.norm(xy - p0)) > 0.005:
        _actor_set_xy(actor, xy)
    return xy


def _bias_opposite_sides(pick_actors: list[tuple[str, object]]) -> None:
    if len(pick_actors) < 2:
        return
    xs = [float(_pose_xyz(actor)[0]) for _label, actor in pick_actors[:2]]
    if xs[0] * xs[1] < 0:
        return
    _label, actor = pick_actors[1]
    pose = _pose_xyz(actor)
    new_x = -abs(float(pose[0])) if xs[0] > 0 else abs(float(pose[0]))
    if abs(new_x) < 0.12:
        new_x = -0.22 if xs[0] > 0 else 0.22
    _actor_set_xy(actor, np.array([new_x, float(pose[1])], dtype=np.float64))
    print(f"[spawn] bias opposite sides: {_label} x {pose[0]:.3f} -> {new_x:.3f}")


def stretch_task_spawns(env) -> None:
    """Nudge pick objects away from place targets so an on_path block can fit."""
    spec = TASK_SPECS.get(getattr(env, "task_name", ""), {})
    min_dist = float(spec.get("stretch_min") or 0.0)
    if min_dist <= 0:
        return
    corridors = iter_corridors(env.task, spec)
    if not corridors:
        return
    pick_actors = list(iter_named_actors(env.task, spec_grasp_names(spec)))
    if spec.get("bias_opposite"):
        _bias_opposite_sides(pick_actors)
        corridors = iter_corridors(env.task, spec)
    placed: list[np.ndarray] = []
    by_label = {label: actor for label, actor in pick_actors}
    for _p0, _p1, label, tgt_name in corridors:
        actor = by_label.get(label)
        if actor is None:
            continue
        if tgt_name in by_label and by_label[tgt_name] is actor:
            continue
        try:
            p1 = target_xy(env.task, tgt_name)
        except Exception:
            p1 = _p1
        before = _pose_xyz(actor)[:2]
        after = _stretch_one(actor, p1, min_dist, placed)
        placed.append(after)
        print(
            f"[spawn] stretch {label}: {before.round(3).tolist()} -> {after.round(3).tolist()} "
            f"vs {tgt_name} {np.asarray(p1[:2]).round(3).tolist()} "
            f"|d|={float(np.linalg.norm(after - p1)):.3f}"
        )


def _t_along(p0: np.ndarray, p1: np.ndarray, xy: np.ndarray) -> float | None:
    delta = np.asarray(p1[:2], dtype=np.float64) - np.asarray(p0[:2], dtype=np.float64)
    dist2 = float(np.dot(delta, delta))
    if dist2 < 1e-12:
        return None
    return float(np.dot(np.asarray(xy[:2], dtype=np.float64) - np.asarray(p0[:2], dtype=np.float64), delta) / dist2)


def choose_and_spawn(
    env,
    obstacle_mode: str,
    place_mode: str,
    corridor_t: float,
    arm: str,
    ee_path: np.ndarray | None = None,
    obstacle_model: str | None = None,
    obstacle_model_id: int = 0,
):
    """Spawn (or skip) the static safety obstacle. Returns (actor, xyz, half, arm)."""
    if obstacle_mode == "none":
        return None, None, None, arm
    model = set_obstacle_model(obstacle_model, obstacle_model_id)
    half_check = _world_half_extents()
    size_m = 2.0 * half_check
    if float(np.max(size_m)) > 0.45:
        print(
            f"[obstacle] {model} world size {size_m.round(3).tolist()} m looks huge; "
            "check scale in model_data (distance uses this OBB)"
        )
    spec = TASK_SPECS[env.task_name]
    arm = resolve_arm(env.task, spec, arm)
    p0, p1, arm_hint = longest_corridor(env.task, spec, env.robot, arm)
    if spec.get("corridors"):
        arm = arm_hint
    table_z = TABLE_Z + float(getattr(env.task, "table_z_bias", 0.0))
    task_keep = keepaways_from_task(env.task, spec)
    gap_range = spec.get("obstacle_bbox_gap")
    if gap_range is not None:
        lo, hi = float(gap_range[0]), float(gap_range[1])
        rng = np.random.RandomState(int(getattr(env, "seed", 0)) + 9183)
        extra = float(rng.uniform(lo, hi))
        task_keep = [(xy, float(r) + extra) for xy, r in task_keep]
        print(f"[obstacle] bbox gap +{extra:.3f} m (task {env.task_name})")
    robot_keep = keepaways_from_robot(env)
    keep = task_keep + robot_keep
    t_pref = float(np.clip(corridor_t, CORRIDOR_T_LO, CORRIDOR_T_HI))
    if keep:
        desc = ", ".join(f"{xy.round(3).tolist()} r={r:.3f}" for xy, r in keep)
        print(f"[obstacle] keepaway {desc}")
    if place_mode == "waypoint" and ee_path is not None and len(ee_path) >= 3:
        xyz, half = waypoint_pose(
            ee_path, obstacle_mode, t_pref, table_z, keep, p0, p1, robot_keep
        )
    else:
        if place_mode == "waypoint":
            print("[obstacle] waypoint path too short; using geometric")
        xyz, half = geometric_pose(
            p0, p1, obstacle_mode, t_pref, table_z, keep, robot_keep
        )
    if xyz is None:
        print(f"[obstacle] skipped spawn  {obstacle_mode} arm={arm} t={t_pref:.2f}")
        return None, None, None, arm
    t_used = _t_along(p0, p1, xyz)
    t_txt = "na" if t_used is None else f"{t_used:.2f}"
    print(
        f"[obstacle] place-mode={place_mode} {obstacle_mode} model={model} "
        f"p0={p0.round(3).tolist()} p1={p1.round(3).tolist()} "
        f"xyz={xyz.round(3).tolist()} arm={arm} t_pref={t_pref:.2f} t={t_txt}"
    )
    actor = spawn_obstacle(env.task, xyz, is_static=True)
    cfg = getattr(actor, "config", None) or _active_cfg
    _center, half_cfg = world_center_and_half(cfg)
    if float(np.max(np.abs(half_cfg))) > 1e-6:
        half = half_cfg
    size_cm = (2.0 * _world_half_extents() * 100.0).round(1).tolist()
    up = "XYZ"[_up_axis()]
    print(
        f"[obstacle] static upright {model} up={up} world_size_cm={size_cm} "
        f"half={np.asarray(half).round(4).tolist()}"
    )
    return actor, xyz, half, arm
