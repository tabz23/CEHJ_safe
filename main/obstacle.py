"""Spawn a static 086_woodenblock and optionally add it to CuRobo MotionGen."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from tasks import TASK_SPECS, resolve_arm, start_target_xy

OBSTACLE_NAME = "safety_obstacle"
OBSTACLE_MODEL = "086_woodenblock"
TABLE_Z = 0.74
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
    raw = np.asarray(cfg.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)
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

# t along object→target. Do not scale the mesh: create_actor uses model_data scale=1.
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
}


def patch_curobo_collision_cache() -> None:
    """Reserve extra cuboid slots so update_world can add the wooden block."""
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


def _model_center() -> np.ndarray:
    path = Path("assets/objects") / OBSTACLE_MODEL / "model_data0.json"
    if path.is_file():
        import json

        data = json.loads(path.read_text())
        return np.asarray(data.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
    return np.zeros(3)


def _model_half_extents() -> np.ndarray:
    path = Path("assets/objects") / OBSTACLE_MODEL / "model_data0.json"
    if path.is_file():
        import json

        data = json.loads(path.read_text())
        ext = np.asarray(data.get("extents", [0.1, 0.1, 0.1]), dtype=np.float64)
        return 0.5 * ext
    return np.array([0.051, 0.051, 0.052], dtype=np.float64)


def _clip_xy(xy: np.ndarray) -> np.ndarray:
    out = xy.copy()
    out[0] = np.clip(out[0], TABLE_XYLIM[0, 0], TABLE_XYLIM[0, 1])
    out[1] = np.clip(out[1], TABLE_XYLIM[1, 0], TABLE_XYLIM[1, 1])
    return out


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
    radius = _xy_radius_from_config(cfg) if cfg else DEFAULT_ACTOR_RADIUS
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


def _t_cap_before_targets(
    p0: np.ndarray,
    p1: np.ndarray,
    dist: float,
    keepaways: list[tuple[np.ndarray, float]],
    block_r: float,
    t_pref: float,
) -> float:
    """Largest t that stays block_r + target_r + gap short of keepaways near p1."""
    t_max = 0.92
    for pt, actor_r in keepaways:
        if np.linalg.norm(np.asarray(pt[:2], dtype=np.float64) - p1) > 0.04:
            continue
        need = block_r + float(actor_r) + KEEPAWAY_GAP
        if dist > 1e-6:
            t_max = min(t_max, 1.0 - need / dist)
    return float(np.clip(min(t_pref, t_max), 0.18, 0.92))


def _along_corridor(p0, delta, t, mode, sign, perp, off_m):
    cand = p0 + t * delta
    if mode == "off_path":
        cand = cand + sign * off_m * perp
    return _clip_xy(cand)


def geometric_pose(
    p0: np.ndarray,
    p1: np.ndarray,
    mode: str,
    level: int,
    table_z: float,
    keepaways: list[tuple[np.ndarray, float]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (xyz, half_extents), or (None, None) if every candidate overlaps a keepaway."""
    cfg = UNSAFE_LEVEL[int(level)]
    p0 = np.asarray(p0[:2], dtype=np.float64)
    p1 = np.asarray(p1[:2], dtype=np.float64)
    delta = p1 - p0
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:
        delta = np.array([0.0, -0.12], dtype=np.float64)
        dist = 0.12
    direction = delta / dist
    perp = np.array([-direction[1], direction[0]], dtype=np.float64)
    sign = 1.0 if p0[0] >= 0 else -1.0
    half = _model_half_extents()
    block_r = float(np.max(half[:2]))
    t0 = _t_cap_before_targets(p0, p1, dist, keepaways, block_r, cfg["t"])
    t_tries = [t0] + [x for x in (0.55, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20) if x < t0 - 1e-3]
    off_m = cfg["off_path_m"]
    xy = None
    for t_try in t_tries:
        cand = _along_corridor(p0, delta, t_try, mode, sign, perp, off_m)
        if _keepaway_ok(cand, keepaways, block_r):
            xy = cand
            break
    if xy is None and mode == "on_path":
        for nudge in (0.03, 0.05, 0.08):
            for s in (1.0, -1.0):
                cand = _clip_xy(p0 + t0 * delta + s * nudge * perp)
                if _keepaway_ok(cand, keepaways, block_r):
                    xy = cand
                    print(f"[obstacle] on_path keepaway nudge {s * nudge:.2f} m")
                    break
            if xy is not None:
                break
    if xy is None:
        print("[obstacle] no spawn pose clears pick/target keepaway; skipping block")
        return None, None
    z = table_z + half[2] + 0.002
    return np.array([xy[0], xy[1], z], dtype=np.float64), half


def waypoint_pose(
    ee_path: np.ndarray,
    mode: str,
    level: int,
    table_z: float,
    keepaways: list[tuple[np.ndarray, float]],
    p0: np.ndarray,
    p1: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Snap onto a recorded EE polyline, else fall back to geometric. Same keepaway as geometric."""
    cfg = UNSAFE_LEVEL[int(level)]
    pts = np.asarray(ee_path, dtype=np.float64).reshape(-1, 3)
    high = pts[pts[:, 2] > table_z + 0.04] if pts.size else pts
    if high.shape[0] < 3:
        return geometric_pose(p0, p1, mode, level, table_z, keepaways)
    t = cfg["t"]
    i = int(np.clip(t * (high.shape[0] - 1), 1, high.shape[0] - 2))
    xy = high[i, :2].copy()
    if mode == "off_path":
        tangent = high[min(i + 1, high.shape[0] - 1), :2] - high[max(i - 1, 0), :2]
        nrm = np.linalg.norm(tangent)
        if nrm < 1e-6:
            return geometric_pose(p0, p1, mode, level, table_z, keepaways)
        perp = np.array([-tangent[1], tangent[0]], dtype=np.float64) / nrm
        sign = 1.0 if np.asarray(p0[:2], dtype=np.float64)[0] >= 0 else -1.0
        xy = xy + sign * cfg["off_path_m"] * perp
    half = _model_half_extents()
    block_r = float(np.max(half[:2]))
    xy = _clip_xy(xy)
    if _keepaway_ok(xy, keepaways, block_r):
        z = table_z + half[2] + 0.002
        return np.array([xy[0], xy[1], z], dtype=np.float64), half
    print("[obstacle] waypoint snap overlaps a keepaway; using geometric")
    return geometric_pose(p0, p1, mode, level, table_z, keepaways)


def spawn_woodenblock(task, xyz: np.ndarray, is_static: bool = True):
    import sapien
    from envs.utils.create_actor import create_actor, create_box

    pose_xyz = np.asarray(xyz, dtype=np.float64) - _model_center()
    pose = sapien.Pose(pose_xyz.tolist(), [1, 0, 0, 0])
    actor = create_actor(
        task,
        pose,
        OBSTACLE_MODEL,
        convex=True,
        is_static=is_static,
        model_id=0,
    )
    if actor is None:
        half = _model_half_extents()
        actor = create_box(
            task,
            pose,
            half_size=half.tolist(),
            color=(0.55, 0.32, 0.12),
            is_static=is_static,
            name=OBSTACLE_NAME,
        )
        print("[obstacle] 086_woodenblock mesh missing; using create_box fallback")
    else:
        actor.actor.set_name(OBSTACLE_NAME)
    return actor


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


def _obstacle_in_base(planner, xyz, quat, half) -> dict:
    origin = planner.robot_origion_pose
    base = np.concatenate([np.asarray(origin.p), np.asarray(origin.q)])
    world = np.concatenate([np.asarray(xyz, dtype=np.float64), np.asarray(quat, dtype=np.float64)])
    p, q = planner._trans_from_world_to_base(base, world)
    bias = np.asarray(planner.frame_bias, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64) + bias
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


def load_collision_spheres(embodiment: str) -> dict:
    rel = COLLISION_YML[embodiment]
    path = Path("assets/embodiments") / rel
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data["collision_spheres"]


def keepaways_from_task(task, spec: dict) -> list[tuple[np.ndarray, float]]:
    """Keep the block off pick object, place target, and extra grasp objects."""
    names = []
    for key in ("object", "target"):
        name = spec.get(key)
        if name and name not in names:
            names.append(name)
    for extra in spec.get("grasp_objects", ()):
        if extra not in names:
            names.append(extra)
    out: list[tuple[np.ndarray, float]] = []
    seen = set()
    for name in names:
        if not hasattr(task, name):
            continue
        entry = _keepaway_entry(getattr(task, name))
        if entry is None:
            continue
        xy, radius = entry
        key = (round(float(xy[0]), 3), round(float(xy[1]), 3))
        if key in seen:
            continue
        seen.add(key)
        out.append((xy, radius))
    return out


def choose_and_spawn(
    env,
    obstacle_mode: str,
    place_mode: str,
    unsafe_level: int,
    arm: str,
    ee_path: np.ndarray | None = None,
):
    """Spawn (or skip) the wooden block. Returns (actor, xyz, half, arm)."""
    if obstacle_mode == "none":
        return None, None, None, arm
    spec = TASK_SPECS[env.task_name]
    arm = resolve_arm(env.task, spec, arm)
    p0, p1 = start_target_xy(env.task, spec, env.robot, arm)
    table_z = TABLE_Z + float(getattr(env.task, "table_z_bias", 0.0))
    keep = keepaways_from_task(env.task, spec)
    if keep:
        desc = ", ".join(f"{xy.round(3).tolist()} r={r:.3f}" for xy, r in keep)
        print(f"[obstacle] keepaway {desc}")
    if place_mode == "waypoint" and ee_path is not None and len(ee_path) >= 3:
        xyz, half = waypoint_pose(ee_path, obstacle_mode, unsafe_level, table_z, keep, p0, p1)
    else:
        if place_mode == "waypoint":
            print("[obstacle] waypoint path too short; using geometric")
        xyz, half = geometric_pose(p0, p1, obstacle_mode, unsafe_level, table_z, keep)
    if xyz is None:
        print(f"[obstacle] skipped spawn  {obstacle_mode} arm={arm} level={unsafe_level}")
        return None, None, None, arm
    print(
        f"[obstacle] place-mode={place_mode} {obstacle_mode} "
        f"p0={p0.round(3).tolist()} p1={p1.round(3).tolist()} "
        f"xyz={xyz.round(3).tolist()} arm={arm} level={unsafe_level}"
    )
    actor = spawn_woodenblock(env.task, xyz, is_static=True)
    return actor, xyz, half, arm
