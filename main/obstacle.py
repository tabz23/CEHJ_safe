"""Spawn a static 086_woodenblock and optionally add it to CuRobo MotionGen."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from tasks import (
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
    level: int,
    table_z: float,
    keepaways: list[tuple[np.ndarray, float]],
    robot_keepaways: list[tuple[np.ndarray, float]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (xyz, half_extents), or (None, None) if every candidate overlaps a keepaway.

    off_path may sit anywhere on the free table (not glued to the object→target
    segment). on_path stays on that segment when possible, then accepts a nearby
    free cell rather than skipping.
    """
    cfg = UNSAFE_LEVEL[int(level)]
    p0 = np.asarray(p0[:2], dtype=np.float64)
    p1 = np.asarray(p1[:2], dtype=np.float64)
    robot_keepaways = list(robot_keepaways or ())
    delta = p1 - p0
    dist = float(np.linalg.norm(delta))
    if dist < 1e-4:
        delta = np.array([0.0, -0.12], dtype=np.float64)
        dist = 0.12
    perp = np.array([-delta[1], delta[0]], dtype=np.float64) / dist
    half = _model_half_extents()
    block_r = float(np.max(half[:2]))
    t0 = _t_cap_before_targets(p0, p1, dist, keepaways, block_r, cfg["t"])
    t_tries: list[float] = []
    for t in (t0, 0.55, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.18, 0.65, 0.75):
        if t not in t_tries:
            t_tries.append(float(t))
    signs = _perp_signs(p0, delta, perp, robot_keepaways)
    off_pref = cfg["off_path_m"]
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
    level: int,
    table_z: float,
    keepaways: list[tuple[np.ndarray, float]],
    p0: np.ndarray,
    p1: np.ndarray,
    robot_keepaways: list[tuple[np.ndarray, float]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Snap onto a recorded EE polyline, else fall back to geometric. Same keepaway as geometric."""
    cfg = UNSAFE_LEVEL[int(level)]
    pts = np.asarray(ee_path, dtype=np.float64).reshape(-1, 3)
    high = pts[pts[:, 2] > table_z + 0.04] if pts.size else pts
    if high.shape[0] < 3:
        return geometric_pose(p0, p1, mode, level, table_z, keepaways, robot_keepaways)
    t = cfg["t"]
    i = int(np.clip(t * (high.shape[0] - 1), 1, high.shape[0] - 2))
    xy = high[i, :2].copy()
    if mode == "off_path":
        tangent = high[min(i + 1, high.shape[0] - 1), :2] - high[max(i - 1, 0), :2]
        nrm = np.linalg.norm(tangent)
        if nrm < 1e-6:
            return geometric_pose(p0, p1, mode, level, table_z, keepaways, robot_keepaways)
        perp = np.array([-tangent[1], tangent[0]], dtype=np.float64) / nrm
        signs = _perp_signs(np.asarray(p0[:2], dtype=np.float64), tangent, perp, robot_keepaways or [])
        xy = xy + signs[0] * cfg["off_path_m"] * perp
    half = _model_half_extents()
    block_r = float(np.max(half[:2]))
    if _in_table(xy, block_r) and _keepaway_ok(xy, keepaways, block_r):
        return _pack_xyz(xy, table_z, half), half
    print("[obstacle] waypoint snap overlaps a keepaway; using geometric")
    return geometric_pose(p0, p1, mode, level, table_z, keepaways, robot_keepaways)


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
    for name, actor in iter_named_actors(task, names):
        entry = _keepaway_entry(actor)
        if entry is None:
            continue
        xy, radius = entry
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
                out.append((xy, 0.06))
        except Exception:
            pass
    return out


STRETCH_XYLIM = np.array([[-0.30, 0.30], [-0.22, 0.16]], dtype=np.float64)
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
    p0, p1, arm_hint = longest_corridor(env.task, spec, env.robot, arm)
    if spec.get("corridors"):
        arm = arm_hint
    table_z = TABLE_Z + float(getattr(env.task, "table_z_bias", 0.0))
    task_keep = keepaways_from_task(env.task, spec)
    robot_keep = keepaways_from_robot(env)
    keep = task_keep + robot_keep
    if keep:
        desc = ", ".join(f"{xy.round(3).tolist()} r={r:.3f}" for xy, r in keep)
        print(f"[obstacle] keepaway {desc}")
    if place_mode == "waypoint" and ee_path is not None and len(ee_path) >= 3:
        xyz, half = waypoint_pose(
            ee_path, obstacle_mode, unsafe_level, table_z, keep, p0, p1, robot_keep
        )
    else:
        if place_mode == "waypoint":
            print("[obstacle] waypoint path too short; using geometric")
        xyz, half = geometric_pose(
            p0, p1, obstacle_mode, unsafe_level, table_z, keep, robot_keep
        )
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
