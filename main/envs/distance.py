"""Min distance from all CuRobo collision spheres (both arms) to the obstacle OBB."""

from __future__ import annotations

import numpy as np

from .obstacle import OBSTACLE_NAME, load_collision_spheres, world_center_and_half


def _link_pose(link):
    if hasattr(link, "get_pose"):
        return link.get_pose()
    if hasattr(link, "pose"):
        return link.pose
    if hasattr(link, "entity"):
        return link.entity.get_pose()
    raise AttributeError(f"cannot read pose from {type(link)}")


def _entity_link(entity, name: str):
    if hasattr(entity, "find_link_by_name"):
        link = entity.find_link_by_name(name)
        if link is not None:
            return link
    for link in entity.get_links():
        n = link.get_name() if hasattr(link, "get_name") else getattr(link, "name", "")
        if n == name:
            return link
    return None


def sphere_world_positions(entity, spheres: dict) -> list[tuple[np.ndarray, float]]:
    out = []
    for link_name, specs in spheres.items():
        link = _entity_link(entity, link_name)
        if link is None:
            continue
        pose = _link_pose(link)
        t = np.asarray(pose.p, dtype=np.float64)
        rmat = pose.to_transformation_matrix()[:3, :3]
        for spec in specs:
            center = np.asarray(spec["center"], dtype=np.float64)
            radius = float(spec["radius"])
            out.append((t + rmat @ center, radius))
    return out


def _resolve_actor(actor):
    """Return something with get_pose() (RoboTwin Actor wrapper or SAPIEN entity)."""
    if actor is None or isinstance(actor, str):
        raise TypeError(f"expected actor object, got {type(actor).__name__}")
    if hasattr(actor, "get_pose"):
        return actor
    inner = getattr(actor, "actor", None)
    if inner is not None and hasattr(inner, "get_pose"):
        return inner
    raise TypeError(f"cannot read pose from {type(actor).__name__}")


def _obb_from_actor(actor, half=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = _resolve_actor(actor).get_pose()
    rot = pose.to_transformation_matrix()[:3, :3]
    cfg = getattr(actor, "config", None) or {}
    center, auto_half = world_center_and_half(cfg)
    origin = np.asarray(pose.p, dtype=np.float64) + rot @ center
    if half is not None:
        half_arr = np.asarray(half, dtype=np.float64)
        # Prefer JSON-scaled AABB if a caller passed unscaled model-unit extents (~1 m).
        if (
            float(np.max(np.abs(auto_half))) > 1e-6
            and float(np.max(np.abs(half_arr))) > 0.4
            and float(np.max(np.abs(auto_half))) < 0.3
        ):
            return origin, rot, auto_half
        return origin, rot, half_arr
    return origin, rot, auto_half


def point_obb_signed_distance(point: np.ndarray, origin: np.ndarray, rot: np.ndarray, half: np.ndarray) -> float:
    """Positive = outside OBB, negative = penetration depth of the point."""
    local = rot.T @ (point - origin)
    delta = np.abs(local) - half
    if np.all(delta <= 0.0):
        return float(np.max(delta))
    return float(np.linalg.norm(np.maximum(delta, 0.0)))


def _robot_link_names(robot) -> set[str]:
    names = set()
    for entity in (robot.left_entity, robot.right_entity):
        for link in entity.get_links():
            n = link.get_name() if hasattr(link, "get_name") else getattr(link, "name", "")
            if n:
                names.add(n)
        if hasattr(entity, "get_name"):
            names.add(entity.get_name())
    return names


def min_robot_obstacle_distance(env) -> tuple[float, bool, str]:
    """Return (d_robot, robot_obstacle_contact, closest_label). Cup is not included."""
    info = distance_info(env)
    return info["d_robot"], info["contact"], info["closest"]


def _finite_min(vals) -> float:
    nums = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(min(nums)) if nums else float("inf")


ALOHA_LINK_PREFIXES = {
    "left": ("fl_", "left_"),
    "right": ("fr_", "right_"),
}


def _arm_collision_spheres(env, side: str, spheres: dict) -> dict:
    """Return one arm's spheres for native bimanual Aloha only.

    The four existing embodiments have separate left/right articulations and
    intentionally keep receiving their original complete sphere dictionary.
    Aloha has one articulation and one combined collision file, so each arm
    must instead be selected by its fl_*/fr_* link prefix.
    """
    if getattr(env, "embodiment", "") != "aloha-agilex":
        return spheres
    prefixes = ALOHA_LINK_PREFIXES[side]
    selected = {name: specs for name, specs in spheres.items() if name.startswith(prefixes)}
    if not selected:
        raise ValueError(f"No aloha-agilex {side} collision spheres found")
    return selected


def _arm_sphere_distance(env, side: str, origin, rot, half, spheres) -> tuple[float, str]:
    entity = env.robot.left_entity if side == "left" else env.robot.right_entity
    prefix = "L" if side == "left" else "R"
    best = float("inf")
    closest = ""
    arm_spheres = _arm_collision_spheres(env, side, spheres)
    for center, radius, link_name in spheres_with_names(entity, arm_spheres):
        d = point_obb_signed_distance(center, origin, rot, half) - radius
        if d < best:
            best = d
            closest = f"{prefix}/{link_name}"
    return best, closest


def distance_info(env) -> dict:
    """Per-arm robot and held-object distances to the wooden block."""
    held = detect_held_by_arm(env)
    hold_dbg = dict(getattr(env, "_cehj_hold_debug", {}) or {})
    left_actor, left_label = held.get("left") or (None, "")
    right_actor, right_label = held.get("right") or (None, "")
    holding = _holding_text(left_label, right_label)
    out = {
        "d_left": float("inf"),
        "d_right": float("inf"),
        "d_left_held": float("inf"),
        "d_right_held": float("inf"),
        "d_min": float("inf"),
        "d_robot": float("inf"),
        "d_held": float("inf"),
        "d_system": float("inf"),
        "contact": False,
        "closest": "",
        "holding": holding,
        "holding_left": left_label,
        "holding_right": right_label,
        "held_actor": left_actor or right_actor,
        "held_left": left_actor,
        "held_right": right_actor,
        "hold_debug": hold_dbg,
    }
    obstacle = getattr(env, "obstacle", None)
    if obstacle is None:
        return out
    spheres = load_collision_spheres(env.embodiment)
    origin, rot, half = _obb_from_actor(obstacle, getattr(env, "obstacle_half", None))
    d_left, closest_l = _arm_sphere_distance(env, "left", origin, rot, half, spheres)
    d_right, closest_r = _arm_sphere_distance(env, "right", origin, rot, half, spheres)
    d_left_held = float("inf")
    d_right_held = float("inf")
    closest = closest_l if d_left <= d_right else closest_r
    if left_actor is not None:
        try:
            d_left_held = _actor_obb_distance(left_actor, origin, rot, half)
            if d_left_held < _finite_min([d_left, d_right, d_right_held]):
                closest = f"held_L/{left_label}"
        except (TypeError, AttributeError):
            d_left_held = float("inf")
            left_actor, left_label = None, ""
    if right_actor is not None:
        try:
            d_right_held = _actor_obb_distance(right_actor, origin, rot, half)
            if d_right_held < _finite_min([d_left, d_right, d_left_held]):
                closest = f"held_R/{right_label}"
        except (TypeError, AttributeError):
            d_right_held = float("inf")
            right_actor, right_label = None, ""
    d_robot = _finite_min([d_left, d_right])
    d_held = _finite_min([d_left_held, d_right_held])
    d_min = _finite_min([d_left, d_right, d_left_held, d_right_held])
    # max robot<->obstacle contact force this physics step (N;  0 = no touch).
    # `contact` keeps the old boolean semantics (direct OR via held object)
    contact_force = _robot_obstacle_contact_force(env)
    contact = contact_force > 0.0
    if not contact and left_actor is not None:
        contact = _held_obstacle_contact(env, left_actor)
    if not contact and right_actor is not None:
        contact = _held_obstacle_contact(env, right_actor)
    holding = _holding_text(left_label, right_label)
    out.update(
        {
            "d_left": d_left,
            "d_right": d_right,
            "d_left_held": d_left_held,
            "d_right_held": d_right_held,
            "d_min": d_min,
            "d_robot": d_robot,
            "d_held": d_held,
            "d_system": d_min,
            "contact": contact,
            "contact_force": contact_force,
            "closest": closest,
            "holding": holding,
            "holding_left": left_label,
            "holding_right": right_label,
            "held_actor": left_actor or right_actor,
            "held_left": left_actor,
            "held_right": right_actor,
        }
    )
    return out


def is_gripper_link(name: str, embodiment: str | None = None) -> bool:
    n = (name or "").lower()
    if n in ("link7", "link8"):
        return True
    if embodiment == "aloha-agilex" and n.startswith(("fl_", "fr_")):
        return n.rsplit("_", 1)[-1] in ("link7", "link8")
    return any(tag in n for tag in ("finger", "hand", "gripper", "robotiq"))


def _aloha_side_from_spheres(spheres: dict) -> str | None:
    """If `spheres` is already filtered to one Aloha arm, return that side.

    Aloha is a single dual-arm articulation, so get_links() includes both
    grippers. The missing-yml fallback must not attach the other arm's fingers.
    Other embodiments keep mixed / unprefixed keys and return None.
    """
    has_l = any(k.startswith(("fl_", "left_")) for k in spheres)
    has_r = any(k.startswith(("fr_", "right_")) for k in spheres)
    if has_l and not has_r:
        return "left"
    if has_r and not has_l:
        return "right"
    return None


def spheres_with_names(entity, spheres: dict):
    out = []
    seen = set()
    for link_name, specs in spheres.items():
        link = _entity_link(entity, link_name)
        if link is None:
            continue
        seen.add(link_name)
        pose = _link_pose(link)
        t = np.asarray(pose.p, dtype=np.float64)
        rmat = pose.to_transformation_matrix()[:3, :3]
        for spec in specs:
            center = t + rmat @ np.asarray(spec["center"], dtype=np.float64)
            out.append((center, float(spec["radius"]), link_name))
    # Grasp links missing from the CuRobo yml (e.g. ARX link7/8).
    try:
        links = entity.get_links()
    except Exception:
        links = []
    embodiment = None
    if any(name.startswith(("fl_", "fr_")) for name in seen):
        embodiment = "aloha-agilex"
    aloha_side = _aloha_side_from_spheres(spheres)
    aloha_prefix = ALOHA_LINK_PREFIXES.get(aloha_side) if aloha_side else None
    for link in links:
        name = link.get_name() if hasattr(link, "get_name") else getattr(link, "name", "")
        if not name or name in seen or not is_gripper_link(name, embodiment):
            continue
        if aloha_prefix is not None and not name.startswith(aloha_prefix):
            continue
        pose = _link_pose(link)
        out.append((np.asarray(pose.p, dtype=np.float64), 0.018, name))
    return out


def _actor_entity_name(actor) -> str:
    inner = getattr(actor, "actor", actor)
    if hasattr(inner, "get_name"):
        return inner.get_name() or ""
    return getattr(inner, "name", "") or ""


def _entity_ids(actor) -> set[int]:
    """Identity keys for a RoboTwin Actor wrapper or SAPIEN entity."""
    ids: set[int] = set()
    cur = actor
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        ids.add(id(cur))
        ent = getattr(cur, "entity", None)
        if ent is not None:
            ids.add(id(ent))
        nxt = getattr(cur, "actor", None)
        if nxt is cur:
            break
        cur = nxt
    return ids


def _holding_text(left_label: str, right_label: str) -> str:
    parts = []
    if left_label:
        parts.append(f"L={left_label}")
    if right_label:
        parts.append(f"R={right_label}")
    return " ".join(parts)


def _iter_task_actors(env):
    from .tasks import TASK_SPECS, iter_named_actors, spec_grasp_names

    spec = TASK_SPECS.get(getattr(env, "task_name", ""), {})
    obstacle = getattr(env, "obstacle", None)
    for name, actor in iter_named_actors(env.task, spec_grasp_names(spec)):
        if actor is None or actor is obstacle:
            continue
        try:
            _resolve_actor(actor)
        except TypeError:
            continue
        yield name, actor


def _entity_side_lookup(env) -> dict[int, str]:
    """Map SAPIEN entity id → left/right."""
    cache = getattr(env, "_cehj_entity_side", None)
    if cache is not None:
        return cache
    cache: dict[int, str] = {}
    if getattr(env, "embodiment", "") == "aloha-agilex":
        # Aloha's left_entity and right_entity are the same articulation.
        # Assign each link entity from its unambiguous fl_*/fr_* prefix.
        for link in env.robot.left_entity.get_links():
            name = link.get_name() if hasattr(link, "get_name") else getattr(link, "name", "")
            side = next(
                (candidate for candidate, prefixes in ALOHA_LINK_PREFIXES.items() if name.startswith(prefixes)),
                None,
            )
            ent = getattr(link, "entity", None)
            if side is not None and ent is not None:
                cache[id(ent)] = side
        env._cehj_entity_side = cache
        return cache

    # Preserve the original separate-articulation lookup for every existing
    # embodiment.
    for side, art in (("left", env.robot.left_entity), ("right", env.robot.right_entity)):
        cache[id(art)] = side
        for link in art.get_links():
            ent = getattr(link, "entity", None)
            if ent is not None:
                cache[id(ent)] = side
    env._cehj_entity_side = cache
    return cache


def _entity_side(env, entity) -> str | None:
    if entity is None:
        return None
    return _entity_side_lookup(env).get(id(entity))


def _arm_is_holding(env, side: str) -> bool:
    try:
        if side == "left":
            if env.robot.is_left_gripper_close():
                return True
            return not env.task.is_left_gripper_open()
        if env.robot.is_right_gripper_close():
            return True
        return not env.task.is_right_gripper_open()
    except Exception:
        return False


def _is_gripper_entity(env, entity) -> bool:
    name = entity.name if hasattr(entity, "name") else ""
    gripper = set(getattr(env.robot, "gripper_name", []) or [])
    return name in gripper or is_gripper_link(name, getattr(env, "embodiment", None))


def _detect_held_from_contacts(env, items: list[tuple[str, object, set]]) -> dict[str, tuple]:
    found: dict[str, tuple] = {}
    try:
        contacts = env.task.scene.get_contacts()
    except Exception:
        return found
    claimed: set[int] = set()
    for contact in contacts:
        bodies = contact.bodies
        for i in range(2):
            obj_ent = bodies[1 - i].entity
            oid = id(obj_ent)
            match = None
            for label, actor, keys in items:
                if oid in keys:
                    match = (label, actor, keys)
                    break
            if match is None:
                continue
            gripper_body = bodies[i]
            if not _is_gripper_entity(env, gripper_body.entity):
                continue
            side = _entity_side(env, gripper_body.entity)
            if side is None or not _arm_is_holding(env, side):
                continue
            if side in found:
                continue
            label, actor, keys = match
            if claimed & keys:
                continue
            found[side] = (actor, label)
            claimed |= keys
    return found


def _detect_held_from_near(env, items: list[tuple[str, object, set]], claimed: set[int] | None = None):
    """Closed gripper whose TCP/fingers sit on a task-object OBB (no PhysX contact needed)."""
    claimed = set(claimed or ())
    found: dict[str, tuple] = {}
    for side in ("left", "right"):
        if not _arm_is_holding(env, side):
            continue
        best = None
        best_d = float("inf")
        for label, actor, keys in items:
            if claimed & keys:
                continue
            metrics = _hold_metrics(env, actor, side)
            if not metrics["near"]:
                continue
            dist = metrics["grip_obb"] if np.isfinite(metrics["grip_obb"]) else metrics["tcp_obb"]
            if not np.isfinite(dist):
                dist = metrics["tcp_obb"]
            if dist < best_d:
                best = (actor, label, metrics, keys)
                best_d = dist
        if best is not None:
            actor, label, metrics, keys = best
            found[side] = (actor, label, metrics)
            claimed |= keys
    return found


# Stay/slip uses TCP / gripper spheres vs the object OBB, not wrist-origin Euclidean.
# Wrist EE is often >10 cm from the object origin while the object is still grasped.
HOLD_TCP_TO_OBB_MAX = 0.04  # TCP may sit in the finger gap, slightly outside the box
HOLD_GRIPPER_TO_OBB_MAX = 0.03  # finger spheres still wrapping the object


def _pose_xyz(pose) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    return arr[:3]


def _ee_position(env, side: str) -> np.ndarray:
    pose = env.robot.get_left_ee_pose() if side == "left" else env.robot.get_right_ee_pose()
    return _pose_xyz(pose)


def _tcp_position(env, side: str) -> np.ndarray:
    if side == "left":
        pose = env.robot.get_left_tcp_pose() if hasattr(env.robot, "get_left_tcp_pose") else env.robot.get_left_ee_pose()
    else:
        pose = env.robot.get_right_tcp_pose() if hasattr(env.robot, "get_right_tcp_pose") else env.robot.get_right_ee_pose()
    return _pose_xyz(pose)


def _gripper_value(env, side: str) -> float:
    try:
        if side == "left":
            return float(env.robot.get_left_gripper_val())
        return float(env.robot.get_right_gripper_val())
    except Exception:
        return float("nan")


def _hold_metrics(env, actor, side: str) -> dict:
    """Geometry of one arm vs one actor. Distances in meters; signed OBB: + outside, - inside."""
    out = {
        "near": False,
        "near_reason": "",
        "origin_dist_ee": float("nan"),
        "origin_dist_tcp": float("nan"),
        "tcp_obb": float("nan"),
        "ee_obb": float("nan"),
        "grip_obb": float("nan"),
        "ee_x": float("nan"),
        "ee_y": float("nan"),
        "ee_z": float("nan"),
        "tcp_x": float("nan"),
        "tcp_y": float("nan"),
        "tcp_z": float("nan"),
        "obj_x": float("nan"),
        "obj_y": float("nan"),
        "obj_z": float("nan"),
    }
    try:
        origin, rot, half = _obb_from_actor(actor)
        obj_p = np.asarray(_resolve_actor(actor).get_pose().p, dtype=np.float64)
        ee = _ee_position(env, side)
        tcp = _tcp_position(env, side)
        out["obj_x"], out["obj_y"], out["obj_z"] = (float(obj_p[0]), float(obj_p[1]), float(obj_p[2]))
        out["ee_x"], out["ee_y"], out["ee_z"] = (float(ee[0]), float(ee[1]), float(ee[2]))
        out["tcp_x"], out["tcp_y"], out["tcp_z"] = (float(tcp[0]), float(tcp[1]), float(tcp[2]))
        out["origin_dist_ee"] = float(np.linalg.norm(obj_p - ee))
        out["origin_dist_tcp"] = float(np.linalg.norm(obj_p - tcp))
        out["ee_obb"] = float(point_obb_signed_distance(ee, origin, rot, half))
        out["tcp_obb"] = float(point_obb_signed_distance(tcp, origin, rot, half))
        entity = env.robot.left_entity if side == "left" else env.robot.right_entity
        grip = float("inf")
        spheres = _arm_collision_spheres(
            env, side, load_collision_spheres(env.embodiment)
        )
        for center, radius, name in spheres_with_names(entity, spheres):
            if not is_gripper_link(name, env.embodiment):
                continue
            d = point_obb_signed_distance(center, origin, rot, half) - radius
            if d < grip:
                grip = d
        if np.isfinite(grip):
            out["grip_obb"] = float(grip)
        if np.isfinite(grip) and grip <= HOLD_GRIPPER_TO_OBB_MAX:
            out["near"] = True
            out["near_reason"] = "grip_obb"
        elif out["tcp_obb"] <= HOLD_TCP_TO_OBB_MAX:
            out["near"] = True
            out["near_reason"] = "tcp_obb"
        else:
            out["near_reason"] = "far"
    except Exception:
        out["near_reason"] = "geom_error"
    return out


def _empty_hold_debug() -> dict:
    return {
        "source": "none",
        "fail": "",
        "side": "",
        "holding": "",
        "contact_grip_obj": False,
        "left_holding": False,
        "right_holding": False,
        "left_grip": float("nan"),
        "right_grip": float("nan"),
        "near": False,
        "near_reason": "",
        "origin_dist_ee": float("nan"),
        "origin_dist_tcp": float("nan"),
        "tcp_obb": float("nan"),
        "ee_obb": float("nan"),
        "grip_obb": float("nan"),
        "L_origin_dist_ee": float("nan"),
        "L_origin_dist_tcp": float("nan"),
        "L_tcp_obb": float("nan"),
        "L_grip_obb": float("nan"),
        "L_near": False,
        "L_source": "none",
        "L_fail": "",
        "R_origin_dist_ee": float("nan"),
        "R_origin_dist_tcp": float("nan"),
        "R_tcp_obb": float("nan"),
        "R_grip_obb": float("nan"),
        "R_near": False,
        "R_source": "none",
        "R_fail": "",
        "holding_left": "",
        "holding_right": "",
        "obj_x": float("nan"),
        "obj_y": float("nan"),
        "obj_z": float("nan"),
        "tcp_x": float("nan"),
        "tcp_y": float("nan"),
        "tcp_z": float("nan"),
        "ee_x": float("nan"),
        "ee_y": float("nan"),
        "ee_z": float("nan"),
    }


def _fill_both_arm_metrics(env, actor, debug: dict) -> None:
    if actor is None:
        return
    for side, prefix in (("left", "L"), ("right", "R")):
        m = _hold_metrics(env, actor, side)
        debug[f"{prefix}_origin_dist_ee"] = m["origin_dist_ee"]
        debug[f"{prefix}_origin_dist_tcp"] = m["origin_dist_tcp"]
        debug[f"{prefix}_tcp_obb"] = m["tcp_obb"]
        debug[f"{prefix}_grip_obb"] = m["grip_obb"]
        debug[f"{prefix}_near"] = m["near"]


def _apply_side_metrics(debug: dict, metrics: dict) -> None:
    for key in (
        "near",
        "near_reason",
        "origin_dist_ee",
        "origin_dist_tcp",
        "tcp_obb",
        "ee_obb",
        "grip_obb",
        "ee_x",
        "ee_y",
        "ee_z",
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "obj_x",
        "obj_y",
        "obj_z",
    ):
        debug[key] = metrics[key]


def detect_held_by_arm(env) -> dict[str, tuple | None]:
    """Per-arm latch: PhysX contact starts it; stay while closed and TCP/fingers on OBB."""
    debug = _empty_hold_debug()
    debug["left_holding"] = _arm_is_holding(env, "left")
    debug["right_holding"] = _arm_is_holding(env, "right")
    debug["left_grip"] = _gripper_value(env, "left")
    debug["right_grip"] = _gripper_value(env, "right")

    items: list[tuple[str, object, set]] = []
    for name, actor in _iter_task_actors(env):
        items.append((name, actor, _entity_ids(actor)))

    latch = getattr(env, "_cehj_held_latch", None)
    if not isinstance(latch, dict) or ("left" not in latch and "right" not in latch):
        latch = {"left": None, "right": None}

    result: dict[str, tuple | None] = {"left": None, "right": None}

    def _finish():
        debug["holding_left"] = result["left"][1] if result["left"] else ""
        debug["holding_right"] = result["right"][1] if result["right"] else ""
        debug["holding"] = _holding_text(debug["holding_left"], debug["holding_right"])
        debug["side"] = "both" if result["left"] and result["right"] else (
            "left" if result["left"] else ("right" if result["right"] else "")
        )
        env._cehj_hold_debug = debug
        env._cehj_held_latch = {
            "left": None if result["left"] is None else {
                "actor": result["left"][0],
                "label": result["left"][1],
                "side": "left",
            },
            "right": None if result["right"] is None else {
                "actor": result["right"][0],
                "label": result["right"][1],
                "side": "right",
            },
        }
        return result

    if not debug["left_holding"] and not debug["right_holding"]:
        debug["fail"] = "grippers_open"
        debug["L_fail"] = "arm_open"
        debug["R_fail"] = "arm_open"
        return _finish()

    if not items:
        debug["fail"] = "no_actors"
        return _finish()

    contacts = _detect_held_from_contacts(env, items)
    debug["contact_grip_obj"] = bool(contacts)
    claimed: set[int] = set()
    for side in ("left", "right"):
        prefix = "L" if side == "left" else "R"
        if not debug[f"{side}_holding"]:
            debug[f"{prefix}_fail"] = "arm_open"
            debug[f"{prefix}_source"] = "none"
            continue
        if side in contacts:
            actor, label = contacts[side]
            metrics = _hold_metrics(env, actor, side)
            result[side] = (actor, label)
            claimed |= _entity_ids(actor)
            debug[f"{prefix}_source"] = "contact"
            debug[f"{prefix}_fail"] = ""
            if side == "left":
                _apply_side_metrics(debug, metrics)
            for key, val in (
                ("origin_dist_ee", metrics["origin_dist_ee"]),
                ("origin_dist_tcp", metrics["origin_dist_tcp"]),
                ("tcp_obb", metrics["tcp_obb"]),
                ("grip_obb", metrics["grip_obb"]),
                ("near", metrics["near"]),
            ):
                debug[f"{prefix}_{key}"] = val
            continue
        prev = latch.get(side)
        if prev is not None:
            metrics = _hold_metrics(env, prev["actor"], side)
            debug[f"{prefix}_near"] = metrics["near"]
            debug[f"{prefix}_tcp_obb"] = metrics["tcp_obb"]
            debug[f"{prefix}_grip_obb"] = metrics["grip_obb"]
            debug[f"{prefix}_origin_dist_ee"] = metrics["origin_dist_ee"]
            debug[f"{prefix}_origin_dist_tcp"] = metrics["origin_dist_tcp"]
            if metrics["near"]:
                result[side] = (prev["actor"], prev["label"])
                claimed |= _entity_ids(prev["actor"])
                debug[f"{prefix}_source"] = "latch"
                debug[f"{prefix}_fail"] = ""
                if side == "left":
                    _apply_side_metrics(debug, metrics)
                continue
            debug[f"{prefix}_fail"] = "not_near"
            debug[f"{prefix}_source"] = "none"

    near_hits = _detect_held_from_near(env, items, claimed)
    for side, hit in near_hits.items():
        if result[side] is not None:
            continue
        actor, label, metrics = hit
        prefix = "L" if side == "left" else "R"
        result[side] = (actor, label)
        claimed |= _entity_ids(actor)
        debug[f"{prefix}_source"] = "near"
        debug[f"{prefix}_fail"] = ""
        debug[f"{prefix}_near"] = metrics["near"]
        debug[f"{prefix}_tcp_obb"] = metrics["tcp_obb"]
        debug[f"{prefix}_grip_obb"] = metrics["grip_obb"]
        debug[f"{prefix}_origin_dist_ee"] = metrics["origin_dist_ee"]
        debug[f"{prefix}_origin_dist_tcp"] = metrics["origin_dist_tcp"]
        if side == "left":
            _apply_side_metrics(debug, metrics)

    for side in ("left", "right"):
        if result[side] is not None or not debug[f"{side}_holding"]:
            continue
        other = "right" if side == "left" else "left"
        if result[other] is None:
            continue
        actor, label = result[other]
        metrics = _hold_metrics(env, actor, side)
        if not metrics["near"]:
            continue
        prefix = "L" if side == "left" else "R"
        result[side] = (actor, label)
        debug[f"{prefix}_source"] = "share"
        debug[f"{prefix}_fail"] = ""
        debug[f"{prefix}_near"] = True
        debug[f"{prefix}_tcp_obb"] = metrics["tcp_obb"]
        debug[f"{prefix}_grip_obb"] = metrics["grip_obb"]
        debug[f"{prefix}_origin_dist_ee"] = metrics["origin_dist_ee"]
        debug[f"{prefix}_origin_dist_tcp"] = metrics["origin_dist_tcp"]

    for side in ("left", "right"):
        prefix = "L" if side == "left" else "R"
        actor = result[side][0] if result[side] else None
        if actor is None and items:
            actor = items[0][1]
        if actor is not None and not np.isfinite(debug.get(f"{prefix}_tcp_obb", float("nan"))):
            m = _hold_metrics(env, actor, side)
            debug[f"{prefix}_origin_dist_ee"] = m["origin_dist_ee"]
            debug[f"{prefix}_origin_dist_tcp"] = m["origin_dist_tcp"]
            debug[f"{prefix}_tcp_obb"] = m["tcp_obb"]
            debug[f"{prefix}_grip_obb"] = m["grip_obb"]
            debug[f"{prefix}_near"] = m["near"]
        if result[side] is None and debug[f"{side}_holding"] and not debug.get(f"{prefix}_fail"):
            debug[f"{prefix}_fail"] = "no_contact"
            debug[f"{prefix}_source"] = "none"

    if result["left"] is None and result["right"] is None:
        debug["fail"] = debug.get("L_fail") or debug.get("R_fail") or "no_contact"
        debug["source"] = "none"
    elif result["left"] and result["right"]:
        debug["source"] = "both"
        debug["fail"] = ""
    else:
        side = "left" if result["left"] else "right"
        prefix = "L" if side == "left" else "R"
        debug["source"] = debug.get(f"{prefix}_source") or "none"
        debug["fail"] = ""
        debug["near"] = debug.get(f"{prefix}_near")
        debug["tcp_obb"] = debug.get(f"{prefix}_tcp_obb")
        debug["grip_obb"] = debug.get(f"{prefix}_grip_obb")
    return _finish()


def detect_held_object(env) -> tuple[object | None, str]:
    """Return one (actor, label) for overlays that still expect a single hold."""
    held = detect_held_by_arm(env)
    if held.get("left"):
        return held["left"]
    if held.get("right"):
        return held["right"]
    return None, ""


def _actor_obb_distance(actor, origin, rot, half) -> float:
    a_o, a_r, a_h = _obb_from_actor(actor)
    d_min = float("inf")
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx * a_h[0], sy * a_h[1], sz * a_h[2]])
                d_min = min(d_min, point_obb_signed_distance(a_o + a_r @ local, origin, rot, half))
                local_b = np.array([sx * half[0], sy * half[1], sz * half[2]])
                d_min = min(d_min, point_obb_signed_distance(origin + rot @ local_b, a_o, a_r, a_h))
    return d_min


def _held_obstacle_contact(env, held_actor) -> bool:
    held_ids = _entity_ids(held_actor)
    try:
        contacts = env.task.scene.get_contacts()
    except Exception:
        return False
    for contact in contacts:
        ents = [contact.bodies[0].entity, contact.bodies[1].entity]
        names = [getattr(e, "name", "") or "" for e in ents]
        if OBSTACLE_NAME not in names:
            continue
        for ent in ents:
            if getattr(ent, "name", "") == OBSTACLE_NAME:
                continue
            if id(ent) in held_ids:
                return True
    return False


def _robot_obstacle_contact(env) -> bool:
    return _robot_obstacle_contact_force(env) > 0.0


def _robot_obstacle_contact_force(env) -> float:
    """Max contact force (N) between any robot link and the obstacle this
    physics step;  0.0 when untouched. impulse/dt per contact point."""
    robot_names = _robot_link_names(env.robot)
    max_f = 0.0
    try:
        contacts = env.task.scene.get_contacts()
    except Exception:
        return 0.0

    dt = float(getattr(env.task.scene, "timestep", 1.0 / 250.0))
    for contact in contacts:
        names = [contact.bodies[0].entity.name, contact.bodies[1].entity.name]
        if OBSTACLE_NAME not in names:
            continue
        other = names[0] if names[1] == OBSTACLE_NAME else names[1]
        if other == OBSTACLE_NAME:
            continue
        if other in robot_names or any(
            tag in other.lower()
            for tag in ("piper", "panda", "franka", "ur5", "arx", "gripper", "finger")
        ):
            for p in contact.points:
                f = float(np.linalg.norm(p.impulse)) / dt
                if f > max_f:
                    max_f = f
    return max_f


def obstacle_corners(actor, half=None) -> np.ndarray:
    origin, rot, half = _obb_from_actor(actor, half)
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]])
                corners.append(origin + rot @ local)
    return np.asarray(corners, dtype=np.float64)


def sphere_aabb_corners(center: np.ndarray, radius: float) -> np.ndarray:
    r = np.array([radius, radius, radius])
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(center + np.array([sx, sy, sz]) * r)
    return np.asarray(corners, dtype=np.float64)
