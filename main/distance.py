"""Min distance from all CuRobo collision spheres (both arms) to the obstacle OBB."""

from __future__ import annotations

import numpy as np

from obstacle import OBSTACLE_NAME, load_collision_spheres, world_center_and_half


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
        return origin, rot, np.asarray(half, dtype=np.float64)
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


def distance_info(env) -> dict:
    """Robot-only and system (robot + held object) distances to the wooden block."""
    held_actor, holding = detect_held_object(env)
    hold_dbg = dict(getattr(env, "_cehj_hold_debug", {}) or {})
    out = {
        "d_robot": float("inf"),
        "d_held": float("inf"),
        "d_system": float("inf"),
        "contact": False,
        "closest": "",
        "holding": holding,
        "held_actor": held_actor,
        "hold_debug": hold_dbg,
    }
    obstacle = getattr(env, "obstacle", None)
    if obstacle is None:
        return out
    spheres = load_collision_spheres(env.embodiment)
    labeled = []
    for side, entity in (("L", env.robot.left_entity), ("R", env.robot.right_entity)):
        for center, radius, link_name in spheres_with_names(entity, spheres):
            labeled.append((center, radius, f"{side}/{link_name}"))
    origin, rot, half = _obb_from_actor(obstacle, getattr(env, "obstacle_half", None))
    d_robot = float("inf")
    closest = ""
    for center, radius, label in labeled:
        d = point_obb_signed_distance(center, origin, rot, half) - radius
        if d < d_robot:
            d_robot = d
            closest = label
    d_held = float("inf")
    if held_actor is not None:
        try:
            d_held = _actor_obb_distance(held_actor, origin, rot, half)
            if d_held < d_robot:
                closest = f"held/{holding}"
        except (TypeError, AttributeError):
            held_actor, holding = None, ""
            d_held = float("inf")
            out["held_actor"] = None
            out["holding"] = ""
    d_system = d_robot if held_actor is None else min(d_robot, d_held)
    contact = _robot_obstacle_contact(env)
    if held_actor is not None and not contact:
        contact = _held_obstacle_contact(env, held_actor)
    out.update(
        {
            "d_robot": d_robot,
            "d_held": d_held,
            "d_system": d_system,
            "contact": contact,
            "closest": closest,
            "holding": holding,
            "held_actor": held_actor,
        }
    )
    return out


def is_gripper_link(name: str) -> bool:
    n = (name or "").lower()
    if n in ("link7", "link8"):
        return True
    return any(tag in n for tag in ("finger", "hand", "gripper", "robotiq"))


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
    for link in links:
        name = link.get_name() if hasattr(link, "get_name") else getattr(link, "name", "")
        if not name or name in seen or not is_gripper_link(name):
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


def _iter_task_actors(env):
    from tasks import TASK_SPECS

    spec = TASK_SPECS.get(getattr(env, "task_name", ""), {})
    names: list[str] = []
    if spec.get("object"):
        names.append(spec["object"])
    for extra in spec.get("grasp_objects", ()):
        if extra not in names:
            names.append(extra)
    if getattr(env, "task_name", "") == "stack_blocks_two" and "block2" not in names:
        names.append("block2")
    seen = set()
    for name in names:
        if not hasattr(env.task, name):
            continue
        actor = getattr(env.task, name)
        if actor is None or actor is getattr(env, "obstacle", None):
            continue
        try:
            _resolve_actor(actor)
        except TypeError:
            continue
        key = id(actor)
        if key in seen:
            continue
        seen.add(key)
        yield name, actor


def _entity_side_lookup(env) -> dict[int, str]:
    """Map SAPIEN entity id → left/right. Link names repeat on both arms."""
    cache = getattr(env, "_cehj_entity_side", None)
    if cache is not None:
        return cache
    cache: dict[int, str] = {}
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
    return name in gripper or is_gripper_link(name)


def _detect_held_from_contacts(env, items: list[tuple[str, object, set]]) -> tuple[object | None, str, str]:
    try:
        contacts = env.task.scene.get_contacts()
    except Exception:
        return None, "", ""
    for contact in contacts:
        bodies = contact.bodies
        for i in range(2):
            obj_ent = bodies[1 - i].entity
            oid = id(obj_ent)
            match = None
            for label, actor, keys in items:
                if oid in keys:
                    match = (label, actor)
                    break
            if match is None:
                continue
            gripper_body = bodies[i]
            if not _is_gripper_entity(env, gripper_body.entity):
                continue
            side = _entity_side(env, gripper_body.entity)
            if side is None or not _arm_is_holding(env, side):
                continue
            label, actor = match
            return actor, label, side
    return None, "", ""


def _detect_held_from_near(env, items: list[tuple[str, object, set]]):
    """Closed gripper whose TCP/fingers sit on a task-object OBB (no PhysX contact needed)."""
    best = None
    best_d = float("inf")
    for label, actor, _keys in items:
        for side in ("left", "right"):
            if not _arm_is_holding(env, side):
                continue
            metrics = _hold_metrics(env, actor, side)
            if not metrics["near"]:
                continue
            dist = metrics["grip_obb"] if np.isfinite(metrics["grip_obb"]) else metrics["tcp_obb"]
            if not np.isfinite(dist):
                dist = metrics["tcp_obb"]
            if dist < best_d:
                best = (actor, label, side, metrics)
                best_d = dist
    return best


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
        for center, radius, name in spheres_with_names(entity, load_collision_spheres(env.embodiment)):
            if not is_gripper_link(name):
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
        "R_origin_dist_ee": float("nan"),
        "R_origin_dist_tcp": float("nan"),
        "R_tcp_obb": float("nan"),
        "R_grip_obb": float("nan"),
        "R_near": False,
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


def detect_held_object(env) -> tuple[object | None, str]:
    """Return (actor, label) while a closed gripper holds a task object.

    PhysX gripper↔object contact starts a latch. Keep it while that arm stays
    closed AND the gripper/TCP is still on the object OBB (so a slip after
    hitting the block clears HOLDING without waiting for an open-gripper command).
    Wrist-origin Euclidean is logged only; it is not the stay/slip test.
    """
    debug = _empty_hold_debug()
    debug["left_holding"] = _arm_is_holding(env, "left")
    debug["right_holding"] = _arm_is_holding(env, "right")
    debug["left_grip"] = _gripper_value(env, "left")
    debug["right_grip"] = _gripper_value(env, "right")

    primary = None
    items: list[tuple[str, object, set]] = []
    for name, actor in _iter_task_actors(env):
        if primary is None:
            primary = actor
        items.append((name, actor, _entity_ids(actor)))
    _fill_both_arm_metrics(env, primary, debug)

    def _finish(actor=None, label=""):
        debug["holding"] = label
        env._cehj_hold_debug = debug
        return actor, label

    if not debug["left_holding"] and not debug["right_holding"]:
        env._cehj_held_latch = None
        debug["fail"] = "grippers_open"
        return _finish()

    if not items:
        env._cehj_held_latch = None
        debug["fail"] = "no_actors"
        return _finish()

    actor, label, side = _detect_held_from_contacts(env, items)
    debug["contact_grip_obj"] = actor is not None
    if actor is not None:
        metrics = _hold_metrics(env, actor, side)
        _apply_side_metrics(debug, metrics)
        _fill_both_arm_metrics(env, actor, debug)
        debug["source"] = "contact"
        debug["side"] = side
        env._cehj_held_latch = {"actor": actor, "label": label, "side": side}
        return _finish(actor, label)

    latch = getattr(env, "_cehj_held_latch", None)
    if latch is not None:
        side = latch["side"]
        metrics = _hold_metrics(env, latch["actor"], side)
        _apply_side_metrics(debug, metrics)
        _fill_both_arm_metrics(env, latch["actor"], debug)
        debug["side"] = side
        arm_ok = _arm_is_holding(env, side)
        if arm_ok and metrics["near"]:
            debug["source"] = "latch"
            env._cehj_held_latch = latch
            return _finish(latch["actor"], latch["label"])
        debug["fail"] = "arm_open" if not arm_ok else "not_near"
        env._cehj_held_latch = None

    near_hit = _detect_held_from_near(env, items)
    if near_hit is not None:
        actor, label, side, metrics = near_hit
        _apply_side_metrics(debug, metrics)
        _fill_both_arm_metrics(env, actor, debug)
        debug["source"] = "near"
        debug["side"] = side
        debug["fail"] = ""
        env._cehj_held_latch = {"actor": actor, "label": label, "side": side}
        return _finish(actor, label)

    debug["fail"] = debug.get("fail") or "no_contact"
    if primary is not None:
        _fill_both_arm_metrics(env, primary, debug)
        closed = "left" if debug["left_holding"] else "right"
        _apply_side_metrics(debug, _hold_metrics(env, primary, closed))
        debug["side"] = closed
    return _finish()


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
    robot_names = _robot_link_names(env.robot)
    try:
        contacts = env.task.scene.get_contacts()
    except Exception:
        return False
    for contact in contacts:
        names = [contact.bodies[0].entity.name, contact.bodies[1].entity.name]
        if OBSTACLE_NAME not in names:
            continue
        other = names[0] if names[1] == OBSTACLE_NAME else names[1]
        if other == OBSTACLE_NAME:
            continue
        if other in robot_names:
            return True
        low = other.lower()
        if any(tag in low for tag in ("piper", "panda", "franka", "ur5", "arx", "gripper", "finger")):
            return True
    return False


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
