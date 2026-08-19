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
    empty = {
        "d_robot": float("inf"),
        "d_held": float("inf"),
        "d_system": float("inf"),
        "contact": False,
        "closest": "",
        "holding": "",
        "held_actor": None,
    }
    obstacle = getattr(env, "obstacle", None)
    if obstacle is None:
        return empty
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
    held_actor, holding = detect_held_object(env)
    d_held = float("inf")
    if held_actor is not None:
        try:
            d_held = _actor_obb_distance(held_actor, origin, rot, half)
            if d_held < d_robot:
                closest = f"held/{holding}"
        except (TypeError, AttributeError):
            held_actor, holding = None, ""
            d_held = float("inf")
    d_system = d_robot if held_actor is None else min(d_robot, d_held)
    contact = _robot_obstacle_contact(env)
    if held_actor is not None and not contact:
        contact = _held_obstacle_contact(env, held_actor)
    return {
        "d_robot": d_robot,
        "d_held": d_held,
        "d_system": d_system,
        "contact": contact,
        "closest": closest,
        "holding": holding,
        "held_actor": held_actor,
    }


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


def _detect_held_from_contacts(env, actors: dict) -> tuple[object | None, str, str]:
    try:
        contacts = env.task.scene.get_contacts()
    except Exception:
        return None, "", ""
    for contact in contacts:
        bodies = contact.bodies
        names = [bodies[0].entity.name, bodies[1].entity.name]
        for i in range(2):
            obj_name = names[1 - i]
            if obj_name not in actors:
                continue
            gripper_body = bodies[i]
            if not _is_gripper_entity(env, gripper_body.entity):
                continue
            side = _entity_side(env, gripper_body.entity)
            if side is None or not _arm_is_holding(env, side):
                continue
            label, actor = actors[obj_name]
            return actor, label, side
    return None, "", ""


def detect_held_object(env) -> tuple[object | None, str]:
    """Return (actor, label) while a closed gripper holds a task object.

    PhysX contact starts a latch; we keep the object marked held until that
    arm opens (contact often drops after lift even though the object moves with
    the gripper).
    """
    if not _arm_is_holding(env, "left") and not _arm_is_holding(env, "right"):
        env._cehj_held_latch = None
        return None, ""

    latch = getattr(env, "_cehj_held_latch", None)
    if latch and not _arm_is_holding(env, latch["side"]):
        latch = None
        env._cehj_held_latch = None

    actors = {}
    for name, actor in _iter_task_actors(env):
        ent = _actor_entity_name(actor)
        if ent:
            actors[ent] = (name, actor)
    if not actors:
        return None, ""

    actor, label, side = _detect_held_from_contacts(env, actors)
    if actor is not None:
        env._cehj_held_latch = {"actor": actor, "label": label, "side": side}
        return actor, label

    if latch and _arm_is_holding(env, latch["side"]):
        return latch["actor"], latch["label"]

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
    held_name = _actor_entity_name(held_actor)
    try:
        contacts = env.task.scene.get_contacts()
    except Exception:
        return False
    for contact in contacts:
        names = [contact.bodies[0].entity.name, contact.bodies[1].entity.name]
        if OBSTACLE_NAME in names and held_name in names:
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
