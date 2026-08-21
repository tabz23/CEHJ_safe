"""Privileged h for HJ-SAC: clearance of robot (plus payload) to the obstacle set.

    h = min distance(robot ∪ payload, obstacles)      [metres, signed]

Obstacle set rules (per phase, not filtered):
  - the target object is NOT an obstacle during approach (distance.py never
    includes it) — grasping it would register as a violation otherwise
  - after grasp, the held object joins the BODY set (its distance to the
    obstacles counts) via distance_info's held-actor terms
  - after release it leaves the body set again
  - the table is a legitimate obstacle with its OWN margin: the gripper
    must approach it within centimetres, so its contribution is
    (link clearance above the table) - table_margin

Self-collision is explicitly out of scope: h covers robot-to-environment
clearance only; arm-arm tangling is the nominal controller's
reachability problem. (Reviewer note: yes, we know.)

Logged alongside the scalar (privileged, sim-only):
  - per-link clearance breakdown (true argmin — to check learned argmin V_i)
  - arm-to-arm distance (to attribute task failures h doesn't see)
"""

from __future__ import annotations

import numpy as np

from main.envs.distance import (
    distance_info,
    load_collision_spheres,
    point_obb_signed_distance,
    spheres_with_names,
    _obb_from_actor,
)

TABLE_HEIGHT = 0.74  # default; the authoritative value lives in FrozenConfig.table_height


def _arm_spheres(env):
    """Collision spheres on the MOVING chain only (link1..link8 per arm).

    Static mount links (base_link) are excluded — their clearance to the
    table is a constant that would dominate the min and make h constant.
    """
    from main.network.body_features import get_arm_spec

    spec = get_arm_spec(env.robot.left_urdf_path)
    allowed = set(spec.link_names)
    spheres = load_collision_spheres(env.embodiment)
    for side, entity in (("L", env.robot.left_entity), ("R", env.robot.right_entity)):
        for center, radius, link_name in spheres_with_names(entity, spheres):
            if link_name in allowed:
                yield f"{side}/{link_name}", np.asarray(center), float(radius)


def compute_h(env, table_margin: float = 0.01, table_height: float = TABLE_HEIGHT) -> tuple[float, dict]:
    """Current h and privileged diagnostics."""
    # --- robot arms ∪ payload vs the spawned obstacle (per-phase rules live
    # inside distance_info: cup never an obstacle, held payload joins body) ---
    info = distance_info(env)
    if getattr(env, "obstacle", None) is not None:
        d_block = info["d_system"]          # arms + held payload vs block
    else:
        d_block = float("inf")

    # --- per-link breakdown: block + table ---
    table_z = table_height + float(env.task.table_z_bias)
    per_link: dict[str, float] = {}
    left_spheres, right_spheres = [], []
    for label, center, radius in _arm_spheres(env):
        (left_spheres if label.startswith("L") else right_spheres).append(
            (center, radius)
        )
        best = float("inf")
        if getattr(env, "obstacle", None) is not None:
            origin, rot, half = _obb_from_actor(
                env.obstacle, getattr(env, "obstacle_half", None)
            )
            best = point_obb_signed_distance(center, origin, rot, half) - radius
        d_tab = center[2] - radius - table_z - table_margin
        per_link[label] = min(best, d_tab)

    d_table = min(
        (c[2] - r - table_z - table_margin) for c, r in left_spheres + right_spheres
    )

    # --- arm-to-arm distance (diagnostic only; NOT in h) ---
    d_arm_arm = float("inf")
    for c1, r1 in left_spheres:
        for c2, r2 in right_spheres:
            d_arm_arm = min(
                d_arm_arm, float(np.linalg.norm(c1 - c2)) - r1 - r2
            )

    h = min(d_block, d_table)
    true_argmin = min(per_link, key=per_link.get) if per_link else ""
    diag = {
        "h": h,
        "d_block": d_block,
        "d_table": d_table,
        "d_held": info["d_held"],
        "d_arm_arm": d_arm_arm,
        "per_link": per_link,
        "true_argmin": true_argmin,
        "holding": info.get("holding", ""),
        "contact": info.get("contact", False),
    }
    return h, diag
