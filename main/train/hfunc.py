"""Privileged h for HJ-SAC: clearance of robot (plus payload) to the obstacle set.

    h = min distance(robot ∪ payload, obstacles)      [metres, signed]

Obstacle set rules (per phase, not filtered):
  - the target object is NOT an obstacle during approach (distance.py never
    includes it) — grasping it would register as a violation otherwise
  - after grasp, the held object joins the BODY set (its distance to the
    obstacles counts) via distance_info's held-actor terms
  - after release it leaves the body set again

The table is NOT part of h: it is a workspace constraint the planner already
handles (update_curobo_world puts it in cuRobo's world model). h scores only
the spawned safety obstacle — h == d_system == d_min, the same number the
sweeps report, and per-link argmin always names the link genuinely closest
to the obstacle. CONSEQUENCE: h is +inf without an obstacle, so spawning is
total (obstacle.py tiered fallback; episodes that fail to spawn are skipped).

Self-collision is explicitly out of scope: h covers robot-to-environment
clearance only; arm-arm tangling is the nominal controller's
reachability problem.

Logged alongside the scalar (privileged, sim-only):
  - per-link obstacle clearance breakdown (true argmin for learned argmin V_i)
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

    aloha-agilex is a native dual-arm articulation: per-arm split-URDF
    specs (arm_spec_paths) plus the side-prefixed sphere selection from
    distance.py (_arm_collision_spheres).
    """
    from main.network.body_features import arm_spec_paths, get_arm_spec
    from main.envs.distance import _arm_collision_spheres

    left_path, right_path = arm_spec_paths(env)
    spheres_all = load_collision_spheres(env.embodiment)
    for side, entity, path in (
        ("L", env.robot.left_entity, left_path),
        ("R", env.robot.right_entity, right_path),
    ):
        spec = get_arm_spec(path)
        allowed = set(spec.link_names)
        spheres = _arm_collision_spheres(
            env, "left" if side == "L" else "right", spheres_all
        )
        for center, radius, link_name in spheres_with_names(entity, spheres):
            if link_name in allowed:
                yield f"{side}/{link_name}", np.asarray(center), float(radius)


def compute_h(env, table_margin: float = 0.01, table_height: float = TABLE_HEIGHT) -> tuple[float, dict]:
    """Current h (obstacle distance only) and privileged diagnostics.

    table_margin/table_height are unused by h itself (kept in the signature
    for the below-table diagnostic and FrozenConfig compatibility).
    """
    # --- robot arms ∪ payload vs the spawned obstacle (per-phase rules live
    # inside distance_info: cup never an obstacle, held payload joins body) ---
    info = distance_info(env)
    if getattr(env, "obstacle", None) is not None:
        d_block = info["d_system"]          # arms + held payload vs block
    else:
        d_block = float("inf")

    # --- per-link breakdown: obstacle only ---
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
        # a link has MULTIPLE collision spheres — keep the MIN per link
        # (a plain assignment keeps only the last sphere and overstates
        # per_link, breaking the true_argmin diagnostic)
        per_link[label] = min(per_link.get(label, float("inf")), best)

    # --- arm-to-arm distance (diagnostic only; NOT in h) ---
    d_arm_arm = float("inf")
    for c1, r1 in left_spheres:
        for c2, r2 in right_spheres:
            d_arm_arm = min(
                d_arm_arm, float(np.linalg.norm(c1 - c2)) - r1 - r2
            )

    # --- below-table fraction (diagnostic for the accepted risk of not
    # scoring the table in h) ---
    table_z = table_height + float(env.task.table_z_bias)
    n_below = sum(
        1 for c, r in left_spheres + right_spheres if c[2] - r < table_z
    )

    h = d_block
    true_argmin = min(per_link, key=per_link.get) if per_link else ""
    diag = {
        "h": h,
        "d_block": d_block,
        "d_held": info["d_held"],
        "d_left": info.get("d_left", float("inf")),
        "d_right": info.get("d_right", float("inf")),
        "d_left_held": info.get("d_left_held", float("inf")),
        "d_right_held": info.get("d_right_held", float("inf")),
        "d_arm_arm": d_arm_arm,
        "per_link": per_link,
        "true_argmin": true_argmin,
        "holding": info.get("holding", ""),
        "holding_left": info.get("holding_left", ""),
        "holding_right": info.get("holding_right", ""),
        "hold_debug": info.get("hold_debug", {}),
        "contact": info.get("contact", False),
        "n_below_table": n_below,
    }
    return h, diag
