"""Per-task registry: which object, target, and arm the expert uses.

RoboTwin task files are not copied. Adding a task is one entry here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SAFETY_TASKS = (
    "place_empty_cup",
    "move_can_pot",
    "place_can_basket",
    "place_container_plate",
    "place_shoe",
    "stack_blocks_two",
    "place_object_stand",
    "place_mouse_pad",
    "click_bell",
    "press_stapler",
)

EMBODIMENTS = ("piper", "franka-panda", "ur5-wsg", "ARX-X5")
# run_all.py --embodiments all: ARX first (stable CuRobo), piper last (weakest IK).
SWEEP_EMBODIMENTS = ("ARX-X5", "franka-panda", "ur5-wsg", "piper")

# object/target are attributes on the RoboTwin task instance after setup_demo.
# arm: "from_object_x" | "from_object_x_inv" | "attr:<name>"
TASK_SPECS: dict[str, dict[str, Any]] = {
    "place_empty_cup": {
        "object": "cup",
        "target": "coaster",
        "arm": "from_object_x",
    },
    "move_can_pot": {
        "object": "can",
        "target": "pot",
        "arm": "from_object_x",
    },
    "place_can_basket": {
        "object": "can",
        "target": "basket",
        "arm": "attr:arm_tag",
    },
    "place_container_plate": {
        "object": "container",
        "target": "plate",
        "arm": "from_object_x",
    },
    "place_shoe": {
        "object": "shoe",
        "target": "target_block",
        "arm": "from_object_x_inv",
    },
    "stack_blocks_two": {
        "object": "block1",
        "target": "block1_target_pose",
        "arm": "from_object_x_inv",
        "grasp_objects": ("block1", "block2"),
    },
    "place_object_stand": {
        "object": "object",
        "target": "displaystand",
        "arm": "from_object_x",
    },
    "place_mouse_pad": {
        "object": "mouse",
        "target": "target",
        "arm": "from_object_x",
    },
    "click_bell": {
        "object": "bell",
        "target": "bell",
        "arm": "from_object_x",
    },
    "press_stapler": {
        "object": "stapler",
        "target": "stapler",
        "arm": "from_object_x_inv",
    },
}


def _xy(value) -> np.ndarray:
    if value is None:
        raise ValueError("missing actor/pose for task spec")
    if hasattr(value, "get_pose"):
        return np.asarray(value.get_pose().p[:2], dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return arr[:2]


def _object_x(task, spec: dict) -> float:
    obj = getattr(task, spec["object"])
    return float(_xy(obj)[0])


def resolve_arm(task, spec: dict, arm: str = "auto") -> str:
    if arm in ("left", "right"):
        return arm
    mode = spec.get("arm", "from_object_x")
    if mode.startswith("attr:"):
        tag = getattr(task, mode.split(":", 1)[1])
        return str(tag).lower()
    x = _object_x(task, spec)
    if mode == "from_object_x_inv":
        return "left" if x < 0 else "right"
    return "right" if x > 0 else "left"


def start_target_xy(task, spec: dict, robot, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Corridor for the obstacle: object → target (cup → coaster).

    Using the current EE pose puts the block on the approach-to-grasp line and
    the expert often dies after the pick. Fall back to EE→object when source
    and target are the same (click_bell).
    """
    p_obj = _xy(getattr(task, spec["object"]))
    p1 = _xy(getattr(task, spec["target"]))
    if np.linalg.norm(p_obj - p1) < 0.04:
        ee = robot.get_left_ee_pose() if arm == "left" else robot.get_right_ee_pose()
        return np.asarray(ee[:2], dtype=np.float64), p1
    return p_obj, p1
