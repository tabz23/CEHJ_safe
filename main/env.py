"""RoboTwin 2.0 environment setup."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import yaml

os.environ.setdefault("SAPIEN_DISABLE_RAY_TRACING", "1")

CEHJ_ROOT = Path(__file__).resolve().parent.parent
ROBOTWIN_ROOT = CEHJ_ROOT / "RoboTwin"

ALIASES = {
    "piper": "piper",
    "franka": "franka-panda",
    "franka-panda": "franka-panda",
    "ur5": "ur5-wsg",
    "ur5-wsg": "ur5-wsg",
    "arx": "ARX-X5",
    "arx-x5": "ARX-X5",
    "ARX-X5": "ARX-X5",
}


def resolve_embodiment(name: str) -> str:
    key = name.strip()
    if key in ALIASES:
        return ALIASES[key]
    if key.lower() in ALIASES:
        return ALIASES[key.lower()]
    raise KeyError(f"Unknown embodiment {name!r}")


def prepare() -> None:
    if not ROBOTWIN_ROOT.is_dir():
        raise FileNotFoundError(f"RoboTwin missing: {ROBOTWIN_ROOT}")
    if str(ROBOTWIN_ROOT) not in sys.path:
        sys.path.insert(0, str(ROBOTWIN_ROOT))
    os.chdir(ROBOTWIN_ROOT)


def force_rasterization() -> None:
    import sapien.core as sapien_core
    import sapien.render
    from envs import _base_task

    orig = sapien.render.set_camera_shader_dir

    def set_camera_shader_dir(name: str = "default"):
        if name == "rt":
            name = "default"
        return orig(name)

    def _noop(*_a, **_k):
        return None

    for mod in (sapien.render, sapien_core.render, _base_task.sapien.render):
        mod.set_camera_shader_dir = set_camera_shader_dir
        mod.set_ray_tracing_samples_per_pixel = _noop
        mod.set_ray_tracing_path_depth = _noop
        mod.set_ray_tracing_denoiser = _noop
    sapien.render.set_camera_shader_dir("default")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle.read(), Loader=yaml.FullLoader)


TABLETOP_QPOS = {
    "piper": {
        "left": (0.35, 1.15, -0.90, 0.0, 0.70, 0.0),
        "right": (-0.35, 1.15, -0.90, 0.0, 0.70, 0.0),
    },
    "franka-panda": {
        "left": (0.0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483),
        "right": (0.0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483),
    },
    "ur5-wsg": {
        "left": (-1.5447, -1.5447, -1.5447, -1.5794, 1.5794, 0.0),
        "right": (-1.5447, -1.5447, -1.5447, -1.5794, 1.5794, 0.0),
    },
    "ARX-X5": {
        "left": (0.1276, 1.1426, 1.4179, -0.9723, 0.0, 0.0),
        "right": (-0.1276, 1.1426, 1.4179, -0.9723, 0.0, 0.0),
    },
}


def _task_args(
    task: str,
    embodiment: str,
    seed: int,
    arm_distance: float,
    cluttered: bool = False,
) -> dict:
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    template = _load_yaml(Path(CONFIGS_PATH) / "demo_clean.yml")
    types = _load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")
    if embodiment not in types or types[embodiment]["file_path"] is None:
        raise KeyError(f"Unknown embodiment {embodiment!r}")

    robot_file = str(Path(types[embodiment]["file_path"]))
    cfg = _load_yaml(Path(robot_file) / "config.yml")
    dual = bool(cfg.get("dual_arm", False))
    args = dict(template)
    args.update(
        {
            "task_name": task,
            "task_config": "demo_clean",
            "seed": seed,
            "now_ep_num": 0,
            "render_freq": 0,
            "save_data": False,
            "collect_data": False,
            "eval_mode": False,
            "need_plan": True,
            "dual_arm": dual,
            "dual_arm_embodied": dual,
            "embodiment": [embodiment] if dual else [embodiment, embodiment, arm_distance],
            "embodiment_name": embodiment,
            "embodiment_dis": None if dual else arm_distance,
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "left_embodiment_config": cfg,
            "right_embodiment_config": cfg,
            "eval_video_save_dir": None,
        }
    )
    args["camera"] = dict(args.get("camera") or {})
    args["camera"].update(
        head_camera_type="D435",
        wrist_camera_type="D435",
        collect_head_camera=True,
        collect_wrist_camera=True,
    )
    args["data_type"] = dict(args.get("data_type") or {})
    args["data_type"].update(rgb=True, depth=True, pointcloud=False, endpose=False, qpos=True)
    dr = dict(args.get("domain_randomization") or {})
    if cluttered:
        dr["cluttered_table"] = True
        dr["clean_background_rate"] = 0.0
    else:
        dr["cluttered_table"] = False
    args["domain_randomization"] = dr
    return args


class Env:
    """One RoboTwin 2.0 task episode."""

    def __init__(
        self,
        task: str = "place_empty_cup",
        embodiment: str = "piper",
        seed: int = 0,
        arm_distance: float = 0.6,
        cluttered: bool = False,
        settle: bool = True,
    ):
        prepare()
        force_rasterization()
        self.task_name = task
        self.embodiment = resolve_embodiment(embodiment)
        self.seed = seed
        self.cluttered = cluttered
        self.obstacle = None
        module = importlib.import_module(f"envs.{task}")
        self.task = getattr(module, task)()
        self.task.setup_demo(
            **_task_args(task, self.embodiment, seed, arm_distance, cluttered=cluttered)
        )
        if settle:
            self.settle_arms()

    @property
    def robot(self):
        return self.task.robot

    def get_obs(self):
        return self.task.get_obs()

    def settle_arms(self, n_steps: int = 400) -> None:
        import numpy as np

        if self.embodiment == "ur5-wsg":
            print(f"Keeping {self.embodiment} at RoboTwin homestate.")
            return
        ready = TABLETOP_QPOS[self.embodiment]
        left = np.asarray(ready["left"], dtype=np.float64)
        right = np.asarray(ready["right"], dtype=np.float64)
        self.robot.set_arm_joints(left, np.zeros_like(left), "left")
        self.robot.set_arm_joints(right, np.zeros_like(right), "right")
        for _ in range(n_steps):
            self.task.scene.step()
        self.task._update_render()

    def close(self) -> None:
        try:
            self.task.close_env(clear_cache=True)
        except Exception as exc:
            print(f"close_env warning: {exc}")
