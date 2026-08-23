"""RoboTwin 2.0 environment setup."""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import yaml

CEHJ_ROOT = Path(__file__).resolve().parent.parent.parent
ROBOTWIN_ROOT = CEHJ_ROOT / "RoboTwin"
if not ROBOTWIN_ROOT.is_dir():
    ROBOTWIN_ROOT = CEHJ_ROOT.parent / "RoboTwin"

# Render observer at 2x, then downsample to stock 320x240 for saved videos.
# 2x SSAA + MSAA 2 is a mild cost bump (one extra camera, every record step).
DEFAULT_MSAA = 2
STOCK_OBSERVER_SIZE = (320, 240)
RECORD_SIZE = (320, 256)
DEFAULT_OBSERVER_SIZE = (640, 480)

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


# Datacenter compute GPUs without RT cores (H100, A100, ...).
_NO_RT_RE = re.compile(
    r"\b(h100|h200|h800|h20|a100|a800|a30|v100|p100|p40|b100|b200|gb200)\b",
    re.I,
)
# Professional / GeForce GPUs with RT cores (RTX, A40, L40S, ...).
_HAS_RT_RE = re.compile(r"(rtx|\ba40\b|\bl40s?\b|\bl4\b|\ba10\b|\ba16\b)", re.I)
# CUDA SMs that expose RT cores SAPIEN can use.
_RT_SMS = {(7, 5), (8, 6), (8, 9), (12, 0)}
_NO_RT_SMS = {(8, 0), (9, 0), (10, 0), (10, 3)}


@lru_cache(maxsize=1)
def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        line = out.strip().splitlines()
        if line:
            return line[0].strip()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return ""


def _gpu_capability() -> tuple[int, int] | None:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return int(major), int(minor)
    except Exception:
        pass
    return None


def gpu_supports_ray_tracing() -> bool:
    """True for RTX / A40 / L40S-class GPUs; False for H100 / A100-class."""
    name = _gpu_name()
    if name and _NO_RT_RE.search(name):
        return False
    if name and _HAS_RT_RE.search(name):
        return True
    cap = _gpu_capability()
    if cap in _RT_SMS:
        return True
    if cap in _NO_RT_SMS:
        return False
    return False


def use_ray_tracing() -> bool:
    return os.environ.get("SAPIEN_DISABLE_RAY_TRACING", "1") != "1"


os.environ.setdefault(
    "SAPIEN_DISABLE_RAY_TRACING",
    "0" if gpu_supports_ray_tracing() else "1",
)


def prepare() -> None:
    if not ROBOTWIN_ROOT.is_dir():
        raise FileNotFoundError(f"RoboTwin missing: {ROBOTWIN_ROOT}")
    if str(ROBOTWIN_ROOT) not in sys.path:
        sys.path.insert(0, str(ROBOTWIN_ROOT))
    os.chdir(ROBOTWIN_ROOT)


def configure_rendering(msaa: int = DEFAULT_MSAA) -> None:
    name = _gpu_name() or "unknown GPU"
    if use_ray_tracing():
        print(f"[env] {name}: ray tracing")
        enable_ray_tracing()
        return
    print(f"[env] {name}: rasterization (no RT cores)")
    force_rasterization(msaa=msaa)


def force_rasterization(msaa: int = DEFAULT_MSAA) -> None:
    import sapien.core as sapien_core
    import sapien.render
    from envs import _base_task

    def _set_msaa(value: int) -> None:
        if int(value) <= 1:
            return
        try:
            sapien.render.set_msaa(int(value))
            print(f"[env] MSAA={sapien.render.get_msaa()}")
        except Exception as exc:
            print(f"[env] set_msaa({value}) failed: {exc}")

    force_rasterization._msaa = int(msaa)
    if getattr(sapien.render.set_camera_shader_dir, "_cehj_raster", False):
        _set_msaa(msaa)
        return

    orig = sapien.render.set_camera_shader_dir
    orig_global = sapien.render.set_global_config

    def set_camera_shader_dir(name: str = "default"):
        if name == "rt":
            name = "default"
        return orig(name)

    def _noop(*_a, **_k):
        return None

    def set_global_config(*args, **kwargs):
        orig_global(*args, **kwargs)
        _set_msaa(getattr(force_rasterization, "_msaa", msaa))

    set_camera_shader_dir._cehj_raster = True
    for mod in (sapien.render, sapien_core.render, _base_task.sapien.render):
        mod.set_camera_shader_dir = set_camera_shader_dir
        mod.set_ray_tracing_samples_per_pixel = _noop
        mod.set_ray_tracing_path_depth = _noop
        mod.set_ray_tracing_denoiser = _noop
        mod.set_global_config = set_global_config
    sapien.render.set_camera_shader_dir("default")
    _set_msaa(msaa)


def enable_ray_tracing(samples_per_pixel: int = 32, path_depth: int = 8) -> None:
    """Use SAPIEN ray tracing shaders (requires an RTX GPU).

    RoboTwin's base task already requests the "rt" shader; this simply
    ensures the request is honoured instead of patched away.
    """
    import sapien.render

    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(samples_per_pixel)
    sapien.render.set_ray_tracing_path_depth(path_depth)
    sapien.render.set_ray_tracing_denoiser("oidn")


def patch_observer_resolution(width: int, height: int) -> None:
    """Replace RoboTwin's 320x240 observer with a finer raster camera after load."""
    from envs.camera.camera import Camera

    if getattr(Camera.load_camera, "_cehj_hires", False):
        Camera._cehj_observer_size = (int(width), int(height))
        return
    orig = Camera.load_camera

    def load_camera(self, scene):
        orig(self, scene)
        w, h = getattr(Camera, "_cehj_observer_size", (width, height))
        old = self.observer_camera
        if old.get_width() == w and old.get_height() == h:
            return
        pose = old.entity.get_pose() if hasattr(old, "entity") else old.get_global_pose()
        new = scene.add_camera(
            name="observer_camera",
            width=int(w),
            height=int(h),
            fovy=float(old.fovy),
            near=float(old.near),
            far=float(old.far),
        )
        new.entity.set_pose(pose)
        try:
            scene.remove_camera(old)
        except Exception:
            pass
        self.observer_camera = new
        print(f"[env] observer camera {w}x{h} (was {old.get_width()}x{old.get_height()})")

    load_camera._cehj_hires = True
    Camera._cehj_observer_size = (int(width), int(height))
    Camera.load_camera = load_camera


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
    # RoboTwin `_camera_config.yml` only lists D435 / Large_D435 / L515.
    # D435_256 was an HJ-encoder leftover and KeyErrors every eval episode.
    args["camera"].setdefault("head_camera_type", "D435")
    args["camera"].setdefault("wrist_camera_type", "D435")
    args["camera"].update(
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

    PHYSICS_FREQ = 250.0  # RoboTwin scene timestep is 1/250 s (_base_task.setup_scene)

    def __init__(
        self,
        task: str = "place_empty_cup",
        embodiment: str = "piper",
        seed: int = 0,
        arm_distance: float = 0.6,
        cluttered: bool = False,
        settle: bool = True,
        msaa: int = DEFAULT_MSAA,
        observer_size: tuple[int, int] = DEFAULT_OBSERVER_SIZE,
        record_size: tuple[int, int] = RECORD_SIZE,
        control_freq: float = 20.0,
        real_time: bool = False,
    ):
        prepare()
        configure_rendering(msaa=msaa)
        if tuple(observer_size) != STOCK_OBSERVER_SIZE:
            patch_observer_resolution(*observer_size)
        self.task_name = task
        self.embodiment = resolve_embodiment(embodiment)
        self.seed = seed
        self.cluttered = cluttered
        self.obstacle = None
        self.record_size = tuple(int(v) for v in record_size)
        self.observer_size = tuple(int(v) for v in observer_size)
        self.control_freq = float(control_freq)
        self.real_time = bool(real_time)
        self._rt_next = None
        self.n_overruns = 0
        self.n_control_steps = 0
        self.n_physics_steps = 0
        self._substep_acc = 0.0
        module = importlib.import_module(f"envs.{task}")
        self.task = getattr(module, task)()
        self.task.setup_demo(
            **_task_args(task, self.embodiment, seed, arm_distance, cluttered=cluttered)
        )
        from .obstacle import stretch_task_spawns

        stretch_task_spawns(self)
        if settle:
            self.settle_arms()

    @property
    def robot(self):
        return self.task.robot

    @property
    def control_dt(self) -> float:
        """Seconds per control step."""
        return 1.0 / self.control_freq

    @property
    def sim_time(self) -> float:
        """Episode time in seconds (physics steps so far / PHYSICS_FREQ)."""
        return self.n_physics_steps / self.PHYSICS_FREQ

    def get_obs(self):
        return self.task.get_obs()

    # HoloBrain GD image normalization (0-255 scale)
    ENCODER_IMG_MEAN = (123.675, 116.28, 103.53)
    ENCODER_IMG_STD = (58.395, 57.12, 57.375)

    @staticmethod
    def camera_extrinsic(cam) -> "np.ndarray":
        """Canonical world->camera extrinsic [4, 4] for a SAPIEN camera.

        This is the single source for every consumer (projection matrices,
        T_cam_world). Verified by the marker projection test.
        """
        import numpy as np

        ext = np.asarray(cam.get_extrinsic_matrix(), dtype=np.float64)
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0.0, 0.0, 0.0, 1.0]])
        return ext

    def get_encoder_obs(self) -> dict:
        """Observation bundle with everything the HoloBrain encoders need.

        Captures the three data cameras (head + both wrists) and the robot
        joint state. Feed directly into encoder.encode_visual_tokens /
        encoder.encode_joint_angles. Everything is in the world frame —
        do NOT pass out_extrinsic; link_pos and visual token positions are
        both world-frame and only relative vectors matter downstream:

            obs = env.get_encoder_obs()
            tokens, pos_mean, pos_max = encoder.encode_visual_tokens(
                obs["imgs"], obs["depths"], obs["image_wh"],
                obs["projection_mat"],
            )
            state_feat = encoder.encode_joint_angles(obs["joint_state"], kin)

        Keys:
            imgs:         [1, 3, 3, H, W] float32 RGB, HoloBrain-normalized.
            depths:       [1, 3, 1, H, W] float32 metric depth (meters).
            image_wh:     [3, 2] int (width, height) per camera.
            projection_mat: [1, 3, 4, 4] float64 K @ T_cam_world per camera
                (built from the canonical Env.camera_extrinsic, the same
                matrix verified by the projection test).
            T_cam_world:  [4, 4] world->head-cam (reference only; the
                encoders and body features stay in the world frame).
            joint_state:  [1, 1, 14] float32 [L6, gripL, R6, gripR].
            rgb:          list of raw uint8 [H, W, 3] frames (visualization).
            camera_names: ["head", "left_wrist", "right_wrist"].
        """
        import numpy as np
        import torch

        # CRITICAL with ray tracing: the RT renderer keeps its own copy of
        # the scene — without this sync, take_picture renders a STALE state
        # (obstacles spawned since the last sync are invisible, arm poses lag)
        self.task._update_render()
        cams = self.task.cameras
        cam_list = [
            cams.static_camera_list[cams.head_camera_id],
            cams.left_camera,
            cams.right_camera,
        ]
        rgbs, depths, projs = [], [], []
        for cam in cam_list:
            cam.take_picture()
            rgba = cam.get_picture("Color")
            rgb = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
            pos = cam.get_picture("Position")
            depth = np.asarray(-pos[:, :, 2], dtype=np.float32)
            K = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)
            # projection_mat = K @ canonical T_cam_world (single source, no drift)
            ext = self.camera_extrinsic(cam)
            proj = np.eye(4, dtype=np.float64)
            proj[:3, :4] = K @ ext[:3]
            rgbs.append(rgb)
            depths.append(depth)
            projs.append(proj)
        h, w = rgbs[0].shape[:2]

        imgs = np.stack(rgbs)[None].astype(np.float32)
        imgs = (imgs - np.array(self.ENCODER_IMG_MEAN)) / np.array(self.ENCODER_IMG_STD)
        left = self.robot.get_left_arm_real_jointState()
        right = self.robot.get_right_arm_real_jointState()
        return {
            "imgs": torch.from_numpy(imgs).permute(0, 1, 4, 2, 3),
            "depths": torch.from_numpy(np.stack(depths)[None, :, None]),
            "image_wh": torch.tensor([[w, h]] * len(cam_list)),
            "projection_mat": torch.from_numpy(np.stack(projs)[None]),
            "T_cam_world": self.camera_extrinsic(cam_list[0]),
            "joint_state": torch.tensor(
                [left + right], dtype=torch.float32
            )[None],
            "rgb": rgbs,
            "camera_names": ["head", "left_wrist", "right_wrist"],
        }

    def step_dtheta(self, dtheta, grip_left=None, grip_right=None) -> float:
        """Apply a per-joint DISPLACEMENT action (actor/critic layout
        [L j1..jn, R j1..jn], n = arm joints + 2 prismatic slots per arm;
        prismatic slots ignored) and advance one control step.

        Bridges to step()'s absolute targets:
        [theta_L + dL, gripL, theta_R + dR, gripR]; grippers hold their
        current opening unless grip_left/right are given (normalized
        [0,1] absolute targets). Embodiment-agnostic: arm joint counts
        come from the robot's real joint states (piper 6, franka 7).
        """
        import numpy as np

        d = np.asarray(dtheta, dtype=np.float64).reshape(-1)
        left = np.array(self.robot.get_left_arm_real_jointState(),
                        dtype=np.float64)
        right = np.array(self.robot.get_right_arm_real_jointState(),
                         dtype=np.float64)
        n_l, n_r = len(left) - 1, len(right) - 1  # gripper scalar is last
        per_arm = d.shape[0] // 2
        a = np.concatenate(
            [
                left[:n_l] + d[:n_l],
                [left[n_l] if grip_left is None else grip_left],
                right[:n_r] + d[per_arm:per_arm + n_r],
                [right[n_r] if grip_right is None else grip_right],
            ]
        )
        return self.step(a)

    def _physics_substeps(self) -> int:
        """Physics steps for one control step, exact on average (250/freq)."""
        self._substep_acc += self.PHYSICS_FREQ / self.control_freq
        n = int(self._substep_acc)
        self._substep_acc -= n
        return n

    def pace(self) -> float:
        """Sleep to hold the wall-clock control rate (real_time mode).

        Returns the overrun in seconds (0.0 if on schedule). If a step's
        computation already exceeded control_dt, the schedule resets
        instead of bursting to catch up.
        """
        import time

        now = time.perf_counter()
        if self._rt_next is None:
            self._rt_next = now + self.control_dt
            return 0.0
        self._rt_next += self.control_dt
        delay = self._rt_next - now
        if delay > 0:
            time.sleep(delay)
            return 0.0
        self.n_overruns += 1
        self._rt_next = now + self.control_dt
        return -delay

    def step(self, action=None) -> float:
        """Apply an action and advance one control step (1/control_freq s).

        Args:
            action: optional vector
                [left_arm(n), left_gripper, right_arm(n), right_gripper],
                n = arm joints per side (piper 6 -> 14-dim, franka 7 ->
                16-dim), gripper values normalized to [0, 1]. `None` holds
                the current drive targets.

        Returns:
            sim_time (seconds) after the step.
        """
        import numpy as np

        if action is not None:
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.shape[0] < 4 or action.shape[0] % 2 != 0:
                raise ValueError(
                    f"action must have 2n+2 dims [Ln, gripL, Rn, gripR], "
                    f"got {action.shape}"
                )
            n = (action.shape[0] - 2) // 2
            zeros = np.zeros(n)
            self.robot.set_arm_joints(action[0:n], zeros, "left")
            self.robot.set_gripper(float(action[n]), "left")
            self.robot.set_arm_joints(action[n + 1:n + 1 + n], zeros, "right")
            self.robot.set_gripper(float(action[2 * n + 1]), "right")
        n = self._physics_substeps()
        for _ in range(n):
            self.task.scene.step()
        self.task._update_render()
        self.n_control_steps += 1
        self.n_physics_steps += n
        if self.real_time:
            self.pace()
        return self.sim_time

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
        self.n_physics_steps += n_steps

    def close(self) -> None:
        try:
            self.task.close_env(clear_cache=True)
        except Exception as exc:
            print(f"close_env warning: {exc}")
