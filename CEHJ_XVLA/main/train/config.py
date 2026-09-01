"""Frozen configuration for HJ-SAC X-VLA baseline runs.

These change what V means, not just how well it trains: dt, ee_step_max
(the EE6D action box), softmin T, gamma, the h definition (margins,
obstacle-set rules), and the encoder variant. Two runs differing in any of
them are NOT comparable. One versioned file is written alongside the
weights and logged as the wandb run config.

Differences from the joint-space parent: the action is X-VLA's EE6D
(per arm [xyz(3), rot6d(6), gripper(1)], 20 total) bounded per-dim by
ee_step_max instead of the URDF-velocity dtheta_max box, and the encoder is
the frozen X-VLA VLM (xvla_ckpt) instead of HoloBrain.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

XVLA_CKPT_DEFAULT = os.environ.get(
    "XVLA_CKPT", "/root/autodl-tmp/XVLA_ckpt/robotwin2"
)

# task -> short English instruction for the X-VLA VLM (their RoboTwin eval
# uses the task name with spaces; we use slightly richer phrasing)
TASK_INSTRUCTIONS = {
    "stack_bowls_two": "stack the bowls",
    "place_bread_basket": "put the bread in the basket",
    "place_container_plate": "put the container on the plate",
    "place_burger_fries": "put the burger and fries on the plate",
    "stack_blocks_two": "stack the blocks",
}


def instruction_for(task: str) -> str:
    return TASK_INSTRUCTIONS.get(task, task.replace("_", " "))


@dataclass
class FrozenConfig:
    # problem definition (change what V means)
    control_dt: float = 0.04            # 25 Hz control: 250/25 = exactly 10 physics steps/tick
    # per-dim EE6D action bound per tick (one arm's 9-dim block, applied to
    # both arms): xyz 0.05 m/tick at 25 Hz (= 1.25 m/s), rot6d 1.0 (the two
    # stored columns are unit vectors — full range, so any delta rotation is
    # representable; the rotation ANGLE is capped separately by ee_rot_max at
    # execution, not by the per-dim box); no gripper channel
    ee_step_max: tuple = (0.05, 0.05, 0.05,
                          1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # no gripper dim
    ee_rot_max: float = 0.5             # rad/tick cap on the executed delta-
                                        # rotation angle (axis-angle norm of
                                        # R_delta, clamped in _execute_ee6d)
    softmin_T: float = 0.02             # metres; softmin temperature
    gamma: float = 0.9                  # HJ discount (annealed -> gamma_final)
    gamma_final: float = 0.999
    gamma_anneal_steps: int = 50_000
    table_margin: float = 0.01          # table margin for cuRobo's world model (NOT in h)
    table_height: float = 0.74          # RoboTwin default table height (m)
    h_scale: float = 20.0               # h/V are trained in h*20 units (~[-1, 1] for
                                        # cm-scale hazards — better conditioned than cm).
                                        # softmin_T and the filter margin stay in
                                        # PHYSICAL metres, scaled at use. Visualization
                                        # converts back to cm via x(100 / h_scale).
    obstacle_mode: str = "on_path"      # safety obstacle spawn mode for collection
    obstacle_t: float | None = None     # corridor t; None = sample per episode via
                                        # run.py's _corridor_t(seed) in U[0.22, 0.48]
    off_path_frac: float = 0.3          # fraction of episodes with off_path obstacle
                                        # (collection AND eval); the rest are on_path
    filter_episode_frac: float = 0.8    # fraction of collection episodes with the filter
    h_include_payload: bool = True      # grasped object joins the body set after grasp
    encoder_variant: str = "X-VLA-RoboTwin2"
    xvla_ckpt: str = XVLA_CKPT_DEFAULT  # X-VLA checkpoint dir (frozen VLM)

    # collection — cyclic schedule over the task x embodiment product (exact
    # balance), obstacle fixed to match the reference sweep geometry
    task: str = "stack_blocks_two"
    embodiment: str = "piper"
    obstacle_model: str = "068_boxdrink"
    task_choices: tuple = (
        "place_container_plate", "place_burger_fries", "stack_blocks_two",
        "stack_bowls_two", "place_bread_basket",
    )
    embodiment_choices: tuple = ("piper", "franka-panda", "ARX-X5", "ur5-wsg",
                             "aloha-agilex")
    obstacle_choices: tuple = ("068_boxdrink",)
    randomize_scenes: bool = True       # sample triples per episode/eval
    hj_hold_ticks: int = 3              # filter intervention block, control ticks (0.12 s)
    n_episodes: int = 25
    max_steps_per_episode: int = 450    # control ticks: 450 * 10 = 4500 physics steps (18 s)
    perturb_prob: float = 0.05          # random EE6D injection probability
    rgbsd_archive_every: int = 25       # low-cadence RGB-D archive stride

    # SAC
    batch_size: int = 64
    lr: float = 3e-4
    tau: float = 0.005                  # target critic Polyak
    alpha_start: float = 0.2            # entropy temperature (annealed -> 0)
    alpha_final: float = 0.0
    alpha_anneal_steps: int = 20_000    # aggressive: unsafe-optimistic bonus
    grad_steps: int = 204800            # 200 epochs x 1024 (real training default)
    eval_every: int = 1024              # one epoch = 1024 grad steps (test runs: 50)
    eval_sweep_every_epochs: int = 0    # rollout sweeps disabled by default —
                                        # round collection records filter metrics
                                        # (RoundMetrics); >0 re-enables sweeps
    eval_ema: float = 0.7               # EMA decay for per-(task,embodiment) metrics
    checkpoint_every: int = 1024        # publish weights every N steps (async collector
                                        # --follow reloads on checkpoint change)
    save_every_epochs: int = 20          # snapshot checkpoint_epoch<N>.pt every N epochs
    eval_seeds: tuple = (1000, 1001)

    # deployment filter: zero threshold — engage when Q(s, a_nom) < 0
    # (predicted penetration), release as soon as Q >= 0. No margin, no
    # hysteresis band — the hj_hold_ticks block provides the commitment.
    filter_margin: float = 0.0
    filter_release_margin: float = 0.0   # metres; engage Q<0, hold until Q>=0
                                        # (same threshold; the 3-tick block is
                                        # the only commitment)
    contact_force_threshold: float = 20.0  # N; collision reporting:
                                        # measured grazes in sim are >= 79 N
                                        # (stiff PD drive), non-contact is 0

    urdf_hash: str = ""                 # filled at run start
    version: int = 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ee_step_max"] = list(d["ee_step_max"])
        d["eval_seeds"] = list(d["eval_seeds"])
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FrozenConfig":
        d = json.loads(Path(path).read_text())
        d["ee_step_max"] = tuple(d["ee_step_max"])
        d["eval_seeds"] = tuple(d["eval_seeds"])
        return cls(**d)

    def hash(self) -> str:
        return hashlib.md5(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:12]


def apply_embodiment_selection(
    cfg: "FrozenConfig", leave_out: str | None = None, only: str | None = None
) -> "FrozenConfig":
    """Leave-one-out cross-validation: restrict the embodiment pool.

    --leave-out X  -> train/eval on all embodiments except X (phase 1)
    --only X       -> train/eval on X alone (phase 2: finetune the phase-1
                      checkpoint on the left-out embodiment)

    EMBODIMENT_IDS are name-keyed so leaving one out does not shift ids.
    The embodiment-agnostic architecture (EE6D action space, no per-joint
    parameters) makes a checkpoint from the 4-embodiment run
    shape-compatible with the 5th.
    """
    if only:
        if only not in cfg.embodiment_choices:
            raise ValueError(f"--only-embodiment {only!r} not a known embodiment")
        cfg.embodiment_choices = (only,)
    elif leave_out:
        if leave_out not in cfg.embodiment_choices:
            # idempotent: the saved config may already be filtered (the
            # two-phase LOO flow passes the flag twice — collect and train)
            return cfg
        cfg.embodiment_choices = tuple(
            e for e in cfg.embodiment_choices if e != leave_out
        )
    return cfg
