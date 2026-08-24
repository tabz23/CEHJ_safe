"""Frozen configuration for HJ-SAC runs.

These change what V means, not just how well it trains: dt, kappa (hence
dtheta_max), softmin T, gamma, the h definition (margins, obstacle-set
rules), and the encoder variant with its feature_level. Two runs differing
in any of them are NOT comparable. One versioned file is written alongside
the weights and logged as the wandb run config.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FrozenConfig:
    # problem definition (change what V means)
    control_dt: float = 0.04            # 25 Hz control: 250/25 = exactly 10 physics steps/tick
    kappa: float = 0.25                 # fraction of URDF vel limit per step
    softmin_T: float = 0.02             # metres; softmin temperature
    gamma: float = 0.9                  # HJ discount (annealed -> gamma_final)
    gamma_final: float = 0.999
    gamma_anneal_steps: int = 50_000
    table_margin: float = 0.01          # table margin for cuRobo's world model (NOT in h)
    table_height: float = 0.74          # RoboTwin default table height (m)
    h_scale: float = 100.0              # h/V are trained in h*h_scale units (= cm; h here
                                        # is ~mm-cm). softmin_T and the filter margin below
                                        # stay in PHYSICAL metres, scaled at use. Watch
                                        # gn_critic: targets 5x larger vs h_scale=20.
    obstacle_mode: str = "on_path"      # safety obstacle spawn mode for collection
    obstacle_t: float | None = None     # corridor t; None = sample per episode via
                                        # run.py's _corridor_t(seed) in U[0.3, 0.7]
    off_path_frac: float = 0.2          # fraction of episodes with off_path obstacle
    filter_episode_frac: float = 0.8    # fraction of collection episodes with the filter
    h_include_payload: bool = True      # grasped object joins the body set after grasp
    encoder_variant: str = "HoloBrain_v0.0_GD"
    feature_level: tuple = (1, 2)

    # collection — cyclic schedule over the task x embodiment product (exact
    # balance), obstacle fixed to match the reference sweep geometry
    task: str = "stack_blocks_two"
    embodiment: str = "piper"
    obstacle_model: str = "068_boxdrink"
    task_choices: tuple = (
        "stack_blocks_two", "stack_bowls_two", "place_burger_fries",
        "place_bread_basket", "place_can_basket",
    )
    embodiment_choices: tuple = ("piper", "franka-panda", "ARX-X5", "ur5-wsg")
    obstacle_choices: tuple = ("068_boxdrink",)
    randomize_scenes: bool = True       # sample triples per episode/eval
    hj_hold_ticks: int = 3              # filter intervention block, control ticks (0.12 s)
    n_episodes: int = 25
    max_steps_per_episode: int = 450    # control ticks: 450 * 10 = 4500 physics steps (18 s)
    perturb_prob: float = 0.05          # random dtheta injection probability
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
    eval_seeds: tuple = (1000, 1001)

    # deployment filter: single margin, engage when Q(s, a_nom) < margin
    # (physical m, scaled by h_scale at use). No hysteresis band — the
    # hj_hold_ticks block provides the commitment.
    filter_margin: float = 0.01

    urdf_hash: str = ""                 # filled at run start
    version: int = 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["feature_level"] = list(d["feature_level"])
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
        d["feature_level"] = tuple(d["feature_level"])
        d["eval_seeds"] = tuple(d["eval_seeds"])
        return cls(**d)

    def hash(self) -> str:
        return hashlib.md5(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:12]
