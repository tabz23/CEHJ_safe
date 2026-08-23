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
    control_dt: float = 0.05            # 20 Hz control
    kappa: float = 0.25                 # fraction of URDF vel limit per step
    softmin_T: float = 0.02             # metres; softmin temperature
    gamma: float = 0.9                  # HJ discount (annealed -> gamma_final)
    gamma_final: float = 0.999
    gamma_anneal_steps: int = 50_000
    table_margin: float = 0.01          # table gets its own clearance margin (m)
    table_height: float = 0.74          # RoboTwin default table height (m); part of h's definition
    h_scale: float = 20.0               # h/V are trained in h*h_scale units (h is ~mm-cm
                                        # here; 20x better-conditioned targets). softmin_T
                                        # and the filter margins below stay in PHYSICAL
                                        # metres and are multiplied by h_scale at use.
    obstacle_mode: str = "on_path"      # safety obstacle spawn mode for collection
    obstacle_t: float = 0.6             # corridor t for on_path spawns (unsafe level 2)
    h_include_payload: bool = True      # grasped object joins the body set after grasp
    encoder_variant: str = "HoloBrain_v0.0_GD"
    feature_level: tuple = (1, 2)

    # collection — each episode samples a random (task, embodiment, obstacle)
    # triple from the choices below (cross-embodiment training: no fixed scene)
    task: str = "stack_blocks_two"
    embodiment: str = "piper"
    obstacle_model: str = "086_woodenblock"
    task_choices: tuple = (
        "stack_blocks_two", "stack_bowls_two", "grab_roller",
        "pick_dual_bottles", "place_bread_basket", "place_burger_fries",
        "place_can_basket", "place_cans_plasticbox",
    )
    embodiment_choices: tuple = ("piper", "franka-panda", "ARX-X5", "ur5-wsg")
    obstacle_choices: tuple = ("086_woodenblock", "059_pencup")
    randomize_scenes: bool = True       # sample triples per episode/eval
    replan_k: int = 60                  # PlanEveryKController window (physics steps); part of the nominal definition
    n_episodes: int = 25
    max_steps_per_episode: int = 400
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

    # deployment filter
    margin_on: float = 0.0              # engage filter below this V (physical m, scaled at use)
    margin_off: float = 0.005           # release above this V (physical m; hysteresis)

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
