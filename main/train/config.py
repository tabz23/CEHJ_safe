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
    kappa: float = 1.0                  # fraction of URDF vel limit per step; 1.0 = actor
                                        # and a full-speed cuRobo tick share the same dtheta box
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
    encoder_variant: str = "HoloBrain_v0.0_GD"
    feature_level: tuple = (1, 2)

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
    perturb_prob: float = 0.05          # random dtheta injection probability
    rgbsd_archive_every: int = 25       # low-cadence RGB-D archive stride

    # SAC
    batch_size: int = 64
    lr: float = 3e-4
    tau: float = 0.005                  # target critic Polyak
    alpha_start: float = 0.2            # entropy temperature (annealed -> alpha_final)
    alpha_final: float = 0.02           # NOT 0: alpha only regularizes the
                                        # actor (the Bellman target is hard),
                                        # and alpha*logp carries the tanh
                                        # barrier -alpha*log(1-u^2) -> +inf at
                                        # the box walls. At alpha=0 the actor
                                        # becomes pure argmax_Q and slams into
                                        # the boundary the critic never saw
                                        # (sat_frac cliff at 20k, bang-bang
                                        # actions) — keep a floor.
    alpha_anneal_steps: int = 20_000
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
                                        # the only commitment)                                        # actor holds until Q >= this
                                        # (hysteresis: engage < 0, release >= 1 cm)
    # ablations: vanilla = both off-features disabled — plain cross-attention
    # over frozen HoloBrain state + scene tokens, same actor/critic heads
    ablate_geometry: bool = False   # no (dist, dir) attention bias / direction
                                    # channel in trunk or critic blocks
    ablate_injection: bool = False  # no analytic body-feature columns injected
    contact_force_threshold: float = 20.0  # N; collision reporting:
                                        # measured grazes in sim are >= 79 N
                                        # (stiff PD drive), non-contact is 0

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


def apply_embodiment_selection(
    cfg: "FrozenConfig", leave_out: str | None = None, only: str | None = None
) -> "FrozenConfig":
    """Leave-one-out cross-validation: restrict the embodiment pool.

    --leave-out X  -> train/eval on all embodiments except X (phase 1)
    --only X       -> train/eval on X alone (phase 2: finetune the phase-1
                      checkpoint on the left-out embodiment)

    EMBODIMENT_IDS are name-keyed so leaving one out does not shift ids.
    The count-invariant architecture (no per-joint parameters) makes a
    checkpoint from the 4-embodiment run shape-compatible with the 5th.
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
