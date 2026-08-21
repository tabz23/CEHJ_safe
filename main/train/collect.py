"""Stage 1 — collection driver for HJ-SAC.

Per control step at 20 Hz:
    obs    = env.get_encoder_obs()
    scene  = encoder.encode_visual_tokens(...)   # frozen
    state  = encoder.encode_joint_angles(...)    # frozen
    body   = extractor.extract()                 # analytic
    h      = hfunc.compute_h(...)                # privileged
    dtheta = clip(theta_goal - theta, +-dtheta_max)   # obstacle-agnostic nominal
    env.step(dtheta);  theta_next = MEASURED qpos

No termination on violation — the avoid-Bellman needs transitions in the
unsafe region. episode_id/done written from step zero. Privileged channels
(per-link h breakdown, arm-arm distance) logged per step.

Usage:
    python -m main.collect            (from CEHJ_safe/)
    python main/collect.py --episodes 25
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "main.train"
CEHJ_ROOT = Path(__file__).resolve().parents[2]

from main.network.body_features import BodyTokenExtractor  # noqa: E402
from main.train.buffer import StepBuffer  # noqa: E402
from main.train.config import FrozenConfig  # noqa: E402
from main.network.encoder import HoloBrainEncoder  # noqa: E402
from main.envs.env import Env  # noqa: E402
from main.train.hfunc import compute_h  # noqa: E402
from main.train.nominal import NominalTracker  # noqa: E402

CKPT_DIR = Path("/root/autodl-tmp/HoloBrain/HoloBrain_v0.0_GD")


def make_kinematics(ckpt_dir: Path):
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        MultiArmKinematics,
    )

    return MultiArmKinematics(
        urdf=str(ckpt_dir / "urdf" / "piper_description_dualarm.urdf"),
        arm_link_keys=[
            [f"left_link{i}" for i in range(1, 7)],
            [f"right_link{i}" for i in range(1, 7)],
        ],
        arm_joint_id=[list(range(6)), list(range(8, 14))],
        finger_keys=[["left_link7"], ["right_link7"]],
    )


def collect(cfg: FrozenConfig, out_root: Path) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_kinematics(CKPT_DIR)

    capacity = cfg.n_episodes * cfg.max_steps_per_episode + cfg.n_episodes
    buf = StepBuffer(out_root / "buffer", capacity=capacity, header=cfg.to_dict())
    h_log = open(out_root / "h_diagnostics.jsonl", "w")

    rng = np.random.RandomState(0)
    t_start = time.time()
    n_written = 0

    for ep in range(cfg.n_episodes):
        seed = ep
        # probe (expert reference path) on its own env, then a fresh env
        probe_env = Env(cfg.task, cfg.embodiment, seed=seed, control_freq=20.0)
        if not cfg.urdf_hash:
            from main.network.body_features import _file_hash

            cfg.urdf_hash = _file_hash(probe_env.robot.left_urdf_path)
            cfg.save(out_root / "config.json")
        tracker = NominalTracker.cached(
            CEHJ_ROOT / "nominal_cache" / f"{cfg.task}_{cfg.embodiment}_s{seed}",
            probe_env, control_freq=20.0,
        )
        env = Env(cfg.task, cfg.embodiment, seed=seed, control_freq=20.0)
        # spawn the safety obstacle (the nominal stays obstacle-agnostic:
        # the probe ran on a block-free env)
        from main.envs.obstacle import choose_and_spawn

        actor, xyz, half, _arm = choose_and_spawn(
            env, cfg.obstacle_mode, "geometric", 0.6, "auto"
        )
        env.obstacle = actor
        env.obstacle_xyz = xyz
        env.obstacle_half = half
        extractor = BodyTokenExtractor(env)
        dtheta_max = extractor.delta_theta_max.astype(np.float32)  # [16]

        for t in range(min(tracker.T, cfg.max_steps_per_episode)):
            obs = env.get_encoder_obs()
            with torch.no_grad():
                tokens, pos_mean, _ = encoder.encode_visual_tokens(
                    obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"]
                )
                state = encoder.encode_joint_angles(obs["joint_state"], kin).squeeze(2)[0]
            body = extractor.extract()
            h, diag = compute_h(env, cfg.table_margin)

            goal = tracker.goal(t)                    # [14] absolute targets
            theta = obs["joint_state"][0, 0].numpy()  # [14] measured
            # 16-dim displacement action (actor/critic layout)
            dtheta = np.zeros(16, dtype=np.float64)
            dtheta[0:6] = np.clip(goal[0:6] - theta[0:6], -dtheta_max[0:6], dtheta_max[0:6])
            dtheta[8:14] = np.clip(goal[7:13] - theta[7:13], -dtheta_max[8:14], dtheta_max[8:14])
            if rng.rand() < cfg.perturb_prob:
                arm_cols = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
                dtheta[arm_cols] = (
                    rng.uniform(-1, 1, 12) * dtheta_max[arm_cols]
                ).astype(np.float64)
            env.step_dtheta(dtheta, grip_left=goal[6], grip_right=goal[13])

            per_link = np.zeros(20, dtype=np.float32)
            for i, name in enumerate(
                [f"L/{n}" for n in extractor.spec.link_names]
                + [f"R/{n}" for n in extractor.spec.link_names]
            ):
                if i < 20:
                    per_link[i] = diag["per_link"].get(name, np.inf)

            buf.append(
                {
                    "scene_tokens": tokens.flatten(1, 2)[0].cpu().half().numpy(),
                    "scene_pos": pos_mean.flatten(1, 2)[0].cpu().numpy(),
                    "state_tokens": state.cpu().half().numpy(),
                    "body_tokens": body["body_tokens"].numpy(),
                    "link_pos": body["link_pos"].numpy(),
                    "qpos": theta.astype(np.float32),
                    "qpos_raw": extractor.raw_qpos(),
                    "dtheta": dtheta.astype(np.float32),
                    "h": np.float32(h),
                    "per_link_h": per_link,
                    "d_arm_arm": np.float32(diag["d_arm_arm"]),
                    "body_mask": body["body_mask"].numpy(),
                    "joint_index": body["joint_index"].numpy().astype(np.int8),
                    "episode_id": np.int32(ep),
                    "done": np.bool_(t == min(tracker.T, cfg.max_steps_per_episode) - 1),
                }
            )
            h_log.write(
                json_dumps({"ep": ep, "t": t, **{k: v for k, v in diag.items() if k != "per_link"}})
                + "\n"
            )
            n_written += 1
            if n_written % 200 == 0:
                buf.flush()
                rate = n_written / (time.time() - t_start)
                print(f"ep {ep} t {t}  steps {n_written}  ({rate:.1f}/s)", end="\r")

        env.close()
        print(f"ep {ep} done: T={tracker.T} probe_success={tracker.success}")

    buf.flush()
    h_log.close()
    print(f"\ncollection done: {n_written} steps in {time.time() - t_start:.0f}s")
    return out_root / "buffer"


def json_dumps(d) -> str:
    import json

    class Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            return str(o)

    return json.dumps(d, cls=Enc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--task", default="place_empty_cup")
    p.add_argument("--embodiment", default="piper")
    p.add_argument("--out", type=Path, default=CEHJ_ROOT / "data" / "smoke")
    args = p.parse_args()
    cfg = FrozenConfig(task=args.task, embodiment=args.embodiment,
                       n_episodes=args.episodes)
    collect(cfg, args.out)


if __name__ == "__main__":
    main()
