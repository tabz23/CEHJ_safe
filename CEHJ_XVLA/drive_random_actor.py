"""Drive one episode with a RANDOM-INIT actor and record the panel video.

Sanity check for the EE6D execution path (workspace clip + 0.5 rad/tick
rotation cap): the filter margin is forced huge so the (random) actor
drives the arms EVERY tick — if the clipped EE6D -> DLS -> step_dtheta
path is broken, the arms fly off and it's visible in the video.

Usage:
    python drive_random_actor.py [--task place_empty_cup] [--embodiment piper]
                                 [--seed 0] [--out /root/autodl-tmp/xvla_random_actor.mp4]
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from main.train.config import FrozenConfig
from main.train.collect import KIN_URDF_DIR, make_kinematics
from main.train.rollout import RolloutController
from main.train.eval_utils import PanelVideoWriter
from main.network.encoder import XVLAEncoder
from main.network.heads import EE6DActor, EE6DTwinCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="place_empty_cup")
    ap.add_argument("--embodiment", default="piper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/autodl-tmp/xvla_random_actor.mp4")
    args = ap.parse_args()

    cfg = FrozenConfig(
        task=args.task, embodiment=args.embodiment,
        # Q(s, a_nom) < margin*h_scale is always true -> the random actor
        # drives every tick (release margin stays 0, never reached)
        filter_margin=1.0,
    )
    encoder = XVLAEncoder(str(cfg.xvla_ckpt), device="cuda")
    actor = EE6DActor().cuda()          # random init — this is the point
    critics = EE6DTwinCritic(
        temperature=cfg.softmin_T * float(cfg.h_scale)).cuda()

    kin = make_kinematics(KIN_URDF_DIR, cfg.embodiment)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    video = PanelVideoWriter(out, fps=25.0, h_scale=float(cfg.h_scale))

    ro = None
    for retry in range(5):  # spawn guarantee: need an obstacle for h
        candidate = RolloutController(
            cfg, args.seed + retry * 1000, mode="filtered", encoder=encoder,
            kin=kin, actor=actor, critics=critics,
            video=video,
            max_steps=int(cfg.max_steps_per_episode * 250.0 / 25.0),
        )
        if candidate.env.obstacle is not None:
            ro = candidate
            break
        print(f"[drive] spawn failed (seed {args.seed + retry * 1000}); re-drawing")
        candidate.env.close()
    if ro is None:
        video.close()
        raise RuntimeError("spawn failed 5x")

    trace = ro.run()
    video.close()
    print(f"[drive] done: success={trace['success']} "
          f"steps={len(trace['h'])} "
          f"intervened={float(np.mean(trace['intervened'])):.2f} "
          f"min_h={min(trace['h']):.2f}")
    print(f"[drive] video: {out}")


if __name__ == "__main__":
    main()
