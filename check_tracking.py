#!/usr/bin/env python3
"""Tracking check: does the arm reach theta + dtheta within one control step?

For a sweep of commanded joint displacements dtheta, this drives a joint to a
new target in a single 20 Hz control step and measures the achieved
displacement after 1, 2 and 3 steps. It answers whether delta_theta_max
(velocity_limit * dt * kappa = 0.0375 rad for piper) is actually trackable
by the sim's PD position drives within one control period.

Usage:
  conda activate RoboTwin
  python check_tracking.py [--embodiment piper] [--task place_empty_cup]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CEHJ_ROOT = Path(__file__).resolve().parent
if str(CEHJ_ROOT) not in sys.path:
    sys.path.insert(0, str(CEHJ_ROOT))

from main.envs.env import Env  # noqa: E402

DTHETAS = [0.01, 0.025, 0.0375, 0.05, 0.075, 0.10, 0.15, 0.20]
JOINTS = [0, 1, 4]  # base rot, shoulder, wrist-ish (per-arm indices)
TRACK_TOL = 0.90  # achieved >= 90% of commanded counts as tracked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="place_empty_cup")
    p.add_argument("--embodiment", default="piper")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = Env(args.task, args.embodiment, args.seed, control_freq=20.0)
    robot = env.robot

    def qpos14():
        return np.array(
            robot.get_left_arm_real_jointState()
            + robot.get_right_arm_real_jointState(),
            dtype=np.float64,
        )

    q0 = qpos14()
    print(f"{args.task} / {env.embodiment}  control 20 Hz (50 ms/step)")
    print(f"{'joint':>6} {'dtheta':>8} {'step1':>8} {'step2':>8} {'step3':>8} {'tracked@1':>10}")

    for j in JOINTS:
        for dtheta in DTHETAS:
            # settle back to q0
            for _ in range(20):
                env.step(q0)
            qa = qpos14()
            a = qa.copy()
            a[j] += dtheta
            achieved = []
            for _ in range(3):
                env.step(a)
                achieved.append(abs(qpos14()[j] - qa[j]))
            r = [min(x / dtheta, 9.99) for x in achieved]
            ok = "YES" if r[0] >= TRACK_TOL else "no"
            print(
                f"L joint{j} {dtheta:>8.4f} "
                f"{r[0]:>7.0%} {r[1]:>7.0%} {r[2]:>7.0%} {ok:>10}"
            )
        print()

    env.close()


if __name__ == "__main__":
    main()
