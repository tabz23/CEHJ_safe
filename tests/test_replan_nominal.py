#!/usr/bin/env python3
"""Verify ReplanNominal: after steering, the nominal replans from the live
state (never chases the pre-steering plan) and recovers to the waypoint.

  conda activate RoboTwin
  python tests/test_replan_nominal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CEHJ_ROOT = Path(__file__).resolve().parents[1]
if str(CEHJ_ROOT) not in sys.path:
    sys.path.insert(0, str(CEHJ_ROOT))

from main.envs.env import Env  # noqa: E402
from main.train.nominal import ReplanNominal  # noqa: E402
from main.network.body_features import BodyTokenExtractor  # noqa: E402
from main.envs.obstacle import choose_and_spawn  # noqa: E402


def main() -> None:
    probe_env = Env("place_empty_cup", "piper", 42, control_freq=20.0)
    tracker = ReplanNominal.probe(probe_env, control_freq=20.0)
    probe_env.close()
    print(f"probe: T={tracker.T} success={tracker.success}")

    env = Env("place_empty_cup", "piper", 42, control_freq=20.0)
    actor, xyz, half, _ = choose_and_spawn(env, "on_path", "geometric", 0.6, "auto")
    env.obstacle, env.obstacle_xyz, env.obstacle_half = actor, xyz, half
    extractor = BodyTokenExtractor(env)
    dtheta_max = extractor.delta_theta_max.astype(np.float64)

    steer_at = tracker.T // 3
    steer_steps = 15
    replans_before_steer = 0
    for t in range(tracker.T):
        if t == steer_at:
            replans_before_steer = tracker.n_replans
            # steer: shove both arms away for steer_steps control steps
            shove = np.zeros(16)
            shove[0] = dtheta_max[0]
            shove[8] = -dtheta_max[8]
            for _ in range(steer_steps):
                env.step_dtheta(shove)
            print(f"steered at t={t}: replans so far {tracker.n_replans}")
            continue
        dtheta, gl, gr = tracker.action(t, env, dtheta_max)
        env.step_dtheta(dtheta, grip_left=gl, grip_right=gr)

    # after steering, the nominal must have replanned from the live state
    new_replans = tracker.n_replans - replans_before_steer
    assert new_replans > 0, "no replans happened after steering"

    # recovery takes time: keep tracking the final waypoint past the probe
    # horizon and require convergence within a bounded number of steps
    err_l = err_r = float("inf")
    recovered_at = None
    for t in range(tracker.T, tracker.T + 200):
        dtheta, gl, gr = tracker.action(t, env, dtheta_max)
        env.step_dtheta(dtheta, grip_left=gl, grip_right=gr)
        err_l = np.linalg.norm(
            np.asarray(env.robot.get_left_ee_pose()[:3])
            - tracker.ee_traj["left"][-1][:3]
        )
        err_r = np.linalg.norm(
            np.asarray(env.robot.get_right_ee_pose()[:3])
            - tracker.ee_traj["right"][-1][:3]
        )
        if err_l < 0.08 and err_r < 0.08:
            recovered_at = t - tracker.T
            break
    print(
        f"replans total={tracker.n_replans} (post-steer +{new_replans})  "
        f"plan failures={tracker.n_plan_fail}"
    )
    print(f"final EE err: L={err_l:.3f} m  R={err_r:.3f} m  recovered_at=+{recovered_at} steps")
    assert recovered_at is not None, "nominal did not recover after steering"
    env.close()
    print("REPLAN NOMINAL TEST PASSED")


if __name__ == "__main__":
    main()
