#!/usr/bin/env python3
"""Tracking test: one-tick commanded vs realized joint displacement.

The drive interpolation in Env.step (theta_d(k) = q0 + d*k/N with velocity
feedforward d/dt) must realize the commanded displacement within one
control tick — before it, a held zero-velocity target realized only ~20%.
With interpolation, piper/franka realize 0.75-0.9 (full-tick) growing with
command magnitude; the residual is drive transient physics, consistent
across actor and nominal paths (both now issue (pos, vel) pairs at 250 Hz).
Threshold 0.75; the per-tick ratio is also logged as a diagnostic.

  conda activate RoboTwin
  python tests/test_tracking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CEHJ_ROOT = Path(__file__).resolve().parents[1]
if str(CEHJ_ROOT) not in sys.path:
    sys.path.insert(0, str(CEHJ_ROOT))

from main.envs.env import Env  # noqa: E402
from main.network.body_features import BodyTokenExtractor  # noqa: E402


def run(embodiment: str) -> bool:
    env = Env("place_empty_cup", embodiment, 0, control_freq=25.0)
    extractor = BodyTokenExtractor(env)
    dmax = extractor.delta_theta_max.astype(np.float64)
    ji = None
    body = extractor.extract()
    ji = body["joint_index"].numpy().astype(np.int64)
    cols = ji[ji >= 0]
    ok = True
    for frac in (0.25, 0.5, 1.0):
        q0 = extractor.raw_qpos().astype(np.float64)
        cmd = np.zeros(2 * extractor.spec.n_joints)
        cmd[cols] = frac * dmax[cols] * 0.5  # half amplitude, all arm joints
        env.step_dtheta(cmd)
        q1 = extractor.raw_qpos().astype(np.float64)
        realized = (q1 - q0)[cols]
        ratio = np.abs(realized).sum() / max(np.abs(cmd[cols]).sum(), 1e-12)
        # per-joint worst case
        worst = np.abs(realized / np.where(np.abs(cmd[cols]) < 1e-12, 1, cmd[cols])).min()
        status = "OK" if ratio >= 0.75 else "FAIL"
        print(f"  {embodiment} frac={frac}: realized/commended={ratio:.3f} "
              f"worst-joint={worst:.3f}  {status}")
        ok = ok and ratio >= 0.75
    env.close()
    return ok


def main() -> None:
    ok = True
    for emb in ("piper", "franka-panda"):
        ok = run(emb) and ok
    assert ok, "tracking below 75% of commanded displacement"
    print("TRACKING TEST PASSED")


if __name__ == "__main__":
    main()
