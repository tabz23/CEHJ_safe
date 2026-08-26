"""Parity test: training-side h vs the sweep-side distance code.

For the same scene (same seed, same embodiment, same obstacle), drives a
fixed action sequence and checks at every tick that:
  - compute_h's h == distance_info["d_system"] (the sweep's d_min)
  - compute_h's per-link breakdown min == min(d_left, d_right) when nothing
    is held (the collaborator's aloha sphere filtering vs hfunc's split-URDF
    filtering must agree)

  conda activate RoboTwin
  python tests/test_h_parity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from main.envs.distance import distance_info
from main.train.collect import make_training_env
from main.train.config import FrozenConfig
from main.train.hfunc import compute_h


def run(embodiment: str) -> bool:
    cfg = FrozenConfig(task="stack_blocks_two", embodiment=embodiment,
                       obstacle_mode="on_path", randomize_scenes=False)
    env = make_training_env(cfg, seed=0)
    n_arm = len(env.robot.get_left_arm_real_jointState()) - 1
    dmax = 0.005  # small fixed displacement per tick, deterministic
    ok = True
    for t in range(40):
        h, diag = compute_h(env)
        info = distance_info(env)
        d_ref = info["d_system"]
        if np.isfinite(d_ref) or np.isfinite(h):
            diff = abs(h - d_ref)
            if diff > 1e-9:
                print(f"  {embodiment} t={t}: h={h:.6f} vs d_system={d_ref:.6f}  MISMATCH")
                ok = False
        # per-link min vs per-arm distances (nothing held in this phase)
        pl = min(v for v in diag["per_link"].values() if np.isfinite(v))
        ref_arms = min(info["d_left"], info["d_right"])
        if np.isfinite(pl) and np.isfinite(ref_arms) and abs(pl - ref_arms) > 1e-6:
            print(f"  {embodiment} t={t}: per_link min {pl:.6f} vs "
                  f"min(dL,dR) {ref_arms:.6f}  MISMATCH")
            ok = False
        # fixed small arm motion so the comparison covers moving states
        left = np.array(env.robot.get_left_arm_real_jointState(), dtype=np.float64)
        right = np.array(env.robot.get_right_arm_real_jointState(), dtype=np.float64)
        a = np.concatenate([left[:n_arm] + dmax * (-1) ** t, [left[n_arm]],
                            right[:n_arm] + dmax * (-1) ** t, [right[n_arm]]])
        env.step(a)
    env.close()
    print(f"  {embodiment}: 40 ticks compared")
    return ok


def main() -> None:
    ok = True
    for emb in ("aloha-agilex", "piper"):
        ok = run(emb) and ok
    assert ok, "h parity mismatch"
    print("H PARITY TEST PASSED")


if __name__ == "__main__":
    main()
