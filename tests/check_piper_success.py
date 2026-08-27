"""Feasibility check: can piper's nominal complete each task at all?

Runs piper nominal (no filter) on the 5 bimanual tasks x 3 seeds through
the collection stack (on-path obstacle present, as in training). Warmup
buffers are success-only, so if piper can't succeed, the LOO pool design
needs to know.

  python tests/check_piper_success.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.network.encoder import HoloBrainEncoder  # noqa: E402
from main.train.buffer import StepBuffer  # noqa: E402
from main.train.collect import CKPT_DIR, make_kinematics  # noqa: E402
from main.train.config import FrozenConfig  # noqa: E402
from main.train.rollout import RolloutController  # noqa: E402

TASKS = [
    "stack_blocks_two", "stack_bowls_two", "place_burger_fries",
    "place_bread_basket", "place_can_basket",
]


def main():
    out = Path("/tmp/piper_check")
    buf = StepBuffer(out / "buffer", capacity=5 * 3 * 452,
                     header=FrozenConfig().to_dict())
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_kinematics(CKPT_DIR, "piper")
    results = {}
    for task in TASKS:
        wins = 0
        for seed in range(3):
            cfg = FrozenConfig(task=task, embodiment="piper",
                               randomize_scenes=False)
            ro = RolloutController(cfg, seed=seed, mode="collect",
                                   encoder=encoder, kin=kin, buf=buf,
                                   episode_id=seed)
            trace = ro.run()
            ok = bool(trace["success"])
            wins += ok
            print(f"piper / {task} / seed{seed}: success={ok} "
                  f"sim={trace['n_physics'] / 250:.1f}s", flush=True)
        results[task] = wins
        print(f"== {task}: {wins}/3 ==", flush=True)
    print("\nsummary (piper nominal, on-path obstacle):")
    for t, w in results.items():
        print(f"  {t:24s} {w}/3")


if __name__ == "__main__":
    main()
