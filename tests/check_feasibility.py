"""Deeper feasibility probe: piper on its failing tasks, more seeds, with
and without the obstacle; franka as a task-feasibility reference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.network.encoder import HoloBrainEncoder  # noqa: E402
from main.train.buffer import StepBuffer  # noqa: E402
from main.train.collect import CKPT_DIR, make_kinematics  # noqa: E402
from main.train.config import FrozenConfig  # noqa: E402
from main.train.rollout import RolloutController  # noqa: E402

CONFIGS = [
    # (embodiment, task, obstacle_mode, seeds)
    ("piper", "place_bread_basket", "on_path", range(6)),
    ("piper", "place_bread_basket", "none", range(6)),
    ("piper", "place_can_basket", "on_path", range(6)),
    ("piper", "place_can_basket", "none", range(6)),
    ("franka-panda", "place_bread_basket", "on_path", range(3)),
    ("franka-panda", "place_can_basket", "on_path", range(3)),
]


def main():
    out = Path("/tmp/feasibility_probe")
    buf = StepBuffer(out / "buffer", capacity=40 * 452,
                     header=FrozenConfig().to_dict())
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kins = {}
    for emb, task, mode, seeds in CONFIGS:
        if emb not in kins:
            kins[emb] = make_kinematics(CKPT_DIR, emb)
        wins = 0
        for seed in seeds:
            cfg = FrozenConfig(task=task, embodiment=emb, obstacle_mode=mode,
                               randomize_scenes=False)
            ro = RolloutController(cfg, seed=seed, mode="collect",
                                   encoder=encoder, kin=kins[emb], buf=buf,
                                   episode_id=seed)
            trace = ro.run()
            ok = bool(trace["success"])
            wins += ok
            print(f"{emb} / {task} / {mode} / seed{seed}: success={ok} "
                  f"sim={trace['n_physics'] / 250:.1f}s", flush=True)
        print(f"== {emb} / {task} / {mode}: {wins}/{len(list(seeds))} ==",
              flush=True)


if __name__ == "__main__":
    main()
