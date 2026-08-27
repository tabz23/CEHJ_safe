"""One-command async online run for single-terminal (e.g. SLURM) use.

Runs the whole warmup + interactive-training flow as one job:

    1. warmup collect (foreground, nominal-only) — creates the buffer at
       --capacity and fills the first --episodes
    2. watch collector (background child process) — keeps collecting with
       the filter, reloading weights whenever the trainer publishes a new
       checkpoint
    3. trainer (foreground) — refreshes the buffer view as it grows

Usage (one srun/sbatch command):
    python main/train/run_online.py --episodes 25 --capacity 90400 \
        --data data/run1 --run runs/run1 [--leave-out ur5-wsg] [--no-wandb]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CEHJ_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=25,
                   help="warmup batch (nominal-only)")
    p.add_argument("--capacity", type=int, default=90400,
                   help="buffer size in steps for the WHOLE run "
                        "(~452 per episode; fixed at creation)")
    p.add_argument("--grad-steps", type=int, default=204800)
    p.add_argument("--eval-every", type=int, default=1024)
    p.add_argument("--leave-out", default=None)
    p.add_argument("--only-embodiment", default=None)
    p.add_argument("--record-video", action="store_true",
                   help="record panel videos of the warmup episodes "
                        "(<data>/videos/)")
    p.add_argument("--success-only", action="store_true",
                   help="warmup keeps only successful episodes (failures "
                        "rolled back and retried; piper cap 10, others 50)")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    args.data = args.data.resolve()
    args.run = args.run.resolve()

    sel = []
    if args.leave_out:
        sel += ["--leave-out", args.leave_out]
    if args.only_embodiment:
        sel += ["--only-embodiment", args.only_embodiment]

    print("[run_online] step 1/3: warmup collection (nominal-only)", flush=True)
    subprocess.run(
        [sys.executable, "main/train/collect.py",
         "--episodes", str(args.episodes),
         "--capacity", str(args.capacity),
         *(["--record-video"] if args.record_video else []),
         *(["--success-only"] if args.success_only else []),
         "--out", str(args.data), *sel],
        cwd=CEHJ_ROOT, check=True,
    )

    print("[run_online] step 2/3: watch collector (background)", flush=True)
    watcher = subprocess.Popen(
        [sys.executable, "main/train/collect.py",
         "--episodes", str(args.episodes),
         "--capacity", str(args.capacity),
         "--watch", "--follow", str(args.run / "checkpoint.pt"),
         "--out", str(args.data), *sel],
        cwd=CEHJ_ROOT,
    )

    print("[run_online] step 3/3: trainer (foreground)", flush=True)
    try:
        cmd = [sys.executable, "main/train/train.py",
               "--data", str(args.data), "--run", str(args.run),
               "--grad-steps", str(args.grad_steps),
               "--eval-every", str(args.eval_every)]
        if args.leave_out:
            cmd += ["--leave-out", args.leave_out]
        if args.only_embodiment:
            cmd += ["--only-embodiment", args.only_embodiment]
        if args.no_wandb:
            cmd += ["--no-wandb"]
        subprocess.run(cmd, cwd=CEHJ_ROOT, check=True)
    finally:
        watcher.terminate()
        try:
            watcher.wait(timeout=30)
        except subprocess.TimeoutExpired:
            watcher.kill()
    print("[run_online] done", flush=True)


if __name__ == "__main__":
    main()
