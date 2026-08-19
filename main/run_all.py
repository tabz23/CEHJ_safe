#!/usr/bin/env python3
"""Sweep safety episodes across tasks, embodiments, and obstacle modes.

Each (task, embodiment, episode) uses a distinct seed so resets get a new
scene. The three obstacle modes for the same (task, embodiment, episode)
share that seed so you can compare none / off_path / on_path on one layout.

Defaults (the main grid we discussed):
  10 tasks × 4 embodiments × 3 obstacle-modes × 1 episode
  place-mode=geometric  plan-mode=ignore_obstacle  no clutter

  python run_all.py
  python run_all.py --preset smoke
  python run_all.py --preset grid --draw-bbox
  python run_all.py --preset discussed --episodes 2 --base-seed 10
  python run_all.py --tasks place_empty_cup --embodiments piper,franka --obstacle-modes on_path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parent
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from argparse import Namespace

from controller import ResidualController
from env import CEHJ_ROOT, resolve_embodiment
from run import CONTROLLERS, run_episode
from tasks import EMBODIMENTS, SAFETY_TASKS, SWEEP_EMBODIMENTS

ALIASES = {
    "piper": "piper",
    "franka": "franka-panda",
    "ur5": "ur5-wsg",
    "arx": "ARX-X5",
}


def _split_embodiments(value: str) -> list[str]:
    if value.strip().lower() in ("all", "*"):
        return list(SWEEP_EMBODIMENTS)
    return _split(value, EMBODIMENTS)


def _split(value: str, allowed: tuple[str, ...]) -> list[str]:
    if value.strip().lower() in ("all", "*"):
        return list(allowed)
    out = []
    for part in value.split(","):
        key = part.strip()
        if not key:
            continue
        if key in ALIASES:
            key = ALIASES[key]
        if key not in allowed and key not in SAFETY_TASKS:
            mapped = ALIASES.get(key.lower(), key)
            key = mapped
        out.append(key)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=("smoke", "grid", "discussed"),
        default="grid",
        help="smoke=cup×piper×3 modes; grid=10×4×3; discussed=grid + on_path replan + waypoint sample",
    )
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--embodiments", default="all")
    parser.add_argument("--obstacle-modes", default="none,off_path,on_path")
    parser.add_argument("--place-mode", default="geometric", choices=("geometric", "waypoint"))
    parser.add_argument(
        "--plan-mode",
        default="ignore_obstacle",
        choices=("ignore_obstacle", "no_ignore_obstacle"),
    )
    parser.add_argument("--unsafe-level", default="2")
    parser.add_argument("--episodes", type=int, default=1, help="Different seeds / scenes per pair.")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--cluttered", action="store_true")
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--output", type=Path, default=CEHJ_ROOT / "outputs" / "ihab")
    parser.add_argument("--record-every", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--arm-distance", type=float, default=0.6)
    parser.add_argument("--controller", default="residual")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _jobs(args: argparse.Namespace) -> list[dict]:
    if args.preset == "smoke":
        tasks = ["place_empty_cup"]
        embs = ["piper"]
        modes = ["none", "off_path", "on_path"]
        extra_plan = []
        extra_place = []
    elif args.preset == "discussed":
        tasks = _split(args.tasks, SAFETY_TASKS)
        embs = [resolve_embodiment(e) for e in _split_embodiments(args.embodiments)]
        modes = ["none", "off_path", "on_path"]
        extra_plan = ["no_ignore_obstacle"]
        extra_place = ["waypoint"]
    else:
        tasks = _split(args.tasks, SAFETY_TASKS)
        embs = [resolve_embodiment(e) for e in _split_embodiments(args.embodiments)]
        modes = [m.strip() for m in args.obstacle_modes.split(",") if m.strip()]
        extra_plan = []
        extra_place = []

    jobs = []
    # Embodiment outer (SWEEP_EMBODIMENTS: ARX → franka → ur5 → piper) so one
    # robot's MotionGen stays hot; switching robots drops other CUDA graphs.
    for ei, emb in enumerate(embs):
        for ti, task in enumerate(tasks):
            for ep in range(max(args.episodes, 1)):
                seed = args.base_seed + ep * 1000 + ti * 10 + ei
                for mode in modes:
                    jobs.append(
                        {
                            "task": task,
                            "embodiment": emb,
                            "seed": seed,
                            "obstacle_mode": mode,
                            "place_mode": args.place_mode,
                            "plan_mode": args.plan_mode,
                        }
                    )
                if extra_plan:
                    jobs.append(
                        {
                            "task": task,
                            "embodiment": emb,
                            "seed": seed,
                            "obstacle_mode": "on_path",
                            "place_mode": args.place_mode,
                            "plan_mode": extra_plan[0],
                        }
                    )
                if extra_place:
                    jobs.append(
                        {
                            "task": task,
                            "embodiment": emb,
                            "seed": seed,
                            "obstacle_mode": "on_path",
                            "place_mode": extra_place[0],
                            "plan_mode": args.plan_mode,
                        }
                    )
    return jobs


def _to_run_args(args: argparse.Namespace, job: dict):
    return Namespace(
        task=job["task"],
        embodiment=job["embodiment"],
        seed=job["seed"],
        arm_distance=args.arm_distance,
        cluttered=args.cluttered,
        no_cluttered=not args.cluttered,
        obstacle_mode=job["obstacle_mode"],
        place_mode=job["place_mode"],
        plan_mode=job["plan_mode"],
        unsafe_level=args.unsafe_level,
        arm="auto",
        draw_bbox=args.draw_bbox,
        controller=args.controller,
        record_every=args.record_every,
        fps=args.fps,
        output=args.output,
    )


def main() -> None:
    args = parse_args()
    ctrl_cls = CONTROLLERS.get(args.controller, ResidualController)
    ctrl_cls.install()
    jobs = _jobs(args)
    print(f"{len(jobs)} episodes  preset={args.preset}  output={args.output}")
    results = []
    for i, job in enumerate(jobs):
        print(
            f"\n=== [{i + 1}/{len(jobs)}] {job['task']} {job['embodiment']} "
            f"seed={job['seed']} {job['obstacle_mode']} {job['place_mode']} {job['plan_mode']} ==="
        )
        if args.dry_run:
            results.append(job)
            continue
        run_ns = _to_run_args(args, job)
        try:
            summary = run_episode(run_ns)
            results.append(summary)
            err = summary.get("play_error") or ""
            if "illegal instruction" in err.lower() or "cuda error" in err.lower():
                print("[run_all] CUDA context is dead; stopping sweep. Restart the process.")
                break
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append({**job, "error": str(exc), "success": False})
            if "illegal instruction" in str(exc).lower() or "cuda error" in str(exc).lower():
                print("[run_all] CUDA context is dead; stopping sweep. Restart the process.")
                break

    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"sweep_{args.preset}_seed{args.base_seed}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    n_ok = sum(1 for r in results if r.get("success"))
    n_err = sum(1 for r in results if r.get("error"))
    print(f"\nDone. {n_ok} success / {len(results)} episodes, {n_err} crashes.")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
