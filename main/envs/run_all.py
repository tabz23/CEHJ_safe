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

if __package__ in (None, ""):
    _cehj = str(Path(__file__).resolve().parents[2])
    if _cehj not in sys.path:
        sys.path.insert(0, _cehj)
    __package__ = "main.envs"
    import main.envs  # noqa: F401  ensure parent package exists for relative imports

from argparse import Namespace

from .controller import ResidualController, controller_class
from .env import CEHJ_ROOT, resolve_embodiment
from .run import CONTROLLERS, run_episode, _corridor_t, _t_tag
from .tasks import BIMANUAL_TASKS, EMBODIMENTS, SAFETY_TASKS, SWEEP_EMBODIMENTS

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
        if key not in allowed and key not in SAFETY_TASKS and key not in BIMANUAL_TASKS:
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
    parser.add_argument(
        "--task-set",
        choices=("safety", "bimanual"),
        default="safety",
        help="Which list --tasks all uses. safety=10 original; bimanual=8 dual-arm.",
    )
    parser.add_argument("--embodiments", default="all")
    parser.add_argument("--obstacle-modes", default="none,off_path,on_path")
    parser.add_argument("--place-mode", default="geometric", choices=("geometric", "waypoint"))
    parser.add_argument(
        "--plan-mode",
        default="ignore_obstacle",
        choices=("ignore_obstacle", "no_ignore_obstacle"),
    )
    parser.add_argument(
        "--unsafe-level",
        default="2",
        help="Ignored. Obstacle t is Uniform[0.3, 0.7] from the seed.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Different seeds / scenes per pair.")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--cluttered", action="store_true")
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--output", type=Path, default=CEHJ_ROOT / "outputs" / "ihab")
    parser.add_argument("--record-every", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--arm-distance", type=float, default=0.6)
    parser.add_argument("--controller", default="residual")
    parser.add_argument(
        "--controllers",
        default="",
        help="Comma list. If set, each scene runs these in order (policy inner). "
        "plan_play_once_everyk is expanded with --replan-ks.",
    )
    parser.add_argument("--replan-k", type=int, default=20)
    parser.add_argument(
        "--replan-ks",
        default="",
        help="Comma list of K for plan_play_once_everyk when using --controllers "
        "(e.g. 20,5,1). Default: --replan-k.",
    )
    parser.add_argument("--replan-max", type=int, default=2500)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episodes whose summary.json already exists under --output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run everything even if --resume would skip it.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-mpc-windows",
        action="store_true",
        help="Do not write mpc_windows/*.mp4 debug clips (episode videos still saved).",
    )
    return parser.parse_args()


def _jobs(args: argparse.Namespace) -> list[dict]:
    task_pool = BIMANUAL_TASKS if args.task_set == "bimanual" else SAFETY_TASKS
    if args.preset == "smoke":
        tasks = ["place_empty_cup"] if args.task_set != "bimanual" else ["place_burger_fries"]
        embs = ["piper"]
        modes = ["none", "off_path", "on_path"]
        extra_plan = []
        extra_place = []
    elif args.preset == "discussed":
        tasks = _split(args.tasks, task_pool)
        embs = [resolve_embodiment(e) for e in _split_embodiments(args.embodiments)]
        modes = ["none", "off_path", "on_path"]
        extra_plan = ["no_ignore_obstacle"]
        extra_place = ["waypoint"]
    else:
        tasks = _split(args.tasks, task_pool)
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


def _policies(args: argparse.Namespace) -> list[dict]:
    raw = str(getattr(args, "controllers", "") or "").strip()
    names = [p.strip() for p in raw.split(",") if p.strip()] if raw else [str(args.controller)]
    ks_raw = str(getattr(args, "replan_ks", "") or "").strip()
    if ks_raw:
        ks = [int(x.strip()) for x in ks_raw.split(",") if x.strip()]
    else:
        ks = [int(getattr(args, "replan_k", 20))]
    out = []
    for name in names:
        if name == "plan_play_once_everyk":
            for k in ks:
                out.append(
                    {
                        "controller": name,
                        "replan_k": int(k),
                        "policy_dir": f"plan_everyk_k{int(k)}",
                    }
                )
        else:
            out.append({"controller": name, "replan_k": None, "policy_dir": name})
    return out


def _expand_jobs(args: argparse.Namespace) -> list[dict]:
    """Scene jobs with policy as the inner loop (same seed, then vanilla / each K)."""
    policies = _policies(args)
    expanded = []
    for scene in _jobs(args):
        for pol in policies:
            expanded.append({**scene, **pol})
    return expanded


def _output_root(args: argparse.Namespace, job: dict) -> Path:
    root = Path(args.output).expanduser()
    policies = _policies(args)
    if len(policies) > 1:
        return root / str(job.get("policy_dir") or job.get("controller") or "policy")
    return root


def _job_out_dir(args: argparse.Namespace, job: dict) -> Path:
    cluttered = bool(args.cluttered)
    t = _corridor_t(job["seed"])
    tag = (
        f"{job['obstacle_mode']}_{job['place_mode']}_{job['plan_mode']}_"
        f"{_t_tag(t)}_seed{job['seed']}"
    )
    if cluttered:
        tag += "_clutter"
    return _output_root(args, job) / job["task"] / job["embodiment"] / tag


def _existing_summary(out_dir: Path) -> dict | None:
    path = out_dir / "summary.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if int(data.get("n_frames") or 0) <= 0 and not data.get("videos"):
        return None
    return data


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
        controller=job.get("controller", args.controller),
        msaa=getattr(args, "msaa", 2),
        video_res=getattr(args, "video_res", "320x240"),
        record_every=args.record_every,
        fps=args.fps,
        output=_output_root(args, job),
        replan_k=int(job["replan_k"] if job.get("replan_k") is not None else getattr(args, "replan_k", 20)),
        replan_max=int(getattr(args, "replan_max", 2500)),
        mpc_window_max=int(getattr(args, "mpc_window_max", 150)),
        no_mpc_windows=bool(getattr(args, "no_mpc_windows", False)),
    )


def main() -> None:
    args = parse_args()
    policies = _policies(args)
    try:
        ctrl_cls = controller_class(policies[0]["controller"])
    except KeyError:
        ctrl_cls = CONTROLLERS.get(policies[0]["controller"], ResidualController)
    ctrl_cls.install()
    jobs = _expand_jobs(args)
    resume = bool(args.resume) and not bool(args.overwrite)
    n_skip = 0
    if resume:
        n_skip = sum(1 for job in jobs if _existing_summary(_job_out_dir(args, job)) is not None)
    pol_txt = ", ".join(
        p["policy_dir"] if p["replan_k"] is None else f"{p['controller']}(k={p['replan_k']})"
        for p in policies
    )
    print(
        f"{len(jobs)} runs  {len(jobs) // max(len(policies), 1)} scenes × {len(policies)} policies  "
        f"preset={args.preset}  policies=[{pol_txt}]  output={args.output}"
        + (f"  resume skip={n_skip} remaining={len(jobs) - n_skip}" if resume else "")
    )
    results = []
    for i, job in enumerate(jobs):
        ctrl = job.get("controller", args.controller)
        k = job.get("replan_k")
        print(
            f"\n=== [{i + 1}/{len(jobs)}] {job['task']} {job['embodiment']} "
            f"seed={job['seed']} {job['obstacle_mode']} {job['place_mode']} {job['plan_mode']} "
            f"ctrl={ctrl}" + (f" k={k}" if k is not None else "") + " ==="
        )
        if resume:
            existing = _existing_summary(_job_out_dir(args, job))
            if existing is not None:
                print(f"[run_all] resume skip  {_job_out_dir(args, job)}")
                results.append(existing)
                continue
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
    tag = "multi" if len(policies) > 1 else str(policies[0]["controller"])
    summary_path = out / f"sweep_{args.preset}_{tag}_seed{args.base_seed}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    n_ok = sum(1 for r in results if r.get("success"))
    n_err = sum(1 for r in results if r.get("error"))
    print(f"\nDone. {n_ok} success / {len(results)} runs, {n_err} crashes.")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
