#!/usr/bin/env python3
"""Time cost of data collection for one expert episode (seconds).

Reports three pieces that make up data collection:

  planning   CuRobo / mplib, a few times per episode (not every sim step)
  images     RGB + depth from cameras, once every recorded frame
  distance   robot-vs-obstacle clearance, once every recorded frame

  python CEHJ/profile_sim.py
  python CEHJ/profile_sim.py --robot piper --task place_empty_cup
  python CEHJ/profile_sim.py --no-head-camera
  python CEHJ/profile_sim.py --compare-no-head-camera
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parent / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from env import CEHJ_ROOT, DEFAULT_MSAA, RECORD_SIZE  # noqa: E402
from obstacle import choose_and_spawn, update_curobo_world  # noqa: E402
from record import (  # noqa: E402
    _put_text,
    draw_debug_bboxes,
    observer_rgb,
    to_record_size,
)
from run import CONTROLLERS, _level, _make_env, _video_res  # noqa: E402
from tasks import SAFETY_TASKS  # noqa: E402
from distance import detect_held_object, distance_info  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    usual = parser.add_argument_group("usual")
    usual.add_argument("--task", default="place_empty_cup", choices=SAFETY_TASKS)
    usual.add_argument(
        "--robot",
        "--embodiment",
        dest="embodiment",
        default="piper",
        help="Robot / embodiment name (same as --embodiment).",
    )
    usual.add_argument("--seed", type=int, default=0)
    usual.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many episodes to run (seeds seed, seed+1, ...).",
    )

    collect = parser.add_argument_group("skip parts of data collection (optional)")
    collect.add_argument(
        "--no-head-camera",
        dest="skip_head",
        action="store_true",
        help="Skip the unused head camera. Videos already use observer + wrists.",
    )
    collect.add_argument(
        "--no-depth",
        dest="skip_depth",
        action="store_true",
        help="Skip depth images.",
    )
    collect.add_argument(
        "--no-observer",
        dest="skip_observer",
        action="store_true",
        help="Skip the extra side / observer camera.",
    )
    collect.add_argument(
        "--compare-no-head-camera",
        dest="compare_skip_head",
        action="store_true",
        help="Run twice (with head camera, then without) and print the time saved.",
    )

    rare = parser.add_argument_group("rarely needed")
    rare.add_argument("--obstacle-mode", choices=("none", "off_path", "on_path"), default="none")
    rare.add_argument("--arm-distance", type=float, default=0.6)
    rare.add_argument("--place-mode", choices=("geometric", "waypoint"), default="geometric")
    rare.add_argument(
        "--plan-mode",
        choices=("ignore_obstacle", "no_ignore_obstacle"),
        default="ignore_obstacle",
    )
    rare.add_argument("--unsafe-level", default="2")
    rare.add_argument("--controller", choices=tuple(CONTROLLERS), default="residual")
    rare.add_argument(
        "--record-every",
        type=int,
        default=8,
        help="Save images/distance every N physics steps (same as run.py).",
    )
    rare.add_argument("--msaa", type=int, default=DEFAULT_MSAA)
    rare.add_argument("--video-res", default=f"{RECORD_SIZE[0]}x{RECORD_SIZE[1]}")
    rare.add_argument("--ssaa", type=int, default=2)
    rare.add_argument("--draw-bbox", action="store_true")
    rare.add_argument("--output", type=Path, default=CEHJ_ROOT / "outputs" / "profile_sim.json")
    rare.add_argument("--no-record", dest="record", action="store_false")
    parser.set_defaults(record=True, skip_head=False, skip_depth=False, skip_observer=False)
    return parser.parse_args()


def _sec(xs: list[float]) -> dict:
    if not xs:
        return {"times": 0, "each_s": None, "total_s": 0.0}
    vals = [float(x) for x in xs]
    return {
        "times": len(vals),
        "each_s": float(statistics.fmean(vals)),
        "total_s": float(sum(vals)),
    }


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _hook_timer(times: list[float], fn, sync: bool = False):
    def wrapped(*args, **kwargs):
        if sync:
            _sync_cuda()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if sync:
            _sync_cuda()
        times.append(time.perf_counter() - t0)
        return out

    return wrapped


def _to_run_ns(args: argparse.Namespace, seed: int) -> Namespace:
    return Namespace(
        task=args.task,
        embodiment=args.embodiment,
        seed=seed,
        arm_distance=args.arm_distance,
        cluttered=False,
        no_cluttered=True,
        obstacle_mode=args.obstacle_mode,
        place_mode=args.place_mode,
        plan_mode=args.plan_mode,
        unsafe_level=args.unsafe_level,
        arm="auto",
        draw_bbox=args.draw_bbox,
        controller=args.controller,
        msaa=args.msaa,
        video_res=args.video_res,
        record_every=args.record_every,
        fps=10.0,
        output=Path("/tmp"),
    )


def np_quat(obs):
    import numpy as np

    if hasattr(obs, "get_pose"):
        return np.array(obs.get_pose().q, dtype=np.float64)
    return np.array(getattr(obs, "actor", obs).get_pose().q, dtype=np.float64)


def _patch_skip_head(cameras) -> None:
    cameras.collect_head_camera = False
    orig = cameras.update_picture

    def update_picture():
        if cameras.collect_wrist_camera:
            cameras.left_camera.take_picture()
            cameras.right_camera.take_picture()
        for cam, name in zip(cameras.static_camera_list, cameras.static_camera_name):
            if name == "head_camera":
                continue
            cam.take_picture()

    cameras.update_picture = update_picture
    cameras._cehj_orig_update_picture = orig


def attach_profile_recorder(env, streams, record_every, overlay, draw_bbox, times, args):
    import numpy as np

    state = {"n": 0, "orig_step": env.task.scene.step, "d_min": []}
    orig_step = env.task.scene.step
    skip_head = bool(args.skip_head)
    skip_depth = bool(args.skip_depth)
    skip_observer = bool(args.skip_observer)
    if skip_head:
        _patch_skip_head(env.task.cameras)
    if skip_depth:
        env.task.data_type["depth"] = False

    def step():
        orig_step()
        state["n"] += 1
        if state["n"] % max(record_every, 1) != 0:
            detect_held_object(env)
            return

        t0 = time.perf_counter()
        info = distance_info(env)
        times["distance"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        env.task._update_render()
        times["update_render"].append(time.perf_counter() - t0)

        _sync_cuda()
        t0 = time.perf_counter()
        env.task.cameras.update_picture()
        _sync_cuda()
        times["take_picture"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        rgb_map = env.task.cameras.get_rgb()
        times["get_rgb"].append(time.perf_counter() - t0)

        if not skip_depth:
            t0 = time.perf_counter()
            env.task.cameras.get_depth()
            times["get_depth"].append(time.perf_counter() - t0)

        observer_s = 0.0
        if skip_observer:
            head = rgb_map.get("head_camera") or {}
            rgb = np.asarray(head.get("rgb", np.zeros((240, 320, 3), np.uint8)), dtype=np.uint8)
        else:
            _sync_cuda()
            t0 = time.perf_counter()
            rgb = observer_rgb(env.task)
            _sync_cuda()
            observer_s = time.perf_counter() - t0
            times["observer"].append(observer_s)

        times["images"].append(
            times["update_render"][-1]
            + times["take_picture"][-1]
            + times["get_rgb"][-1]
            + (times["get_depth"][-1] if not skip_depth else 0.0)
            + observer_s
        )

        rec_wh = getattr(env, "record_size", RECORD_SIZE)
        debug = draw_debug_bboxes(env, rgb) if draw_bbox else None
        rgb = to_record_size(rgb, rec_wh)
        hold_txt = info["holding"] if info["holding"] else "none"
        lines = [
            f"HOLDING={hold_txt} contact={int(info['contact'])} {info['closest']}",
            f"d_robot={info['d_robot']} d_sys={info['d_system']}",
        ]
        streams["agent_rgb"].append(_put_text(rgb, lines))
        if draw_bbox:
            streams["debug_bbox"].append(_put_text(to_record_size(debug, rec_wh), lines))
        left = rgb_map.get("left_camera") or {}
        right = rgb_map.get("right_camera") or {}
        if "rgb" in left:
            streams["left_wrist_rgb"].append(np.asarray(left["rgb"], dtype=np.uint8))
        if "rgb" in right:
            streams["right_wrist_rgb"].append(np.asarray(right["rgb"], dtype=np.uint8))
        state["d_min"].append(info["d_system"])
        print(f"recorded {len(streams['agent_rgb'])} frames", end="\r")

    env.task.scene.step = step
    return state


def make_env(args: argparse.Namespace, seed: int):
    run_ns = _to_run_ns(args, seed)
    record_wh = _video_res(args.video_res)
    ssaa = max(int(args.ssaa), 1)
    if ssaa == 2:
        env, ctrl = _make_env(run_ns, seed)
    else:
        from env import Env

        ctrl_cls = CONTROLLERS[args.controller]
        ctrl_cls.install()
        env = Env(
            args.task,
            args.embodiment,
            seed,
            args.arm_distance,
            cluttered=False,
            settle=True,
            msaa=int(args.msaa),
            observer_size=(record_wh[0] * ssaa, record_wh[1] * ssaa),
            record_size=record_wh,
        )
        ctrl = ctrl_cls(env)
        ctrl.attach()
    return env, ctrl


def profile_episode(args: argparse.Namespace, seed: int) -> dict:
    env, ctrl = make_env(args, seed)
    embodiment = env.embodiment
    level = _level(args.unsafe_level, seed)
    actor, xyz, half, _arm = choose_and_spawn(
        env, args.obstacle_mode, args.place_mode, level, "auto"
    )
    env.obstacle = actor
    env.obstacle_xyz = xyz
    env.obstacle_half = half
    update_curobo_world(env.robot)
    if args.plan_mode == "no_ignore_obstacle" and xyz is not None:
        update_curobo_world(env.robot, xyz, np_quat(env.obstacle), half)

    times = {
        "physics": [],
        "step_total": [],
        "plan": [],
        "distance": [],
        "images": [],
        "update_render": [],
        "take_picture": [],
        "get_rgb": [],
        "get_depth": [],
        "observer": [],
    }

    orig_step = env.task.scene.step

    def physics_step():
        t0 = time.perf_counter()
        orig_step()
        times["physics"].append(time.perf_counter() - t0)

    env.task.scene.step = physics_step
    orig_plan = ctrl.plan
    ctrl.plan = _hook_timer(times["plan"], orig_plan, sync=True)
    ctrl.attach()

    rec = None
    streams = {"agent_rgb": [], "left_wrist_rgb": [], "right_wrist_rgb": []}
    if args.draw_bbox:
        streams["debug_bbox"] = []
    if args.record:
        rec = attach_profile_recorder(
            env, streams, args.record_every, {}, args.draw_bbox, times, args
        )

    hooked = env.task.scene.step

    def total_step():
        t0 = time.perf_counter()
        hooked()
        times["step_total"].append(time.perf_counter() - t0)

    env.task.scene.step = total_step

    t_ep0 = time.perf_counter()
    play_error = None
    try:
        env.task.play_once()
    except Exception as exc:
        play_error = str(exc)
        print(f"play_once failed: {exc}")
    episode_s = time.perf_counter() - t_ep0
    if rec is not None:
        env.task.scene.step = rec["orig_step"]
    n_record = 0 if rec is None else len(rec.get("d_min") or [])
    env.close()

    n_steps = len(times["step_total"])
    tags = (
        "physics",
        "step_total",
        "plan",
        "distance",
        "images",
        "update_render",
        "take_picture",
        "get_rgb",
        "get_depth",
        "observer",
    )
    parts = {k: _sec(times[k]) for k in tags}
    collect_s = (
        (parts["plan"]["total_s"] or 0.0)
        + (parts["images"]["total_s"] or 0.0)
        + (parts["distance"]["total_s"] or 0.0)
    )
    out = {
        "task": args.task,
        "embodiment": embodiment,
        "seed": seed,
        "skip_head": bool(args.skip_head),
        "skip_depth": bool(args.skip_depth),
        "skip_observer": bool(args.skip_observer),
        "n_steps": n_steps,
        "n_record_frames": n_record,
        "episode_s": episode_s,
        "data_collection_s": collect_s,
        "play_error": play_error,
        **parts,
    }
    return out


def _line(name: str, block: dict, extra: str = "") -> str:
    n = block.get("times") or 0
    if n == 0:
        return f"  {name:<28}  (did not run)"
    each = block["each_s"]
    total = block["total_s"]
    suffix = f"  {extra}" if extra else ""
    return f"  {name:<28}  {n:4d} times   {each:7.4f} s each   {total:7.3f} s total{suffix}"


def _print_summary(summary: dict) -> None:
    collect = summary["data_collection_s"]
    ep = summary["episode_s"]
    pct = 0.0 if not ep else 100.0 * collect / ep
    extras = []
    if summary["skip_head"]:
        extras.append("no head camera")
    if summary["skip_depth"]:
        extras.append("no depth")
    if summary["skip_observer"]:
        extras.append("no observer")
    extra = f"  [{', '.join(extras)}]" if extras else ""
    print(
        f"{summary['task']} / {summary['embodiment']}  seed={summary['seed']}{extra}"
    )
    print(
        f"  episode {ep:.3f} s   "
        f"{summary['n_steps']} physics steps   "
        f"{summary['n_record_frames']} recorded frames"
    )
    print()
    print("Data collection")
    print(_line("planning", summary["plan"], "(a few times per episode)"))
    print(_line("generating images", summary["images"], "(once per recorded frame)"))
    print(_line("  scene update", summary["update_render"]))
    print(_line("  wrists+head render", summary["take_picture"]))
    print(_line("  RGB read", summary["get_rgb"]))
    print(_line("  depth read", summary["get_depth"]))
    print(_line("  extra observer cam", summary["observer"]))
    print(_line("calculating distance", summary["distance"], "(once per recorded frame)"))
    print(
        f"  {'DATA COLLECTION TOTAL':<28}  {collect:7.3f} s   "
        f"({pct:.0f}% of episode)"
    )
    print()
    print("Not data collection")
    print(_line("physics", summary["physics"]))


def main() -> None:
    args = parse_args()
    jobs = [dict(vars(args))]
    if args.compare_skip_head:
        second = dict(vars(args))
        second["skip_head"] = True
        jobs.append(second)

    results = []
    for job_i, job in enumerate(jobs):
        ns = Namespace(**job)
        label = "no head camera" if ns.skip_head else "full"
        for ep in range(max(args.episodes, 1)):
            seed = args.seed + ep
            print(f"\n=== {label}  episode {ep + 1}/{args.episodes}  seed={seed} ===")
            summary = profile_episode(ns, seed)
            results.append(summary)
            _print_summary(summary)

    if args.compare_skip_head and len(results) >= 2:
        full = next((r for r in results if not r["skip_head"]), None)
        skip = next((r for r in results if r["skip_head"]), None)
        if full and skip:
            print("\n=== with head camera vs without (seconds) ===")
            for key, label in (
                ("images", "images each frame"),
                ("take_picture", "wrists+head render each frame"),
                ("data_collection_s", "data collection whole episode"),
                ("episode_s", "whole episode"),
            ):
                if key == "data_collection_s" or key == "episode_s":
                    a, b = full[key], skip[key]
                else:
                    a, b = full[key]["each_s"], skip[key]["each_s"]
                if a is None or b is None:
                    continue
                print(f"  {label:<32}  with={a:.4f}  without={b:.4f}  saved={a - b:.4f}")

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "episodes": results}, handle, indent=2, default=str)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
