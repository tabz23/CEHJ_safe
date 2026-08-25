#!/usr/bin/env python3
"""Run one RoboTwin expert episode with optional path obstacle.

Params and options (also `python run.py --help`):

  --task              One of the CEHJ tasks (default: place_empty_cup).
                      Safety set: place_empty_cup, move_can_pot, place_can_basket,
                      place_container_plate, place_shoe, stack_blocks_two,
                      place_object_stand, place_mouse_pad, click_bell, press_stapler
                      Bimanual set: place_container_plate, place_burger_fries, place_cans_plasticbox,
                      stack_blocks_two, place_can_basket, place_bread_basket,
                      grab_roller, pick_dual_bottles, stack_bowls_two

  --embodiment        Dual-arm robot (default: piper).
                      piper | franka | ur5 | arx | aloha
                      (Aloha is the native bimanual aloha-agilex articulation.)

  --seed              Scene RNG. Same seed => same object poses. Change seed
                      (or use run_all.py) to get a new scene on reset.

  --cluttered / --no-cluttered
                      Extra RoboTwin-OD clutter off the task objects.
                      Default: --no-cluttered.

  --obstacle-mode     none | off_path | on_path   (default: none)
                      none     : no safety obstacle
                      off_path : obstacle on table, beside the expert EE→target line
                      on_path  : obstacle on that line (should hit if plan ignores it)

  --obstacle-model    RoboTwin-OD assets/objects/<name> (default: 086_woodenblock).
                      Always spawned static. Distance / CuRobo use the scaled OBB.
                      Presets (world AABB):
                        086_woodenblock   cube    10.3 cm
                        068_boxdrink      box     11.0 x 15.4 x 11.6 cm
                        105_sauce-can     can     10.0 x 11.6 x 10.0 cm
                        059_pencup        cup     9.8 x 11.7 x 9.8 cm
                        071_can           can     7.1 x 9.6 x 7.1 cm
                        101_milk-tea      cup     13.6 x 15.5 x 13.6 cm
                        023_tissue-box    box     11.6 x 6.3 x 6.8 cm
                        038_milk-box      carton  6.9 x 12.2 x 6.5 cm
                        004_fluted-block  block   9.2 x 6.5 x 9.0 cm
                        073_rubikscube    cube    6.5 x 6.8 x 7.7 cm
                      Any other objects/<name> folder also works.

  --place-mode        geometric | waypoint   (default: geometric)
                      geometric : spawn between current EE and target (no extra plan)
                      waypoint  : plan once without the block, snap onto that EE path,
                                  reset same seed, run again

  --plan-mode         ignore_obstacle | no_ignore_obstacle
                      (default: ignore_obstacle)
                      ignore_obstacle      : CuRobo world = table only (can hit the obstacle)
                      no_ignore_obstacle   : add the obstacle to CuRobo MotionGen and replan around it

  --unsafe-level      ignored (kept so old scripts still parse). Placement t is
                      Uniform[0.3, 0.7] from the seed, same for none/off_path/on_path.

  --arm               auto | left | right   (default: auto)
                      auto uses the same rule as the task expert.

  --draw-bbox         Write only debug_bbox.mp4 (CuRobo spheres + obstacle OBB).
                      Does not save agent_rgb / wrist RGB.

  --controller        residual | nominal | vanilla_play_once | plan_play_once_everyk
                      (default: residual; residual is 0 today)
                      vanilla_play_once     : current play_once (one plan per Action)
                      plan_play_once_everyk : receding horizon, replan every --replan-k steps

  --replan-k          For plan_play_once_everyk only (default 20).

Examples:
  python run.py --task place_empty_cup --embodiment piper --obstacle-mode none --seed 0
  python run.py --task place_empty_cup --embodiment piper --obstacle-mode on_path \\
      --place-mode geometric --plan-mode ignore_obstacle --draw-bbox
  python run.py --task place_empty_cup --embodiment piper --obstacle-mode on_path \\
      --place-mode waypoint --plan-mode no_ignore_obstacle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    _cehj = str(Path(__file__).resolve().parents[2])
    if _cehj not in sys.path:
        sys.path.insert(0, _cehj)
    __package__ = "main.envs"
    import main.envs  # noqa: F401  ensure parent package exists for relative imports

from main.timing import EpisodeClock, log_time

from .controller import (
    CONTROLLER_NAMES,
    CuroboIKController,
    ResidualController,
    controller_class,
    make_controller,
)
from .env import CEHJ_ROOT, DEFAULT_MSAA, RECORD_SIZE, Env
from .obstacle import (
    OBSTACLE_MODEL,
    OBSTACLE_PRESETS,
    choose_and_spawn,
    update_curobo_world,
)
from .record import (
    EpisodeStepTimeout,
    attach_recorder,
    detach_recorder,
    save_video,
    write_hold_trace,
)
from .tasks import ALL_TASKS, resolve_arm, TASK_SPECS

CONTROLLERS = {
    "residual": ResidualController,
    "nominal": CuroboIKController,
    "vanilla_play_once": controller_class("vanilla_play_once"),
    "plan_play_once_everyk": controller_class("plan_play_once_everyk"),
}

DEFAULT_OUTPUT = CEHJ_ROOT / "outputs" / "ihab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="place_empty_cup", choices=ALL_TASKS)
    parser.add_argument("--embodiment", default="piper")
    parser.add_argument("--seed", type=int, default=0, help="Scene RNG. Change for a new layout.")
    parser.add_argument("--arm-distance", type=float, default=0.6)
    parser.add_argument("--cluttered", action="store_true")
    parser.add_argument("--no-cluttered", action="store_true")
    parser.add_argument(
        "--obstacle-mode",
        choices=("none", "off_path", "on_path"),
        default="none",
    )
    parser.add_argument(
        "--obstacle-model",
        default=OBSTACLE_MODEL,
        help="RoboTwin-OD assets/objects/<name>, always static. "
        + "; ".join(f"{k}={v}" for k, v in OBSTACLE_PRESETS.items())
        + ". Any other objects/ folder also works.",
    )
    parser.add_argument(
        "--obstacle-model-id",
        type=int,
        default=0,
        help="model_data<id>.json variant (default 0).",
    )
    parser.add_argument("--place-mode", choices=("geometric", "waypoint"), default="geometric")
    parser.add_argument(
        "--plan-mode",
        choices=("ignore_obstacle", "no_ignore_obstacle"),
        default="ignore_obstacle",
    )
    parser.add_argument(
        "--unsafe-level",
        default="2",
        help="Ignored. Obstacle t is Uniform[0.3, 0.7] from the seed.",
    )
    parser.add_argument("--arm", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--controller", choices=CONTROLLER_NAMES, default="residual")
    parser.add_argument(
        "--replan-k",
        type=int,
        default=20,
        help="For plan_play_once_everyk: execute this many joint steps, then replan.",
    )
    parser.add_argument(
        "--mpc-window-max",
        type=int,
        default=20,
        help="Max K-step window clips to save (plan_play_once_everyk). 0 = no cap.",
    )
    parser.add_argument(
        "--mpc-window-stride",
        type=int,
        default=200,
        help="Save one window clip every this many physics steps (2000 steps → 10 clips).",
    )
    parser.add_argument(
        "--no-mpc-windows",
        action="store_true",
        help="Do not write mpc_windows/*.mp4 debug clips (episode videos still saved).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Stop play_once after this many physics steps. 0 = no limit.",
    )
    parser.add_argument(
        "--msaa",
        type=int,
        default=DEFAULT_MSAA,
        help="SAPIEN raster MSAA. Default 2. 1 is stock; 4/8 is slower.",
    )
    parser.add_argument(
        "--video-res",
        default=f"{RECORD_SIZE[0]}x{RECORD_SIZE[1]}",
        help="Saved video WxH (default 320x240). Observer renders at 2x then downsamples.",
    )
    parser.add_argument("--record-every", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list-params", action="store_true", help="Print param list and exit.")
    return parser.parse_args()


def _corridor_t(seed: int) -> float:
    """Same t for none / off_path / on_path of one seed. Uniform in [0.3, 0.7]."""
    rng = np.random.RandomState(int(seed) + 917)
    return float(rng.uniform(0.30, 0.70))


def _t_tag(t: float) -> str:
    return f"t{int(round(float(t) * 100.0)):02d}"


def _video_res(value: str) -> tuple[int, int]:
    text = str(value).lower().replace(" ", "")
    if "x" not in text:
        raise ValueError("--video-res must look like 320x240")
    w, h = text.split("x", 1)
    width, height = int(w), int(h)
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("--video-res must be even positive integers")
    return width, height


def _controller_tag(args: argparse.Namespace) -> str:
    name = str(getattr(args, "controller", "residual"))
    if name == "plan_play_once_everyk":
        return f"plan_everyk_k{int(getattr(args, 'replan_k', 20))}"
    return name


def _make_env(args: argparse.Namespace, seed: int) -> tuple[object, object]:
    cluttered = bool(args.cluttered) and not bool(args.no_cluttered)
    ctrl_cls = controller_class(args.controller)
    ctrl_cls.install()
    record_wh = _video_res(getattr(args, "video_res", f"{RECORD_SIZE[0]}x{RECORD_SIZE[1]}"))
    env = Env(
        args.task,
        args.embodiment,
        seed,
        args.arm_distance,
        cluttered=cluttered,
        settle=True,
        msaa=int(getattr(args, "msaa", DEFAULT_MSAA)),
        observer_size=(record_wh[0] * 2, record_wh[1] * 2),
        record_size=record_wh,
    )
    ctrl = make_controller(
        args.controller,
        env,
        replan_k=int(getattr(args, "replan_k", 20)),
        replan_max=int(getattr(args, "replan_max", 2500)),
    )
    ctrl.attach()
    return env, ctrl


def _record_ee_path(env, arm: str, record_every: int) -> np.ndarray:
    path = []
    orig = env.task.scene.step
    n = {"i": 0}

    def step():
        orig()
        n["i"] += 1
        if n["i"] % max(record_every, 1) != 0:
            return
        pose = env.robot.get_left_ee_pose() if arm == "left" else env.robot.get_right_ee_pose()
        path.append(np.asarray(pose[:3], dtype=np.float64))

    env.task.scene.step = step
    try:
        env.task.play_once()
    except Exception as exc:
        print(f"[run] waypoint probe play_once failed: {exc}")
    env.task.scene.step = orig
    return np.asarray(path, dtype=np.float64) if path else np.zeros((0, 3))


def run_episode(args: argparse.Namespace) -> dict:
    cluttered = bool(args.cluttered) and not bool(args.no_cluttered)
    t = _corridor_t(args.seed)
    ee_path = None
    arm = args.arm

    if args.obstacle_mode != "none" and args.place_mode == "waypoint":
        print("[run] waypoint probe: expert with no extra obstacle...")
        probe, _ = _make_env(args, args.seed)
        spec = TASK_SPECS[args.task]
        arm = resolve_arm(probe.task, spec, args.arm)
        ee_path = _record_ee_path(probe, arm, args.record_every)
        print(f"[run] recorded {len(ee_path)} EE waypoints on {arm}")
        probe.close()

    CuroboIKController.reset_session_timers()
    t_env = time.perf_counter()
    env, ctrl = _make_env(args, args.seed)
    dt_env = time.perf_counter() - t_env
    log_time("setup env+robot+planners", dt_env)

    clock = EpisodeClock()
    ctrl.clock = clock
    clock.t_setup_env = dt_env
    clock.t_curobo_warmup = CuroboIKController._session_warmup_s
    clock.t_curobo_reuse = CuroboIKController._session_reuse_s
    clock.n_curobo_warmup = CuroboIKController._session_warmup_n
    clock.n_curobo_reuse = CuroboIKController._session_reuse_n
    clock.wrap_scene(env.task.scene)
    ctrl_tag = _controller_tag(args)
    replan_k = int(getattr(args, "replan_k", 20)) if args.controller == "plan_play_once_everyk" else None
    print(
        f"{args.task} / {env.embodiment} seed={args.seed} "
        f"obstacle={args.obstacle_mode} model={getattr(args, 'obstacle_model', OBSTACLE_MODEL)} "
        f"place={args.place_mode} "
        f"plan={args.plan_mode} cluttered={cluttered} t={t:.2f} "
        f"controller={ctrl_tag} planner={type(env.robot.left_planner).__name__}"
    )

    t_spawn = time.perf_counter()
    actor, xyz, half, arm = choose_and_spawn(
        env,
        args.obstacle_mode,
        args.place_mode,
        t,
        arm,
        ee_path=ee_path,
        obstacle_model=getattr(args, "obstacle_model", OBSTACLE_MODEL),
        obstacle_model_id=int(getattr(args, "obstacle_model_id", 0) or 0),
    )
    clock.t_spawn = time.perf_counter() - t_spawn
    log_time("spawn obstacle", clock.t_spawn, mode=args.obstacle_mode)

    env.obstacle = actor
    env.obstacle_xyz = xyz
    env.obstacle_half = half
    env.obstacle_model = getattr(args, "obstacle_model", OBSTACLE_MODEL)
    t_world = time.perf_counter()
    update_curobo_world(env.robot)
    if args.plan_mode == "no_ignore_obstacle" and xyz is not None:
        obs = env.obstacle
        if hasattr(obs, "get_pose"):
            quat = np.array(obs.get_pose().q, dtype=np.float64)
        else:
            quat = np.array(getattr(obs, "actor", obs).get_pose().q, dtype=np.float64)
        update_curobo_world(env.robot, xyz, quat, half)
        print("[run] CuRobo world includes safety_obstacle")
    elif xyz is not None:
        print("[run] CuRobo ignores safety_obstacle")
    clock.t_world = time.perf_counter() - t_world
    log_time("update CuRobo world", clock.t_world)

    tag = (
        f"{args.obstacle_mode}_{args.place_mode}_{args.plan_mode}_"
        f"{_t_tag(t)}_seed{args.seed}"
    )
    if cluttered:
        tag += "_clutter"
    out_dir = Path(args.output).expanduser() / args.task / env.embodiment / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if (
        args.controller == "plan_play_once_everyk"
        and hasattr(ctrl, "window_dir")
        and not bool(getattr(args, "no_mpc_windows", False))
    ):
        ctrl.window_dir = out_dir / "mpc_windows"
        ctrl.window_draw_bbox = bool(args.draw_bbox)
        ctrl.window_fps = 1.0
        ctrl.window_max = int(getattr(args, "mpc_window_max", 20))
        ctrl.window_stride_steps = max(1, int(getattr(args, "mpc_window_stride", 200)))
        n_at_2k = max(1, 2000 // ctrl.window_stride_steps)
        print(
            f"[run] MPC window clips → {ctrl.window_dir}  "
            f"every {ctrl.window_stride_steps} steps  max={ctrl.window_max}  "
            f"(2000 steps → {n_at_2k} clips)  "
            f"1 fps ({ctrl.k} steps ≈ {ctrl.k}s of playback)"
        )
    elif args.controller == "plan_play_once_everyk":
        print("[run] MPC window clips disabled (--no-mpc-windows)")

    streams = {"debug_bbox": []} if args.draw_bbox else {}
    overlay = {
        "obstacle_mode": args.obstacle_mode,
        "obstacle_model": getattr(args, "obstacle_model", OBSTACLE_MODEL),
        "plan_mode": args.plan_mode,
        "place_mode": args.place_mode,
        "seed": args.seed,
        "controller": ctrl_tag,
        "replan_k": replan_k,
        "clock": clock,
    }
    max_steps = int(getattr(args, "max_steps", 3000) or 0)
    rec = attach_recorder(
        env,
        streams,
        args.record_every,
        overlay,
        args.draw_bbox,
        clock=clock,
        max_steps=max_steps,
    )
    play_error = None
    timed_out = False
    clock.start()
    try:
        try:
            env.task.play_once()
        except EpisodeStepTimeout as exc:
            timed_out = True
            play_error = str(exc)
            print(f"play_once timeout: {exc}")
        except Exception as exc:
            play_error = str(exc)
            print(f"play_once failed: {exc}")
    finally:
        clock.stop()
        detach_recorder(env, rec)
    print()

    success = False
    t_ok = time.perf_counter()
    try:
        success = bool(env.task.check_success())
    except Exception as exc:
        print(f"check_success warning: {exc}")
    clock.t_check_success = time.perf_counter() - t_ok
    log_time("check_success", clock.t_check_success, success=success)
    plan_success = getattr(env.task, "plan_success", None)

    def _min_finite(vals):
        nums = [d for d in vals if np.isfinite(d)]
        return float(min(nums)) if nums else float("inf")

    d_min = _min_finite(rec["d_min"])
    d_robot = _min_finite(rec.get("d_robot", rec["d_min"]))
    d_system = _min_finite(rec.get("d_system", rec["d_min"]))
    d_left = _min_finite(rec.get("d_left", []))
    d_right = _min_finite(rec.get("d_right", []))
    d_left_held = _min_finite(rec.get("d_left_held", []))
    d_right_held = _min_finite(rec.get("d_right_held", []))
    held_labels = [h for h in rec.get("holding", []) if h]
    held_left = [h for h in rec.get("holding_left", []) if h]
    held_right = [h for h in rec.get("holding_right", []) if h]
    any_contact = bool(any(rec["contact"]))
    print(
        f"success={success} plan_success={plan_success} "
        f"dL={d_left} dR={d_right} dLh={d_left_held} dRh={d_right_held} dmin={d_min} "
        f"contact={any_contact} held={held_labels[-1] if held_labels else None} "
        f"frames={len(streams.get('debug_bbox') or [])} "
        f"t_ep={clock.t_episode:.2f}s plan={clock.t_plan:.2f}s "
        f"(first={clock.t_plan_initial:.2f}s n={clock.n_plan_initial} "
        f"replan={clock.t_plan_replan:.2f}s n={clock.n_replan}) "
        f"phys={clock.t_physics:.2f}s cam={clock.t_render:.2f}s dist={clock.t_metric:.2f}s "
        f"n_plan={clock.n_plan}"
    )

    outputs = {}
    n_frames = len(streams.get("debug_bbox") or [])
    if n_frames:
        for name, frames in streams.items():
            if not frames:
                continue
            t_vid = time.perf_counter()
            outputs[name] = str(save_video(frames, out_dir / f"{name}.mp4", args.fps))
            dt_vid = time.perf_counter() - t_vid
            clock.add("video_encode", dt_vid)
            log_time(f"episode mp4 {name}", dt_vid, n_frames=len(frames))
    else:
        print("[run] no frames recorded")
    t_hold = time.perf_counter()
    hold_trace = str(write_hold_trace(rec.get("csv_rows", []), out_dir / "hold_trace.csv"))
    clock.t_hold_trace = time.perf_counter() - t_hold
    log_time("hold_trace.csv", clock.t_hold_trace, n_rows=len(rec.get("csv_rows") or []))
    plans_path = out_dir / "plans.jsonl"
    with plans_path.open("w", encoding="utf-8") as handle:
        for event in clock.plan_events:
            handle.write(json.dumps(event) + "\n")
    timing = clock.summary()

    summary = {
        "task": args.task,
        "embodiment": env.embodiment,
        "seed": args.seed,
        "cluttered": cluttered,
        "obstacle_mode": args.obstacle_mode,
        "obstacle_model": getattr(args, "obstacle_model", OBSTACLE_MODEL),
        "obstacle_model_id": int(getattr(args, "obstacle_model_id", 0) or 0),
        "place_mode": args.place_mode,
        "plan_mode": args.plan_mode,
        "controller": args.controller,
        "controller_tag": ctrl_tag,
        "replan_k": replan_k,
        "replan_max": int(getattr(args, "replan_max", 2500)) if args.controller == "plan_play_once_everyk" else None,
        "obstacle_t": t,
        "unsafe_level": getattr(args, "unsafe_level", None),
        "arm": arm,
        "success": success,
        "plan_success": plan_success,
        "d_min": None if not np.isfinite(d_min) else d_min,
        "d_robot": None if not np.isfinite(d_robot) else d_robot,
        "d_system": None if not np.isfinite(d_system) else d_system,
        "d_left": None if not np.isfinite(d_left) else d_left,
        "d_right": None if not np.isfinite(d_right) else d_right,
        "d_left_held": None if not np.isfinite(d_left_held) else d_left_held,
        "d_right_held": None if not np.isfinite(d_right_held) else d_right_held,
        "held_object": held_labels[-1] if held_labels else None,
        "held_left": held_left[-1] if held_left else None,
        "held_right": held_right[-1] if held_right else None,
        "contact": any_contact,
        "play_error": play_error,
        "timed_out": timed_out,
        "max_steps": max_steps if max_steps > 0 else None,
        "obstacle_xyz": None if xyz is None else xyz.tolist(),
        "obstacle_half": None if half is None else np.asarray(half, dtype=np.float64).tolist(),
        "videos": outputs,
        "hold_trace": hold_trace,
        "plans": str(plans_path),
        "n_frames": n_frames,
        "n_steps": rec.get("n", 0),
        "n_plans": clock.n_plan,
        "n_plan_ok": clock.n_plan_ok,
        "n_plan_fail": clock.n_plan_fail,
        "n_plan_initial": clock.n_plan_initial,
        "n_replans": clock.n_replan,
        "n_mpc_windows": clock.n_mpc_windows,
        "timing": timing,
    }
    clock.dump()
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    env.close()
    return summary


def main() -> None:
    args = parse_args()
    if args.list_params:
        print(__doc__)
        return
    summary = run_episode(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "videos"}, indent=2, default=str))
    for name, path in summary.get("videos", {}).items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
