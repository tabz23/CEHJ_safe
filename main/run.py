#!/usr/bin/env python3
"""Run one RoboTwin expert episode with optional path obstacle.

Params and options (also `python run.py --help`):

  --task              One of the 10 safety tasks (default: place_empty_cup).
                      place_empty_cup, move_can_pot, place_can_basket,
                      place_container_plate, place_shoe, stack_blocks_two,
                      place_object_stand, place_mouse_pad, click_bell, press_stapler

  --embodiment        Dual-arm robot (default: piper).
                      piper | franka | ur5 | arx
                      ( Dual-arm Piper / Franka / UR5 / ARX )

  --seed              Scene RNG. Same seed => same object poses. Change seed
                      (or use run_all.py) to get a new scene on reset.

  --cluttered / --no-cluttered
                      Extra RoboTwin-OD clutter off the task objects.
                      Default: --no-cluttered.

  --obstacle-mode     none | off_path | on_path   (default: none)
                      none     : no wooden block
                      off_path : block on table, beside the expert EE→target line
                      on_path  : block on that line (should hit if plan ignores it)

  --place-mode        geometric | waypoint   (default: geometric)
                      geometric : spawn between current EE and target (no extra plan)
                      waypoint  : plan once without the block, snap onto that EE path,
                                  reset same seed, run again

  --plan-mode         ignore_obstacle | no_ignore_obstacle
                      (default: ignore_obstacle)
                      ignore_obstacle      : CuRobo world = table only (can hit the block)
                      no_ignore_obstacle   : add the block to CuRobo MotionGen and replan around it

  --unsafe-level      1 | 2 | 3 | random   (default: 2)
                      How on-the-line / large the on_path block is (or how far off_path is).

  --arm               auto | left | right   (default: auto)
                      auto uses the same rule as the task expert.

  --draw-bbox         Also write debug_bbox.mp4 (CuRobo spheres + obstacle OBB).

  --controller        residual | nominal   (default: residual; residual is 0 today)

Examples:
  python run.py --task place_empty_cup --embodiment piper --obstacle-mode none --seed 0
  python run.py --task place_empty_cup --embodiment piper --obstacle-mode on_path \\
      --place-mode geometric --plan-mode ignore_obstacle --unsafe-level 3 --draw-bbox
  python run.py --task place_empty_cup --embodiment piper --obstacle-mode on_path \\
      --place-mode waypoint --plan-mode no_ignore_obstacle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MAIN_DIR = Path(__file__).resolve().parent
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from controller import CuroboIKController, ResidualController
from env import CEHJ_ROOT, Env
from obstacle import choose_and_spawn, update_curobo_world
from record import attach_recorder, detach_recorder, save_video
from tasks import SAFETY_TASKS, resolve_arm, TASK_SPECS

CONTROLLERS = {
    "residual": ResidualController,
    "nominal": CuroboIKController,
}

DEFAULT_OUTPUT = CEHJ_ROOT / "outputs" / "ihab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="place_empty_cup", choices=SAFETY_TASKS)
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
    parser.add_argument("--place-mode", choices=("geometric", "waypoint"), default="geometric")
    parser.add_argument(
        "--plan-mode",
        choices=("ignore_obstacle", "no_ignore_obstacle"),
        default="ignore_obstacle",
    )
    parser.add_argument("--unsafe-level", default="2", help="1, 2, 3, or random")
    parser.add_argument("--arm", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--controller", choices=tuple(CONTROLLERS), default="residual")
    parser.add_argument("--record-every", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list-params", action="store_true", help="Print param list and exit.")
    return parser.parse_args()


def _level(value: str, seed: int) -> int:
    if str(value).lower() == "random":
        rng = np.random.RandomState(seed + 917)
        return int(rng.choice([1, 2, 3]))
    lvl = int(value)
    if lvl not in (1, 2, 3):
        raise ValueError("--unsafe-level must be 1, 2, 3, or random")
    return lvl


def _make_env(args: argparse.Namespace, seed: int) -> tuple[object, object]:
    cluttered = bool(args.cluttered) and not bool(args.no_cluttered)
    ctrl_cls = CONTROLLERS[args.controller]
    ctrl_cls.install()
    env = Env(
        args.task,
        args.embodiment,
        seed,
        args.arm_distance,
        cluttered=cluttered,
        settle=True,
    )
    ctrl = ctrl_cls(env)
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
    level = _level(args.unsafe_level, args.seed)
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

    env, ctrl = _make_env(args, args.seed)
    print(
        f"{args.task} / {env.embodiment} seed={args.seed} "
        f"obstacle={args.obstacle_mode} place={args.place_mode} "
        f"plan={args.plan_mode} cluttered={cluttered} level={level} "
        f"controller={type(ctrl).__name__} planner={type(env.robot.left_planner).__name__}"
    )

    actor, xyz, half, arm = choose_and_spawn(
        env,
        args.obstacle_mode,
        args.place_mode,
        level,
        arm,
        ee_path=ee_path,
    )
    env.obstacle = actor
    env.obstacle_xyz = xyz
    env.obstacle_half = half
    # Cached MotionGen may still hold a cuboid from a previous episode.
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

    streams = {"agent_rgb": [], "left_wrist_rgb": [], "right_wrist_rgb": []}
    if args.draw_bbox:
        streams["debug_bbox"] = []
    overlay = {
        "obstacle_mode": args.obstacle_mode,
        "plan_mode": args.plan_mode,
        "place_mode": args.place_mode,
        "seed": args.seed,
    }
    rec = attach_recorder(env, streams, args.record_every, overlay, args.draw_bbox)
    play_error = None
    try:
        try:
            env.task.play_once()
        except Exception as exc:
            play_error = str(exc)
            print(f"play_once failed: {exc}")
    finally:
        detach_recorder(env, rec)
    print()

    success = False
    try:
        success = bool(env.task.check_success())
    except Exception as exc:
        print(f"check_success warning: {exc}")
    plan_success = getattr(env.task, "plan_success", None)

    def _min_finite(vals):
        nums = [d for d in vals if np.isfinite(d)]
        return float(min(nums)) if nums else float("inf")

    d_min = _min_finite(rec["d_min"])
    d_robot = _min_finite(rec.get("d_robot", rec["d_min"]))
    d_system = _min_finite(rec.get("d_system", rec["d_min"]))
    held_labels = [h for h in rec.get("holding", []) if h]
    any_contact = bool(any(rec["contact"]))
    print(
        f"success={success} plan_success={plan_success} "
        f"d_robot={d_robot} d_sys={d_system} contact={any_contact} "
        f"held={held_labels[-1] if held_labels else None} frames={len(streams['agent_rgb'])}"
    )

    tag = (
        f"{args.obstacle_mode}_{args.place_mode}_{args.plan_mode}_"
        f"l{level}_seed{args.seed}"
    )
    if cluttered:
        tag += "_clutter"
    out_dir = args.output.expanduser() / args.task / env.embodiment / tag
    outputs = {}
    if streams["agent_rgb"]:
        for name, frames in streams.items():
            outputs[name] = str(save_video(frames, out_dir / f"{name}.mp4", args.fps))
    else:
        print("[run] no frames recorded")

    summary = {
        "task": args.task,
        "embodiment": env.embodiment,
        "seed": args.seed,
        "cluttered": cluttered,
        "obstacle_mode": args.obstacle_mode,
        "place_mode": args.place_mode,
        "plan_mode": args.plan_mode,
        "unsafe_level": level,
        "arm": arm,
        "success": success,
        "plan_success": plan_success,
        "d_min": None if not np.isfinite(d_min) else d_min,
        "d_robot": None if not np.isfinite(d_robot) else d_robot,
        "d_system": None if not np.isfinite(d_system) else d_system,
        "held_object": held_labels[-1] if held_labels else None,
        "contact": any_contact,
        "play_error": play_error,
        "obstacle_xyz": None if xyz is None else xyz.tolist(),
        "videos": outputs,
        "n_frames": len(streams["agent_rgb"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    env.close()
    return summary


def main() -> None:
    args = parse_args()
    if args.list_params:
        print(__doc__)
        return
    summary = run_episode(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "videos"}, indent=2))
    for name, path in summary.get("videos", {}).items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
