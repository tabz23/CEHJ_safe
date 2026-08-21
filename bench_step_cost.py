#!/usr/bin/env python3
"""Benchmark the average per-step cost of a CEHJ_safe episode.

Breaks one simulation step down into:

  physics    — scene.step() (PhysX simulation)
  plan       — controller.plan() (CuRobo IK + MotionGen, a few calls per episode)
  render     — camera image capture (observer Color + wrist cams via get_obs,
               every --record-every steps)
  distance   — distance_info() + detect_held_object() safety metrics

Reports calls, total time, mean time per call, and the cost amortized over
all simulation steps.

Usage:
  conda activate RoboTwin
  python bench_step_cost.py --task place_empty_cup --embodiment piper --seed 0
  python bench_step_cost.py --task stack_blocks_two --embodiment arx --record-every 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CEHJ_ROOT = Path(__file__).resolve().parent
if str(CEHJ_ROOT) not in sys.path:
    sys.path.insert(0, str(CEHJ_ROOT))

from main.envs.controller import CuroboIKController, ResidualController
from main.envs.distance import detect_held_object, distance_info
from main.envs.env import Env
from main.envs.record import observer_rgb

CONTROLLERS = {
    "residual": ResidualController,
    "nominal": CuroboIKController,
}


class Timer:
    def __init__(self) -> None:
        self.n = 0
        self.t = 0.0

    def add(self, dt: float) -> None:
        self.n += 1
        self.t += dt

    @property
    def mean_ms(self) -> float:
        return 1e3 * self.t / max(self.n, 1)

    def amortized_ms(self, steps: int) -> float:
        return 1e3 * self.t / max(steps, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="place_empty_cup")
    p.add_argument("--embodiment", default="piper")
    p.add_argument("--controller", choices=tuple(CONTROLLERS), default="residual")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arm-distance", type=float, default=0.6)
    p.add_argument("--record-every", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ctrl_cls = CONTROLLERS[args.controller]
    ctrl_cls.install()
    env = Env(args.task, args.embodiment, args.seed, args.arm_distance)
    ctrl = ctrl_cls(env)
    ctrl.attach()

    timers = {
        "physics": Timer(),          # scene.step (PhysX)
        "plan": Timer(),             # ctrl.plan (CuRobo motion gen)
        "render_observer": Timer(),  # observer camera Color picture
        "render_wrist": Timer(),     # task.get_obs (wrist cameras)
        "dist_hold": Timer(),        # detect_held_object (every step)
        "dist_info": Timer(),        # distance_info (recorded steps)
    }

    # --- instrument planning ---
    orig_plan = ctrl.plan

    def timed_plan(arm, target_pose, **kw):
        t0 = time.perf_counter()
        result = orig_plan(arm, target_pose, **kw)
        timers["plan"].add(time.perf_counter() - t0)
        return result

    ctrl.plan = timed_plan

    # --- instrument the simulation step ---
    orig_step = env.task.scene.step
    state = {"n": 0}
    record_every = max(args.record_every, 1)

    def step():
        t0 = time.perf_counter()
        orig_step()
        timers["physics"].add(time.perf_counter() - t0)
        state["n"] += 1

        t0 = time.perf_counter()
        detect_held_object(env)
        timers["dist_hold"].add(time.perf_counter() - t0)

        if state["n"] % record_every == 0:
            t0 = time.perf_counter()
            distance_info(env)
            timers["dist_info"].add(time.perf_counter() - t0)

            t0 = time.perf_counter()
            env.task.get_obs()
            timers["render_wrist"].add(time.perf_counter() - t0)

            t0 = time.perf_counter()
            observer_rgb(env.task)
            timers["render_observer"].add(time.perf_counter() - t0)

    env.task.scene.step = step

    # --- run one episode ---
    t0 = time.perf_counter()
    try:
        env.task.play_once()
    except Exception as exc:
        print(f"play_once failed: {exc}")
    wall = time.perf_counter() - t0
    env.task.scene.step = orig_step
    steps = state["n"]

    # --- report ---
    groups = [
        ("physics (scene.step)", timers["physics"]),
        ("plan (CuRobo)", timers["plan"]),
        ("render observer", timers["render_observer"]),
        ("render wrist", timers["render_wrist"]),
        ("distance held-obj", timers["dist_hold"]),
        ("distance info", timers["dist_info"]),
    ]
    print(f"\n{'=' * 78}")
    print(f"{args.task} / {env.embodiment} seed={args.seed} record_every={args.record_every}")
    print(f"sim steps: {steps}   wall time: {wall:.2f}s   avg step wall: {1e3 * wall / max(steps, 1):.2f} ms")
    print(f"{'-' * 78}")
    print(f"{'component':<22}{'calls':>7}{'total(s)':>10}{'mean(ms)':>10}{'ms/step':>9}{'%wall':>8}")
    acc = 0.0
    for name, tm in groups:
        acc += tm.t
        print(
            f"{name:<22}{tm.n:>7}{tm.t:>10.2f}{tm.mean_ms:>10.2f}"
            f"{tm.amortized_ms(steps):>9.2f}{100 * tm.t / wall:>7.1f}%"
        )
    print(f"{'-' * 78}")
    print(
        f"{'TOTAL (instrumented)':<22}{'':>7}{acc:>10.2f}{'':>10}"
        f"{1e3 * acc / max(steps, 1):>9.2f}{100 * acc / wall:>7.1f}%"
    )
    env.close()


if __name__ == "__main__":
    main()
