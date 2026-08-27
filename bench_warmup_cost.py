#!/usr/bin/env python3
"""Benchmark the average per-step cost of one WARMUP collection episode.

Same structure as bench_step_cost.py but for the warmup path
(RolloutController mode='collect', nominal-only, no filter). Breaks one
control tick down into:

  physics      — scene.step() (PhysX), runs at 250 Hz
  plan         — ctrl.plan() (CuRobo, a few calls per episode)
  encoder_obs  — env.get_encoder_obs(): camera renders + resize + obs dict
  compute_h    — distance_info + h (every control tick)
  encode_vis   — HoloBrainEncoder.encode_visual_tokens (the frozen forward)
  encode_state — encode_joint_angles (robot state encoder)
  body         — BodyTokenExtractor.extract (analytic features + Jacobians)
  buffer       — StepBuffer.append (memmap write)

Usage:
  conda activate RoboTwin
  python bench_warmup_cost.py --task place_bread_basket --embodiment piper
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CEHJ_ROOT = Path(__file__).resolve().parent
if str(CEHJ_ROOT) not in sys.path:
    sys.path.insert(0, str(CEHJ_ROOT))


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="place_bread_basket")
    p.add_argument("--embodiment", default="piper")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from main.train.buffer import StepBuffer
    from main.train.collect import CKPT_DIR, make_kinematics
    from main.train.config import FrozenConfig
    from main.network.encoder import HoloBrainEncoder
    import main.train.rollout as rollout_mod
    from main.train.rollout import RolloutController

    cfg = FrozenConfig(task=args.task, embodiment=args.embodiment,
                       n_episodes=1)
    kin = make_kinematics(CKPT_DIR, args.embodiment)
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    buf = StepBuffer(Path("/tmp/bench_warmup_buf"),
                     capacity=cfg.max_steps_per_episode + 2,
                     header=cfg.to_dict())

    ro = RolloutController(cfg, seed=args.seed, mode="collect",
                           encoder=encoder, kin=kin, buf=buf,
                           max_steps=int(cfg.max_steps_per_episode
                                         * 250.0 / 25.0))
    env = ro.env

    timers = {k: Timer() for k in (
        "physics", "plan", "encoder_obs", "compute_h", "encode_vis",
        "encode_state", "body", "buffer")}

    # physics
    orig_scene_step = env.task.scene.step
    def physics_step():
        t0 = time.perf_counter()
        orig_scene_step()
        timers["physics"].add(time.perf_counter() - t0)
    env.task.scene.step = physics_step
    # RolloutController.run() captures env.task.scene.step as its _orig_step,
    # so this physics timing wrapper stays inside the tick hook chain.

    # plan
    orig_plan = ro.ctrl.plan
    def timed_plan(*a, **kw):
        t0 = time.perf_counter()
        out = orig_plan(*a, **kw)
        timers["plan"].add(time.perf_counter() - t0)
        return out
    ro.ctrl.plan = timed_plan

    # encoder obs (renders cameras)
    orig_geo = env.get_encoder_obs
    def timed_geo(*a, **kw):
        t0 = time.perf_counter()
        out = orig_geo(*a, **kw)
        timers["encoder_obs"].add(time.perf_counter() - t0)
        return out
    env.get_encoder_obs = timed_geo

    # compute_h
    orig_compute_h = rollout_mod.compute_h
    def timed_compute_h(*a, **kw):
        t0 = time.perf_counter()
        out = orig_compute_h(*a, **kw)
        timers["compute_h"].add(time.perf_counter() - t0)
        return out
    rollout_mod.compute_h = timed_compute_h

    # encoder forwards
    orig_evis = encoder.encode_visual_tokens
    def timed_evis(*a, **kw):
        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_evis(*a, **kw)
        torch.cuda.synchronize()
        timers["encode_vis"].add(time.perf_counter() - t0)
        return out
    encoder.encode_visual_tokens = timed_evis

    orig_estate = encoder.encode_joint_angles
    def timed_estate(*a, **kw):
        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_estate(*a, **kw)
        torch.cuda.synchronize()
        timers["encode_state"].add(time.perf_counter() - t0)
        return out
    encoder.encode_joint_angles = timed_estate

    # body extractor
    orig_extract = ro.extractor.extract
    def timed_extract(*a, **kw):
        t0 = time.perf_counter()
        out = orig_extract(*a, **kw)
        timers["body"].add(time.perf_counter() - t0)
        return out
    ro.extractor.extract = timed_extract

    # buffer append
    orig_append = buf.append
    def timed_append(*a, **kw):
        t0 = time.perf_counter()
        out = orig_append(*a, **kw)
        timers["buffer"].add(time.perf_counter() - t0)
        return out
    buf.append = timed_append

    # run() re-hooks scene.step: capture orig AFTER our physics wrapper by
    # pre-setting _orig_step is not possible — instead wrap run's hook chain:
    # run() does self._orig_step = self.env.task.scene.step (our physics_step),
    # so physics timing survives.
    t0 = time.perf_counter()
    trace = ro.run()
    wall = time.perf_counter() - t0

    steps = trace["n_physics"]
    n_ticks = len(trace["t"])
    print(f"\n{'=' * 84}")
    print(f"WARMUP {args.task} / {args.embodiment} seed={args.seed} "
          f"success={trace['success']}")
    print(f"physics steps: {steps}   control ticks: {n_ticks}   "
          f"wall: {wall:.2f}s   avg tick wall: {1e3 * wall / max(n_ticks, 1):.2f} ms")
    print(f"{'-' * 84}")
    print(f"{'component':<22}{'calls':>7}{'total(s)':>10}{'mean(ms)':>10}"
          f"{'ms/tick':>9}{'%wall':>8}")
    acc = 0.0
    for name in ("physics", "plan", "encoder_obs", "compute_h", "encode_vis",
                 "encode_state", "body", "buffer"):
        tm = timers[name]
        acc += tm.t
        print(f"{name:<22}{tm.n:>7}{tm.t:>10.2f}{tm.mean_ms:>10.2f}"
              f"{1e3 * tm.t / max(n_ticks, 1):>9.2f}{100 * tm.t / wall:>7.1f}%")
    print(f"{'-' * 84}")
    print(f"{'TOTAL (instrumented)':<22}{'':>7}{acc:>10.2f}{'':>10}"
          f"{1e3 * acc / max(n_ticks, 1):>9.2f}{100 * acc / wall:>7.1f}%")


if __name__ == "__main__":
    main()
