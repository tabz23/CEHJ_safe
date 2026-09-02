"""Generate high-res paper figure screenshots: one per embodiment, each on a
different task (all 5 embodiments x all 5 tasks covered).

Plays the nominal task with the safety obstacle in scene and captures the
observer camera at 1920x1440 at several sim-time points per episode; the
best candidate per pair is then chosen manually.

Usage (from CEHJ_safe/):
    python scripts/gen_paper_figs.py [--out /root/autodl-tmp/figs]
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

_CEHJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CEHJ))
_ROBOTWIN = _CEHJ / "RoboTwin"
if not _ROBOTWIN.is_dir():
    _ROBOTWIN = _CEHJ.parent / "RoboTwin"
sys.path.insert(0, str(_ROBOTWIN))  # envs.* (RoboTwin) imports
# RoboTwin opens ./assets/... at MODULE IMPORT time — chdir before any
# main.* / envs.* import (env.py does the same at env creation)
import os

os.chdir(_ROBOTWIN)

from main.train.config import FrozenConfig
from main.train.collect import ARM_DISTANCE
from main.envs.controller import CuroboIKController
from main.envs.env import Env
from main.envs.obstacle import choose_and_spawn, update_curobo_world
from main.envs.record import observer_rgb

PAIRS = [
    ("piper", "place_empty_cup"),
    ("franka-panda", "stack_blocks_two"),
    ("aloha-agilex", "place_bread_basket"),
    ("ARX-X5", "place_container_plate"),
    ("ur5-wsg", "stack_bowls_two"),
]
# physics steps at 250 Hz -> capture times 1.0/2.5/4.0/6.0 s
CAPTURE_STEPS = [250, 625, 1000, 1500]


def shoot(embodiment: str, task: str, seed: int, out_dir: Path,
          resolution=(1920, 1440)) -> list[Path]:
    # make_training_env with a hi-res observer: Env's ctor applies
    # patch_observer_resolution itself, so the size must go through the
    # constructor (patching beforehand gets overridden)
    CuroboIKController.install()
    env = Env(task, embodiment, seed,
              arm_distance=ARM_DISTANCE.get(embodiment, 0.6),
              control_freq=25.0, observer_size=resolution)
    from main.envs.run import _corridor_t

    cfg = FrozenConfig(task=task, embodiment=embodiment)
    actor, xyz, half, _arm = choose_and_spawn(
        env, cfg.obstacle_mode, "geometric", _corridor_t(seed), "auto",
        obstacle_model=getattr(cfg, "obstacle_model", "086_woodenblock"),
    )
    env.obstacle = actor
    env.obstacle_xyz = xyz
    env.obstacle_half = half
    update_curobo_world(env.robot)
    if env.obstacle is None:
        env.close()
        raise RuntimeError("obstacle spawn failed")
    task_obj = env.task
    orig_step = task_obj.scene.step
    state = {"n": 0}
    saved = []

    def hooked_step():
        orig_step()
        state["n"] += 1
        if state["n"] in CAPTURE_STEPS:
            rgb = observer_rgb(task_obj)
            t = state["n"] / 250.0
            p = out_dir / f"{embodiment}_{task}_t{t:.1f}s.png"
            from PIL import Image

            Image.fromarray(rgb).save(p)
            saved.append(p)
            print(f"  captured {p.name} ({rgb.shape[1]}x{rgb.shape[0]})")

    task_obj.scene.step = hooked_step
    try:
        task_obj.play_once()
    except Exception as exc:
        print(f"  play_once ended early: {type(exc).__name__}: "
              f"{str(exc)[:120]}")
    finally:
        task_obj.scene.step = orig_step
        env.close()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/figs")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out_dir = Path(args.out) / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    for embodiment, task in PAIRS:
        for attempt in range(3):
            seed = args.seed + attempt * 1000
            print(f"[fig] {embodiment} / {task} (seed {seed})")
            try:
                saved = shoot(embodiment, task, seed, out_dir)
                if saved:
                    break
            except Exception:
                traceback.print_exc()
        else:
            print(f"[fig] WARNING: no frame for {embodiment}/{task}")


if __name__ == "__main__":
    main()
