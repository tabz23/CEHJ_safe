"""Record vanilla-tick rollouts (nominal only, no filter) for the five
bimanual tasks: observer camera at 10 fps of SIM time (1:1 playback).

  conda activate RoboTwin
  python tests/record_vanilla_tick.py
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main.envs.controller import TickChunkedController  # noqa: E402
from main.envs.record import save_video  # noqa: E402
from main.train.collect import make_training_env  # noqa: E402
from main.train.config import FrozenConfig  # noqa: E402

TASKS = [
    "place_container_plate",
    "place_burger_fries",
    "stack_blocks_two",
    "place_bread_basket",
    "stack_bowls_two",
]
OUT = Path(__file__).resolve().parents[1] / "outputs" / "ihab" / "vanilla_tick"


class _Timeout(Exception):
    pass


def record(task: str, embodiment: str = "franka-panda", seed: int = 0,
           max_physics: int = 4500) -> dict:
    cfg = FrozenConfig(task=task, embodiment=embodiment, randomize_scenes=False)
    env = make_training_env(cfg, seed)
    ctrl = TickChunkedController(env)
    ctrl.attach()

    frames = []
    orig_step = env.task.scene.step
    state = {"i": 0}

    def hooked():
        orig_step()
        state["i"] += 1
        if state["i"] > max_physics:
            raise _Timeout()
        if state["i"] % 10 == 0:  # 250/10 = 25 fps of sim time (1:1 playback)
            # sync the RT scene first — take_picture alone renders a STALE
            # copy (the render-lag flashes), worst exactly when the arm
            # moves fast between frames
            env.task.scene.update_render()
            cam = env.task.cameras.observer_camera
            cam.take_picture()
            rgba = cam.get_picture("Color")
            frames.append((rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3])

    env.task.scene.step = hooked
    success = False
    try:
        env.task.play_once()
        success = bool(env.task.check_success())
    except _Timeout:
        print(f"[{task}] timeout at {max_physics} physics steps")
    except Exception:
        traceback.print_exc()
    env.task.scene.step = orig_step

    out_dir = OUT / task / embodiment
    out_dir.mkdir(parents=True, exist_ok=True)
    video = None
    if frames:
        video = str(save_video(frames, out_dir / "agent_rgb.mp4", 25.0))
    sim_s = state["i"] / env.PHYSICS_FREQ
    print(f"[{task}] success={success} sim={sim_s:.1f}s "
          f"frames={len(frames)} -> {video}")
    env.close()
    return {"task": task, "success": success, "sim_s": sim_s,
            "video": video}


def main() -> None:
    results = []
    for task in TASKS:
        results.append(record(task))
    print("\nsummary:")
    for r in results:
        print(f"  {r['task']:24s} success={r['success']} "
              f"sim={r['sim_s']:.1f}s  {r['video']}")


if __name__ == "__main__":
    main()
