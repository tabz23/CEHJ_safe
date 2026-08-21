"""Run the task expert with ResidualController (default) and record videos.

    conda activate rbtw128
    python /storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ/main/envs/evaluation.py
    python .../evaluation.py --embodiment ur5 --task place_dual_shoes
    python .../evaluation.py --controller nominal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    _cehj = str(Path(__file__).resolve().parents[2])
    if _cehj not in sys.path:
        sys.path.insert(0, _cehj)
    __package__ = "main.envs"
    import main.envs  # noqa: F401

from .controller import CuroboIKController, ResidualController
from .env import CEHJ_ROOT, Env, RECORD_SIZE

CONTROLLERS = {
    "residual": ResidualController,
    "nominal": CuroboIKController,
}

DEFAULT_OUTPUT = CEHJ_ROOT / "outputs" / "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_dual_shoes")
    parser.add_argument("--embodiment", default="piper")
    parser.add_argument(
        "--controller",
        choices=tuple(CONTROLLERS),
        default="residual",
        help="Low-level controller. residual = CuRobo path + joint residual (default).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm-distance", type=float, default=0.6)
    parser.add_argument("--record-every", type=int, default=8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _observer_rgb(task) -> np.ndarray:
    cam = task.cameras.observer_camera
    cam.take_picture()
    rgba = cam.get_picture("Color")
    rgb = (rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
    tw, th = RECORD_SIZE
    if rgb.shape[1] == tw and rgb.shape[0] == th:
        return rgb
    import cv2

    return cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)


def attach_recorder(task, streams: dict, record_every: int) -> None:
    state = {"n": 0}
    orig_step = task.scene.step

    def step():
        orig_step()
        state["n"] += 1
        if state["n"] % max(record_every, 1) != 0:
            return
        obs = task.get_obs()["observation"]
        streams["agent_rgb"].append(_observer_rgb(task))
        streams["left_wrist_rgb"].append(np.asarray(obs["left_camera"]["rgb"], dtype=np.uint8))
        streams["right_wrist_rgb"].append(np.asarray(obs["right_camera"]["rgb"], dtype=np.uint8))
        print(f"recorded {len(streams['agent_rgb'])} frames", end="\r")

    task.scene.step = step


def save_video(frames, out_path: Path, fps: float) -> Path:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video = np.stack(frames, axis=0)
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            ffmpeg_params=["-crf", "17", "-preset", "medium"],
            macro_block_size=1,
        )
        for frame in video:
            writer.append_data(frame)
        writer.close()
        print(f"Saved {len(video)} frames to {out_path}")
        return out_path
    except Exception as exc:
        print(f"imageio failed ({exc}); trying OpenCV")

    import cv2

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (video.shape[2], video.shape[1]),
    )
    for frame in video[:, :, :, ::-1]:
        writer.write(frame)
    writer.release()
    print(f"Saved {len(video)} frames to {out_path}")
    return out_path


def evaluate(args: argparse.Namespace) -> dict[str, Path]:
    ctrl_cls = CONTROLLERS[args.controller]
    ctrl_cls.install()
    env = Env(args.task, args.embodiment, args.seed, args.arm_distance)
    ctrl = ctrl_cls(env)
    ctrl.attach()
    print(
        f"{args.task} / {env.embodiment}  controller={type(ctrl).__name__}  "
        f"planner={type(ctrl.robot.left_planner).__name__}"
    )

    streams = {"agent_rgb": [], "left_wrist_rgb": [], "right_wrist_rgb": [], "head_rgb": []}
    attach_recorder(env.task, streams, args.record_every)
    try:
        env.task.play_once()
    except Exception as exc:
        print(f"play_once failed: {exc}")
    print()
    if not streams["agent_rgb"]:
        env.close()
        raise RuntimeError("No frames recorded")

    try:
        print(f"success={env.task.check_success()}")
    except Exception as exc:
        print(f"check_success warning: {exc}")

    out_dir = args.output.expanduser() / args.task / env.embodiment
    outputs = {name: save_video(frames, out_dir / f"{name}.mp4", args.fps) for name, frames in streams.items()}
    env.close()
    return outputs


def main() -> None:
    args = parse_args()
    outputs = evaluate(args)
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
