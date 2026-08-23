#!/usr/bin/env python3
"""Tangent-perturbation demo for the ReplanNominal (training controller).

Runs stack_blocks_two with the run.py-style env (on_path obstacle), and at
one point during the rollout shoves the active arm TANGENT to the planned
path (lateral offset, ~10 cm). The video shows whether the nominal chases
the pre-perturbation plan (bad) or replans from the live state (good).

  conda activate RoboTwin
  python record_tangent_perturb.py [--seed 0] [--offset 0.10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CEHJ_ROOT = Path(__file__).resolve().parent
if str(CEHJ_ROOT) not in sys.path:
    sys.path.insert(0, str(CEHJ_ROOT))

from main.envs.env import Env  # noqa: E402
from main.network.body_features import BodyTokenExtractor  # noqa: E402
from main.train.collect import make_training_env  # noqa: E402
from main.train.config import FrozenConfig  # noqa: E402
from main.train.nominal import ReplanNominal  # noqa: E402


def _ee(env, arm):
    return np.asarray(
        env.robot.get_left_ee_pose() if arm == "left" else env.robot.get_right_ee_pose()
    )


def _tangent(env, arm, waypoint):
    """Unit vector perpendicular to the planned EE direction (horizontal)."""
    p = _ee(env, arm)
    d = np.asarray(waypoint[:3]) - p[:3]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    d = d / n
    t = np.cross(d, [0.0, 0.0, 1.0])
    tn = np.linalg.norm(t)
    if tn < 1e-3:  # direction nearly vertical: pick x
        return np.array([1.0, 0.0, 0.0])
    return t / tn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stack_blocks_two")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=float, default=0.15, help="tangent offset (m)")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--no-perturb", action="store_true")
    ap.add_argument("--out", type=Path, default=CEHJ_ROOT / "runs" / "videos")
    args = ap.parse_args()

    cfg = FrozenConfig(task=args.task, embodiment="piper", obstacle_mode="on_path")

    # probe the expert EE trajectory
    probe_env = make_training_env(cfg, args.seed)
    tracker = ReplanNominal.cached(
        CEHJ_ROOT / "nominal_cache" / f"{cfg.task}_{cfg.embodiment}_s{args.seed}",
        probe_env, control_freq=20.0,
    )
    print(f"probe: T={tracker.T} success={tracker.success}")

    env = make_training_env(cfg, args.seed)
    extractor = BodyTokenExtractor(env)
    dtheta_max = extractor.delta_theta_max.astype(np.float64)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    video = imageio.get_writer(
        str(out_dir / f"tangent_perturb_{args.task}_s{args.seed}.mp4"),
        fps=5, codec="libx264", format="FFMPEG", pixelformat="yuv420p",
        macro_block_size=1,
    )

    cam = env.task.cameras.observer_camera
    perturb_at = 10**9 if args.no_perturb else tracker.T // 3
    mode = "NOMINAL"
    offset_dir = None

    def grab(t, err_l, err_r):
        cam.take_picture()
        rgba = cam.get_picture("Color")
        img = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3].copy()
        import cv2

        lines = [
            f"t={t * 0.05:4.1f}s  step {t}/{tracker.T}  replans={tracker.n_replans}",
            f"MODE: {mode}" + (f" (tangent {args.offset:.2f}m)" if mode == "PERTURB" else ""),
            f"EE err L={err_l:.3f}m  R={err_r:.3f}m",
        ]
        y = 18
        for line in lines:
            color = (0, 255, 255) if "PERTURB" in line else (255, 255, 0)
            cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            y += 16
        video.append_data(img)

    T = tracker.T
    t = 0
    cap = args.max_steps
    while not tracker.done() and t < cap:
        t += 1
        if t == perturb_at:
            mode = "PERTURB"
            # sustained tangent drag via the arm's own Jacobian: dtheta =
            # pinv(J_ee) @ (tangent * step). No IK feasibility issues, so
            # the shove reaches the full offset (reach permitting).
            errs = {
                a: np.linalg.norm(
                    _ee(env, a)[:3] - tracker.ee_traj[a][tracker.seg[a]][:3]
                )
                for a in ("left", "right")
            }
            arm = max(errs, key=errs.get)
            col = 0 if arm == "left" else 8
            tok = 5 if arm == "left" else 12  # link6 (EE) token
            offset_dir = _tangent(env, arm, tracker.ee_traj[arm][tracker.seg[arm]])
            g = [env.robot.get_left_gripper_val(), env.robot.get_right_gripper_val()]
            start_ee = _ee(env, arm)[:3].copy()
            moved = 0.0
            step_size = 0.04  # m of commanded EE motion per control step
            budget = int(args.offset / 0.01) + 20  # generous; stops on stall
            stall = 0
            for _s in range(budget):
                if moved >= args.offset:
                    break
                prev = moved
                body = extractor.extract()
                Je = body["Jlin"][tok, :, col : col + 6].numpy()  # [3, 6]
                dq = np.linalg.lstsq(Je, offset_dir * step_size, rcond=1e-2)[0]
                dq = np.clip(dq, -dtheta_max[col : col + 6], dtheta_max[col : col + 6])
                d = np.zeros(16)
                d[col : col + 6] = dq
                env.step_dtheta(d, grip_left=float(g[0]), grip_right=float(g[1]))
                t += 1
                moved = float(np.linalg.norm(_ee(env, arm)[:3] - start_ee))
                stall = stall + 1 if moved - prev < 0.001 else 0
                if stall >= 15:
                    print(f"perturb stalled at {moved:.3f} m (reach limit)")
                    break
                if t % 4 == 0:
                    grab(t, errs["left"], errs["right"])
            print(f"perturb: shoved {arm} tangent by {moved:.3f} m")
            mode = "NOMINAL (replanned)"
            continue

        dtheta, gl, gr = tracker.action(t, env, dtheta_max)
        env.step_dtheta(dtheta, grip_left=gl, grip_right=gr)
        if t % 4 == 0:
            err_l = np.linalg.norm(_ee(env, "left")[:3] - tracker.ee_traj["left"][tracker.seg["left"]][:3])
            err_r = np.linalg.norm(_ee(env, "right")[:3] - tracker.ee_traj["right"][tracker.seg["right"]][:3])
            grab(t, err_l, err_r)

    # success check + final overlay frames
    try:
        success = bool(env.task.check_success())
    except Exception:
        success = False
    for _ in range(10):
        cam.take_picture()
        rgba = cam.get_picture("Color")
        img = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3].copy()
        import cv2

        cv2.putText(
            img,
            f"TASK {'SUCCESS' if success else 'INCOMPLETE'} (check_success={success})",
            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0) if success else (0, 0, 255), 1, cv2.LINE_AA,
        )
        video.append_data(img)
    video.close()
    print(f"replans={tracker.n_replans} fails={tracker.n_plan_fail}  success={success}")
    print(f"video: {out_dir / f'tangent_perturb_{args.task}_s{args.seed}.mp4'}")
    env.close()


if __name__ == "__main__":
    main()
