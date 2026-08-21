#!/usr/bin/env python3
"""Load the HoloBrain encoder and encode one real simulation step.

Builds a CEHJ_safe RoboTwin env, steps the simulator once, captures the
three data cameras (head + 2 wrists: RGB, metric depth, intrinsics,
extrinsics) and the 14-DoF joint state, then runs the pretrained
HoloBrain-0 (GD) robot state encoder and spatial enhancer on them.

Usage:
  conda activate RoboTwin
  python test_encoder_step.py --task place_empty_cup --embodiment piper --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

CEHJ_ROOT = Path(__file__).resolve().parents[1]
if str(CEHJ_ROOT) not in sys.path:
    sys.path.insert(0, str(CEHJ_ROOT))

from main.envs.env import Env  # noqa: E402
from main.network.encoder import HoloBrainEncoder  # noqa: E402

CKPT_DIR = Path("/root/autodl-tmp/HoloBrain/HoloBrain_v0.0_GD")
# HoloBrain robotwin config: input is normalized with these stats (0-255 scale)
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="place_empty_cup")
    p.add_argument("--embodiment", default="piper")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=Path, default=CKPT_DIR)
    return p.parse_args()


def make_piper_kinematics(ckpt_dir: Path):
    """Dual-arm piper kinematics matching HoloBrain's agilex_ro config."""
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        MultiArmKinematics,
    )

    return MultiArmKinematics(
        urdf=str(ckpt_dir / "urdf" / "piper_description_dualarm.urdf"),
        arm_link_keys=[
            [f"left_link{i}" for i in range(1, 7)],
            [f"right_link{i}" for i in range(1, 7)],
        ],
        arm_joint_id=[list(range(6)), list(range(8, 14))],
        finger_keys=[["left_link7"], ["right_link7"]],
    )


def grab_camera(cam):
    """Return (rgb uint8 [H,W,3], depth m [H,W], projection_mat [4,4])."""
    cam.take_picture()
    rgba = cam.get_picture("Color")
    rgb = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
    position = cam.get_picture("Position")  # camera frame, z forward = -z in front
    depth = np.asarray(-position[:, :, 2], dtype=np.float32)

    K = np.asarray(cam.get_intrinsic_matrix(), dtype=np.float64)
    ext = np.asarray(cam.get_extrinsic_matrix(), dtype=np.float64)  # world->cam
    if ext.shape == (4, 4):
        ext = ext[:3, :]
    proj = np.eye(4, dtype=np.float64)
    proj[:3, :4] = K @ ext
    return rgb, depth, proj


def main() -> None:
    args = parse_args()

    # ---------------- env + one simulation step ----------------
    env = Env(args.task, args.embodiment, args.seed)
    env.task.scene.step()
    env.task.scene.update_render()

    # ---------------- cameras ----------------
    cams = env.task.cameras
    head_cam = cams.static_camera_list[cams.head_camera_id]
    cam_list = [head_cam, cams.left_camera, cams.right_camera]
    rgbs, depths, projs = [], [], []
    for cam in cam_list:
        rgb, depth, proj = grab_camera(cam)
        rgbs.append(rgb)
        depths.append(depth)
        projs.append(proj)
    h, w = rgbs[0].shape[:2]
    print(f"captured {len(cam_list)} cameras at {w}x{h} "
          f"(depth range {min(d.min() for d in depths):.3f}..{max(d.max() for d in depths):.3f} m)")

    imgs = np.stack(rgbs)[None].astype(np.float32)          # [1, NC, H, W, 3]
    imgs = (imgs - IMG_MEAN) / IMG_STD
    imgs = torch.from_numpy(imgs).permute(0, 1, 4, 2, 3)    # [1, NC, 3, H, W]
    depths_t = torch.from_numpy(np.stack(depths)[None, :, None])  # [1, NC, 1, H, W]
    image_wh = torch.tensor([[w, h]] * len(cam_list))
    proj_t = torch.from_numpy(np.stack(projs)[None])        # [1, NC, 4, 4]

    # ---------------- joint state ----------------
    left = env.robot.get_left_arm_real_jointState()    # 6 arm + gripper
    right = env.robot.get_right_arm_real_jointState()  # 6 arm + gripper
    joint_state = torch.tensor([left + right], dtype=torch.float32)[None]  # [1, 1, 14]
    print(f"joint state [L6, gripL, R6, gripR]: {np.round(joint_state[0, 0].numpy(), 3).tolist()}")

    # ---------------- encoder ----------------
    encoder = HoloBrainEncoder(str(args.ckpt), device="cuda")
    kin = make_piper_kinematics(args.ckpt)

    with torch.no_grad():
        t0 = time.perf_counter()
        state_feat = encoder.encode_joint_angles(joint_state, kin)
        torch.cuda.synchronize()
        t_state = time.perf_counter() - t0

        # warmup + timed visual encoding (full 4-scale path)
        encoder.encode_visual(imgs, depths_t, image_wh, proj_t)
        encoder.encode_visual_tokens(imgs, depths_t, image_wh, proj_t)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fused, depth_prob = encoder.encode_visual(imgs, depths_t, image_wh, proj_t)
        torch.cuda.synchronize()
        t_vis = time.perf_counter() - t0

        # efficient decoder-level token path (world frame — no out_extrinsic)
        t0 = time.perf_counter()
        tokens, pos_mean, pos_max = encoder.encode_visual_tokens(
            imgs, depths_t, image_wh, proj_t
        )
        torch.cuda.synchronize()
        t_tok = time.perf_counter() - t0

    print(f"state feature:  {tuple(state_feat.shape)}  ({1e3 * t_state:.1f} ms)")
    print(f"spatial fused:  {[tuple(f.shape) for f in fused]}")
    print(f"depth prob:     {tuple(depth_prob.shape)}  ({1e3 * t_vis:.1f} ms full 4-scale)")
    print(f"vision tokens:  {tuple(tokens.shape)}")
    print(f"token pos:      {tuple(pos_mean.shape)}  ({1e3 * t_tok:.1f} ms levels [1,2] only)")
    print(f"state feat stats: mean {state_feat.mean():.4f} std {state_feat.std():.4f}")
    print(f"tokens stats:     mean {tokens.mean():.4f} std {tokens.std():.4f}")

    # cross-check: full-path positions restricted to [1,2] must match
    ref_mean, _ = encoder.spatial_token_positions(
        depth_prob, image_wh, proj_t, fused, feature_level=[1, 2],
    )
    print(f"pos cross-check max err: {(ref_mean - pos_mean).abs().max().item():.2e}")

    # world-frame token positions (link_pos and vision tokens share one frame)
    world_tp = pos_mean[0, 0]
    print(
        "world-frame tokens: x "
        f"[{world_tp[:, 0].min():.2f}, {world_tp[:, 0].max():.2f}]  "
        f"y [{world_tp[:, 1].min():.2f}, {world_tp[:, 1].max():.2f}]  "
        f"z [{world_tp[:, 2].min():.2f}, {world_tp[:, 2].max():.2f}] (m)"
    )

    # sanity: dump the head camera frame
    from PIL import Image

    out = Path("/tmp/encoder_step_head.png")
    Image.fromarray(rgbs[0]).save(out)
    print(f"head camera frame saved to {out}")
    print("ENCODER STEP TEST PASSED")
    env.close()


if __name__ == "__main__":
    main()
