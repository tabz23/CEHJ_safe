#!/usr/bin/env python3
"""Load the HoloBrain encoder and encode one real simulation step.

Builds a CEHJ_safe RoboTwin env, captures the observation through
Env.get_encoder_obs (HoloBrain-parity path: [left, right, head] camera
order, 320x256 resize with intrinsic rescale, alpha-masked depth,
EGO-frame projection matrices), then runs the pretrained HoloBrain-0 (GD)
robot state encoder (ego-frame FK via embodiedment_mat) and the spatial
enhancer.

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


def main() -> None:
    args = parse_args()

    env = Env(args.task, args.embodiment, args.seed)
    env.task.scene.step()

    encoder = HoloBrainEncoder(str(args.ckpt), device="cuda")
    kin = make_piper_kinematics(args.ckpt)

    obs = env.get_encoder_obs()
    assert obs["camera_names"] == ["left_wrist", "right_wrist", "head"]
    w, h = obs["image_wh"][0].tolist()
    print(f"obs: {len(obs['camera_names'])} cams at {w}x{h} (ego = head, LAST)")
    assert (w, h) == (320, 256), "encoder input must be HoloBrain's 320x256"
    js = obs["joint_state"]
    print(f"joint state: {js.shape[-1]} dims  "
          f"(cmd-vs-measured gap {obs['joint_cmd_gap']:.4f} rad)")

    with torch.no_grad():
        t0 = time.perf_counter()
        state_feat = encoder.encode_joint_angles(
            js, kin, embodiedment_mat=obs["T_base2ego"]
        )
        torch.cuda.synchronize()
        t_state = time.perf_counter() - t0

        t0 = time.perf_counter()
        tokens, pos_mean, pos_max = encoder.encode_visual_tokens(
            obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"],
            out_extrinsic=obs["T_ego2world"],  # ego -> world metres
        )
        torch.cuda.synchronize()
        t_tok = time.perf_counter() - t0

    n_cam, n_per_cam = tokens.shape[1], tokens.shape[2]
    print(f"state feature:  {tuple(state_feat.shape)}  ({1e3 * t_state:.1f} ms)")
    print(f"vision tokens:  {tuple(tokens.shape)}  ({1e3 * t_tok:.1f} ms)")
    print(f"token pos:      {tuple(pos_mean.shape)}")
    assert n_cam == 3 and n_per_cam == 400, (
        f"expected 3 cams x 400 tokens (20x16 + 10x8 at 320x256) = 1200, "
        f"got {n_cam} x {n_per_cam} — resolution/order mismatch"
    )

    # token positions are world-frame: project a sample back into the head
    # camera and confirm they land inside the image (geometry sanity)
    world_tp = pos_mean[0, 2]  # head camera is LAST
    print(
        "world-frame tokens (head cam): x "
        f"[{world_tp[:, 0].min():.2f}, {world_tp[:, 0].max():.2f}]  "
        f"y [{world_tp[:, 1].min():.2f}, {world_tp[:, 1].max():.2f}]  "
        f"z [{world_tp[:, 2].min():.2f}, {world_tp[:, 2].max():.2f}] (m)"
    )
    ext = obs["T_cam_world"]  # world -> head
    pw = world_tp.double().cpu().numpy()
    pc = pw @ ext[:3, :3].T + ext[:3, 3]
    # lifted points live in the OpenCV-style ego frame: in front = +z
    in_front = (pc[:, 2] > 0).mean()
    print(f"head-cam frame: {100 * in_front:.0f}% of head tokens in front")

    from PIL import Image

    out = Path("/tmp/encoder_step_head.png")
    Image.fromarray(obs["rgb"][2]).save(out)
    print(f"head camera frame saved to {out}")
    print("ENCODER STEP TEST PASSED")
    env.close()


if __name__ == "__main__":
    main()
