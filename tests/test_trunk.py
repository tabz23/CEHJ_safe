#!/usr/bin/env python3
"""Test GeometricTrunk: shapes, finiteness, padding invariance, timing.

  conda activate RoboTwin
  python test_trunk.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

CEHJ_ROOT = Path(__file__).resolve().parents[1]
if str(CEHJ_ROOT) not in sys.path:
    sys.path.insert(0, str(CEHJ_ROOT))

from main.envs.env import Env  # noqa: E402
from main.network.encoder import HoloBrainEncoder, RobotInjection  # noqa: E402
from main.network.body_features import BodyTokenExtractor  # noqa: E402
from main.network.trunk import GeometricTrunk  # noqa: E402
from tests.test_encoder_step import make_piper_kinematics, CKPT_DIR  # noqa: E402


def main() -> None:
    env = Env("place_empty_cup", "piper", 0, control_freq=20.0)
    extractor = BodyTokenExtractor(env)
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_piper_kinematics(CKPT_DIR)
    injection = RobotInjection().cuda().eval()
    trunk = GeometricTrunk().cuda().eval()

    # ---- gather inputs ----
    obs = env.get_encoder_obs(kin)
    state_tokens = encoder.encode_joint_angles(obs["joint_state"], kin,
                            embodiedment_mat=obs["T_base2ego"]).squeeze(2)
    tokens, pos_mean, _ = encoder.encode_visual_tokens(
        obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"],
        out_extrinsic=obs["T_ego2world"],
    )
    body = extractor.extract()
    with torch.no_grad():
        btok, bpos, bmask = injection(
            state_tokens,
            body["body_tokens"][None],
            body["link_pos"][None],
            body["body_mask"][None],
            body["joint_index"][None],
        )
        scene, scene_pos = injection.adapt_scene(tokens, pos_mean)
    scene_mask = torch.ones(scene.shape[:2], dtype=torch.bool, device=scene.device)
    print(f"body {tuple(btok.shape)}  scene {tuple(scene.shape)}")

    # ---- forward ----
    with torch.no_grad():
        out = trunk(btok, bpos, bmask, scene, scene_pos, scene_mask)
    assert out.shape == btok.shape
    assert torch.isfinite(out).all(), "NaN/Inf in trunk output"
    print(f"trunk out {tuple(out.shape)}  finite OK")

    # ---- padding invariance (constant shapes => must be bit-identical) ----
    # 2a: garbage in padded body slots
    pad20 = ~body["body_mask"][None].cuda()[0]          # [20] padded body slots
    pad22 = torch.cat([pad20, torch.zeros(2, dtype=torch.bool, device=pad20.device)])
    btok_g = btok.clone()
    btok_g[:, pad22] = 100.0
    bpos_g = bpos.clone()
    bpos_g[:, pad22] = -999.0
    with torch.no_grad():
        out_g = trunk(btok_g, bpos_g, bmask, scene, scene_pos, scene_mask)
    valid = bmask[0]
    assert torch.isfinite(out_g).all()
    assert torch.equal(out[0, valid], out_g[0, valid]), (
        f"body-padding garbage leaked: max diff "
        f"{(out[0, valid] - out_g[0, valid]).abs().max().item():.2e}"
    )
    print("CHECK padding invariance (body padding garbage): bit-identical  OK")

    # 2b: garbage in scene slots that are masked out (same M, so exact)
    scene_g = scene.clone()
    scene_pos_g = scene_pos.clone()
    scene_mask_g = scene_mask.clone()
    scene_g[:, :50] = 1000.0
    scene_pos_g[:, :50] = -1000.0
    scene_mask_g[:, :50] = False
    scene_z = scene.clone()          # reference: same slots zeroed + masked
    scene_pos_z = scene_pos.clone()
    scene_mask_z = scene_mask.clone()
    scene_z[:, :50] = 0.0
    scene_pos_z[:, :50] = 0.0
    scene_mask_z[:, :50] = False
    with torch.no_grad():
        out_ref = trunk(btok, bpos, bmask, scene_z, scene_pos_z, scene_mask_z)
        out_sg = trunk(btok, bpos, bmask, scene_g, scene_pos_g, scene_mask_g)
    assert torch.equal(out_ref, out_sg), (
        f"masked scene garbage leaked: max diff "
        f"{(out_ref - out_sg).abs().max().item():.2e}"
    )
    print("CHECK padding invariance (masked scene garbage): bit-identical  OK")

    # ---- timing ----
    with torch.no_grad():
        for _ in range(5):
            trunk(btok, bpos, bmask, scene, scene_pos, scene_mask)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            trunk(btok, bpos, bmask, scene, scene_pos, scene_mask)
        torch.cuda.synchronize()
    print(f"trunk forward: {1e3 * (time.perf_counter() - t0) / 20:.1f} ms  (M={scene.shape[1]})")

    # ---- ablation mode: head-averaged direction channel ----
    trunk_avg = GeometricTrunk(dir_per_head=False).cuda().eval()
    with torch.no_grad():
        out_avg = trunk_avg(btok, bpos, bmask, scene, scene_pos, scene_mask)
    assert out_avg.shape == btok.shape and torch.isfinite(out_avg).all()
    print("dir_per_head=False mode: OK")

    env.close()
    print("TRUNK TEST PASSED")


if __name__ == "__main__":
    main()
