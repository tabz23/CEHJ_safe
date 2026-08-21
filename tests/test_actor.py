#!/usr/bin/env python3
"""Test TokenActor: bounds, scatter correctness, padding & count invariance.

  conda activate RoboTwin
  python test_actor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

CEHJ_ROOT = Path(__file__).resolve().parents[1]
if str(CEHJ_ROOT) not in sys.path:
    sys.path.insert(0, str(CEHJ_ROOT))

from main.envs.env import Env  # noqa: E402
from main.network.encoder import HoloBrainEncoder, RobotInjection  # noqa: E402
from main.network.body_features import BodyTokenExtractor  # noqa: E402
from main.network.trunk import GeometricTrunk  # noqa: E402
from main.network.heads import TokenActor  # noqa: E402
from tests.test_encoder_step import make_piper_kinematics, CKPT_DIR  # noqa: E402


def main() -> None:
    env = Env("place_empty_cup", "piper", 0, control_freq=20.0)
    extractor = BodyTokenExtractor(env)
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_piper_kinematics(CKPT_DIR)
    injection = RobotInjection().cuda().eval()
    trunk = GeometricTrunk().cuda().eval()
    actor = TokenActor().cuda().eval()

    # ---- full pipeline to trunk output ----
    obs = env.get_encoder_obs()
    state_tokens = encoder.encode_joint_angles(obs["joint_state"], kin).squeeze(2)
    tokens, pos_mean, _ = encoder.encode_visual_tokens(
        obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"]
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
        body_out = trunk(btok, bpos, bmask, scene, scene_pos, scene_mask)

    joint_index = body["joint_index"][None].cuda()
    dtheta_max = body["delta_theta_max"].cuda()  # [16]

    # ---- forward ----
    with torch.no_grad():
        dtheta, logp, n_act = actor(body_out, joint_index, dtheta_max)
    print(f"dtheta {tuple(dtheta.shape)}  logp {tuple(logp.shape)}  n_act {n_act.item()}")
    assert dtheta.shape == (1, 16) and logp.shape == (1,)
    assert n_act.item() == 12, f"n_act {n_act.item()} != 12 (6 joints x 2 arms)"
    assert torch.isfinite(logp).all()

    # ---- bounds: |dtheta| <= dtheta_max elementwise ----
    assert (dtheta.abs() <= dtheta_max + 1e-6).all()
    print("CHECK |dtheta| <= dtheta_max elementwise  OK")

    # ---- scatter correctness: zero at every column not in joint_index ----
    valid_cols = set(body["joint_index"][body["joint_index"] >= 0].tolist())
    all_cols = set(range(16))
    zero_cols = sorted(all_cols - valid_cols)
    assert zero_cols == [6, 7, 14, 15], f"unexpected free cols {zero_cols}"
    assert (dtheta[0, zero_cols] == 0).all(), "nonzero action in unmapped column"
    extractor.assert_action(dtheta.cpu())  # critic-boundary guard
    print(f"CHECK dtheta == 0 at unmapped columns {zero_cols}  OK")

    # ---- deterministic mode: identical outputs across calls, no sampling ----
    with torch.no_grad():
        d1, _, _ = actor(body_out, joint_index, dtheta_max, deterministic=True)
        d2, _, _ = actor(body_out, joint_index, dtheta_max, deterministic=True)
    assert torch.equal(d1, d2), "deterministic mode is not deterministic"
    assert (d1.abs() <= dtheta_max + 1e-6).all()
    print("CHECK deterministic=True: reproducible, still bounded  OK")

    # ---- padding invariance ----
    pad20 = ~body["body_mask"][None].cuda()[0]
    pad22 = torch.cat([pad20, torch.zeros(2, dtype=torch.bool, device=pad20.device)])
    btok_g = btok.clone(); btok_g[:, pad22] = 100.0
    bpos_g = bpos.clone(); bpos_g[:, pad22] = -999.0
    torch.manual_seed(0)
    with torch.no_grad():
        dtheta1, logp1, _ = actor(trunk(btok, bpos, bmask, scene, scene_pos, scene_mask),
                                  joint_index, dtheta_max)
    torch.manual_seed(0)
    with torch.no_grad():
        dtheta2, logp2, _ = actor(trunk(btok_g, bpos_g, bmask, scene, scene_pos, scene_mask),
                                  joint_index, dtheta_max)
    assert torch.equal(dtheta1, dtheta2) and torch.equal(logp1, logp2), (
        f"padding garbage leaked: {(dtheta1 - dtheta2).abs().max().item():.2e}"
    )
    print("CHECK padding invariance: bit-identical  OK")

    # ---- count invariance: same weights at 14/16/18 valid tokens ----
    for n_valid in (14, 16, 18):
        fake_body = torch.randn(1, n_valid + 2, 256, device="cuda")
        fake_ji = torch.full((1, n_valid), -1, dtype=torch.long, device="cuda")
        n_arm = (n_valid // 2)
        fake_ji[0, : n_arm - 1] = torch.arange(n_arm - 1, device="cuda")
        fake_ji[0, n_arm - 1 : 2 * n_arm - 2] = torch.arange(8, 8 + n_arm - 1, device="cuda")
        with torch.no_grad():
            d, lp, na = actor(fake_body, fake_ji, dtheta_max)
        assert d.shape == (1, 16) and torch.isfinite(lp).all()
        assert na.item() == 2 * (n_arm - 1)
        print(f"CHECK count invariance: {n_valid} valid tokens -> n_act {na.item()}  OK")

    env.close()
    print("ACTOR TEST PASSED")


if __name__ == "__main__":
    main()
