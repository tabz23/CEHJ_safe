#!/usr/bin/env python3
"""Test RobotInjection: shapes, residual anchoring, padding invariance.

  conda activate RoboTwin
  python test_injection.py
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
from tests.test_encoder_step import make_piper_kinematics, CKPT_DIR  # noqa: E402


def main() -> None:
    env = Env("place_empty_cup", "piper", 0, control_freq=20.0)
    extractor = BodyTokenExtractor(env)
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_piper_kinematics(CKPT_DIR)
    injection = RobotInjection().cuda().eval()

    # ---- gather inputs ----
    obs = env.get_encoder_obs()
    state_tokens = encoder.encode_joint_angles(obs["joint_state"], kin).squeeze(2)  # [1,14,256]
    body = extractor.extract()
    B = 1
    body_tokens = body["body_tokens"][None]
    link_pos = body["link_pos"][None]
    body_mask = body["body_mask"][None]
    joint_index = body["joint_index"][None]
    print(f"state_tokens {tuple(state_tokens.shape)}  body_tokens {tuple(body_tokens.shape)}")

    # ---- norm report (frozen half vs analytic columns) ----
    frozen_rms = state_tokens.pow(2).mean(dim=-1).sqrt().mean().item()
    body_rms = body_tokens[:, body["body_mask"]].pow(2).mean(dim=-1).sqrt().mean().item()
    print(f"rms: frozen state tokens {frozen_rms:.3f}   analytic body cols {body_rms:.3f}")

    # ---- forward ----
    with torch.no_grad():
        tok, pos, mask = injection(state_tokens, body_tokens, link_pos, body_mask, joint_index)
    tok, pos, mask = tok.cpu(), pos.cpu(), mask.cpu()
    print(f"tok {tuple(tok.shape)}  pos {tuple(pos.shape)}  mask {tuple(mask.shape)}")
    assert tok.shape == (B, 22, 256) and pos.shape == (B, 22, 3) and mask.shape == (B, 22)
    assert mask[0, :14].all() and not mask[0, 14:20].any(), "body block mask wrong"
    assert mask[0, 20:22].all(), "residuals must be valid"

    # ---- CHECK 1: residual anchoring (indexed by arm, not token order) ----
    grip_idx = [6, 13]  # piper layout: (joint_index==-1)&body_mask
    flagged = ((joint_index[0] == -1) & body_mask[0]).nonzero().flatten().tolist()
    assert flagged == grip_idx, f"gripper flags at {flagged}"
    assert torch.allclose(pos[0, 20], link_pos[0, grip_idx[0]]), "res0 not at left gripper"
    assert torch.allclose(pos[0, 21], link_pos[0, grip_idx[1]]), "res1 not at right gripper"
    assert pos[0, 20:22].abs().max() > 0.01, "res_pos fell back to origin!"
    print("CHECK residual anchoring: res_pos == gripper link_pos (per arm, not origin)  OK")

    # ---- CHECK 1b: missing-gripper fallback (no hard failure) ----
    import warnings as _w

    mask_shy = body_mask.clone()
    mask_shy[0, 13] = False  # drop the right gripper token
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        with torch.no_grad():
            _, pos_fb, mask_fb = injection(
                state_tokens, body_tokens, link_pos, mask_shy, joint_index
            )
    pos_fb = pos_fb.cpu()
    assert any("gripper missing" in str(w.message) for w in caught), "no fallback warning"
    # arm-1 residual must fall back to the arm tip = last valid arm-1 token (idx 12)
    assert torch.allclose(pos_fb[0, 21], link_pos[0, 12]), (
        f"fallback anchor wrong: {pos_fb[0, 21]} vs tip {link_pos[0, 12]}"
    )
    assert mask_fb[0, 21], "residual slot must stay valid on fallback"
    print("CHECK missing-gripper fallback: warns + anchors at arm tip, slot stays valid  OK")

    # ---- CHECK 2: padding invariance ----
    body_tokens_g = body_tokens.clone()
    link_pos_g = link_pos.clone()
    body_tokens_g[:, ~body["body_mask"]] = torch.randn_like(body_tokens_g[:, ~body["body_mask"]]) * 100
    link_pos_g[:, ~body["body_mask"]] = -999.0
    with torch.no_grad():
        tok2, pos2, mask2 = injection(state_tokens, body_tokens_g, link_pos_g, body_mask, joint_index)
    tok2, pos2, mask2 = tok2.cpu(), pos2.cpu(), mask2.cpu()
    valid = mask[0]
    assert torch.equal(tok[0, valid], tok2[0, valid]), "valid tokens changed by padded garbage"
    assert torch.equal(tok[0, 20:22], tok2[0, 20:22]), "residual tokens changed by padded garbage"
    assert torch.equal(pos[0, valid], pos2[0, valid]), "valid pos changed by padded garbage"
    assert torch.equal(mask, mask2)
    print("CHECK padding invariance: garbage in padded slots -> bit-identical valid outputs  OK")

    # ---- scene adapter (raw 4-D in, flattened pair out) ----
    z4 = torch.randn(B, 3, 400, 256)
    p4 = torch.randn(B, 3, 400, 3)
    z_flat, pos_flat = injection.adapt_scene(z4, p4)
    assert z_flat.shape == (B, 1200, 256) and pos_flat.shape == (B, 1200, 3)
    assert torch.equal(pos_flat.cpu(), p4.flatten(1, 2)), "pos flatten mismatch"
    print(f"scene adapter: {tuple(z4.shape)} -> {tuple(z_flat.shape)} + {tuple(pos_flat.shape)}  OK")

    env.close()
    print("INJECTION TEST PASSED")


if __name__ == "__main__":
    main()
