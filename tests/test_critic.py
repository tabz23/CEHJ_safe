#!/usr/bin/env python3
"""Test TwinCritic: shapes, boundary assert, invariances, softmin bound.

  conda activate RoboTwin
  python test_critic.py
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
from main.network.heads import PolicyEncoder, TwinCritic  # noqa: E402
from tests.test_encoder_step import make_piper_kinematics, CKPT_DIR  # noqa: E402


def main() -> None:
    env = Env("place_empty_cup", "piper", 0, control_freq=20.0)
    extractor = BodyTokenExtractor(env)
    encoder = HoloBrainEncoder(str(CKPT_DIR), device="cuda")
    kin = make_piper_kinematics(CKPT_DIR)
    injection = RobotInjection().cuda().eval()
    trunk = GeometricTrunk().cuda().eval()
    policy_enc = PolicyEncoder(injection, trunk).cuda().eval()
    critics = TwinCritic().cuda().eval()

    # ---- full pipeline to Encoded ----
    obs = env.get_encoder_obs()
    state_tokens = encoder.encode_joint_angles(obs["joint_state"], kin,
                            embodiedment_mat=obs["T_base2ego"]).squeeze(2)
    tokens, pos_mean, _ = encoder.encode_visual_tokens(
        obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"],
        out_extrinsic=obs["T_ego2world"],
    )
    body = extractor.extract()
    with torch.no_grad():
        enc = policy_enc(
            state_tokens,
            body["body_tokens"][None], body["link_pos"][None],
            body["body_mask"][None], body["joint_index"][None],
            tokens, pos_mean,
        )
    print(f"encoded: body {tuple(enc.body.shape)}  scene {tuple(enc.scene.shape)}")

    joint_index = body["joint_index"][None]
    Jlin = body["Jlin"][None]
    Jang = body["Jang"][None]
    dtheta = torch.zeros(1, 16, device="cuda")
    dtheta[0, 0] = 0.01
    dtheta[0, 8] = -0.01

    # ---- forward ----
    with torch.no_grad():
        q1, q2, v1, v2 = critics(enc, dtheta, Jlin, Jang, joint_index)
    assert q1.shape == (1,) and v1.shape == (1, 22)
    assert torch.isfinite(q1).all() and torch.isfinite(v1).all()
    assert not torch.equal(q1, q2), "twins should differ at init"
    print(f"Q1 {q1.item():.4f}  Q2 {q2.item():.4f}  finite, twins differ  OK")

    # ---- boundary assert: stale value at an unmapped column must fire ----
    critics_dbg = TwinCritic(debug=True).cuda().eval()
    bad = dtheta.clone()
    bad[0, 6] = 0.005  # prismatic gripper column — never actor-written
    try:
        with torch.no_grad():
            critics_dbg(enc, bad, Jlin, Jang, joint_index)
        raise SystemExit("FAIL: boundary assert did not fire")
    except AssertionError as e:
        print(f"CHECK boundary assert fires (debug=True): {e}  OK")

    # ---- softmin bound over VALID tokens only, with tightness ----
    with torch.no_grad():
        _, _, v1, _ = critics(enc, dtheta, Jlin, Jang, joint_index)
        qmin = critics.qmin(enc, dtheta, Jlin, Jang, joint_index)
    v_min_valid = v1.masked_fill(~enc.mask, float("inf")).min(dim=-1).values
    assert (v1[0, ~enc.mask[0]] == 1e4).all(), "padding not +1e4"
    T = critics.c1.temperature
    assert (q1 <= v_min_valid + 1e-6).all(), (
        f"softmin bound violated: Q {q1.item()} > min valid V {v_min_valid.item()}"
    )
    gap = (v_min_valid - q1).item()
    assert gap < 3 * T, (
        f"softmin degenerate: min V - Q = {gap:.4f} >= 3*T = {3 * T:.4f} "
        f"(T mis-scaled; Q is -T*log(N), not a minimum)"
    )
    print(f"CHECK Q <= min valid V_i ({q1.item():.4f} <= {v_min_valid.item():.4f}) "
          f"and gap {gap:.4f} < 3T={3 * T:.4f}  OK")

    # ---- padding invariance ----
    pad20 = ~body["body_mask"][None][0]
    bt_g = body["body_tokens"][None].clone(); bt_g[0, pad20] = 100.0
    lp_g = body["link_pos"][None].clone(); lp_g[0, pad20] = -999.0
    with torch.no_grad():
        enc2 = policy_enc(
            state_tokens, bt_g, lp_g,
            body["body_mask"][None], body["joint_index"][None],
            tokens, pos_mean,
        )
        q1b, q2b, v1b, _ = critics(enc2, dtheta, Jlin, Jang, joint_index)
    assert torch.equal(q1, q1b) and torch.equal(q2, q2b), (
        f"Q changed by padding garbage: {(q1 - q1b).abs().max().item():.2e}"
    )
    assert torch.equal(v1[0, enc.mask[0]], v1b[0, enc2.mask[0]])
    print("CHECK padding invariance: bit-identical  OK")

    # ---- count invariance (same weights, 14/16/18 tokens) ----
    from main.network.heads import CriticHead  # noqa: E402
    from main.network.heads import Encoded  # noqa: E402

    critic = CriticHead().cuda().eval()
    for n_valid in (14, 16, 18):
        n_arm = n_valid // 2
        ji = torch.full((1, n_valid), -1, dtype=torch.long, device="cuda")
        ji[0, : n_arm - 1] = torch.arange(n_arm - 1, device="cuda")
        ji[0, n_arm - 1 : 2 * n_arm - 2] = torch.arange(8, 8 + n_arm - 1, device="cuda")
        M_s = 50
        enc_fake = Encoded(
            body=torch.randn(1, n_valid + 2, 256, device="cuda"),
            pos=torch.randn(1, n_valid + 2, 3, device="cuda"),
            mask=torch.ones(1, n_valid + 2, dtype=torch.bool, device="cuda"),
            scene=torch.randn(1, M_s, 256, device="cuda"),
            scene_pos=torch.randn(1, M_s, 3, device="cuda"),
            scene_mask=torch.ones(1, M_s, dtype=torch.bool, device="cuda"),
        )
        Jl = torch.randn(1, n_valid, 3, 16, device="cuda")
        Ja = torch.randn(1, n_valid, 3, 16, device="cuda")
        dt = torch.zeros(1, 16, device="cuda")
        with torch.no_grad():
            q, v = critic(enc_fake, dt, Jl, Ja, ji)
        assert q.shape == (1,) and torch.isfinite(q).all()
        print(f"CHECK count invariance: {n_valid} valid tokens -> Q {q.item():.3f}  OK")

    env.close()
    print("CRITIC TEST PASSED")


if __name__ == "__main__":
    main()
