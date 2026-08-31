"""Actor / critic heads on top of the trunk.

TokenActor: per-token SAC policy head on actuated body tokens only,
scattered into the action vector by joint_index. Shared weights across all
joints and both arms — the action dimension follows the embodiment
(count-invariant), it is not baked into the parameters.

in :  body [B, 22, 256]        trunk output
      joint_index [B, 20]      action column per token, -1 for gripper/padding
      dtheta_max [A] or [B, A] per-step displacement limit, A = 2*n_joint
out:  dtheta [B, A]            the action (|dtheta| <= dtheta_max by tanh)
      logp   [B]               for SAC
      n_act  [B]               active dims, for the per-sample entropy target

Design notes:
  - residual tokens are dropped first: they carry value heads, not action
    heads — they correspond to no actuator.
  - reparameterized Gaussian + tanh: gradients flow to mu/std, and the
    action is bounded so scaling by dtheta_max makes every action feasible
    by construction (commanded == executed; the value function learns the
    true dynamics; URDF limits enter analytically, not learned).
  - logp uses torch.where(actuated, logp, 0), NOT multiplication — a
    non-finite padded slot survives * 0 but not where().
  - scatter by joint_index, never by position: after the gripper merge,
    token 7 is the right arm's first link but action column 8.
  - joint space, not task space: theta' = theta + dtheta is exact,
    continuous, always defined; IK stays in the nominal controller.
  - target entropy is per-sample (-n_act), because A varies with the
    embodiment and gripper columns are never written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .trunk import TrunkBlock

_LOG_2PI_HALF = 0.5 * math.log(2 * math.pi)


@dataclass
class Encoded:
    """The shared encoding both critics consume — one object, so the twins
    can't drift. Produced once by PolicyEncoder (injection + trunk)."""

    body: torch.Tensor        # [B, 22, 256] trunk output
    pos: torch.Tensor         # [B, 22, 3] world metres
    mask: torch.Tensor        # [B, 22] bool
    scene: torch.Tensor       # [B, M, 256]
    scene_pos: torch.Tensor   # [B, M, 3] world metres
    scene_mask: torch.Tensor  # [B, M] bool


class PolicyEncoder(nn.Module):
    """injection + trunk, run once; returns the Encoded shared object."""

    def __init__(self, injection: nn.Module, trunk: nn.Module):
        super().__init__()
        self.injection = injection
        self.trunk = trunk

    def forward(self, state_tokens, body_tokens, link_pos, body_mask,
                joint_index, scene_tokens, scene_pos) -> Encoded:
        btok, bpos, bmask = self.injection(
            state_tokens, body_tokens, link_pos, body_mask, joint_index
        )
        scene, spos = self.injection.adapt_scene(scene_tokens, scene_pos)
        smask = torch.ones(scene.shape[:2], dtype=torch.bool, device=scene.device)
        body_out = self.trunk(btok, bpos, bmask, scene, spos, smask)
        return Encoded(body_out, bpos, bmask, scene, spos, smask)


class CriticHead(nn.Module):
    """Per-twin critic: action enters here, as per-token Cartesian effects.

    dp_i = Jlin_i @ dtheta aggregates every upstream joint's contribution
    into one quantity attached to link i — joint-space dtheta does not
    localize (dtheta_3 moves every downstream link). Grippers get action
    effects but no action head: no actuator of their own, but they move,
    and they're frequently the binding link. Residual tokens get zeros.

    Softmin readout: V(s) = min_i V_i is exact (mins commute), softmin is
    conservative in the safe direction (softmin <= min), spreads gradient
    across heads, and is count-invariant. Padding must LOSE the min:
    masked_fill with +1e4 (NOT finfo.min — that sign is for softmax).

    temperature is PART OF THE PROBLEM DEFINITION (it sets how conservative
    h's soft version is): V is in metres, the interesting range is about
    [-0.2, 0.5], and at T=1.0 the softmin degenerates to -T*log(N) rather
    than approximating a minimum. Default 0.02 (2 cm). It must be identical
    between training and evaluation — freeze it next to dt and dtheta_max.

    post_action_geometry: if True, the critic's blocks compute the
    attention bias from pos + dp (the post-action configuration) instead of
    the pre-action pos — the action's effect enters the attention geometry,
    not just the token features.
    """

    def __init__(self, dim: int = 256, n_actions: int = 16, depth: int = 2,
                 temperature: float = 0.02, post_action_geometry: bool = True,
                 debug: bool = False, geo: bool = True):
        super().__init__()
        self.n_actions = n_actions
        self.temperature = temperature
        # post-action bias requires the geometric attention; ablating the
        # trunk's geometry forces it off
        self.post_action_geometry = post_action_geometry and geo
        self.debug = debug
        self._layout_cache: dict = {}
        self.act_proj = nn.Linear(8, dim)
        self.blocks = nn.ModuleList(
            [TrunkBlock(dim, geo=geo) for _ in range(depth)]
        )
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )

    def _unmapped_cols(self, joint_index, A: int, dev) -> torch.Tensor:
        """Layout mask of action columns no token maps to. Layout changes
        only with the embodiment, so it is cached by content."""
        key = (A, tuple(joint_index.reshape(-1, joint_index.shape[-1])[0].tolist()))
        present = self._layout_cache.get(key)
        if present is None:
            present = torch.zeros(A, dtype=torch.bool, device=dev)
            present[joint_index[joint_index >= 0]] = True
            self._layout_cache[key] = present
        return ~present

    def forward(self, enc: Encoded, dtheta, Jlin, Jang, joint_index):
        B, M_body, C = enc.body.shape
        M = joint_index.shape[-1]
        n_res = M_body - M
        dev = enc.body.device
        if joint_index.dim() == 1:
            joint_index = joint_index.unsqueeze(0).expand(B, -1)
        joint_index = joint_index.to(dev)
        dtheta = dtheta.to(dev).to(enc.body.dtype)
        Jlin = Jlin.to(dev).to(enc.body.dtype)
        Jang = Jang.to(dev).to(enc.body.dtype)

        # critic boundary: action must be exactly zero at unmapped columns.
        # Layout is cached (changes only with the embodiment); the assert
        # itself is gated behind debug — it forces a GPU sync per call.
        if self.debug:
            unmapped = self._unmapped_cols(joint_index, dtheta.shape[-1], dev)
            assert bool((dtheta[..., unmapped] == 0).all()), (
                f"dtheta non-zero at unmapped columns "
                f"{unmapped.nonzero().flatten().tolist()}"
            )

        # Step 1 — action to per-token Cartesian effects
        dp = torch.einsum("bnca,ba->bnc", Jlin, dtheta)       # [B, M, 3]
        dw = torch.einsum("bnca,ba->bnc", Jang, dtheta)       # [B, M, 3]
        valid = joint_index >= 0
        dth = torch.where(
            valid,
            dtheta.gather(1, joint_index.clamp_min(0)),
            torch.zeros_like(joint_index, dtype=dtheta.dtype),
        )                                                     # [B, M]
        eff = torch.cat(
            [dth.unsqueeze(-1), dp, dw, valid.to(dtheta.dtype).unsqueeze(-1)],
            dim=-1,
        )                                                     # [B, M, 8]
        eff = torch.cat(
            [eff, eff.new_zeros(B, n_res, 8)], dim=1
        )                                                     # [B, 22, 8]
        body = enc.body + self.act_proj(eff)

        # query positions: post-action configuration if enabled (residual
        # tokens have no Jlin row — their dp is zero)
        if self.post_action_geometry:
            dp_full = torch.cat(
                [dp, dp.new_zeros(B, n_res, 3)], dim=1
            )
            pos_q = enc.pos + dp_full
        else:
            pos_q = enc.pos

        # Step 2 — own late blocks, same masking rules: tokens re-consult
        # the scene with the action applied
        for blk in self.blocks:
            body = blk(body, pos_q, enc.mask, enc.scene,
                       enc.scene_pos, enc.scene_mask)

        # Step 3 — softmin readout (+1e4: padding must LOSE a min)
        V = self.value_head(body).squeeze(-1)                 # [B, 22]
        V = V.masked_fill(~enc.mask, 1e4)
        T = self.temperature
        Q = -T * torch.logsumexp(-V / T, dim=-1)              # [B]
        return Q, V


class TwinCritic(nn.Module):
    """Two CriticHeads with separate late blocks and value heads, both
    consuming the same Encoded object. Min-of-twins is only pessimistic if
    the errors decorrelate — sharing late blocks would defeat it, and for
    an avoid problem pessimism is the correct conservative direction."""

    def __init__(self, dim: int = 256, n_actions: int = 16, depth: int = 2,
                 temperature: float = 0.02, post_action_geometry: bool = True,
                 debug: bool = False, geo: bool = True):
        super().__init__()
        self.c1 = CriticHead(dim, n_actions, depth, temperature,
                             post_action_geometry, debug, geo=geo)
        self.c2 = CriticHead(dim, n_actions, depth, temperature,
                             post_action_geometry, debug, geo=geo)

    def forward(self, enc: Encoded, dtheta, Jlin, Jang, joint_index):
        q1, v1 = self.c1(enc, dtheta, Jlin, Jang, joint_index)
        q2, v2 = self.c2(enc, dtheta, Jlin, Jang, joint_index)
        return q1, q2, v1, v2

    def qmin(self, enc: Encoded, dtheta, Jlin, Jang, joint_index):
        q1, q2, _, _ = self.forward(enc, dtheta, Jlin, Jang, joint_index)
        return torch.minimum(q1, q2)


class TokenActor(nn.Module):
    """Per-token SAC actor head (shared weights, count-invariant).

    The action width is NOT a constructor constant: it is read from
    dtheta_max at forward time (the extractor produces dtheta_max and
    joint_index consistently per embodiment — piper A=16, franka A=18).
    """

    def __init__(self, dim: int = 256, n_actions: int = None):
        super().__init__()
        self.head = nn.Linear(dim, 2)

    def forward(self, body, joint_index, dtheta_max, deterministic: bool = False):
        B = body.shape[0]
        dev = body.device
        if joint_index.dim() == 1:
            joint_index = joint_index.unsqueeze(0).expand(B, -1)
        if dtheta_max.dim() == 1:
            dtheta_max = dtheta_max.unsqueeze(0).expand(B, -1)
        joint_index = joint_index.to(dev)
        dtheta_max = dtheta_max.to(dev).to(body.dtype)

        # drop residual tokens via joint_index's own length (the extractor's
        # padded M) — never a hardcoded residual count
        body_only = body[:, : joint_index.shape[1]]
        mu, log_std = self.head(body_only).unbind(-1)  # [B, M] each
        log_std = log_std.clamp(-8, 2)                 # keep std in a sane range
        std = log_std.exp()

        if deterministic:
            pre = mu
        else:
            pre = mu + std * torch.randn_like(mu)      # reparameterized sample
        u = torch.tanh(pre)                            # bounded to [-1, 1]

        normal_ll = (
            -0.5 * ((pre - mu) / std) ** 2 - log_std - _LOG_2PI_HALF
        )
        # tanh change-of-variables correction
        logp_tok = normal_ll - torch.log(1 - u.pow(2) + 1e-6)
        valid = joint_index >= 0
        # where, not multiply: non-finite padded slots survive *0
        logp = torch.where(valid, logp_tok, torch.zeros_like(logp_tok)).sum(-1)

        # scatter by joint_index (only valid entries; deterministic writes);
        # width from dtheta_max — the embodiment's true action dimension
        act = u.new_zeros(B, dtheta_max.shape[-1])
        bidx, tidx = valid.nonzero(as_tuple=True)
        act[bidx, joint_index[bidx, tidx]] = u[bidx, tidx]

        # scale LAST: |dtheta| <= dtheta_max by construction
        dtheta = act * dtheta_max
        n_act = valid.sum(-1)                          # per-sample entropy target
        return dtheta, logp, n_act
