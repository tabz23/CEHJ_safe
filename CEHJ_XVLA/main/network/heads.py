"""EE6D actor + twin critics for the X-VLA baseline.

Same HJ-SAC structure as the joint-space heads (TokenActor/TwinCritic in the
main project) but the action is X-VLA's EE6D: per arm [dx, dy, dz, drot6d(6)]
— the filter drops the gripper: 9 dims per arm, 18 total per tick.

Representation in:  scene tokens [B, M, 256] (X-VLA adapter output) +
per-arm proprio tokens [B, 2, 256] (XVLAEncoder.encode_proprio). A plain
cross-attention trunk (no geometric bias — X-VLA carries no metric 3D)
fuses them; heads read out per-arm actions/values from the arm tokens.

Actor: Gaussian mean/logstd per arm over the 9-dim EE6D delta, tanh-bounded
by per-dim limits (ee_step_max). Critic: per-arm value heads + softmin readout
(same conservative trick as the joint-space critic: V = softmin over arms).
"""

from __future__ import annotations

import math

import torch
from torch import nn

_LOG_2PI_HALF = 0.5 * math.log(2 * math.pi)
EE6D_ARM_DIM = 9   # xyz(3) + rot6d(6) — no gripper: the safety
                     # filter never drives the gripper
N_ACTION = 18      # two arms


class EE6DTrunk(nn.Module):
    """Plain cross-attention: arm tokens consult the scene tokens."""

    def __init__(self, dim: int = 256, heads: int = 8, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({
                "norm_q": nn.RMSNorm(dim),
                "cross": nn.MultiheadAttention(dim, heads, batch_first=True),
                "norm_ffn": nn.RMSNorm(dim),
                "ffn": nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                     nn.Linear(dim * 4, dim)),
            }) for _ in range(depth)]
        )

    def forward(self, arms, scene, scene_mask=None):
        # arms [B, 2, D], scene [B, M, D]
        for blk in self.blocks:
            q = blk["norm_q"](arms)
            arms = arms + blk["cross"](q, scene, scene,
                                       key_padding_mask=scene_mask)[0]
            arms = arms + blk["ffn"](blk["norm_ffn"](arms))
        return arms


class EE6DActor(nn.Module):
    """Gaussian + tanh over EE6D deltas, per arm."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.trunk = EE6DTrunk(dim)
        self.head = nn.Linear(dim, EE6D_ARM_DIM * 2)  # mean + logstd per arm

    def forward(self, arm_tokens, scene, scene_mask, step_max, deterministic=False):
        """arm_tokens [B,2,D] -> a [B, 18] EE6D delta, tanh-bounded by step_max."""
        B = arm_tokens.shape[0]
        h = self.trunk(arm_tokens, scene, scene_mask)         # [B, 2, D]
        mu, logstd = self.head(h).chunk(2, dim=-1)            # [B, 2, 9]
        logstd = logstd.clamp(-5, 2)
        if deterministic:
            a = torch.tanh(mu)
            logp = torch.zeros(B, device=mu.device)
        else:
            std = logstd.exp()
            eps = torch.randn_like(mu)
            a_raw = mu + eps * std
            a = torch.tanh(a_raw)
            logp = (-0.5 * (eps ** 2) - logstd - _LOG_2PI_HALF).sum(-1).sum(-1)
            logp = logp - (2 * (math.log(2.0) - a_raw
                              - torch.nn.functional.softplus(-2 * a_raw))
                           ).sum(-1).sum(-1)
        return (a * step_max.view(B, 2, EE6D_ARM_DIM)).view(B, N_ACTION), logp


class EE6DTwinCritic(nn.Module):
    """Two per-arm value heads over trunked arm tokens + the EE6D action.

    Q(s, a): the action enters as a per-arm vector appended to each arm
    token (no Jacobian localization — EE6D is already Cartesian per arm).
    V(s) = softmin over arms of min-twin per-arm values.
    """

    def __init__(self, dim: int = 256, temperature: float = 0.4, depth: int = 2):
        super().__init__()
        self.temperature = temperature
        self.trunks = nn.ModuleList([EE6DTrunk(dim, depth=depth) for _ in range(2)])
        self.act_proj = nn.Linear(EE6D_ARM_DIM, dim)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
            for _ in range(2)
        ])

    def forward(self, arm_tokens, scene, scene_mask, action):
        """action [B, 18] -> per-arm Q [B], V [B] (softmin over arms)."""
        B = arm_tokens.shape[0]
        a_arm = action.view(B, 2, EE6D_ARM_DIM)
        act = self.act_proj(a_arm)
        vs = []
        qs = []
        for trunk, head in zip(self.trunks, self.heads):
            h = trunk(arm_tokens + act, scene, scene_mask)
            v_arm = head(h).squeeze(-1)                        # [B, 2]
            # softmin over arms (conservative: <= min)
            v = -self.temperature * torch.logsumexp(
                -v_arm / self.temperature, dim=-1)
            q = v                                             # per-tick Q
            vs.append(v)
            qs.append(q)
        return qs[0], qs[1], vs[0], vs[1]

    def qmin(self, arm_tokens, scene, scene_mask, action):
        q1, q2, _, _ = self.forward(arm_tokens, scene, scene_mask, action)
        return torch.minimum(q1, q2)
