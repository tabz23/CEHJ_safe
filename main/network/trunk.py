"""Geometric trunk: body tokens gather local scene context.

Injection produced two token sets in one 256-d space: 22 body tokens with
world positions, and ~1200 scene tokens with world positions. The trunk
lets each body part find out what's around it. Output is the same 22 body
tokens, now carrying local scene context. No pooling — heads consume the
sequence.

Three blocks, each three residual sub-layers (pre-norm):

    body = body + cross_attn(body, scene, rel)   # what's near me
    body = body + self_attn(body)                # what the rest of the body does
    body = body + ffn(body)

Cross-attention geometric bias:
    rel  = scene_pos[:, None, :, :] - pos[:, :, None, :]   # [B, 22, M, 3] METRES
    dist = ||rel||;  dir = rel / dist
    - bias: small MLP on (log dist, dir) -> per-head additive logit
    - output channel: attention-weighted direction A @ dir is concatenated
      onto each token's output (two states with identical h sit on opposite
      sides of the BRT boundary depending on obstacle direction).

Rules this keeps:
  - scene validity masked only, never body validity (padded body queries
    attend normally; their outputs are discarded downstream). Masking a
    whole query row -> softmax(all -inf) -> NaN that survives all masking.
  - positions from `pos` (22 tokens, incl. residual anchors), never the
    20-entry link_pos.
  - rel stays in metres: h is a distance with a zero failure threshold on
    every robot; normalizing by reach would couple the threshold to the
    embodiment.
  - no distance cutoff: the bias MLP learns smooth decay; a hard threshold
    is a discontinuity in an input V depends on.
  - keys/values computed once per scene token (dominant cost M*d^2, not
    22*M*d^2); only the small bias MLP runs on the pair grid.
"""

from __future__ import annotations

import torch
from torch import nn


class GeoCrossAttention(nn.Module):
    """Cross-attention with geometric logit bias + direction output channel.

    dir_per_head=True keeps one attention-weighted direction per head
    (3*heads dims out — richer, heads may track different obstacles, but
    there is no single "direction to what I'm attending to" and head
    identity is arbitrary). dir_per_head=False averages over heads (3 dims
    — a clean geometric quantity for h_dot ≈ ∇h · p_dot). Ablation knob.
    """

    def __init__(self, dim: int = 256, heads: int = 8, bias_hidden: int = 32,
                 dir_per_head: bool = True):
        super().__init__()
        self.heads = heads
        self.dh = dim // heads
        self.dir_per_head = dir_per_head
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        # per-head additive logit from (log dist, dir) — "physically nearby
        # matters" as initialization rather than something RL must discover
        self.bias_mlp = nn.Sequential(
            nn.Linear(4, bias_hidden),
            nn.SiLU(),
            nn.Linear(bias_hidden, heads),
        )
        # value aggregate + attention-weighted direction
        self.out_proj = nn.Linear(dim + (3 * heads if dir_per_head else 3), dim)

    def forward(self, body, pos, scene, scene_pos, scene_mask):
        B, N, C = body.shape
        M = scene.shape[1]
        H = self.heads

        q = self.q_proj(body).view(B, N, H, self.dh).transpose(1, 2)   # B,H,N,dh
        k = self.k_proj(scene).view(B, M, H, self.dh).transpose(1, 2)  # B,H,M,dh
        v = self.v_proj(scene).view(B, M, H, self.dh).transpose(1, 2)

        # pair grid, METRES. 1 mm floor bounds |dir| <= 1 and keeps
        # log dist sane for near-coincident tokens (arm in view of a scene
        # token) — physically meaningless, numerically essential.
        rel = scene_pos[:, None, :, :] - pos[:, :, None, :]            # B,N,M,3
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-3)          # B,N,M,1
        dir_ = rel / dist
        bias_in = torch.cat([dist.log(), dir_], dim=-1)
        logit_bias = self.bias_mlp(bias_in).permute(0, 3, 1, 2)        # B,H,N,M

        logits = (q @ k.transpose(-1, -2)) * self.dh**-0.5 + logit_bias
        if scene_mask is not None:
            # scene validity only; finfo.min is fp16-safe (never overflows
            # to -inf) and still softmaxes to exactly 0
            logits = logits.masked_fill(
                ~scene_mask[:, None, None, :],
                torch.finfo(logits.dtype).min,
            )
        A = logits.softmax(dim=-1)                                     # B,H,N,M

        val = (A @ v).transpose(1, 2).reshape(B, N, C)                 # B,N,C
        direction = torch.einsum("bhnm,bnmc->bhnc", A, dir_)           # B,H,N,3
        if self.dir_per_head:
            direction = direction.transpose(1, 2).reshape(B, N, H * 3)
        else:
            direction = direction.mean(dim=1)                          # B,N,3
        return self.out_proj(torch.cat([val, direction], dim=-1))


class TrunkBlock(nn.Module):
    """cross_attn -> self_attn -> ffn, each pre-norm residual."""

    def __init__(self, dim: int = 256, heads: int = 8, ffn_dim: int = 1024,
                 dir_per_head: bool = True):
        super().__init__()
        self.norm_cross = nn.RMSNorm(dim)
        self.cross = GeoCrossAttention(dim, heads, dir_per_head=dir_per_head)
        self.norm_self = nn.RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ffn = nn.RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, body, pos, mask, scene, scene_pos, scene_mask):
        # what's near me
        body = body + self.cross(
            self.norm_cross(body), pos, scene, scene_pos, scene_mask
        )
        # what the rest of the body is doing (padded KEYS masked, padded
        # queries attend normally — their outputs are discarded downstream)
        x = self.norm_self(body)
        body = body + self.self_attn(
            x, x, x, key_padding_mask=~mask, need_weights=False
        )[0]
        body = body + self.ffn(self.norm_ffn(body))
        return body


class GeometricTrunk(nn.Module):
    """3 trunk blocks.

    in :  body [B, 22, 256]   pos [B, 22, 3]   mask [B, 22]
          scene [B, M, 256]   scene_pos [B, M, 3]   scene_mask [B, M]
    out:  body [B, 22, 256]   (same tokens, now with local scene context)
    """

    def __init__(self, dim: int = 256, heads: int = 8, depth: int = 3,
                 dir_per_head: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TrunkBlock(dim, heads, dir_per_head=dir_per_head) for _ in range(depth)]
        )

    def forward(self, body, pos, mask, scene, scene_pos, scene_mask):
        # inputs must already be on the module device (place once at the
        # batch boundary — copying 1200-token scene tensors per forward is
        # wasteful and per-call .to() hides real device bugs)
        for blk in self.blocks:
            body = blk(body, pos, mask, scene, scene_pos, scene_mask)
        return body
