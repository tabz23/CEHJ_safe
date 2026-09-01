"""Joint-space actor + twin critics for the X-VLA baseline.

Same action semantics as the joint-space parent (TokenActor/TwinCritic in
the main project): the action is a per-tick JOINT displacement dtheta,
bounded by dtheta_max = URDF velocity limit x control_dt x kappa
(kappa = 1.0 — a saturated actor tick and a full-speed cuRobo tick share
the same box). Per arm the layout is the spec's [j1..j7] slots (7 for
franka, 6 for piper — the 7th slot is dropped, never written), followed by
the 2 prismatic gripper slots, which the filter never drives (the task
script owns the gripper) and which stay exactly zero.

Representation in:  scene tokens [B, M, 256] (X-VLA adapter output) +
per-arm proprio tokens [B, 2, 256] (XVLAEncoder.encode_proprio). X-VLA has
no robot-state encoder, so there are no per-link body tokens — a plain
cross-attention trunk (EE6DTrunk, no geometric bias) fuses the two arm
tokens with the scene; heads read out per-arm actions/values from the arm
tokens.

Actor: Gaussian mean/logstd per arm over MAX_ARM_JOINTS slots, tanh,
scattered into the action vector by joint_cols (the slot -> action-column
map, from the BodyTokenExtractor's joint_index), scaled by dtheta_max read
at forward time (the action width follows the embodiment; it is not baked
into the parameters). Padded slots and prismatic columns are never written.

Critic: the action enters as per-token Cartesian effects, like the parent's
CriticHead — dp = Jlin @ dtheta localizes a joint-space action onto the
links (dtheta_3 moves every downstream link). With only two arm tokens, the
effect feature is evaluated at each arm's TERMINAL link (the merged gripper
token — its Jacobian row aggregates every upstream joint of that arm), plus
the arm's mean commanded dtheta. Twin heads, softmin readout over the arms
(softmin <= min — conservative in the safe direction).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

_LOG_2PI_HALF = 0.5 * math.log(2 * math.pi)
MAX_ARM_JOINTS = 7   # per-arm action slots (franka 7-DoF; 6-DoF arms drop
                     # the 7th slot — see joint_cols)


def arm_layout(joint_index, body_mask, max_slots: int = MAX_ARM_JOINTS):
    """Pack the extractor's joint_index/body_mask into the actor/critic
    layout (numpy; joint_index/body_mask [20] or [B, 20]).

    Returns (joint_cols, ee_rows):
      joint_cols [B, 2, max_slots] int64 — action column of arm-joint slot
        k of each arm, -1 where the arm has fewer than max_slots joints
        (6-DoF arms: slot 6 is -1 — "throw the additional joint").
      ee_rows [B, 2] int64 — body-token row of each arm's terminal link
        (the merged gripper token; its Jlin row aggregates the whole arm).
    """
    ji = np.asarray(joint_index, dtype=np.int64)
    bm = np.asarray(body_mask).astype(bool)
    if ji.ndim == 1:
        ji, bm = ji[None], bm[None]
    B = ji.shape[0]
    cols = np.full((B, 2, max_slots), -1, dtype=np.int64)
    ee = np.zeros((B, 2), dtype=np.int64)
    for b in range(B):
        # actuated columns in token order: left arm slots, then right arm
        # slots (the extractor's layout); the gripper token is -1
        act = ji[b][ji[b] >= 0]
        n = act.shape[0] // 2
        cols[b, 0, :n] = act[:n]
        cols[b, 1, :n] = act[n: 2 * n]
        rows = np.nonzero(bm[b])[0]
        nl = rows.shape[0] // 2
        ee[b, 0] = rows[nl - 1]
        ee[b, 1] = rows[2 * nl - 1]
    return cols, ee


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


class JointActor(nn.Module):
    """Per-arm SAC actor head over joint displacements (shared weights
    across the two arm tokens; count-invariant within an arm's slots).

    The action width is NOT a constructor constant: it is read from
    dtheta_max at forward time (the extractor produces dtheta_max and
    joint_index consistently per embodiment — piper A=16, franka A=18).

    in :  arm_tokens [B, 2, 256]   proprio tokens
          scene [B, M, 256]        adapter output
          joint_cols [B, 2, 7]     action column per slot, -1 = dropped
          dtheta_max [A] or [B, A] per-step displacement limit, A = 2*n_joint
    out:  dtheta [B, A]            the action (|dtheta| <= dtheta_max by tanh)
          logp   [B]               for SAC
          n_act  [B]               active slots, for the per-sample entropy
    """

    def __init__(self, dim: int = 256, depth: int = 2):
        super().__init__()
        self.trunk = EE6DTrunk(dim, depth=depth)
        self.head = nn.Linear(dim, MAX_ARM_JOINTS * 2)  # mean + logstd

    def forward(self, arm_tokens, scene, scene_mask, dtheta_max, joint_cols,
                deterministic: bool = False):
        B = arm_tokens.shape[0]
        dev = arm_tokens.device
        if joint_cols.dim() == 2:
            joint_cols = joint_cols.unsqueeze(0).expand(B, -1, -1)
        if dtheta_max.dim() == 1:
            dtheta_max = dtheta_max.unsqueeze(0).expand(B, -1)
        joint_cols = joint_cols.to(dev)
        dtheta_max = dtheta_max.to(dev).to(arm_tokens.dtype)

        h = self.trunk(arm_tokens, scene, scene_mask)    # [B, 2, D]
        mu, log_std = self.head(h).chunk(2, dim=-1)      # [B, 2, 7] each
        log_std = log_std.clamp(-8, 2)                   # keep std sane
        std = log_std.exp()

        if deterministic:
            pre = mu
        else:
            pre = mu + std * torch.randn_like(mu)        # reparameterized
        u = torch.tanh(pre)                              # bounded [-1, 1]

        normal_ll = (
            -0.5 * ((pre - mu) / std) ** 2 - log_std - _LOG_2PI_HALF
        )
        # tanh change-of-variables correction
        logp_tok = normal_ll - torch.log(1 - u.pow(2) + 1e-6)
        valid = joint_cols >= 0                          # [B, 2, 7]
        # where, not multiply: a non-finite dropped slot survives *0
        logp = torch.where(valid, logp_tok, torch.zeros_like(logp_tok)
                           ).sum(-1).sum(-1)

        # scatter by joint_cols (only valid slots; deterministic writes);
        # width from dtheta_max — the embodiment's true action dimension.
        # Dropped slots, prismatic columns and padding are never written.
        act = u.new_zeros(B, dtheta_max.shape[-1])
        bidx, armidx, slotidx = valid.nonzero(as_tuple=True)
        act[bidx, joint_cols[bidx, armidx, slotidx]] = u[bidx, armidx, slotidx]

        # scale LAST: |dtheta| <= dtheta_max by construction
        dtheta = act * dtheta_max
        n_act = valid.sum(-1).sum(-1)                    # per-sample entropy target
        return dtheta, logp, n_act


class JointTwinCritic(nn.Module):
    """Two per-arm value heads over trunked arm tokens + the joint action.

    Q(s, a): the action enters as its Cartesian effect at each arm's
    terminal link (dp = Jlin_ee @ dtheta, dw = Jang_ee @ dtheta) plus the
    arm's mean commanded dtheta — appended to the arm token via act_proj
    (shared between the twins, like the previous EE6D critic). The terminal
    link's Jacobian row aggregates every upstream joint of that arm.
    V(s) = softmin over arms of the per-arm values (conservative: <= min).
    The temperature is PART OF THE PROBLEM DEFINITION (physical metres,
    h_scale-scaled at construction) — it must match between training and
    evaluation and is a plain attribute, NOT in the state_dict.
    """

    def __init__(self, dim: int = 256, temperature: float = 0.4, depth: int = 2):
        super().__init__()
        self.temperature = temperature
        self.trunks = nn.ModuleList([EE6DTrunk(dim, depth=depth) for _ in range(2)])
        self.act_proj = nn.Linear(8, dim)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
            for _ in range(2)
        ])

    def forward(self, arm_tokens, scene, scene_mask, dtheta, Jlin, Jang,
                ee_rows, joint_cols):
        """dtheta [B, A], Jlin/Jang [B, 20, 3, A], ee_rows [B, 2],
        joint_cols [B, 2, 7] -> per-twin Q [B], V [B]."""
        B = arm_tokens.shape[0]
        dev = arm_tokens.device
        if ee_rows.dim() == 1:
            ee_rows = ee_rows.unsqueeze(0).expand(B, -1)
        if joint_cols.dim() == 2:
            joint_cols = joint_cols.unsqueeze(0).expand(B, -1, -1)
        dtheta = dtheta.to(dev).to(arm_tokens.dtype)
        Jlin = Jlin.to(dev).to(arm_tokens.dtype)
        Jang = Jang.to(dev).to(arm_tokens.dtype)
        ee_rows = ee_rows.to(dev)
        joint_cols = joint_cols.to(dev)

        # action to Cartesian effects, evaluated at each arm's terminal
        # link (its J row aggregates every upstream joint of that arm)
        dp = torch.einsum("bnca,ba->bnc", Jlin, dtheta)       # [B, 20, 3]
        dw = torch.einsum("bnca,ba->bnc", Jang, dtheta)       # [B, 20, 3]
        bidx = torch.arange(B, device=dev)[:, None]
        dp_ee = dp[bidx, ee_rows]                             # [B, 2, 3]
        dw_ee = dw[bidx, ee_rows]                             # [B, 2, 3]

        valid = joint_cols >= 0
        dth = torch.where(
            valid,
            dtheta.gather(1, joint_cols.clamp_min(0).flatten(1)
                        ).view(B, 2, MAX_ARM_JOINTS),
            torch.zeros(B, 2, MAX_ARM_JOINTS, dtype=dtheta.dtype, device=dev),
        )
        dth_mean = dth.sum(-1) / valid.sum(-1).clamp_min(1)   # [B, 2]
        eff = torch.cat(
            [dth_mean.unsqueeze(-1), dp_ee, dw_ee,
             valid.any(-1, keepdim=True).to(dtheta.dtype)],
            dim=-1,
        )                                                     # [B, 2, 8]

        qs, vs = [], []
        for trunk, head in zip(self.trunks, self.heads):
            h = trunk(arm_tokens + self.act_proj(eff), scene, scene_mask)
            v_arm = head(h).squeeze(-1)                        # [B, 2]
            # softmin over arms (conservative: <= min)
            v = -self.temperature * torch.logsumexp(
                -v_arm / self.temperature, dim=-1)
            qs.append(v)
            vs.append(v)
        return qs[0], qs[1], vs[0], vs[1]

    def qmin(self, arm_tokens, scene, scene_mask, dtheta, Jlin, Jang,
             ee_rows, joint_cols):
        q1, q2, _, _ = self.forward(arm_tokens, scene, scene_mask, dtheta,
                                    Jlin, Jang, ee_rows, joint_cols)
        return torch.minimum(q1, q2)
