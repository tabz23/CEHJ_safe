"""Deployment filter: least-restrictive override with hysteresis.

    a = a_nom  if V(s) >= margin  else  actor(s, deterministic=True)

Engages below margin_on, releases above margin_off (> margin_on) — a state
near threshold doesn't chatter between branches.
"""

from __future__ import annotations

import torch


class SafetyFilter:
    def __init__(self, actor, critics, joint_index, dtheta_max,
                 margin_on: float = 0.0, margin_off: float = 0.02):
        self.actor = actor
        self.critics = critics
        self.joint_index = joint_index
        self.dtheta_max = dtheta_max
        self.margin_on = margin_on
        self.margin_off = margin_off
        self.engaged = False
        self.n_interventions = 0
        self.n_steps = 0

    @torch.no_grad()
    def value(self, enc, Jlin, Jang):
        """V(s) = min-twin Q under the deterministic actor."""
        dtheta_pi, _, _ = self.actor(
            enc.body, self.joint_index, self.dtheta_max, deterministic=True
        )
        return self.critics.qmin(enc, dtheta_pi, Jlin, Jang, self.joint_index)

    @torch.no_grad()
    def action(self, enc, Jlin, Jang, a_nom):
        """Filtered action; tracks hysteresis state and intervention rate."""
        self.n_steps += 1
        dtheta_pi, _, _ = self.actor(
            enc.body, self.joint_index, self.dtheta_max, deterministic=True
        )
        V = self.critics.qmin(enc, dtheta_pi, Jlin, Jang, self.joint_index)
        v = float(V.item())
        if not self.engaged and v < self.margin_on:
            self.engaged = True
        elif self.engaged and v > self.margin_off:
            self.engaged = False
        if self.engaged:
            self.n_interventions += 1
            return dtheta_pi, v, True
        return a_nom, v, False

    @property
    def intervention_rate(self) -> float:
        return self.n_interventions / max(self.n_steps, 1)
