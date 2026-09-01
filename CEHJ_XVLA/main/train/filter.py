"""Deployment filter: least-restrictive override on the NOMINAL's action.

    intervene  iff  Q(s, a_nom) < margin

The trigger scores the planner's own EE6D action for the upcoming tick —
"if the nominal does what it wants and the robot then behaves optimally
safely, does it stay clear" — not the safe actor's rescue ability.

Single margin, no hysteresis: commitment comes from the hj_hold_ticks
intervention block instead (the RolloutController re-checks Q(s, a_nom) at
each block boundary and resumes the nominal when it clears the margin).

enc is an _Enc namespace (arm_tokens [1,2,D], scene [1,M,D], scene_mask)
produced by RolloutController; a_nom and actor actions are EE6D deltas
[1, 20] bounded per-dim by step_max.
"""

from __future__ import annotations

import torch


class SafetyFilter:
    def __init__(self, actor, critics, step_max, margin: float,
                 hold_ticks: int = 3, release_margin: float | None = None):
        self.actor = actor
        self.critics = critics
        self.step_max = step_max          # [1, 20] tensor on cuda
        self.margin = float(margin)       # in h*h_scale units
        # hysteresis: once engaged, hold until Q clears the release margin
        # (default = margin, i.e. single threshold)
        self.release_margin = (float(margin) if release_margin is None
                               else float(release_margin))
        self.hold_ticks = int(hold_ticks)
        self.n_interventions = 0
        self.n_switches = 0               # chatter diagnostic
        self.n_steps = 0
        self._was_engaged = False

    @torch.no_grad()
    def q_nom(self, enc, a_nom) -> torch.Tensor:
        """Min-twin Q at the nominal action — the quantity being thresholded."""
        return self.critics.qmin(enc.arm_tokens, enc.scene, enc.scene_mask,
                                 a_nom)

    @torch.no_grad()
    def q_actor(self, enc) -> torch.Tensor:
        """V(s) = min-twin Q under the deterministic safe actor (diagnostic)."""
        a_pi, _, _ = self.actor(enc.arm_tokens, enc.scene, enc.scene_mask,
                                self.step_max, deterministic=True)
        return self.critics.qmin(enc.arm_tokens, enc.scene, enc.scene_mask,
                                 a_pi)

    @torch.no_grad()
    def actor_action(self, enc) -> torch.Tensor:
        """The safe actor's deterministic EE6D delta for this tick."""
        a_pi, _, _ = self.actor(enc.arm_tokens, enc.scene, enc.scene_mask,
                                self.step_max, deterministic=True)
        return a_pi

    def track(self, engaged: bool) -> None:
        """Bookkeeping per tick: intervention rate and mode-switch count."""
        self.n_steps += 1
        if engaged:
            self.n_interventions += 1
        if engaged != self._was_engaged:
            self.n_switches += 1
        self._was_engaged = engaged

    @property
    def intervention_rate(self) -> float:
        return self.n_interventions / max(self.n_steps, 1)
