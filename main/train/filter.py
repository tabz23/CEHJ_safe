"""Deployment filter: least-restrictive override on the NOMINAL's action.

    intervene  iff  Q(s, a_nom) < margin

The trigger scores the planner's own action for the upcoming tick — "if the
nominal does what it wants and the robot then behaves optimally safely, does
it stay clear" — not the safe actor's rescue ability (the old Q(s, pi_safe)
trigger, which only goes unsafe once the safe policy is already losing).

Single margin, no hysteresis: commitment comes from the hj_hold_ticks
intervention block instead (the RolloutController re-checks Q(s, a_nom) at
each block boundary and resumes the nominal when it clears the margin).
"""

from __future__ import annotations

import torch


class SafetyFilter:
    def __init__(self, actor, critics, joint_index, dtheta_max,
                 margin: float, hold_ticks: int = 3,
                 release_margin: float | None = None):
        self.actor = actor
        self.critics = critics
        self.joint_index = joint_index
        self.dtheta_max = dtheta_max
        self.margin = float(margin)     # in h*h_scale units
        # hysteresis: once engaged, hold until Q clears the release margin
        # (default = margin, i.e. single threshold as before)
        self.release_margin = (float(margin) if release_margin is None
                               else float(release_margin))
        self.hold_ticks = int(hold_ticks)
        self.n_interventions = 0
        self.n_switches = 0             # chatter diagnostic
        self.n_steps = 0
        self._was_engaged = False

    @torch.no_grad()
    def q_nom(self, enc, a_nom, Jlin, Jang) -> torch.Tensor:
        """Min-twin Q at the nominal action — the quantity being thresholded."""
        return self.critics.qmin(enc, a_nom, Jlin, Jang, self.joint_index)

    @torch.no_grad()
    def q_actor(self, enc, Jlin, Jang) -> torch.Tensor:
        """V(s) = min-twin Q under the deterministic safe actor (diagnostic)."""
        dtheta_pi, _, _ = self.actor(
            enc.body, self.joint_index, self.dtheta_max, deterministic=True
        )
        return self.critics.qmin(enc, dtheta_pi, Jlin, Jang, self.joint_index)

    @torch.no_grad()
    def actor_action(self, enc) -> torch.Tensor:
        """The safe actor's deterministic action for this tick."""
        dtheta_pi, _, _ = self.actor(
            enc.body, self.joint_index, self.dtheta_max, deterministic=True
        )
        return dtheta_pi

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
