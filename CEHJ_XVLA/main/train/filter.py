"""Deployment filter: least-restrictive override on the NOMINAL's action.

    intervene  iff  Q(s, a_nom) < margin

The trigger scores the planner's own joint displacement for the upcoming
tick — "if the nominal does what it wants and the robot then behaves
optimally safely, does it stay clear" — not the safe actor's rescue
ability.

Single margin, no hysteresis: commitment comes from the hj_hold_ticks
intervention block instead (the RolloutController re-checks Q(s, a_nom) at
each block boundary and resumes the nominal when it clears the margin).

enc is a namespace (arm_tokens [1,2,D], scene [1,M,D], scene_mask)
produced by RolloutController; a_nom and actor actions are joint
displacements dtheta [1, A] bounded by dtheta_max; Jlin/Jang [1, 20, 3, A]
are the extractor's block-diagonal per-link Jacobians at the current state.
The slot layout (joint_cols/ee_rows/dtheta_max) is fixed per embodiment
and held by the filter, mirroring the parent's joint_index/dtheta_max.
"""

from __future__ import annotations

import torch


class SafetyFilter:
    def __init__(self, actor, critics, joint_cols, ee_rows, dtheta_max,
                 margin: float, hold_ticks: int = 3,
                 release_margin: float | None = None):
        self.actor = actor
        self.critics = critics
        self.joint_cols = joint_cols      # [1, 2, 7] long on cuda
        self.ee_rows = ee_rows            # [1, 2] long on cuda
        self.dtheta_max = dtheta_max      # [1, A] tensor on cuda
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
    def q_nom(self, enc, a_nom, Jlin, Jang) -> torch.Tensor:
        """Min-twin Q at the nominal action — the quantity being thresholded."""
        return self.critics.qmin(enc.arm_tokens, enc.scene, enc.scene_mask,
                                 a_nom, Jlin, Jang, self.ee_rows,
                                 self.joint_cols)

    @torch.no_grad()
    def q_actor(self, enc, Jlin, Jang) -> torch.Tensor:
        """V(s) = min-twin Q under the deterministic safe actor (diagnostic)."""
        dtheta_pi, _, _ = self.actor(enc.arm_tokens, enc.scene,
                                     enc.scene_mask, self.dtheta_max,
                                     self.joint_cols, deterministic=True)
        return self.critics.qmin(enc.arm_tokens, enc.scene, enc.scene_mask,
                                 dtheta_pi, Jlin, Jang, self.ee_rows,
                                 self.joint_cols)

    @torch.no_grad()
    def actor_action(self, enc) -> torch.Tensor:
        """The safe actor's deterministic dtheta for this tick."""
        dtheta_pi, _, _ = self.actor(enc.arm_tokens, enc.scene,
                                     enc.scene_mask, self.dtheta_max,
                                     self.joint_cols, deterministic=True)
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
