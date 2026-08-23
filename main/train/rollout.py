"""Rollout driver on PlanEveryKController: task-native execution with hooks.

The task's play_once owns the stage machine (grip timing, stage advance,
check_success). PlanEveryKController makes it receding-horizon: plan,
execute a window of k physics steps, replan from the LIVE qpos. This is
the same env + controller stack as run.py --controller plan_play_once_everyk.

Modes:

  nominal   — plain receding-horizon rollout; per-tick h capture. If
              actor/critics are given, V is evaluated (shadow, never
              intervenes) at each window boundary.
  filtered  — 5 Hz safety filter: each window is one 0.2 s filter period
              (k=50 physics steps). Evaluate V at window start; if the
              planned action is unsafe (V < margin), the safe actor drives
              exactly this 0.2 s window; the next window replans from the
              steered state.
  collect   — nominal + per-tick dataset writes to a StepBuffer, with
              random dtheta perturbations at window boundaries
              (perturb_prob). dtheta stored per entry is the MEASURED
              arm-joint displacement to the next entry (never commanded).

Interleave points:

  scene.step wrapper — per-control-tick obs/h capture (trace/video/buffer)
  window boundary    — filtered mode: filter decision; collect mode:
                       perturbation injection
"""

from __future__ import annotations

import numpy as np
import torch

from main.network.body_features import BodyTokenExtractor  # noqa: E402
from main.train.collect import make_training_env  # noqa: E402
from main.train.hfunc import compute_h  # noqa: E402


class RolloutTimeout(Exception):
    """Raised from the scene.step hook when max_steps physics steps pass."""


EMBODIMENT_IDS = {"piper": 0, "franka-panda": 1, "ARX-X5": 2, "ur5-wsg": 3}


class RolloutController:
    """PlanEveryKController + per-tick capture + window-level filter."""

    def __init__(self, cfg, seed, mode="nominal", encoder=None, kin=None,
                 policy_enc=None, actor=None, critics=None, k: int | None = None,
                 buf=None, episode_id: int = 0, perturb_prob: float = 0.0,
                 rng=None, video=None, max_steps: int = 2500):
        from main.envs.controller import PlanEveryKController
        from main.train.filter import SafetyFilter

        self.cfg = cfg
        self.env = make_training_env(cfg, seed)
        self.extractor = BodyTokenExtractor(self.env)
        self.encoder = encoder
        self.kin = kin
        self.policy_enc = policy_enc
        self.mode = mode
        self.k = int(k if k is not None else getattr(cfg, "replan_k", 60))
        self.ctrl = PlanEveryKController(self.env, k=self.k)
        self.ctrl.attach()
        self.buf = buf
        self.episode_id = int(episode_id)
        self.perturb_prob = float(perturb_prob)
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.video = video
        self.video_fps = 10.0           # frames per SIM second (every 2nd tick)
        self._next_frame_t = 0.0
        self.max_steps = int(max_steps)
        self._pending = None
        self._n_physics = 0

        self.flt = None
        if actor is not None and critics is not None:
            # built for filtered mode AND for nominal-mode shadow V.
            # Margins are physical metres in the config; V lives in
            # h*h_scale units, so scale the thresholds to match.
            hs = float(getattr(cfg, "h_scale", 1.0))
            self.flt = SafetyFilter(
                actor, critics,
                self.extractor.extract()["joint_index"][None].cuda(),
                torch.from_numpy(
                    self.extractor.delta_theta_max.astype(np.float32)
                ).cuda(),
                margin_on=cfg.margin_on * hs, margin_off=cfg.margin_off * hs,
            )
        self.trace = {"t": [], "h": [], "V_t": [], "V": [], "intervened": []}
        self._tick_acc = 0.0
        self._orig_step = None
        self._last_ratio = 0.0

    # ---------------- capture ----------------
    def _capture_tick(self):
        obs = self.env.get_encoder_obs()
        h, diag = compute_h(self.env, self.cfg.table_margin, self.cfg.table_height)
        # h is trained/evaluated in h*h_scale units (config keeps metres
        # for physical margins; see FrozenConfig.h_scale)
        h = float(h) * float(getattr(self.cfg, "h_scale", 1.0))
        t_now = self._n_physics / self.env.PHYSICS_FREQ
        self.trace["t"].append(t_now)
        self.trace["h"].append(float(h))
        if self.mode == "collect" and self.buf is not None:
            self._collect_tick(obs, h, diag)
        if self.video is not None and t_now >= self._next_frame_t - 1e-9:
            # sim-time-based frame capture: 10 fps of simulation in BOTH
            # nominal (20 Hz tick capture) and filtered (5 Hz window
            # capture) modes
            self._next_frame_t = t_now + 1.0 / self.video_fps
            cam = self.env.task.cameras.observer_camera
            cam.take_picture()
            rgba = cam.get_picture("Color")
            frame = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
            v_now = self.trace["V"][-1] if self.trace["V"] else float("nan")
            engaged = bool(self.trace["intervened"][-1]) if self.trace["intervened"] else False
            self.video.add(
                frame, t_now, len(self.trace["t"]), -1, float(h), v_now,
                "FILTER" if engaged else "NOMINAL", self._last_ratio,
            )
        return obs

    def _collect_tick(self, obs, h, diag):
        """Stash this state; the previous stashed entry is finalized with the
        measured displacement to this one and written to the buffer.

        Cross-embodiment: variable-width fields are zero-padded to the
        buffer maxima (state tokens 16, action side 18). Jlin/Jang and
        dtheta_max are stored per step — a mixed-embodiment buffer cannot
        recompute them from qpos at sample time."""
        from main.train.buffer import MAX_ACTION, MAX_STATE_TOKENS

        with torch.no_grad():
            tokens, pos_mean, _ = self.encoder.encode_visual_tokens(
                obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"]
            )
            state = self.encoder.encode_joint_angles(
                obs["joint_state"], self.kin
            ).squeeze(2)[0]
        body = self.extractor.extract()
        per_link = np.zeros(20, dtype=np.float32)
        names = [f"L/{n}" for n in self.extractor.spec.link_names] + [
            f"R/{n}" for n in self.extractor.spec.link_names
        ]
        for i, name in enumerate(names[:20]):
            per_link[i] = diag["per_link"].get(name, np.inf)

        state_np = state.cpu().half().numpy()           # [N, 256], N in {14, 16}
        state_pad = np.zeros((MAX_STATE_TOKENS, state_np.shape[1]), np.float16)
        state_pad[: state_np.shape[0]] = state_np

        qpos_np = obs["joint_state"][0, 0].numpy().astype(np.float32)
        qpos_pad = np.zeros(MAX_STATE_TOKENS, np.float32)
        qpos_pad[: qpos_np.shape[0]] = qpos_np

        qraw = self.extractor.raw_qpos()                # [2*n_joint]
        qraw_pad = np.zeros(MAX_ACTION, np.float32)
        qraw_pad[: qraw.shape[0]] = qraw

        dmax = self.extractor.delta_theta_max.astype(np.float32)
        dmax_pad = np.zeros(MAX_ACTION, np.float32)
        dmax_pad[: dmax.shape[0]] = dmax

        Jlin = body["Jlin"].numpy()                     # [20, 3, 2*n_joint]
        Jang = body["Jang"].numpy()
        Jlin_pad = np.zeros((20, 3, MAX_ACTION), np.float16)
        Jang_pad = np.zeros((20, 3, MAX_ACTION), np.float16)
        Jlin_pad[:, :, : Jlin.shape[2]] = Jlin
        Jang_pad[:, :, : Jang.shape[2]] = Jang

        entry = {
            "scene_tokens": tokens.flatten(1, 2)[0].cpu().half().numpy(),
            "scene_pos": pos_mean.flatten(1, 2)[0].cpu().numpy(),
            "state_tokens": state_pad,
            "body_tokens": body["body_tokens"].numpy(),
            "link_pos": body["link_pos"].numpy(),
            "Jlin": Jlin_pad,
            "Jang": Jang_pad,
            "qpos": qpos_pad,
            "qpos_raw": qraw_pad,
            "dtheta_max": dmax_pad,
            "body_mask": body["body_mask"].numpy(),
            "joint_index": body["joint_index"].numpy().astype(np.int8),
            "embodiment_id": np.int8(EMBODIMENT_IDS[self.cfg.embodiment]),
            "h": np.float32(h),
            "per_link_h": per_link,
            "d_arm_arm": np.float32(diag["d_arm_arm"]),
            "episode_id": np.int32(self.episode_id),
            "done": np.bool_(False),
        }
        if self._pending is not None:
            ji = self._pending["joint_index"].astype(np.int64)
            valid = ji >= 0
            # measured arm-joint displacement to the new state; action
            # columns == raw qpos layout (both a*n_joint + k per spec)
            delta = entry["qpos_raw"].astype(np.float64) - self._pending[
                "qpos_raw"
            ].astype(np.float64)
            dtheta = np.zeros_like(delta)
            dtheta[ji[valid]] = delta[ji[valid]]
            self._pending["dtheta"] = dtheta.astype(np.float32)
            self.buf.append(self._pending)
        self._pending = entry

    def _flush_pending(self):
        if self._pending is not None and self.buf is not None:
            self._pending["dtheta"] = np.zeros_like(
                self._pending["qpos_raw"], dtype=np.float32
            )
            self._pending["done"] = np.bool_(True)
            self.buf.append(self._pending)
            self._pending = None

    def _model_inputs(self, obs):
        with torch.no_grad():
            tokens, pos_mean, _ = self.encoder.encode_visual_tokens(
                obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"]
            )
            state = self.encoder.encode_joint_angles(obs["joint_state"], self.kin).squeeze(2)
            body = self.extractor.extract()
            enc = self.policy_enc(
                state, body["body_tokens"][None], body["link_pos"][None],
                body["body_mask"][None], body["joint_index"][None],
                tokens, pos_mean,
            )
        return body, enc

    # ---------------- window-level hooks ----------------
    def _filtered_play_seq(self, task, left_arm, right_arm, left_g, right_g,
                           n_chunk, grip_i):
        # obs for the MODEL at the window boundary; the trace/video capture
        # runs independently in the per-tick scene.step hook (all modes)
        obs = self.env.get_encoder_obs()
        body, enc = self._model_inputs(obs)
        with torch.no_grad():
            action, V, engaged = self.flt.action(
                enc, body["Jlin"][None], body["Jang"][None], None
            )
        self.trace["V_t"].append(self._n_physics / self.env.PHYSICS_FREQ)
        self.trace["V"].append(float(V))
        self.trace["intervened"].append(bool(engaged))
        if engaged:
            a = action[0].cpu().numpy()
            self._last_ratio = float(
                np.abs(a).max() / max(self.extractor.delta_theta_max.max(), 1e-9)
            )
            # drive the actor's action for exactly 0.2 s (4 control ticks)
            for _ in range(max(1, round(int(n_chunk) / 12.5))):
                self.env.step_dtheta(a)
            return True
        return self._orig_play_seq(
            task, left_arm, right_arm, left_g, right_g, n_chunk, grip_i
        )

    def _shadow_play_seq(self, task, left_arm, right_arm, left_g, right_g,
                         n_chunk, grip_i):
        """Nominal window, but evaluate V at the window start (never act)."""
        obs = self.env.get_encoder_obs()
        body, enc = self._model_inputs(obs)
        with torch.no_grad():
            V = self.flt.value(enc, body["Jlin"][None], body["Jang"][None])
        self.trace["V_t"].append(self._n_physics / self.env.PHYSICS_FREQ)
        self.trace["V"].append(float(V))
        self.trace["intervened"].append(False)
        return self._orig_play_seq(
            task, left_arm, right_arm, left_g, right_g, n_chunk, grip_i
        )

    def _collect_play_seq(self, task, left_arm, right_arm, left_g, right_g,
                          n_chunk, grip_i):
        """Nominal window; with prob perturb_prob first drive one tick of
        uniform random arm displacement (the next window replans from the
        disturbed state — coverage of off-nominal configurations)."""
        if self.rng.rand() < self.perturb_prob:
            ji = self.extractor.extract()["joint_index"].numpy().astype(np.int64)
            cols = ji[ji >= 0]
            dmax = np.asarray(self.extractor.delta_theta_max, dtype=np.float64)
            dtheta = np.zeros(2 * self.extractor.spec.n_joints)
            dtheta[cols] = self.rng.uniform(-1, 1, len(cols)) * dmax[cols]
            # one control tick through env.step -> hooked capture fires
            self.env.step_dtheta(dtheta)
        return self._orig_play_seq(
            task, left_arm, right_arm, left_g, right_g, n_chunk, grip_i
        )

    # ---------------- main loop ----------------
    def run(self, max_steps: int | None = None):
        from main.envs.controller import PlanEveryKController

        if max_steps is not None:
            self.max_steps = int(max_steps)
        self._orig_play_seq = PlanEveryKController._play_seq.__get__(
            self.ctrl, PlanEveryKController
        )
        if self.mode == "filtered" and self.flt is not None:
            self.ctrl._play_seq = self._filtered_play_seq
        elif self.mode == "nominal" and self.flt is not None:
            self.ctrl._play_seq = self._shadow_play_seq
        elif self.mode == "collect":
            self.ctrl._play_seq = self._collect_play_seq

        self._orig_step = self.env.task.scene.step
        env = self.env

        def hooked_step():
            self._orig_step()
            self._n_physics += 1
            if self._n_physics > self.max_steps:
                raise RolloutTimeout(
                    f"rollout exceeded {self.max_steps} physics steps"
                )
            # control_freq/PHYSICS_FREQ per physics step -> one capture per
            # control tick (20 Hz), NOT per physics step
            self._tick_acc += env.control_freq / env.PHYSICS_FREQ
            if self._tick_acc >= 1.0:
                self._tick_acc -= 1.0
                self._capture_tick()  # per control tick (20 Hz), all modes

        env.task.scene.step = hooked_step
        success = False
        try:
            if self.mode == "collect":
                self._capture_tick()  # initial state s_0
            env.task.play_once()
            success = bool(env.task.check_success())
        except RolloutTimeout as exc:
            print(f"[rollout] {exc}")
        except Exception:
            import traceback

            traceback.print_exc()
        env.task.scene.step = self._orig_step
        if self.mode == "collect":
            self._flush_pending()
        self.trace["success"] = success
        self.trace["n_physics"] = self._n_physics
        if self.flt is not None:
            self.trace["interventions"] = self.flt.n_interventions
            self.trace["intervention_rate"] = self.flt.intervention_rate
        env.close()
        return self.trace
