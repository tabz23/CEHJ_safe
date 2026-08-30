"""Rollout driver on TickChunkedController: vanilla nominal + per-tick hooks.

The task's play_once owns the stage machine. TickChunkedController walks each
cuRobo plan in fixed tick chunks (exactly PHYSICS_FREQ/control_freq physics
steps — 10 at 25 Hz) and calls the per-tick hook at every boundary:

  hook False -> play the next tick of plan rows (open-loop within the Action)
  hook True  -> an intervention just drove the arm for hj_hold_ticks ticks
                (actor re-queried every tick); the controller discards the
                stale plan and replans the same target from live qpos.

Filter trigger: Q(s, a_nom) < margin — the critic scores the NOMINAL's action
for the upcoming tick (a_nom = pos[min(i_end, T-1)] - qpos_now, per arm),
which is the textbook least-restrictive semantics.

Modes:
  filtered — deployment stack: filter active every tick.
  collect  — dataset writes to a StepBuffer. If models are given AND
             filter_active, the filter runs (DAgger rounds); otherwise the
             nominal runs untouched and only commanded actions are recorded.

Capture: a scene.step hook records h/obs at every control tick (all stepping
paths — plan chunks, actor ticks, perturbations — go through scene.step).
The buffer's dtheta is the COMMANDED action per tick (planner a_nom, actor
dtheta, or the sampled perturbation), with action_source marking which.
"""

from __future__ import annotations

import numpy as np
import torch

from main.network.body_features import BodyTokenExtractor  # noqa: E402
from main.train.collect import make_training_env  # noqa: E402
from main.train.hfunc import compute_h  # noqa: E402


class RolloutTimeout(Exception):
    """Raised from the scene.step hook when max_steps physics steps pass."""

EMBODIMENT_IDS = {"piper": 0, "franka-panda": 1, "ARX-X5": 2,
                 "ur5-wsg": 3, "aloha-agilex": 4}

# action_source values stored in the buffer
SRC_PLANNER, SRC_ACTOR, SRC_PERTURB = 0, 1, 2


class RolloutController:
    """TickChunkedController + per-tick capture + tick-level filter."""

    def __init__(self, cfg, seed, mode="filtered", encoder=None, kin=None,
                 policy_enc=None, actor=None, critics=None,
                 filter_active: bool = True,
                 buf=None, episode_id: int = 0, perturb_prob: float = 0.0,
                 rng=None, video=None, max_steps: int = 4500):
        from main.envs.controller import TickChunkedController
        from main.train.filter import SafetyFilter

        self.cfg = cfg
        self.env = make_training_env(cfg, seed)
        self.extractor = BodyTokenExtractor(self.env, kappa=cfg.kappa)
        self.encoder = encoder
        self.kin = kin
        self.policy_enc = policy_enc
        self.mode = mode
        self.ctrl = TickChunkedController(self.env)
        self.ctrl.attach()
        self.ctrl.tick_hook = self._tick_hook
        self.buf = buf
        self.episode_id = int(episode_id)
        self.perturb_prob = float(perturb_prob)
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.video = video
        self.video_fps = 25.0           # every control tick, 1:1 sim time —
        # a 3-tick block always gets 3 frames (never aliased)
        self._next_frame_t = 0.0
        self._last_diag = None
        self.max_steps = int(max_steps)
        self._pending = None
        self._n_physics = 0
        # commanded action + source for the CURRENT tick (read by
        # _collect_tick when finalizing the pending entry)
        self._n_act = 2 * self.extractor.spec.n_joints
        self._tick_action = np.zeros(self._n_act, dtype=np.float64)
        self._tick_source = SRC_PLANNER

        self.flt = None
        self.filter_active = bool(filter_active)
        if actor is not None and critics is not None:
            hs = float(getattr(cfg, "h_scale", 1.0))
            self.flt = SafetyFilter(
                actor, critics,
                self.extractor.extract()["joint_index"][None].cuda(),
                torch.from_numpy(
                    self.extractor.delta_theta_max.astype(np.float32)
                ).cuda(),
                margin=float(getattr(cfg, "filter_margin", 0.0)) * hs,
                hold_ticks=int(getattr(cfg, "hj_hold_ticks", 3)),
            )
        self.trace = {"t": [], "h": [], "V_t": [], "V": [], "intervened": []}
        self._tick_acc = 0.0
        self._orig_step = None
        self._last_ratio = 0.0
        self._block_info = None      # (tick-in-block, hold_ticks, engaged_s)
        self._engaged_ticks = 0

    # ---------------- a_nom assembly ----------------
    def _anom(self, ctx) -> torch.Tensor:
        """Nominal's commanded displacement for the upcoming tick, per arm:
        a_nom = pos[min(i_end, T-1)] - qpos_now — referenced to the LIVE qpos
        (not the plan's start), so it is the same physical quantity as the
        actor's dtheta. Arm-only rows; prismatic columns stay zero."""
        from main.envs.controller import _path_arrays

        i, spt = ctx["i"], self.ctrl.steps_per_tick
        qraw = self.extractor.raw_qpos().astype(np.float64)
        nj = self.extractor.spec.n_joints
        n_arm = nj - 2  # spec order: arm joints, then the 2 prismatic
        a = np.zeros(self._n_act, dtype=np.float64)
        for arm_i, key in ((0, "left_arm"), (1, "right_arm")):
            plan = ctx.get(key)
            if plan is None:
                continue
            pos, _ = _path_arrays(plan)
            i_end = min(i + spt - 1, len(pos) - 1)
            a[arm_i * nj: arm_i * nj + n_arm] = (
                pos[i_end] - qraw[arm_i * nj: arm_i * nj + n_arm]
            )
        return torch.from_numpy(a.astype(np.float32))[None].cuda()

    # ---------------- per-tick hook ----------------
    def _tick_hook(self, ctx) -> bool:
        """Called by the controller at every tick boundary. Returns True iff
        an intervention just drove the arm (controller then replans)."""
        self._tick_source = SRC_PLANNER
        self._tick_action = self._anom(ctx).cpu().numpy()[0].astype(np.float64)
        # panel ratio on EVERY tick: the nominal's demanded step as a
        # fraction of the actor's bound (interventions overwrite it with
        # the actor's own action ratio)
        dmax = np.asarray(self.extractor.delta_theta_max, dtype=np.float64)
        self._last_ratio = float(
            np.abs(self._tick_action).max() / max(dmax.max(), 1e-9)
        )
        if self.mode == "collect" and (self.flt is None or not self.filter_active):
            # nominal-only collection episode: no filter, perturbation only
            self._perturb_maybe()
            return False
        if self.flt is None or not self.filter_active:
            return False

        body, enc = self._policy_inputs_cached()
        a_nom_t = torch.from_numpy(self._tick_action.astype(np.float32))[None].cuda()
        with torch.no_grad():
            q_nom = float(self.flt.q_nom(
                enc, a_nom_t, body["Jlin"][None].cuda(), body["Jang"][None].cuda()
            ))
        t_now = self._n_physics / self.env.PHYSICS_FREQ
        self.trace["V_t"].append(t_now)
        self.trace["V"].append(q_nom)
        engaged = q_nom < self.flt.margin
        self.trace["intervened"].append(engaged)
        self.flt.track(engaged)
        if not engaged:
            self._block_info = None
            return False

        # intervention: actor drives hj_hold_ticks, re-queried every tick
        self._engaged_ticks = getattr(self, "_engaged_ticks", 0)
        for k in range(self.flt.hold_ticks):
            body, enc = self._policy_inputs_cached()
            with torch.no_grad():
                a = self.flt.actor_action(enc)
            a_np = a[0].cpu().numpy()
            self._tick_source = SRC_ACTOR
            self._tick_action = a_np.astype(np.float64)
            self._last_ratio = float(
                np.abs(a_np).max() / max(self.extractor.delta_theta_max.max(), 1e-9)
            )
            # panel block counter: tick-in-block + cumulative engaged time
            self._engaged_ticks += 1
            self._block_info = (
                k + 1, self.flt.hold_ticks,
                self._engaged_ticks * self.env.control_dt,
            )
            self.env.step_dtheta(a_np)
        return True

    def _perturb_maybe(self):
        """Collection coverage: uniform random dtheta for one tick. Rate
        matches the old per-window 5%: p_tick = perturb_prob * spt/60."""
        spt = self.ctrl.steps_per_tick
        if self.rng.rand() >= self.perturb_prob * (spt / 60.0):
            return
        ji = self.extractor.extract()["joint_index"].numpy().astype(np.int64)
        cols = ji[ji >= 0]
        dmax = np.asarray(self.extractor.delta_theta_max, dtype=np.float64)
        dtheta = np.zeros(self._n_act)
        dtheta[cols] = self.rng.uniform(-1, 1, len(cols)) * dmax[cols]
        self._tick_source = SRC_PERTURB
        self._tick_action = dtheta
        self.env.step_dtheta(dtheta)

    # ---------------- capture ----------------
    def _capture_tick(self):
        obs = self.env.get_encoder_obs(self.kin)
        h, diag = compute_h(self.env, self.cfg.table_margin, self.cfg.table_height)
        # h is trained/evaluated in h*h_scale units (config keeps metres
        # for physical margins; see FrozenConfig.h_scale)
        h = float(h) * float(getattr(self.cfg, "h_scale", 1.0))
        self._last_diag = diag
        t_now = self._n_physics / self.env.PHYSICS_FREQ
        self.trace["t"].append(t_now)
        self.trace["h"].append(float(h))
        # encode ONCE per tick: the buffer write, the hook's Q(s, a_nom) eval
        # (same sim state — the hook fires at this exact boundary), and the
        # actor all consume this encoding
        encoded = self._encode_obs(obs)
        if self.mode == "collect" and self.buf is not None:
            self._collect_tick(obs, h, diag, encoded)
        if self.flt is not None and self.filter_active:
            tokens, pos_mean, state, body = encoded
            with torch.no_grad():
                enc = self.policy_enc(
                    state, body["body_tokens"][None], body["link_pos"][None],
                    body["body_mask"][None], body["joint_index"][None],
                    tokens, pos_mean,
                )
            self._obs_cache = (self._n_physics, body, enc)
        else:
            self._obs_cache = None
        if self.video is not None and t_now >= self._next_frame_t - 1e-9:
            # sim-time-based frame capture: 10 fps of simulation
            self._next_frame_t = t_now + 1.0 / self.video_fps
            cam = self.env.task.cameras.observer_camera
            cam.take_picture()
            rgba = cam.get_picture("Color")
            frame = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
            v_now = self.trace["V"][-1] if self.trace["V"] else float("nan")
            engaged = bool(self.trace["intervened"][-1]) if self.trace["intervened"] else False
            from main.envs.controller import current_skill

            extras = {
                "d_left": diag.get("d_left"),
                "d_right": diag.get("d_right"),
                "d_left_held": diag.get("d_left_held"),
                "d_right_held": diag.get("d_right_held"),
                "true_argmin": diag.get("true_argmin", ""),
                "holding": diag.get("holding", ""),
                "holding_left": diag.get("holding_left", ""),
                "holding_right": diag.get("holding_right", ""),
                "hold_debug": diag.get("hold_debug", {}),
                "contact": diag.get("contact", False),
                "block": self._block_info,
                # cumulative actor intervention so far: (ticks, seconds)
                "intervened": (self._engaged_ticks,
                               self._engaged_ticks * self.env.control_dt),
                "skill": current_skill(self.env.task),
            }
            self.video.add(
                frame, t_now, len(self.trace["t"]), -1, float(h), v_now,
                "FILTER" if engaged else "NOMINAL", self._last_ratio,
                extras=extras,
            )
        return obs

    def _collect_tick(self, obs, h, diag, encoded):
        """Stash this state; the previous stashed entry is finalized with the
        COMMANDED action of the tick that led here (planner a_nom, actor
        dtheta, or sampled perturbation — action_source marks which) and
        written to the buffer.

        Cross-embodiment: variable-width fields are zero-padded to the
        buffer maxima (state tokens 16, action side 18). Jlin/Jang and
        dtheta_max are stored per step — a mixed-embodiment buffer cannot
        recompute them from qpos at sample time."""
        from main.train.buffer import MAX_ACTION, MAX_STATE_TOKENS

        # encoding comes from _capture_tick (computed once per tick)
        tokens, pos_mean, state, body = encoded
        state = state[0]                            # [N, D], drop batch
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

        dtheta_pad = np.zeros(MAX_ACTION, np.float32)
        dtheta_pad[: self._tick_action.shape[0]] = self._tick_action

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
            "task_id": np.int8(
                list(self.cfg.task_choices).index(self.cfg.task)
                if self.cfg.task in self.cfg.task_choices else -1
            ),
            "h": np.float32(h),
            "per_link_h": per_link,
            "d_arm_arm": np.float32(diag["d_arm_arm"]),
            "episode_id": np.int32(self.episode_id),
            "done": np.bool_(False),
        }
        if self._pending is not None:
            # the pending entry's action = the action COMMANDED during the
            # tick that produced this state (registered by the tick hook)
            self._pending["dtheta"] = dtheta_pad
            self._pending["action_source"] = np.int8(self._tick_source)
            # realized-vs-commanded: how much of the stored command the
            # drive actually achieved (low ratio = the critic's action
            # semantics drift from physics)
            cmd = dtheta_pad
            mask = (np.abs(cmd) > 0) & (self._pending["dtheta_max"] > 0)
            denom = float(np.linalg.norm(cmd[mask]))
            if denom > 1e-6:
                # aggregate as summed norms, not per-tick ratios — per-tick
                # ratios explode on near-zero commands and skew the mean
                achieved = qraw_pad - self._pending["qpos_raw"]
                self.trace["real_num"] = (
                    self.trace.get("real_num", 0.0)
                    + float(np.linalg.norm(achieved[mask]))
                )
                self.trace["real_den"] = self.trace.get("real_den", 0.0) + denom
            self.buf.append(self._pending)
        self._pending = entry

    def _flush_pending(self):
        if self._pending is not None and self.buf is not None:
            from main.train.buffer import MAX_ACTION

            self._pending["dtheta"] = np.zeros(MAX_ACTION, np.float32)
            self._pending["action_source"] = np.int8(SRC_PLANNER)
            self._pending["done"] = np.bool_(True)
            self.buf.append(self._pending)
            self._pending = None

    def _policy_inputs_cached(self):
        """Body features + policy encoding for the current state, reusing the
        capture's computation when it is at most one tick old. The boundary
        capture normally fires at the exact chunk end; after intervention
        blocks the plan cursor and the global step counter can sit a few
        physics steps apart, so the freshest capture may lag the hook by
        < one tick (<= 12 ms of sim — the state barely moves, and the Q
        trigger tolerates it; a_nom is always computed from LIVE qpos)."""
        cache = getattr(self, "_obs_cache", None)
        spt = self.ctrl.steps_per_tick
        if cache is not None and 0 <= self._n_physics - cache[0] <= spt:
            return cache[1], cache[2]
        return self._model_inputs(self.env.get_encoder_obs(self.kin))

    def _encode_obs(self, obs):
        """obs -> (tokens, pos_mean, state, body) — the frozen encoder plus
        analytic body features. Runs ONCE per tick; the buffer write and the
        policy encoding both consume this (see _capture_tick)."""
        with torch.no_grad():
            tokens, pos_mean, _ = self.encoder.encode_visual_tokens(
                obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"],
                out_extrinsic=obs["T_ego2world"],  # ego -> world metres
            )
            state = self.encoder.encode_joint_angles(
                obs["joint_state"], self.kin,
                embodiedment_mat=obs["T_base2ego"],
            ).squeeze(2)                          # [B, N, D]
            body = self.extractor.extract()
        return tokens, pos_mean, state, body

    def _model_inputs(self, obs):
        tokens, pos_mean, state, body = self._encode_obs(obs)
        with torch.no_grad():
            enc = self.policy_enc(
                state, body["body_tokens"][None], body["link_pos"][None],
                body["body_mask"][None], body["joint_index"][None],
                tokens, pos_mean,
            )
        return body, enc

    # ---------------- main loop ----------------
    def run(self, max_steps: int | None = None):
        if max_steps is not None:
            self.max_steps = int(max_steps)
        self._orig_step = self.env.task.scene.step
        env = self.env

        def hooked_step():
            self._orig_step()
            self._n_physics += 1
            if self._n_physics > self.max_steps:
                raise RolloutTimeout(
                    f"rollout exceeded {self.max_steps} physics steps"
                )
            self._tick_acc += env.control_freq / env.PHYSICS_FREQ
            # epsilon: 10x float(0.1) sums to 0.9999999999999999 — without the
            # epsilon the capture drifts one physics step late and the hook's
            # obs cache never hits
            if self._tick_acc >= 1.0 - 1e-9:
                self._tick_acc -= 1.0
                self._capture_tick()

        env.task.scene.step = hooked_step
        success = False
        try:
            if self.mode == "collect":
                self._capture_tick()  # initial state s_0
            env.task.play_once()
            success = bool(env.task.check_success())
        except RolloutTimeout as exc:
            print(f"[rollout] {exc}")
        except AssertionError as exc:
            if "buffer full" in str(exc):
                # buffer-full is a normal termination signal in watch mode —
                # but restore the step hook and release the env first
                env.task.scene.step = self._orig_step
                env.close()
                raise
            import traceback

            traceback.print_exc()
        except Exception:
            import traceback

            traceback.print_exc()
        env.task.scene.step = self._orig_step
        if self.mode == "collect":
            self._flush_pending()
        self.trace["success"] = success
        self.trace["n_physics"] = self._n_physics
        self.trace["mean_realized_ratio"] = (
            self.trace.get("real_num", 0.0) / self.trace["real_den"]
            if self.trace.get("real_den") else float("nan")
        )
        if self.flt is not None:
            self.trace["interventions"] = self.flt.n_interventions
            self.trace["intervention_rate"] = self.flt.intervention_rate
            self.trace["mode_switches"] = self.flt.n_switches
        env.close()
        return self.trace

