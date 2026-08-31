"""Rollout driver on TickChunkedController: vanilla nominal + per-tick hooks.

X-VLA EE6D variant of the joint-space HJ-SAC rollout. The task's play_once
owns the stage machine. TickChunkedController walks each cuRobo plan in
fixed tick chunks (exactly PHYSICS_FREQ/control_freq physics steps — 10 at
25 Hz) and calls the per-tick hook at every boundary:

  hook False -> play the next tick of plan rows (open-loop within the Action)
  hook True  -> an intervention just drove the arms for hj_hold_ticks ticks
                (actor re-queried every tick); the controller discards the
                stale plan and replans the same target from live qpos.

Filter trigger: Q(s, a_nom) < margin — the critic scores the NOMINAL's EE6D
action for the upcoming tick. a_nom is the EE6D delta from the LIVE measured
EE pose to the plan-row EE target (FK of pos[min(i_end, T-1)] per arm), the
EE6D analogue of the parent's a_nom = pos[i_end] - qpos_now.

Action space (X-VLA "ee6d", 20 dims): per arm [dxyz(3), drot6d(6), dgrip(1)]
— a per-tick DELTA on the current absolute EE pose, bounded per-dim by
cfg.ee_step_max. The rot6d block encodes the relative rotation
R_delta = R_cur^T @ R_target (first two columns, row-major: the X-VLA
client layout [R00,R01,R10,R11,R20,R21]). The gripper channel lives in
X-VLA's 1-2g space (+1 = closed, -1 = open).

Execution: delta -> absolute target (R_new = R_delta @ R_cur,
p_new = p_cur + dxyz, grip thresholded X-VLA-style: 1-2*(g > 0.7)) ->
task.take_action([left 8, right 8], action_type='ee'). Verified against
RoboTwin envs/_base_task.py take_action: for action_type='ee' the arm block
is 7 wide [x,y,z, qw,qx,qy,qz] (planner convention, t3d wxyz), gripper at
index 7 / 15, clipped to [0,1] in Robot.set_gripper. NOTE: one take_action
call plans a short cuRobo path and plays it out internally (its own
scene.step loop), so an intervention tick spans as many physics steps as
the path needs; per-tick capture still fires through the scene.step hook
and intermediate ticks inherit the same commanded action.

NOTE on quat conventions: X-VLA's own client.py round-trips through a
scipy xyzw <-> t3d wxyz permutation that cancels between proprio encoding
and execution. We use the TRUE rotation everywhere (proprio rot6d is the
actual EE rotation; the executed quat is the true wxyz of the target) —
the frozen VLM never sees proprio (only our learned adapter does), so
there is no parity constraint with X-VLA's permuted convention.

Modes:
  filtered — deployment stack: filter active every tick.
  collect  — dataset writes to a StepBuffer. If models are given AND
             filter_active, the filter runs (DAgger rounds); otherwise the
             nominal runs untouched and only commanded actions are recorded.

Capture: a scene.step hook records h/obs at every control tick (all stepping
paths — plan chunks, actor take_action ticks, perturbations — go through
scene.step). The buffer's action is the COMMANDED EE6D delta per tick
(planner a_nom, actor action, or the sampled perturbation), with
action_source marking which.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from main.train.collect import make_training_env  # noqa: E402
from main.train.config import instruction_for  # noqa: E402
from main.train.hfunc import compute_h  # noqa: E402
from main.envs.distance import obstacle_contact  # noqa: E402


class RolloutTimeout(Exception):
    """Raised from the scene.step hook when max_steps physics steps pass."""

EMBODIMENT_IDS = {"piper": 0, "franka-panda": 1, "ARX-X5": 2,
                 "ur5-wsg": 3, "aloha-agilex": 4}

# action_source values stored in the buffer
SRC_PLANNER, SRC_ACTOR, SRC_PERTURB = 0, 1, 2

EE6D_ARM_DIM = 10  # dxyz(3) + drot6d(6) + dgripper(1)
N_ACTION = 20      # two arms


# ---------------- rotation helpers (true rotations, wxyz quats) ----------------
def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion (SAPIEN/RoboTwin t3d convention) -> 3x3 rotation."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> wxyz quaternion (Shepperd's method)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, \
            (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, \
            (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, \
            0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, \
            (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> 6d (first two columns; X-VLA layout: row-major of
    R[:, :2] = [R00,R01,R10,R11,R20,R21])."""
    return R[:, :2].astype(np.float64).reshape(6)


def rot6d_to_mat(v6: np.ndarray) -> np.ndarray:
    """6d -> 3x3 rotation via Gram-Schmidt (X-VLA rotate6D_to_quat layout)."""
    a1 = v6[[0, 2, 4]]
    a2 = v6[[1, 3, 5]]
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _ee_link_indices(kin) -> tuple[int, int]:
    """Flat FK-matrix indices of the two end links (last arm link per arm).

    MultiArmKinematics keys: arm0 links, arm0 fingers, arm1 links, arm1
    fingers. DualArmKinematics keys: left arm, left fingers, right arm,
    right fingers — the same interleaving.
    """
    if hasattr(kin, "arm_link_keys"):
        n_links = [len(k) for k in kin.arm_link_keys]
        n_fingers = [len(k) for k in kin.finger_keys]
    else:
        n_links = [len(kin.left_arm_link_keys), len(kin.right_arm_link_keys)]
        n_fingers = [len(kin.left_finger_keys), len(kin.right_finger_keys)]
    i_left = n_links[0] - 1
    i_right = n_links[0] + n_fingers[0] + n_links[1] - 1
    return i_left, i_right


def _pose_to_mat(pose7: np.ndarray) -> np.ndarray:
    """[x,y,z, qw,qx,qy,qz] -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_mat(np.asarray(pose7[3:], dtype=np.float64))
    T[:3, 3] = np.asarray(pose7[:3], dtype=np.float64)
    return T


class RolloutController:
    """TickChunkedController + per-tick capture + tick-level EE6D filter."""

    def __init__(self, cfg, seed, mode="filtered", encoder=None, kin=None,
                 actor=None, critics=None,
                 filter_active: bool = True,
                 buf=None, episode_id: int = 0, perturb_prob: float = 0.0,
                 rng=None, video=None, max_steps: int = 4500):
        from main.envs.controller import TickChunkedController
        from main.train.filter import SafetyFilter

        self.cfg = cfg
        self.env = make_training_env(cfg, seed)
        self.encoder = encoder
        if self.encoder is not None:
            self.encoder.set_instruction(instruction_for(cfg.task))
        self.kin = kin
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
        # per-dim EE6D action bound, both arms: [10] per arm -> [20]
        self.step_max = np.tile(
            np.asarray(cfg.ee_step_max, dtype=np.float64), 2
        )
        assert self.step_max.shape == (N_ACTION,)
        self._step_max_t = torch.from_numpy(
            self.step_max.astype(np.float32)
        )[None].cuda()
        # commanded action + source for the CURRENT tick (read by
        # _collect_tick when finalizing the pending entry)
        self._tick_action = np.zeros(N_ACTION, dtype=np.float64)
        self._tick_source = SRC_PLANNER

        # FK -> measured-EE-pose calibration: T_meas = T_fk @ T_off with a
        # CONSTANT T_off per arm (RoboTwin's endpose transform is a fixed
        # offset from the URDF end link). Computed once from the live state,
        # same spirit as env.compute_T_base2world.
        self._ee_idx = _ee_link_indices(kin)
        self._T_base2world = self.env.compute_T_base2world(kin)
        self._T_off = self._calibrate_ee()

        self.flt = None
        self.filter_active = bool(filter_active)
        if actor is not None and critics is not None:
            hs = float(getattr(cfg, "h_scale", 1.0))
            self.flt = SafetyFilter(
                actor, critics, self._step_max_t,
                margin=float(getattr(cfg, "filter_margin", 0.0)) * hs,
                hold_ticks=int(getattr(cfg, "hj_hold_ticks", 3)),
            )
        self.trace = {"t": [], "h": [], "V_t": [], "V": [], "intervened": [],
                      "contact_force": [], "contact_touch": []}
        self._tick_acc = 0.0
        self._last_force_touch = (0.0, False)
        self._orig_step = None
        self._last_ratio = 0.0
        self._block_info = None      # (tick-in-block, hold_ticks, engaged_s)
        self._engaged_ticks = 0
        self._obs_cache = None

    # ---------------- EE FK (plan-row -> measured-EE-convention pose) ----------------
    def _kin_input(self, q_left: np.ndarray, q_right: np.ndarray) -> torch.Tensor:
        """Assemble the kinematics input layout [Larm, gripL, Rarm, gripR]
        (gripper slots are dummies — arm-link FK does not depend on them)."""
        js = np.concatenate([q_left, [0.0], q_right, [0.0]])
        return torch.tensor(js[None], dtype=torch.float32)

    def _fk_ee_world(self, q_left: np.ndarray, q_right: np.ndarray):
        """FK both arms -> measured-EE-convention 4x4 poses (via _T_off)."""
        mats = self.kin.joint_state_to_robot_state(
            self._kin_input(q_left, q_right), return_matrix=True
        )
        mats = mats[0].double().numpy()                # [n_keys, 4, 4]
        out = []
        for arm_i, idx in enumerate(self._ee_idx):
            T = self._T_base2world @ mats[idx] @ self._T_off[arm_i]
            out.append(T)
        return out

    def _calibrate_ee(self):
        """Per-arm constant offset between FK end-link pose and RoboTwin's
        get_ee_pose (endpose convention), from the live settled state."""
        robot = self.env.robot
        q_left = np.asarray(robot.get_left_arm_real_jointState()[:-1],
                            dtype=np.float64)
        q_right = np.asarray(robot.get_right_arm_real_jointState()[:-1],
                             dtype=np.float64)
        mats = self.kin.joint_state_to_robot_state(
            self._kin_input(q_left, q_right), return_matrix=True
        )[0].double().numpy()
        offs = []
        for arm_i, idx in enumerate(self._ee_idx):
            T_fk = self._T_base2world @ mats[idx]
            meas = robot.get_left_ee_pose() if arm_i == 0 \
                else robot.get_right_ee_pose()
            T_meas = _pose_to_mat(np.asarray(meas, dtype=np.float64))
            offs.append(np.linalg.inv(T_fk) @ T_meas)
        return offs

    def _arm_qpos(self):
        robot = self.env.robot
        q_left = np.asarray(robot.get_left_arm_real_jointState()[:-1],
                            dtype=np.float64)
        q_right = np.asarray(robot.get_right_arm_real_jointState()[:-1],
                             dtype=np.float64)
        return q_left, q_right

    def _ee_state(self) -> dict:
        """LIVE EE poses + gripper openings, without the camera renders
        (get_xvla_obs renders 3 cameras; a_nom / EE6D execution need only
        the sim state). Same dict keys minus the images."""
        robot = self.env.robot
        grips = robot.get_normal_real_gripper_val()
        return {
            "ee_poses": (
                np.asarray(robot.get_left_ee_pose(), dtype=np.float64),
                np.asarray(robot.get_right_ee_pose(), dtype=np.float64),
            ),
            "grippers": (float(grips[0]), float(grips[1])),
        }

    # ---------------- proprio / a_nom ----------------
    def _proprio(self, obs) -> np.ndarray:
        """X-VLA EE6D proprio [20]: per arm [xyz, rot6d(wxyz quat), 1-2*g]."""
        out = np.zeros(N_ACTION, dtype=np.float64)
        for arm_i in range(2):
            ee = obs["ee_poses"][arm_i]
            out[arm_i * 10: arm_i * 10 + 3] = ee[:3]
            out[arm_i * 10 + 3: arm_i * 10 + 9] = mat_to_rot6d(
                quat_wxyz_to_mat(ee[3:])
            )
            out[arm_i * 10 + 9] = 1.0 - 2.0 * obs["grippers"][arm_i]
        return out

    def _anom(self, ctx) -> np.ndarray:
        """Nominal's commanded EE6D delta for the upcoming tick, per arm:
        from the LIVE measured EE pose to the plan-row EE target
        (FK of pos[min(i_end, T-1)]). Gripper delta from the plan's gripper
        rows at the same cursor, in the 1-2g channel."""
        from main.envs.controller import _path_arrays

        i, spt = ctx["i"], self.ctrl.steps_per_tick
        obs = self._ee_state()
        a = np.zeros(N_ACTION, dtype=np.float64)
        q_left, q_right = self._arm_qpos()
        for arm_i, key in ((0, "left_arm"), (1, "right_arm")):
            plan = ctx.get(key)
            if plan is None:
                continue
            pos, _ = _path_arrays(plan)
            i_end = min(i + spt - 1, len(pos) - 1)
            if arm_i == 0:
                T_tgt = self._fk_ee_world(pos[i_end], q_right)[0]
            else:
                T_tgt = self._fk_ee_world(q_left, pos[i_end])[1]
            T_cur = _pose_to_mat(obs["ee_poses"][arm_i])
            a[arm_i * 10: arm_i * 10 + 3] = T_tgt[:3, 3] - T_cur[:3, 3]
            a[arm_i * 10 + 3: arm_i * 10 + 9] = mat_to_rot6d(
                T_cur[:3, :3].T @ T_tgt[:3, :3]
            )
            g = ctx.get("left_g" if arm_i == 0 else "right_g")
            if g is not None:
                gi = min(ctx["g_i"] + spt - 1, int(g["num_step"]) - 1)
                # 1-2g channel: delta = (1-2*g_tgt) - (1-2*g_now)
                a[arm_i * 10 + 9] = 2.0 * (
                    obs["grippers"][arm_i] - float(g["result"][gi])
                )
        return a

    # ---------------- EE6D execution ----------------
    def _execute_ee6d(self, action: np.ndarray) -> None:
        """EE6D delta -> absolute target -> task.take_action(action_type='ee').

        take_action layout (verified at RoboTwin envs/_base_task.py
        take_action, action_type='ee'): 16 dims =
        [left xyz(3) + quat(4), left_grip, right xyz(3) + quat(4),
        right_grip]; the quat goes straight into the planner (t3d wxyz);
        gripper values are clipped to [0,1] in Robot.set_gripper. The
        gripper command follows X-VLA's client: 1 - 2*(g > 0.7) on the
        absolute 1-2g target.
        """
        obs = self._ee_state()
        out = np.zeros(16, dtype=np.float64)
        for arm_i in range(2):
            d = action[arm_i * 10: (arm_i + 1) * 10]
            ee = obs["ee_poses"][arm_i]
            R_new = rot6d_to_mat(d[3:9]) @ quat_wxyz_to_mat(ee[3:])
            p_new = ee[:3] + d[:3]
            q_new = mat_to_quat_wxyz(R_new)
            g_cur = 1.0 - 2.0 * obs["grippers"][arm_i]
            g_cmd = 1.0 - 2.0 * (g_cur + d[9] > 0.7)
            out[arm_i * 8: arm_i * 8 + 3] = p_new
            out[arm_i * 8 + 3: arm_i * 8 + 7] = q_new
            out[arm_i * 8 + 7] = g_cmd
        task = self.env.task
        # take_action sets eval_success=True (and then no-ops forever) if the
        # success check trips inside an intervention — keep the actor live
        # for the rest of the episode; run() does the final check itself
        prev_success = getattr(task, "eval_success", False)
        task.take_action(out, action_type="ee")
        task.eval_success = prev_success

    # ---------------- per-tick hook ----------------
    def _tick_hook(self, ctx) -> bool:
        """Called by the controller at every tick boundary. Returns True iff
        an intervention just drove the arms (controller then replans)."""
        self._tick_source = SRC_PLANNER
        self._tick_action = self._anom(ctx)
        # panel ratio on EVERY tick: the nominal's demanded step as a
        # fraction of the actor's bound (interventions overwrite it with
        # the actor's own action ratio)
        self._last_ratio = float(
            np.abs(self._tick_action / self.step_max).max()
        )
        if self.mode == "collect" and (self.flt is None or not self.filter_active):
            # nominal-only collection episode: no filter, perturbation only
            self._perturb_maybe()
            return False
        if self.flt is None or not self.filter_active:
            return False

        enc = self._policy_inputs_cached()
        a_nom_t = torch.from_numpy(
            self._tick_action.astype(np.float32)
        )[None].cuda()
        with torch.no_grad():
            q_nom = float(self.flt.q_nom(enc, a_nom_t))
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
        for k in range(self.flt.hold_ticks):
            enc = self._policy_inputs_cached()
            with torch.no_grad():
                a = self.flt.actor_action(enc)
            a_np = a[0].cpu().numpy()
            self._tick_source = SRC_ACTOR
            self._tick_action = a_np.astype(np.float64)
            self._last_ratio = float(
                np.abs(a_np / self.step_max).max()
            )
            # panel block counter: tick-in-block + cumulative engaged time
            self._engaged_ticks += 1
            self._block_info = (
                k + 1, self.flt.hold_ticks,
                self._engaged_ticks * self.env.control_dt,
            )
            self._execute_ee6d(a_np)
        return True

    def _perturb_maybe(self):
        """Collection coverage: uniform random EE6D delta for one tick. Rate
        matches the old per-window 5%: p_tick = perturb_prob * spt/60."""
        spt = self.ctrl.steps_per_tick
        if self.rng.rand() >= self.perturb_prob * (spt / 60.0):
            return
        action = self.rng.uniform(-1, 1, N_ACTION) * self.step_max
        self._tick_source = SRC_PERTURB
        self._tick_action = action
        self._execute_ee6d(action)

    # ---------------- capture ----------------
    def _capture_tick(self):
        obs = self.env.get_xvla_obs()
        h, diag = compute_h(self.env, self.cfg.table_margin, self.cfg.table_height)
        # h is trained/evaluated in h*h_scale units (config keeps metres
        # for physical margins; see FrozenConfig.h_scale)
        h = float(h) * float(getattr(self.cfg, "h_scale", 1.0))
        self._last_diag = diag
        t_now = self._n_physics / self.env.PHYSICS_FREQ
        self.trace["t"].append(t_now)
        self.trace["h"].append(float(h))
        force, touch = obstacle_contact(self.env)
        self.trace["contact_force"].append(float(force))
        self.trace["contact_touch"].append(bool(touch))
        self._last_force_touch = (force, touch)
        # encode ONCE per tick: the buffer write, the hook's Q(s, a_nom) eval
        # (same sim state — the hook fires at this exact boundary), and the
        # actor all consume this encoding
        encoded = self._encode_obs(obs)
        if self.mode == "collect" and self.buf is not None:
            self._collect_tick(obs, h, encoded)
        if self.flt is not None and self.filter_active:
            self._obs_cache = (self._n_physics, encoded)
        else:
            self._obs_cache = None
        if self.video is not None and t_now >= self._next_frame_t - 1e-9:
            # sim-time-based frame capture: every control tick
            self._next_frame_t = t_now + 1.0 / self.video_fps
            cam = self.env.task.cameras.observer_camera
            cam.take_picture()
            rgba = cam.get_picture("Color")
            frame = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
            # debug bbox overlay (obstacle + held objects) — the videos are
            # for debugging; the boxes make filter behavior legible
            from main.envs.record import draw_debug_bboxes

            frame = draw_debug_bboxes(self.env, frame)
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
                "contact_force": self._last_force_touch[0],
                "contact_touch": self._last_force_touch[1],
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

    def _collect_tick(self, obs, h, encoded):
        """Stash this state; the previous stashed entry is finalized with the
        COMMANDED EE6D action of the tick that led here (planner a_nom,
        actor action, or sampled perturbation — action_source marks which)
        and written to the buffer."""
        tokens, proprio, _arm_tokens = encoded
        entry = {
            "scene_tokens": tokens[0].cpu().half().numpy(),   # [200, 256]
            "proprio": proprio.astype(np.float32),            # [20]
            "embodiment_id": np.int8(EMBODIMENT_IDS[self.cfg.embodiment]),
            "task_id": np.int8(
                list(self.cfg.task_choices).index(self.cfg.task)
                if self.cfg.task in self.cfg.task_choices else -1
            ),
            "h": np.float32(h),
            "episode_id": np.int32(self.episode_id),
            "done": np.bool_(False),
        }
        if self._pending is not None:
            # the pending entry's action = the action COMMANDED during the
            # tick that produced this state (registered by the tick hook)
            self._pending["action"] = self._tick_action.astype(np.float32)
            self._pending["action_source"] = np.int8(self._tick_source)
            # realized-vs-commanded (xyz block only): how much of the
            # commanded EE translation the drive actually achieved
            cmd = np.concatenate(
                [self._pending["action"][0:3], self._pending["action"][10:13]]
            )
            denom = float(np.linalg.norm(cmd))
            if denom > 1e-6:
                achieved = np.concatenate(
                    [proprio[0:3] - self._pending["proprio"][0:3],
                     proprio[10:13] - self._pending["proprio"][10:13]]
                )
                self.trace["real_num"] = (
                    self.trace.get("real_num", 0.0)
                    + float(np.linalg.norm(achieved))
                )
                self.trace["real_den"] = self.trace.get("real_den", 0.0) + denom
            self.buf.append(self._pending)
        self._pending = entry

    def _flush_pending(self):
        if self._pending is not None and self.buf is not None:
            self._pending["action"] = np.zeros(N_ACTION, np.float32)
            self._pending["action_source"] = np.int8(SRC_PLANNER)
            self._pending["done"] = np.bool_(True)
            self.buf.append(self._pending)
            self._pending = None

    def _policy_inputs_cached(self):
        """Policy encoding for the current state, reusing the capture's
        computation when it is at most one tick old. The boundary capture
        normally fires at the exact chunk end; after intervention blocks the
        plan cursor and the global step counter can sit a few physics steps
        apart, so the freshest capture may lag the hook by < one tick
        (<= 12 ms of sim — the state barely moves, and the Q trigger
        tolerates it; a_nom is always computed from the LIVE EE pose)."""
        cache = self._obs_cache
        spt = self.ctrl.steps_per_tick
        if cache is not None and 0 <= self._n_physics - cache[0] <= spt:
            return cache[1]
        return self._encode_obs(self.env.get_xvla_obs())

    def _encode_obs(self, obs):
        """obs -> (scene_tokens [1,200,D], proprio np [20], enc namespace)
        — the frozen X-VLA encoder plus the learned proprio adapter. Runs
        ONCE per tick; the buffer write and the policy encoding both consume
        this (see _capture_tick)."""
        with torch.no_grad():
            tokens, _pos = self.encoder.encode_scene(obs["images"])
            proprio = self._proprio(obs)
            arm_tokens = self.encoder.encode_proprio(
                torch.from_numpy(proprio.astype(np.float32))[None]
            )
        enc = SimpleNamespace(
            arm_tokens=arm_tokens, scene=tokens, scene_mask=None
        )
        return tokens, proprio, enc

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
