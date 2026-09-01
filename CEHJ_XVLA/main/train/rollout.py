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

Action space (X-VLA "ee6d", 18 dims): per arm [dxyz(3), drot6d(6)] — a
per-tick DELTA on the current absolute EE pose, bounded per-dim by
cfg.ee_step_max (0.05 m xyz; rot6d full unit-column range). The rot6d block
encodes the relative rotation R_delta = R_cur^T @ R_target (first two
columns, row-major: the X-VLA client layout [R00,R01,R10,R11,R20,R21]). No
gripper channel — the safety filter never drives the gripper.

Execution: the EE6D delta becomes a per-tick JOINT displacement via the
arm's geometric Jacobian at the live qpos (pytorch_kinematics SerialChain
per arm on the same dual-arm URDF as the a_nom FK): Δx = [dxyz, axis-angle
of R_delta] (6-dim EE twist, world frame), Δq_arm = J⁺ Δx with damped
least squares (λ = 0.05) near singularities, then env.step_dtheta — the
SAME per-tick velocity-feedforward stepping (10 physics substeps at 250 Hz)
the parent's joint-space actor uses, so actor ticks, planner ticks, and
perturbations all share the tick structure. The commanded delta is clipped
before the DLS solve: the translation target p_ee + dxyz is clipped into
the arm's workspace (reach sphere about the static base, z >= table_height
+ table_margin), and the rotation angle is capped at cfg.ee_rot_max
(0.5 rad/tick) — so argmax_a Q(s,a) can never command an unreachable or
physically absurd per-tick jump. NOTE: the measured-EE frame differs from
the FK link frame by the constant calibration offset T_off — the Jacobian
is computed at the FK link, so rotational deltas carry a small lever-arm
error (~|T_off translation|); acceptable for the baseline.

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
paths — plan chunks, actor step_dtheta ticks, perturbations — go through
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
from main.network.body_features import BodyTokenExtractor  # noqa: E402
from main.envs.distance import obstacle_contact  # noqa: E402


class RolloutTimeout(Exception):
    """Raised from the scene.step hook when max_steps physics steps pass."""

EMBODIMENT_IDS = {"piper": 0, "franka-panda": 1, "ARX-X5": 2,
                 "ur5-wsg": 3, "aloha-agilex": 4}

# action_source values stored in the buffer
SRC_PLANNER, SRC_ACTOR, SRC_PERTURB = 0, 1, 2

EE6D_ARM_DIM = 9   # dxyz(3) + drot6d(6) — the filter never
                      # drives the gripper (task script owns it)
N_ACTION = 18      # two arms, no gripper channels


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


def mat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> axis-angle vector (axis * angle, radians)."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-4:
        # near-180 deg: axis from the symmetric part of R + I
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        axis *= np.sign(np.array([A[2, 1] - A[1, 2], A[0, 2] - A[2, 0],
                                  A[1, 0] - A[0, 1]]) + 1e-12)
        axis /= max(np.linalg.norm(axis), 1e-9)
        return axis * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


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

        # per-arm geometric Jacobians for EE6D intervention execution:
        # SerialChain from the shared base to each end link (verified:
        # ndof == arm joints on all four URDF embodiments). aloha's
        # DualArmKinematics chain is checked the same way.
        import pytorch_kinematics as pk

        if hasattr(kin, "arm_link_keys"):
            ee_names = [k[-1] for k in kin.arm_link_keys]
        else:
            ee_names = [kin.left_arm_link_keys[-1], kin.right_arm_link_keys[-1]]
        self._serial = []
        q_left, q_right = self._arm_qpos()
        self._n_arm = (len(q_left), len(q_right))
        assert self._n_arm[0] == self._n_arm[1], \
            f"asymmetric arms unsupported: {self._n_arm}"
        for arm_i, ee_name in enumerate(ee_names):
            sc = pk.SerialChain(kin.chain, ee_name)
            n_dof = len(sc.get_joint_parameter_names())
            assert n_dof == self._n_arm[arm_i], (
                f"serial chain to {ee_name} has {n_dof} DOF, expected "
                f"{self._n_arm[arm_i]} (mobile base / extra joints in the "
                f"chain are unsupported)"
            )
            self._serial.append(sc)
        self.dls_lambda = 0.05    # damped-least-squares lambda for J+
        # sim-consistent Jacobians (the parent's BodyTokenExtractor path:
        # joint axes from FK composed with the live base pose, link points
        # from the sim — verified against SAPIEN there)
        self._extractor = BodyTokenExtractor(self.env)
        # workspace clip geometry: STATIC per-arm base positions (all four
        # embodiments are fixed-base) + the URDF-sampled reach, used to clip
        # the actor's EE translation target into the reachable workspace
        # before the DLS solve (_execute_ee6d)
        self._ws_base = []
        for spec, entity in zip(
            self._extractor.specs,
            (self.env.robot.left_entity, self.env.robot.right_entity),
        ):
            links = {l.get_name(): l for l in entity.get_links()}
            self._ws_base.append(
                np.asarray(links[spec.base_link].get_pose().p,
                           dtype=np.float64)
            )
        self._ws_reach = [s.l_reach for s in self._extractor.specs]

        self.flt = None
        self.filter_active = bool(filter_active)
        if actor is not None and critics is not None:
            hs = float(getattr(cfg, "h_scale", 1.0))
            self.flt = SafetyFilter(
                actor, critics, self._step_max_t,
                margin=float(getattr(cfg, "filter_margin", 0.0)) * hs,
                release_margin=float(getattr(cfg, "filter_release_margin", 0.01)) * hs,
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
        """X-VLA EE6D proprio [20]: per arm [xyz, rot6d(wxyz quat), 1-2*g].
        (input encoding — includes the gripper; the ACTION does not)"""
        out = np.zeros(20, dtype=np.float64)  # X-VLA proprio layout
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
        (FK of pos[min(i_end, T-1)]). No gripper channel — the filter never
        drives it."""
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
            a[arm_i * 9: arm_i * 9 + 3] = T_tgt[:3, 3] - T_cur[:3, 3]
            a[arm_i * 9 + 3: arm_i * 9 + 9] = mat_to_rot6d(
                T_cur[:3, :3].T @ T_tgt[:3, :3]
            )
        return a

    # ---------------- EE6D execution ----------------
    @staticmethod
    def _skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def _arm_jacobian(self, arm_i: int, q_arm: np.ndarray) -> np.ndarray:
        """Jacobian [6, n_arm] of the arm's EE (the measured endpose point) in
        world frame, from the verified BodyTokenExtractor path: per-link
        Jacobians are sim-consistent (FK axes composed with the live base
        pose, link points from SAPIEN); evaluate at the EE point via
        J_ee = J_link - skew(p_ee - p_link) @ J_ang."""
        body = self._extractor.extract()
        Jlin = body["Jlin"].numpy().astype(np.float64)   # [tok, 3, 2n]
        Jang = body["Jang"].numpy().astype(np.float64)
        lp = body["link_pos"].numpy().astype(np.float64)
        spec = self._extractor.specs[arm_i]
        n_link = len(spec.arm_links) + (1 if spec.gripper_links else 0)
        n_joint = spec.n_joints
        ee_tok = arm_i * n_link + len(spec.arm_links) - 1   # last arm link
        col = arm_i * n_joint
        Jl = Jlin[ee_tok][:, col:col + n_joint]
        Ja = Jang[ee_tok][:, col:col + n_joint]
        p_ee = np.asarray(self._ee_state()["ee_poses"][arm_i][:3],
                          dtype=np.float64)
        Jl_ee = Jl - self._skew(p_ee - lp[ee_tok]) @ Ja
        return np.vstack([Jl_ee, Ja])

    def _clip_dxyz(self, arm_i: int, p_ee: np.ndarray, dxyz: np.ndarray) -> np.ndarray:
        """Clip the EE translation so the TARGET p_ee + dxyz stays in the
        workspace: inside the arm's reach sphere about its (static) base and
        above the table. Returns the clipped delta."""
        p_tgt = p_ee + dxyz
        v = p_tgt - self._ws_base[arm_i]
        r = float(np.linalg.norm(v))
        r_max = self._ws_reach[arm_i]
        if r > r_max:
            p_tgt = self._ws_base[arm_i] + v * (r_max / r)
        z_min = self.cfg.table_height + self.cfg.table_margin
        if p_tgt[2] < z_min:
            p_tgt[2] = z_min
        return p_tgt - p_ee

    def _aa_clamped(self, v6: np.ndarray) -> np.ndarray:
        """rot6d block -> axis-angle of R_delta, angle clamped to
        cfg.ee_rot_max (rad/tick). Gram-Schmidt in rot6d_to_mat makes this
        well-defined for ANY raw 6-vector the tanh actor can emit."""
        aa = mat_to_axis_angle(rot6d_to_mat(v6))
        n = float(np.linalg.norm(aa))
        r_max = float(self.cfg.ee_rot_max)
        if n > r_max:
            aa = aa * (r_max / n)
        return aa

    def _execute_ee6d(self, action: np.ndarray) -> None:
        """EE6D delta -> per-tick joint displacement -> env.step_dtheta.

        Per arm: dx = [dxyz, axis-angle(R_delta)] (world frame), dq = J+ dx
        with damped least squares (lambda = dls_lambda) so near-singular J
        stays bounded. The commanded delta is clipped before the solve:
        translation so the EE target stays in the reachable workspace
        (_clip_dxyz), rotation angle to cfg.ee_rot_max (_aa_clamped). No
        gripper channel — the filter never drives the gripper; it holds its
        current opening (task script owns timing). step_dtheta runs the same
        10-substep velocity-feedforward tick the parent's joint-space actor
        uses."""
        q_left, q_right = self._arm_qpos()
        dq = np.zeros(self._n_arm[0] + self._n_arm[1], dtype=np.float64)
        ee = self._ee_state()["ee_poses"]
        for arm_i, q_arm in enumerate((q_left, q_right)):
            d = action[arm_i * 9: (arm_i + 1) * 9]
            p_ee = np.asarray(ee[arm_i][:3], dtype=np.float64)
            dx = np.concatenate([self._clip_dxyz(arm_i, p_ee, d[:3]),
                                 self._aa_clamped(d[3:9])])
            J = self._arm_jacobian(arm_i, q_arm)
            n = self._n_arm[arm_i]
            lam2 = self.dls_lambda ** 2
            dq_arm = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), dx)
            dq[arm_i * n: (arm_i + 1) * n] = dq_arm[:n]  # arm joints only
            # (the extractor's J has prismatic gripper columns after them)
        self.env.step_dtheta(dq)

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
        if self.flt._was_engaged:
            engaged = q_nom < self.flt.release_margin
        else:
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
            # cache the policy encoding (enc namespace), not the raw tuple
            self._obs_cache = (self._n_physics, encoded[2])
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

            frame = draw_debug_bboxes(self.env, frame, include_links=False)
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
        feats, proprio, _arm_tokens = encoded
        entry = {
            "scene_tokens": feats[0].cpu().half().numpy(),    # [200, 1024]
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
            # 18-dim action: arm blocks are 9 wide (dxyz + drot6d)
            cmd = np.concatenate(
                [self._pending["action"][0:3], self._pending["action"][9:12]]
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
        return self._encode_obs(self.env.get_xvla_obs())[2]  # enc namespace

    def _encode_obs(self, obs):
        """obs -> (scene_raw [1,200,1024], proprio np [20], enc namespace)
        — the frozen X-VLA VLM features, the raw proprio, and the policy
        encoding (learned adapter + proprio_enc applied). Runs ONCE per
        tick; the buffer stores the RAW features/proprio (adapter and
        proprio_enc train at train time), the filter consumes the adapted
        tokens (see _capture_tick)."""
        with torch.no_grad():
            feats = self.encoder.encode_scene_features(obs["images"])
            scene = self.encoder.adapter(feats)               # [1, 200, 256]
            proprio = self._proprio(obs)
            arm_tokens = self.encoder.encode_proprio(
                torch.from_numpy(proprio.astype(np.float32))[None]
            )
        enc = SimpleNamespace(
            arm_tokens=arm_tokens, scene=scene, scene_mask=None
        )
        return feats, proprio, enc

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
