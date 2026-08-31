"""Body-token features from RoboTwin.

One token per link, with the two gripper finger links merged into a single
gripper token (mean of both fingers — HoloBrain's own FK convention). Token
layout per arm is [arm links..., gripper], giving HoloBrain's 14-entry
layout [L_arm(6), L_gripper, R_arm(6), R_gripper], so body token i pairs
1:1 with HoloBrain robot-state token i. 16 columns per token:

    cols 0:3   (link_pos - base_pos) / L_reach      world frame, SAPIEN poses
    cols 3:7   link quaternion (wxyz, SAPIEN convention)
    cols 7:10  linear  velocity (Jlin @ qvel) / 0.5
    cols 10:13 angular velocity (Jang @ qvel) / pi
    col  13    chain depth (tree depth; base link itself is not a token,
               so tokens span (0, 1], with 1 at the chain tip)
    col  14    arm id (0 left / 1 right)
    col  15    gripper opening, normalized; 0 on non-gripper tokens

Also returned (outside the token block): link_pos (world, metres),
block-diagonal Jlin/Jang [n_tokens, 3, 2*n_joint] (left arm joints in
columns 0..n-1, right arm in n..2n-1), joint_index (action column per
token; -1 for the merged gripper token and padding — the gripper's
orientation is taken from its wrist parent link, not averaged over the
mirrored fingers), body_mask, is_actuated (== joint_index >= 0), and
T_cam_world of the fixed head camera.

delta_theta_max is a per-control-step displacement limit: URDF
<limit velocity> * control_dt * kappa. dt and kappa are problem-definition
parameters (constructor args), not URDF facts. Joints with a missing/zero/
implausible velocity attribute fall back to DEFAULT_JOINT_VEL and are logged.

Per-embodiment URDF facts (chain order, limits, L_reach) are parsed once
and cached by file content hash. One spec is built per arm, so mirrored
left/right URDFs are handled correctly.

Constants 0.5 and pi are global (identical for every embodiment); L_reach is
per-embodiment on purpose. Features are extracted in the world frame; the
camera extrinsic is handed back separately (no pre-transform).
"""

from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET

import numpy as np
import torch

from pathlib import Path

logger = logging.getLogger(__name__)

LIN_VEL_SCALE = 0.5          # global constant, NOT per-embodiment
ANG_VEL_SCALE = np.pi        # global constant, NOT per-embodiment
TOKEN_DIM = 16
DEFAULT_JOINT_VEL = 2.0      # rad/s (or m/s prismatic) fallback <limit velocity>
DEFAULT_KAPPA = 0.25         # fraction of the velocity limit usable per step

_SPEC_CACHE: dict = {}


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _parse_urdf_joints(urdf_path: str) -> tuple[list[dict], str]:
    """Actuated joints (revolute/prismatic) with limits, in tree order.

    Handles branching chains (e.g. two gripper fingers). Joints are ordered
    depth-first from the base link, revolute children before prismatic ones —
    for the supported arms (grippers branching off the chain tip) this puts
    arm links first and gripper links last, but an embodiment with a
    prismatic joint near the base would order differently; downstream code
    must use link_names / arm_links / gripper_links rather than assuming it.
    """
    root = ET.parse(urdf_path).getroot()
    joints = []
    for j in root.iter("joint"):
        if j.get("type") not in ("revolute", "prismatic"):
            continue
        limit = j.find("limit")
        axis = j.find("axis")
        vel = None
        if limit is not None and limit.get("velocity") is not None:
            try:
                vel = float(limit.get("velocity"))
            except ValueError:
                vel = None
        joints.append(
            dict(
                name=j.get("name"),
                type=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                axis=np.fromstring(axis.get("xyz"), sep=" ")
                if axis is not None
                else np.array([0.0, 0.0, 1.0]),
                lower=float(limit.get("lower")) if limit is not None else -np.pi,
                upper=float(limit.get("upper")) if limit is not None else np.pi,
                vel=vel,
            )
        )
    # arm base: walk from the GLOBAL root (a link that is a parent but
    # never a child — e.g. 'world' in the ur5 URDF) through fixed joints;
    # the parent of the first ACTUATED joint on that walk is the arm base.
    # (A plain "parent never appearing as a child" rule breaks when a
    # fixed world joint sits above the arm base.)
    all_children = {
        j.find("child").get("link") for j in root.iter("joint")
    }
    all_parents = {
        j.find("parent").get("link") for j in root.iter("joint")
    }
    global_roots = [l for l in all_parents if l not in all_children]
    if len(global_roots) != 1:
        raise ValueError(f"expected a single URDF root, got {global_roots}")
    joints_by_parent: dict[str, list] = {}
    for j in root.iter("joint"):
        joints_by_parent.setdefault(j.find("parent").get("link"), []).append(j)
    base = None
    queue = [global_roots[0]]
    while queue and base is None:
        link = queue.pop(0)
        for j in joints_by_parent.get(link, []):
            if j.get("type") in ("revolute", "prismatic"):
                base = j.find("parent").get("link")
                break
            queue.append(j.find("child").get("link"))
    if base is None:
        raise ValueError(f"no actuated joint reachable from {global_roots}")

    # traverse with ALL joints (fixed included) so branch links reachable
    # only through fixed joints (e.g. Franka's panda_hand -> fingers) are
    # covered; only actuated joints are kept in `ordered`.
    all_joints = []
    for j in root.iter("joint"):
        all_joints.append(
            (
                j.get("type"),
                j.find("parent").get("link"),
                j.find("child").get("link"),
            )
        )
    by_parent: dict[str, list[int]] = {}
    for idx, (jtype, parent, child) in enumerate(all_joints):
        by_parent.setdefault(parent, []).append(idx)

    act_by_name = {j["name"]: j for j in joints}
    ordered = []

    def walk(link: str) -> None:
        kids = by_parent.get(link, [])
        # actuated revolute first, then prismatic, then fixed
        def sort_key(idx):
            t = all_joints[idx][0]
            return {"revolute": 0, "prismatic": 1}.get(t, 2)

        for idx in sorted(kids, key=sort_key):
            jtype, parent, child = all_joints[idx]
            jname = root.findall("joint")[idx].get("name")
            if jname in act_by_name:
                ordered.append(act_by_name[jname])
            walk(child)

    walk(base)
    if len(ordered) != len(joints):
        raise ValueError("URDF tree walk did not cover all actuated joints")
    return ordered, base


def arm_spec_paths(env) -> tuple[str, str]:
    """Per-arm URDF paths for ArmChainSpec.

    aloha-agilex is a native dual-arm articulation (one URDF, one entity,
    fl_*/fr_* links): the body extractor and h use per-arm split URDFs
    (tests/gen_franka_dualarm_urdf.py). All other embodiments have one
    URDF per arm already.
    """
    if getattr(env, "embodiment", "") == "aloha-agilex":
        assets = Path(__file__).resolve().parents[2] / "assets" / "urdf"
        return (str(assets / "aloha_agilex_left.urdf"),
                str(assets / "aloha_agilex_right.urdf"))
    return (env.robot.left_urdf_path, env.robot.right_urdf_path)


class ArmChainSpec:
    """Per-arm kinematic chain facts parsed (once) from a URDF."""

    def __init__(self, urdf_path: str, n_reach_samples: int = 4096, seed: int = 0):
        import pytorch_kinematics as pk

        self.urdf_path = urdf_path
        self.chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read())
        self.chain.to(dtype=torch.float32)

        self.joints, self.base_link = _parse_urdf_joints(urdf_path)
        self.link_names = [j["child"] for j in self.joints]   # token order
        self.arm_links = [j["child"] for j in self.joints if j["type"] == "revolute"]
        self.gripper_links = [j["child"] for j in self.joints if j["type"] == "prismatic"]
        self.n_joints = len(self.joints)

        # joint velocity limits (rad/s or m/s), with logged fallback
        vels, fallbacks = [], []
        for j in self.joints:
            if j["vel"] is not None and 0.1 < j["vel"] < 10:
                vels.append(j["vel"])
            else:
                vels.append(DEFAULT_JOINT_VEL)
                fallbacks.append(j["name"])
        if fallbacks:
            logger.warning(
                "URDF %s: missing/implausible <limit velocity> on %s; "
                "falling back to %.1f",
                urdf_path, fallbacks, DEFAULT_JOINT_VEL,
            )
        self.joint_vel_limits = np.array(vels, dtype=np.float32)

        # tree depth per link and ancestor joints (branch-aware)
        child2joint = {j["child"]: j for j in self.joints}
        depth, self.ancestors = {}, {}
        max_depth = 0
        for name in self.link_names:
            idx, anc = 0, []
            link = name
            while link in child2joint:
                j = child2joint[link]
                anc.insert(0, self.joints.index(j))
                link = j["parent"]
                idx += 1
            depth[name] = idx
            max_depth = max(max_depth, idx)
            self.ancestors[name] = anc
        self.chain_depth = {n: d / max(max_depth, 1) for n, d in depth.items()}

        # link center-of-mass offsets (for exact velocity cross-checks;
        # SAPIEN reports body velocity at the CoM, not the frame origin)
        link_coms = {}
        for lk in ET.parse(urdf_path).getroot().iter("link"):
            inertial = lk.find("inertial")
            origin = inertial.find("origin") if inertial is not None else None
            if origin is not None and origin.get("xyz"):
                link_coms[lk.get("name")] = np.fromstring(origin.get("xyz"), sep=" ")
        self.com_offsets = {
            name: link_coms.get(name, np.zeros(3)) for name in self.link_names
        }

        gj = [j for j in self.joints if j["type"] == "prismatic"]
        self.gripper_joint = gj[0] if gj else None
        self.gripper_range = (
            (self.gripper_joint["lower"], self.gripper_joint["upper"])
            if self.gripper_joint
            else (0.0, 1.0)
        )

        # FK tensor input is ordered by pk's own traversal, which must match
        # our tree order (checked now, not after a wrong L_reach is sampled)
        pk_names = list(self.chain.get_joint_parameter_names())
        mine = [j["name"] for j in self.joints]
        assert pk_names == mine, f"joint order mismatch: pk={pk_names} spec={mine}"

        self.l_reach = self._sample_l_reach(n_reach_samples, seed)

    def _sample_l_reach(self, n_samples: int, seed: int) -> float:
        """Max EE(gripper-tip)-to-base distance over sampled configurations.

        EE and base positions both come from pk FK (URDF-root frame), so any
        fixed pedestal above base_link cancels out.
        """
        rng = np.random.RandomState(seed)
        q = np.zeros((n_samples, self.n_joints), dtype=np.float32)
        for k, j in enumerate(self.joints):
            if j["type"] == "revolute":
                q[:, k] = rng.uniform(j["lower"], j["upper"], n_samples)
        fk = self.chain.forward_kinematics(torch.from_numpy(q))
        ee = fk[self.link_names[-1]].get_matrix()[:, :3, 3].numpy()
        if self.base_link in fk:
            base_p = fk[self.base_link].get_matrix()[:, :3, 3].numpy()
        else:
            base_p = 0.0
        return float(np.linalg.norm(ee - base_p, axis=-1).max())


def get_arm_spec(urdf_path: str) -> ArmChainSpec:
    """URDF-derived per-arm spec, computed once, cached by file content hash."""
    key = _file_hash(urdf_path)
    if key not in _SPEC_CACHE:
        _SPEC_CACHE[key] = ArmChainSpec(urdf_path)
    return _SPEC_CACHE[key]


def _pose_matrix(pose) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float64)


class BodyTokenExtractor:
    """Extracts per-link body tokens from a CEHJ_safe Env.

    Token order is fixed: [left arm links..., left gripper, right arm
    links..., right gripper] — 14 tokens for dual 6-DoF + gripper arms,
    matching HoloBrain's robot-state token order. The two finger links of
    each gripper are merged into one token (mean of both). Output is padded
    to `max_tokens`; `body_mask` marks valid entries.

    Jlin/Jang are block-diagonal over both arms' joints: token of the left
    arm uses columns [0, n_joint), right arm [n_joint, 2*n_joint), so the
    critic can compute dp = Jlin @ dtheta with the full 2*n_joint action.
    The two prismatic gripper slots per arm are included; actors emitting
    only 6 arm joints per arm must zero-pad those slots.

    Args:
        control_dt: control period (s); defaults to env.control_dt if present.
        kappa: fraction of each URDF velocity limit usable per control step.

    Note: construct AFTER the env is reset/settled — the startup FK
    consistency check reads live link poses.
    """

    def __init__(self, env, max_tokens: int = 20, control_dt: float | None = None,
                 kappa: float = DEFAULT_KAPPA):
        self.env = env
        self.max_tokens = max_tokens
        self.control_dt = (
            float(control_dt) if control_dt is not None
            else float(getattr(env, "control_dt", 1.0 / 20.0))
        )
        self.kappa = float(kappa)

        # one spec per arm (hash-cached, so identical URDFs share an object;
        # aloha-agilex gets per-arm split URDFs — see arm_spec_paths)
        left_path, right_path = arm_spec_paths(env)
        self.specs = [get_arm_spec(left_path), get_arm_spec(right_path)]
        self.spec = self.specs[0]  # convenience alias (chain layout)
        for spec in self.specs:
            if spec.n_joints != self.spec.n_joints or len(spec.link_names) != len(self.spec.link_names):
                raise ValueError("left/right arm chain layouts differ")

        n_tokens = 2 * (len(self.spec.arm_links) + (1 if self.spec.gripper_links else 0))
        if n_tokens > max_tokens:
            raise ValueError(f"embodiment needs {n_tokens} tokens > max_tokens={max_tokens}")

        # per-control-step displacement limits (problem definition, not URDF)
        self.delta_theta_max = np.concatenate(
            [spec.joint_vel_limits * self.control_dt * self.kappa for spec in self.specs]
        ).astype(np.float32)

        # action columns of the prismatic (gripper) joints in the 2*n_joint
        # action vector. The actor never writes these, but Jlin reads them —
        # they must be exactly zero (see assert_action).
        self.prismatic_cols = []
        for arm_id, spec in enumerate(self.specs):
            col = arm_id * spec.n_joints
            self.prismatic_cols += [
                col + k for k, j in enumerate(spec.joints) if j["type"] == "prismatic"
            ]

        # startup consistency: pk FK composed to world == SAPIEN link poses
        self.assert_fk_consistency()

        # cache static base world poses (per arm) for offline FK/Jacobian
        # recomputation from stored qpos — valid because the arm bases are
        # static mounts in these tasks (a mobile base would need per-step
        # base poses stored in the buffer)
        self._base_T = []
        for spec, entity in zip(
            self.specs, (self.env.robot.left_entity, self.env.robot.right_entity)
        ):
            _, _, links = self._arm_state(spec, entity)
            self._base_T.append(
                torch.from_numpy(_pose_matrix(links[spec.base_link].get_pose())).float()
            )

    def raw_qpos(self) -> np.ndarray:
        """Current raw joint qpos, [16]: per-arm 8 actuated joints in spec
        order (joint1..joint8). Store this in the buffer alongside the
        14-dim policy-layout qpos."""
        out = []
        for spec, entity in zip(
            self.specs, (self.env.robot.left_entity, self.env.robot.right_entity)
        ):
            qpos, _, _ = self._arm_state(spec, entity)
            out.append(qpos)
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def jacobian_batch(self, qpos_raw: torch.Tensor):
        """Batched Jlin/Jang from raw qpos alone (no simulator).

        qpos_raw: [B, 16] raw per-arm 8-joint qpos (see raw_qpos()).
        Returns Jlin, Jang [B, max_tokens, 3, 2*n_joint], block-diagonal,
        padded token layout (arm links + merged gripper midpoint rows,
        zero padding) — identical layout to extract()["Jlin"].
        """
        qpos_raw = torch.as_tensor(qpos_raw, dtype=torch.float32).cpu()
        B = qpos_raw.shape[0]
        n_joint = self.spec.n_joints
        n_tok_arm = len(self.spec.arm_links) + (1 if self.spec.gripper_links else 0)
        Jlin = qpos_raw.new_zeros(B, self.max_tokens, 3, 2 * n_joint)
        Jang = qpos_raw.new_zeros(B, self.max_tokens, 3, 2 * n_joint)

        for arm_id, (spec, T_wb_t) in enumerate(zip(self.specs, self._base_T)):
            q = qpos_raw[:, arm_id * n_joint : (arm_id + 1) * n_joint]
            fk = spec.chain.forward_kinematics(q)  # batched, arm-base frame
            R_wb, p_wb = T_wb_t[:3, :3], T_wb_t[:3, 3]

            # joint world positions / axes per batch
            jp, jz = [], []
            for j in spec.joints:
                M = fk[j["child"]].get_matrix()              # [B, 4, 4]
                axis = torch.from_numpy(
                    np.asarray(j["axis"], dtype=np.float32)
                ).reshape(1, 3, 1)
                jp.append((R_wb @ M[:, :3, 3].T).T + p_wb)   # [B, 3]
                jz.append((R_wb @ (M[:, :3, :3] @ axis).squeeze(-1).T).T)  # [B, 3]

            link_idx = {name: k for k, name in enumerate(spec.link_names)}
            n_link = len(spec.link_names)
            Jl = qpos_raw.new_zeros(B, n_link, 3, n_joint)
            Ja = qpos_raw.new_zeros(B, n_link, 3, n_joint)
            for k, name in enumerate(spec.link_names):
                M = fk[name].get_matrix()
                p_k = (R_wb @ M[:, :3, 3].T + p_wb[:, None]).T   # [B, 3]
                for jj in spec.ancestors[name]:
                    j = spec.joints[jj]
                    if j["type"] == "revolute":
                        Jl[:, k, :, jj] = torch.cross(jz[jj], p_k - jp[jj], dim=-1)
                        Ja[:, k, :, jj] = jz[jj]
                    else:
                        Jl[:, k, :, jj] = jz[jj]

            # merge finger rows (midpoint convention from extract())
            rows_l, rows_a = [], []
            for name in spec.arm_links:
                rows_l.append(Jl[:, link_idx[name]])
                rows_a.append(Ja[:, link_idx[name]])
            if spec.gripper_links:
                rows_l.append(
                    torch.stack([Jl[:, link_idx[f]] for f in spec.gripper_links]).mean(0)
                )
                rows_a.append(
                    torch.stack([Ja[:, link_idx[f]] for f in spec.gripper_links]).mean(0)
                )
            Jl_tok = torch.stack(rows_l, dim=1)               # [B, n_tok_arm, 3, 8]
            Ja_tok = torch.stack(rows_a, dim=1)
            off = arm_id * n_tok_arm
            col = arm_id * n_joint
            Jlin[:, off : off + n_tok_arm, :, col : col + n_joint] = Jl_tok
            Jang[:, off : off + n_tok_arm, :, col : col + n_joint] = Ja_tok
        return Jlin, Jang


    def assert_action(self, action, atol: float = 0.0) -> None:
        """Critic-boundary check: prismatic gripper action slots are zero.

        Jlin has all 2*n_joint columns populated, so dp = Jlin @ dtheta
        reads the prismatic (gripper) slots. The actor never writes them,
        so any nonzero value there is stale garbage that silently
        contributes to every link's action effect.
        """
        cols = self.prismatic_cols
        if not cols:
            return
        vals = action[..., cols]
        if isinstance(vals, torch.Tensor):
            bad = int((vals.abs() > atol).sum())
        else:
            bad = int((np.abs(vals) > atol).sum())
        assert bad == 0, (
            f"{bad} non-zero entr{'y' if bad == 1 else 'ies'} in prismatic "
            f"gripper action columns {cols} (max "
            f"{float(np.abs(np.asarray(vals)).max()):.3g})"
        )

    # ------------------------------------------------------------------
    # consistency checks
    # ------------------------------------------------------------------
    def _arm_state(self, spec, entity):
        names = [j.get_name() for j in entity.get_active_joints()]
        missing = [j["name"] for j in spec.joints if j["name"] not in names]
        if missing:
            raise RuntimeError(
                f"URDF joints {missing} not found in SAPIEN active joints "
                f"{names}. Mimic/passive gripper joints must be mapped "
                f"explicitly — extend _arm_state for this embodiment."
            )
        idx = [names.index(j["name"]) for j in spec.joints]
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64)[idx]
        qvel = np.asarray(entity.get_qvel(), dtype=np.float64)[idx]
        links = {l.get_name(): l for l in entity.get_links()}
        return qpos, qvel, links

    def _fk_world(self, spec, qpos, base_pose_mat):
        """pk FK (URDF-root frame) composed to world via the base link pose."""
        fk = spec.chain.forward_kinematics(
            torch.from_numpy(np.asarray(qpos, dtype=np.float32))
        )
        R_wb, p_wb = base_pose_mat[:3, :3], base_pose_mat[:3, 3]
        out = {}
        for name in spec.link_names + [spec.base_link]:
            M = fk[name].get_matrix()[0].numpy()
            out[name] = (R_wb @ M[:3, :3], R_wb @ M[:3, 3] + p_wb)
        return out

    def assert_fk_consistency(self, atol: float = 1e-4) -> None:
        """Startup check: pk-FK-to-world reproduces SAPIEN link poses.

        Catches URDF/simulator mismatches (e.g. pk root sitting on a
        different link than the spec's base_link) that would silently shift
        every Jacobian.
        """
        for spec, entity in zip(
            self.specs, (self.env.robot.left_entity, self.env.robot.right_entity)
        ):
            qpos, _, links = self._arm_state(spec, entity)
            T_wb = _pose_matrix(links[spec.base_link].get_pose())
            fk_world = self._fk_world(spec, qpos, T_wb)
            for name in spec.link_names:
                sim_p = np.asarray(links[name].get_pose().p, dtype=np.float64)
                err = np.abs(fk_world[name][1] - sim_p).max()
                if err > atol:
                    raise RuntimeError(
                        f"FK/sim mismatch on link {name!r}: max err {err:.2e} "
                        f"(atol {atol}). pk FK root does not coincide with "
                        f"SAPIEN base link {spec.base_link!r}."
                    )

    def velocity_cross_check(self) -> float:
        """Compare Jlin @ qvel against SAPIEN link linear velocities.

        Run while the robot is moving; returns the max absolute error (m/s).
        The Jacobian is evaluated at each link's center of mass (URDF
        inertial origin), since SAPIEN reports body velocity at the CoM.
        Catches joint-ordering mistakes across all links in one shot.
        Diagnostic: if this returns a large error, first re-check without
        the CoM offset — some SAPIEN versions report link-origin velocity.
        """
        max_err = 0.0
        for spec, entity in zip(
            self.specs, (self.env.robot.left_entity, self.env.robot.right_entity)
        ):
            qpos, qvel, links = self._arm_state(spec, entity)
            fk = spec.chain.forward_kinematics(
                torch.from_numpy(qpos.astype(np.float32))
            )
            T_wb = _pose_matrix(links[spec.base_link].get_pose())
            R_wb, p_wb = T_wb[:3, :3], T_wb[:3, 3]
            joint_pz = []
            for j in spec.joints:
                M = fk[j["child"]].get_matrix()[0].numpy()
                joint_pz.append(
                    (R_wb @ M[:3, 3] + p_wb, R_wb @ (M[:3, :3] @ j["axis"]))
                )
            for k, name in enumerate(spec.link_names):
                pose = links[name].get_pose()
                T_l = _pose_matrix(pose)
                p_eval = T_l[:3, 3] + T_l[:3, :3] @ spec.com_offsets.get(
                    name, np.zeros(3)
                )
                v_jac = np.zeros(3)
                for jj in spec.ancestors[name]:
                    j = spec.joints[jj]
                    p_j, z_j = joint_pz[jj]
                    if j["type"] == "revolute":
                        v_jac += np.cross(z_j, p_eval - p_j) * qvel[jj]
                    else:
                        v_jac += z_j * qvel[jj]
                v_sim = np.asarray(links[name].get_linear_velocity(), dtype=np.float64)
                max_err = max(max_err, float(np.abs(v_sim - v_jac).max()))
        return max_err

    # ------------------------------------------------------------------
    # jacobian
    # ------------------------------------------------------------------
    def _jacobian(self, spec, qpos, links):
        """World-frame geometric Jacobian per link: Jlin/Jang [n_link, 3, n_joint]."""
        fk = spec.chain.forward_kinematics(
            torch.from_numpy(np.asarray(qpos, dtype=np.float32))
        )
        T_wb = _pose_matrix(links[spec.base_link].get_pose())
        R_wb, p_wb = T_wb[:3, :3], T_wb[:3, 3]

        n_link, n_joint = len(spec.link_names), spec.n_joints
        Jlin = np.zeros((n_link, 3, n_joint), dtype=np.float64)
        Jang = np.zeros((n_link, 3, n_joint), dtype=np.float64)

        joint_p, joint_z = [], []
        for j in spec.joints:
            M = fk[j["child"]].get_matrix()[0].numpy()
            joint_p.append(R_wb @ M[:3, 3] + p_wb)
            joint_z.append(R_wb @ (M[:3, :3] @ j["axis"]))

        for k, name in enumerate(spec.link_names):
            p_k = np.asarray(links[name].get_pose().p, dtype=np.float64)
            for jj in spec.ancestors[name]:  # tree ancestors, incl. own joint
                j = spec.joints[jj]
                if j["type"] == "revolute":
                    Jlin[k, :, jj] = np.cross(joint_z[jj], p_k - joint_p[jj])
                    Jang[k, :, jj] = joint_z[jj]
                else:  # prismatic
                    Jlin[k, :, jj] = joint_z[jj]
        return Jlin, Jang

    # ------------------------------------------------------------------
    # extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract(self) -> dict:
        """Current body tokens + aux outputs (all world frame, SI units).

        Token layout per arm: [arm links (6), gripper] — the two gripper
        finger links are merged into ONE gripper token (mean of both
        fingers), matching HoloBrain's 14-entry layout
        [L_arm(6), L_gripper, R_arm(6), R_gripper], so body token i pairs
        1:1 with HoloBrain robot-state token i.
        """
        spec = self.spec
        n_link = len(spec.arm_links) + (1 if spec.gripper_links else 0)
        n_joint = spec.n_joints
        D = TOKEN_DIM

        tokens = np.zeros((self.max_tokens, D), dtype=np.float32)
        link_pos = np.zeros((self.max_tokens, 3), dtype=np.float32)
        Jlin = np.zeros((self.max_tokens, 3, 2 * n_joint), dtype=np.float32)
        Jang = np.zeros((self.max_tokens, 3, 2 * n_joint), dtype=np.float32)
        body_mask = np.zeros(self.max_tokens, dtype=bool)
        is_actuated = np.zeros(self.max_tokens, dtype=bool)
        joint_index = np.full(self.max_tokens, -1, dtype=np.int64)

        for arm_id, (arm_spec, entity) in enumerate(
            zip(self.specs, (self.env.robot.left_entity, self.env.robot.right_entity))
        ):
            qpos, qvel, links = self._arm_state(arm_spec, entity)
            Jl, Ja = self._jacobian(arm_spec, qpos, links)
            link_idx = {name: k for k, name in enumerate(arm_spec.link_names)}
            base_pos = np.asarray(
                links[arm_spec.base_link].get_pose().p, dtype=np.float64
            )
            grip_lo, grip_hi = arm_spec.gripper_range
            grip_q = (
                qpos[arm_spec.joints.index(arm_spec.gripper_joint)]
                if arm_spec.gripper_joint else 0.0
            )
            grip_open = float(
                np.clip((grip_q - grip_lo) / max(grip_hi - grip_lo, 1e-9), 0, 1)
            )

            off = arm_id * n_link
            col = arm_id * n_joint

            def _fill(i, name, p, q_wxyz, Jl_row, Ja_row, grip_val, actuated):
                tokens[i, 0:3] = (p - base_pos) / arm_spec.l_reach
                tokens[i, 3:7] = q_wxyz
                tokens[i, 7:10] = (Jl_row @ qvel) / LIN_VEL_SCALE
                tokens[i, 10:13] = (Ja_row @ qvel) / ANG_VEL_SCALE
                tokens[i, 13] = arm_spec.chain_depth[name]
                tokens[i, 14] = arm_id
                tokens[i, 15] = grip_val
                link_pos[i] = p
                Jlin[i, :, col : col + n_joint] = Jl_row
                Jang[i, :, col : col + n_joint] = Ja_row
                body_mask[i] = True
                is_actuated[i] = actuated

            # arm links: token k <-> action column col + k (see joint_index)
            for k, name in enumerate(arm_spec.arm_links):
                ki = link_idx[name]
                pose = links[name].get_pose()
                _fill(
                    off + k, name,
                    np.asarray(pose.p, dtype=np.float64),
                    np.asarray(pose.q, dtype=np.float64),
                    Jl[ki], Ja[ki], 0.0, True,
                )
                joint_index[off + k] = col + k

            # merged gripper token: midpoint of the finger links (HoloBrain's
            # own FK convention); ORIENTATION taken from the wrist (finger
            # parent) — the midpoint orientation of mirrored fingers is
            # meaningless, and naive quaternion averaging has a q/-q
            # double-cover hole. The gripper carries no action column (-1).
            fingers = arm_spec.gripper_links
            if fingers:
                p_f = np.mean(
                    [np.asarray(links[f].get_pose().p, dtype=np.float64) for f in fingers],
                    axis=0,
                )
                wrist = arm_spec.gripper_joint["parent"]
                q_f = np.asarray(links[wrist].get_pose().q, dtype=np.float64)
                Jl_f = np.mean([Jl[link_idx[f]] for f in fingers], axis=0)
                Ja_f = np.mean([Ja[link_idx[f]] for f in fingers], axis=0)
                i = off + len(arm_spec.arm_links)
                tokens[i, 0:3] = (p_f - base_pos) / arm_spec.l_reach
                tokens[i, 3:7] = q_f
                tokens[i, 7:10] = (Jl_f @ qvel) / LIN_VEL_SCALE
                tokens[i, 10:13] = (Ja_f @ qvel) / ANG_VEL_SCALE
                tokens[i, 13] = arm_spec.chain_depth[fingers[0]]
                tokens[i, 14] = arm_id
                tokens[i, 15] = grip_open
                link_pos[i] = p_f
                Jlin[i, :, col : col + n_joint] = Jl_f
                Jang[i, :, col : col + n_joint] = Ja_f
                body_mask[i] = True
                is_actuated[i] = False  # gripper carries no learned action
                # joint_index stays -1: no action column for this token

        cams = self.env.task.cameras
        head_cam = cams.static_camera_list[cams.head_camera_id]
        # canonical source, shared with projection_mat in get_encoder_obs
        T_cam_world = self.env.camera_extrinsic(head_cam)

        return {
            "body_tokens": torch.from_numpy(tokens),           # [max_tokens, 16]
            "link_pos": torch.from_numpy(link_pos),            # [max_tokens, 3] world m
            "Jlin": torch.from_numpy(Jlin),                    # [max_tokens, 3, 2*n_joint]
            "Jang": torch.from_numpy(Jang),                    # [max_tokens, 3, 2*n_joint]
            "body_mask": torch.from_numpy(body_mask),          # [max_tokens]
            "is_actuated": torch.from_numpy(is_actuated),      # [max_tokens] == joint_index >= 0
            "joint_index": torch.from_numpy(joint_index),      # [max_tokens] action column per token, -1 for gripper/padding
            "prismatic_cols": list(self.prismatic_cols),       # action cols the actor must leave zero
            "T_cam_world": torch.from_numpy(T_cam_world),      # [4, 4] world->head
            # per-embodiment meta
            "l_reach": [s.l_reach for s in self.specs],
            "delta_theta_max": torch.from_numpy(self.delta_theta_max),  # [2*n_joint]
            "joint_vel_limits": [
                torch.from_numpy(s.joint_vel_limits) for s in self.specs
            ],
            "link_names": spec.link_names,
            "token_names": spec.arm_links + ["gripper"],
            "gripper_links": spec.gripper_links,
            "control_dt": self.control_dt,
            "kappa": self.kappa,
        }
