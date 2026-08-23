"""Stepwise nominal controllers for the training pipeline.

NominalTracker (legacy): probe once, replay the recorded joint trajectory.
Time-indexed — after the safety actor steers, it keeps chasing the OLD
path instead of replanning from the live state. Kept for reference; do not
use for filtered rollouts.

ReplanNominal: receding-horizon nominal aligned with PlanEveryKController.
The probe records EE-pose waypoints at control rate; at rollout the
nominal replans from the LIVE qpos toward the current waypoint every K
control steps (and whenever the current plan is exhausted). After the safe
actor steers, the next replan starts from the steered state — it resumes
the task from where the robot actually is, never chases a stale plan.

Planning uses the task's own RoboTwin planners (cuRobo, table-only world)
— obstacle-agnostic by construction, same as the expert.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# legacy replay nominal (kept for reference)
# ----------------------------------------------------------------------
class NominalTracker:
    """Probe once, replay recorded joint targets. Time-indexed: chases the
    old path after interventions. Superseded by ReplanNominal."""

    def __init__(self, traj: np.ndarray, success: bool):
        self.traj = traj
        self.success = bool(success)
        self.T = len(traj)

    def goal(self, t: int) -> np.ndarray:
        return self.traj[min(t, self.T - 1)]


# ----------------------------------------------------------------------
# receding-horizon nominal
# ----------------------------------------------------------------------
class ReplanNominal:
    """Plan from live qpos toward probe EE waypoints, every K control steps.

    probe(env) records, at control rate, both arms' EE poses ([x,y,z,qw..])
    and normalized gripper values while the expert plays once.

    action(t, env) returns (dtheta16, grip_left, grip_right): each arm is
    replanned toward its waypoint whenever its plan is exhausted, its EE
    error exceeds ee_tol and K control steps elapsed — always seeded from
    the current measured qpos.
    """

    def __init__(self, ee_traj: dict, grip_traj: np.ndarray, success: bool,
                 k: int = 2, ee_tol: float = 0.03, stall_window: int = 8,
                 ee_tol_grasp: float = 0.008):
        self.ee_traj = ee_traj          # {"left": [T,7], "right": [T,7]}
        self.grip_traj = grip_traj      # [T, 2] normalized gripper values
        self.success = bool(success)
        self.T = len(grip_traj)
        self.k = max(1, int(k))
        self.ee_tol = float(ee_tol)
        self.ee_tol_grasp = float(ee_tol_grasp)
        self.stall_window = int(stall_window)
        self.paths = {"left": None, "right": None}   # joint waypoint arrays
        self.path_i = {"left": 0.0, "right": 0.0}    # float index (250Hz rate)
        # progress-gated waypoint cursors: the nominal resumes the SAME
        # waypoint after an intervention instead of skipping ahead in time
        self.seg = {"left": 0, "right": 0}
        self._err_hist = {"left": [], "right": []}
        self.n_replans = 0
        self.n_plan_fail = 0

    def done(self) -> bool:
        return all(s >= self.T - 1 for s in self.seg.values())

    # ---------------- safety: skip unsafe waypoints ----------------
    def waypoint_clearance(self, env, arm: str) -> np.ndarray | None:
        """Signed distance (m) from each of this arm's EE waypoints to the
        spawned obstacle. None if there is no obstacle."""
        from main.envs.distance import _obb_from_actor, point_obb_signed_distance

        obs = getattr(env, "obstacle", None)
        if obs is None:
            return None
        origin, rot, half = _obb_from_actor(
            obs, getattr(env, "obstacle_half", None)
        )
        return np.array(
            [
                point_obb_signed_distance(np.asarray(wp[:3]), origin, rot, half)
                for wp in self.ee_traj[arm]
            ]
        )

    def skip_unsafe(self, env, margin: float = 0.0) -> dict:
        """Advance each arm's cursor PAST waypoints inside the unsafe region.

        The whole point of steering is that the current waypoint is unsafe —
        resuming it would drive straight back in and make the steering
        pointless. After an intervention the nominal instead resumes at the
        first upcoming waypoint that clears the obstacle set by `margin`,
        planning fresh from wherever the robot was steered to. Waypoints
        that are unsafe are skipped permanently (their task content — e.g. a
        grasp — is sacrificed to safety, which is the correct trade).
        """
        skipped = {}
        for arm in ("left", "right"):
            cl = self.waypoint_clearance(env, arm)
            if cl is None:
                continue
            seg = self.seg[arm]
            j = seg
            while j < self.T - 1 and cl[j] <= margin:
                j += 1
            if j != seg:
                skipped[arm] = j - seg
                self.seg[arm] = j
                self.paths[arm] = None
                self._err_hist[arm] = []
        return skipped

    # ---------------- probe ----------------
    @classmethod
    def probe(cls, env, control_freq: float = 20.0) -> "ReplanNominal":
        """Run the expert once; record EE poses + gripper values per control step."""
        physics_freq = env.PHYSICS_FREQ
        ee = {"left": [], "right": []}
        grip = []
        orig_step = env.task.scene.step
        state = {"n": 0}

        def step():
            orig_step()
            state["n"] += 1
            # record at control boundaries (same cadence as collection)
            if state["n"] >= round((len(grip) + 1) * physics_freq / control_freq):
                ee["left"].append(np.asarray(env.robot.get_left_ee_pose(), dtype=np.float32))
                ee["right"].append(np.asarray(env.robot.get_right_ee_pose(), dtype=np.float32))
                grip.append(
                    [
                        env.robot.get_left_gripper_val(),
                        env.robot.get_right_gripper_val(),
                    ]
                )

        env.task.scene.step = step
        success = False
        try:
            env.task.play_once()
            success = bool(env.task.check_success())
        except Exception as exc:
            print(f"[nominal] probe play_once failed: {exc}")
        finally:
            env.task.scene.step = orig_step  # restore even on failure
        ee = {a: np.asarray(v, dtype=np.float32) for a, v in ee.items()}
        return cls(ee, np.asarray(grip, dtype=np.float32), success)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path.with_suffix(".npz"),
            left=self.ee_traj["left"],
            right=self.ee_traj["right"],
            grip=self.grip_traj,
            meta=json.dumps({"success": self.success, "T": self.T}),
        )
        return path

    @classmethod
    def load(cls, path: str | Path, k: int = 2) -> "ReplanNominal":
        z = np.load(Path(path).with_suffix(".npz"))
        meta = json.loads(str(z["meta"]))
        return cls(
            {"left": z["left"], "right": z["right"]}, z["grip"], meta["success"], k=k
        )

    @classmethod
    def cached(cls, path: str | Path, env, control_freq: float = 20.0, k: int = 2):
        path = Path(path)
        if path.with_suffix(".npz").exists():
            return cls.load(path, k=k)
        tracker = cls.probe(env, control_freq)
        tracker.save(path)
        env.close()
        return tracker

    # ---------------- rollout ----------------
    def _ee_err(self, env, arm: str, target) -> float:
        cur = env.robot.get_left_ee_pose() if arm == "left" else env.robot.get_right_ee_pose()
        return float(np.linalg.norm(np.asarray(target[:3]) - np.asarray(cur[:3])))

    def _replan(self, env, arm: str, target) -> None:
        self.n_replans += 1
        plan = env.robot.left_plan_path if arm == "left" else env.robot.right_plan_path
        result = plan(target)
        if result and result.get("status") == "Success" and len(result.get("position", [])):
            self.paths[arm] = np.asarray(result["position"], dtype=np.float64)
            self.path_i[arm] = 0.0
        else:
            self.n_plan_fail += 1
            self.paths[arm] = None  # hold

    def action(self, t: int, env, dtheta_max: np.ndarray, physics_freq: float = 250.0):
        """Nominal 16-dim dtheta + gripper targets.

        Receding horizon aligned with PlanEveryKController: each plan runs
        to completion, then a fresh plan is issued from the LIVE state. The
        waypoint cursor is PROGRESS-GATED — it advances only when the EE
        reaches the current waypoint, so after the safe actor steers, the
        nominal resumes the same waypoint from wherever the robot is,
        instead of skipping ahead in time.
        """
        dtheta = np.zeros(16, dtype=np.float64)
        step_advance = physics_freq / 20.0  # path points per control step (12.5)
        # per-arm grip targets from each arm's OWN cursor — a shared index
        # misaligns grasp timing when the two cursors diverge
        g_left = self.grip_traj[min(self.seg["left"], self.T - 1)][0]
        g_right = self.grip_traj[min(self.seg["right"], self.T - 1)][1]
        real_grips = env.robot.get_normal_real_gripper_val()  # [left, right] normalized
        for arm_id, arm in enumerate(("left", "right")):
            seg = self.seg[arm]
            wp = self.ee_traj[arm][min(seg, self.T - 1)]
            err = self._ee_err(env, arm, wp)
            hist = self._err_hist[arm]
            hist.append(err)
            if len(hist) > self.stall_window:
                hist.pop(0)
            progress = hist[0] - err if len(hist) == self.stall_window else None
            # Grip gating: the cursor may only advance when the REAL gripper
            # has converged to the timeline's target opening — for closing
            # (grasp) AND for opening (release). Otherwise the arm moves on
            # while the fingers are still opening (~0.1/step rate limit) and
            # the block is dragged/dropped at the wrong position.
            g_arm = g_left if arm == "left" else g_right
            grasp_wp = g_arm < 0.3
            tol = self.ee_tol_grasp if grasp_wp else self.ee_tol
            grip_done = abs(real_grips[arm_id] - g_arm) < 0.1
            at_goal = err <= tol and grip_done
            # everyk's no-progress stop: close enough but not improving —
            # accept and advance (never at grasp waypoints)
            close_enough = err <= max(self.ee_tol * 2.0, 0.04)
            no_progress = progress is not None and progress < 0.002
            if at_goal or (close_enough and no_progress and not grasp_wp and grip_done):
                if seg < self.T - 1:
                    self.seg[arm] = seg + 1  # waypoint reached: progress
                self.paths[arm] = None
                self._err_hist[arm] = []
                continue
            path = self.paths[arm]
            exhausted = path is None or self.path_i[arm] >= len(path)
            stalled = (
                progress is not None and progress < 0.002
            )
            if path is None or exhausted or stalled:
                self._replan(env, arm, wp)
                self._err_hist[arm] = [err]
                path = self.paths[arm]
            if path is None:
                continue  # plan failed: hold
            i = min(int(self.path_i[arm]), len(path) - 1)
            goal_q = path[i]
            self.path_i[arm] += step_advance
            cur_q = np.asarray(
                env.robot.get_left_arm_real_jointState()[:6]
                if arm == "left"
                else env.robot.get_right_arm_real_jointState()[:6]
            )
            col = arm_id * 8
            dtheta[col : col + 6] = np.clip(
                goal_q - cur_q,
                -dtheta_max[col : col + 6],
                dtheta_max[col : col + 6],
            )
        return dtheta, float(g_left), float(g_right)
