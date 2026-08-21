"""CuRobo IK / motion controller for a RoboTwin Env.

Call `CuroboIKController.install()` (or `ResidualController.install()`) before
`Env(...)` so `Robot.set_planner` loads CuRobo (mplib RRT is a plan fallback).
The default eval controller is `ResidualController`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from .env import Env, prepare

ResidualFn = Callable[..., np.ndarray]


class CuroboIKController:
    """Nominal controller: CuRobo IK + motion gen to a gripper pose.

    Subclass and override `plan()` to change the joint path. Call `attach()`
    so `task.play_once()` uses this controller instead of the robot planners
    directly.
    """

    _curobo_cache: dict = {}

    @staticmethod
    def drop_other_curobo(yml_path: str) -> None:
        """Keep MotionGen only for this robot YAML folder. Multiple robots' CUDA
        graphs on one GPU caused H100 illegal-instruction after reuse."""
        family = str(Path(str(yml_path)).resolve().parent)
        cache = CuroboIKController._curobo_cache
        drop = [k for k in list(cache) if str(Path(k[0]).resolve().parent) != family]
        if not drop:
            return
        for key in drop:
            cache.pop(key, None)
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        print(f"[controller] dropped CuRobo cache for {len(drop)} other-arm planner(s)")

    def __init__(self, env: Env):
        self.env = env
        self.robot = env.robot
        self._left_plan = env.robot.left_plan_path
        self._right_plan = env.robot.right_plan_path
        self._n_plan = 0

    @staticmethod
    def install() -> None:
        prepare()
        from .obstacle import patch_curobo_collision_cache
        from envs.robot.planner import CuroboPlanner, MplibPlanner
        from envs.robot.robot import Robot

        patch_curobo_collision_cache()
        if getattr(Robot.set_planner, "_cehj_installed", False):
            return

        orig_set = Robot.set_planner
        orig_curobo_init = CuroboPlanner.__init__

        def _cached_curobo_init(self, robot_origion_pose, active_joints_name, all_joints, yml_path=None):
            CuroboIKController.drop_other_curobo(str(yml_path))
            p = np.asarray(getattr(robot_origion_pose, "p"), dtype=np.float64)
            key = (str(yml_path), tuple(np.round(p, 3).tolist()))
            hit = CuroboIKController._curobo_cache.get(key)
            if hit is not None:
                self.yml_path = yml_path
                self.robot_origion_pose = robot_origion_pose
                self.active_joints_name = active_joints_name
                self.all_joints = all_joints
                self.frame_bias = hit["frame_bias"]
                self.motion_gen = hit["motion_gen"]
                self.motion_gen_batch = hit["motion_gen_batch"]
                print(f"[controller] reuse CuRobo ({Path(str(yml_path)).name})")
                return
            print(f"[controller] warmup CuRobo ({Path(str(yml_path)).name}) ...")
            orig_curobo_init(
                self,
                robot_origion_pose,
                active_joints_name,
                all_joints,
                yml_path=yml_path,
            )
            CuroboIKController._curobo_cache[key] = {
                "frame_bias": self.frame_bias,
                "motion_gen": self.motion_gen,
                "motion_gen_batch": self.motion_gen_batch,
            }

        CuroboPlanner.__init__ = _cached_curobo_init

        def _mplib_pair(robot, scene=None):
            robot.communication_flag = False
            robot.left_mplib_planner = MplibPlanner(
                robot.left_urdf_path,
                robot.left_srdf_path,
                robot.left_move_group,
                robot.left_entity_origion_pose,
                robot.left_entity,
                "mplib_RRT",
                scene,
            )
            robot.right_mplib_planner = MplibPlanner(
                robot.right_urdf_path,
                robot.right_srdf_path,
                robot.right_move_group,
                robot.right_entity_origion_pose,
                robot.right_entity,
                "mplib_RRT",
                scene,
            )
            return robot.left_mplib_planner, robot.right_mplib_planner

        def set_planner(self, scene=None):
            try:
                orig_set(self, scene)
            except Exception as exc:
                print(f"[controller] CuRobo init failed ({exc}); using mplib RRT.")
                left, right = _mplib_pair(self, scene)
                self.left_planner = left
                self.right_planner = right
                return
            if getattr(self, "left_mplib_planner", None) is None:
                try:
                    _mplib_pair(self, scene)
                except Exception as exc:
                    print(f"[controller] mplib fallback not attached: {exc}")

        set_planner._cehj_installed = True
        Robot.set_planner = set_planner

    def attach(self) -> None:
        """Make `play_once` / `move` go through `self.plan`."""
        self.robot.left_plan_path = lambda target_pose, **kw: self.plan("left", target_pose, **kw)
        self.robot.right_plan_path = lambda target_pose, **kw: self.plan("right", target_pose, **kw)

    def plan(self, arm: str, target_pose, **kwargs):
        """IK + collision-aware path to a gripper pose `[x,y,z, qw,qx,qy,qz]`."""
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        self._n_plan += 1
        stats = self._plan_stats(arm, target_pose, **kwargs)
        nominal = self._left_plan if arm == "left" else self._right_plan
        result = nominal(target_pose, **kwargs)
        ok = bool(result) and result.get("status") == "Success"
        n_wp = 0 if not ok else len(result.get("position", []))
        print(
            f"[controller] plan #{self._n_plan} {arm} "
            f"curobo/rrt={'ok' if ok else result.get('status') if result else None} "
            f"n_wp={n_wp} {stats}"
        )
        if ok:
            return result
        result = self._plan_screw(arm, target_pose, **kwargs)
        if result and result.get("status") == "Success":
            print(f"[controller] plan #{self._n_plan} {arm} screw fallback ok")
            return result
        lifted = list(target_pose)
        lifted[2] += 0.08
        result = self._plan_screw(arm, lifted, **kwargs)
        if result and result.get("status") == "Success":
            print(f"[controller] plan #{self._n_plan} {arm} world-Z fallback ok")
        else:
            print(f"[controller] plan #{self._n_plan} {arm} all fallbacks failed")
        return result

    def _plan_stats(self, arm: str, target_pose, **kwargs) -> str:
        import numpy as np

        cur = np.asarray(
            self.robot.get_left_ee_pose() if arm == "left" else self.robot.get_right_ee_pose(),
            dtype=np.float64,
        )
        tgt = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        dxyz = tgt[:3] - cur[:3]
        entity = self.robot.left_entity if arm == "left" else self.robot.right_entity
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64)[:6]
        return (
            f"xyz {cur[:3].round(3).tolist()} -> {tgt[:3].round(3).tolist()} "
            f"dxyz {dxyz.round(3).tolist()} |d|={np.linalg.norm(dxyz):.3f} "
            f"quat {cur[3:].round(3).tolist()} -> {tgt[3:].round(3).tolist()} "
            f"qpos {qpos.round(3).tolist()} "
            f"constraint={kwargs.get('constraint_pose')}"
        )

    def _plan_screw(self, arm: str, target_pose, **kwargs):
        mplib = (
            self.robot.left_mplib_planner if arm == "left" else self.robot.right_mplib_planner
        )
        if mplib is None:
            return {"status": "Fail"}
        qpos = kwargs.get("last_qpos")
        if qpos is None:
            entity = self.robot.left_entity if arm == "left" else self.robot.right_entity
            qpos = entity.get_qpos()
        pose = self.robot._trans_from_gripper_to_endlink(target_pose, arm_tag=arm)
        return mplib.plan_screw(qpos, pose, arms_tag=arm, log=False)


class ResidualController(CuroboIKController):
    """Nominal CuRobo path plus a joint-space residual.

    `play_once` still picks gripper waypoints. This controller plans the
    nominal joint path, then adds `delta()` to `result["position"]`.

    Override `delta()` or pass `residual_fn(arm, result, target_pose, **kwargs)`
    returning a `(T, n)` offset. The default residual is zeros, so behavior
    matches `CuroboIKController` until a policy is plugged in.
    """

    def __init__(self, env: Env, residual_fn: ResidualFn | None = None):
        super().__init__(env)
        self.residual_fn = residual_fn

    def delta(self, arm: str, result: dict, target_pose=None, **kwargs) -> np.ndarray:
        pos = np.asarray(result["position"], dtype=np.float64)
        if self.residual_fn is None:
            return np.zeros_like(pos)
        return np.asarray(
            self.residual_fn(arm, result, target_pose, **kwargs), dtype=np.float64
        )

    def plan(self, arm: str, target_pose, **kwargs):
        result = super().plan(arm, target_pose, **kwargs)
        if not result or result.get("status") != "Success":
            return result
        pos = np.asarray(result["position"], dtype=np.float64)
        dq = np.asarray(self.delta(arm, result, target_pose=target_pose, **kwargs), dtype=np.float64)
        if dq.shape != pos.shape:
            raise ValueError(
                f"residual shape {dq.shape} does not match path {pos.shape} ({arm})"
            )
        if not np.any(dq):
            return result
        new_pos = self._clip_qpos(arm, pos + dq)
        out = dict(result)
        out["position"] = new_pos
        out["velocity"] = self._recompute_velocity(new_pos, result.get("velocity"))
        print(
            f"[controller] residual {arm} |dq|={float(np.linalg.norm(dq)):.4f} "
            f"max|dq|={float(np.max(np.abs(dq))):.4f}"
        )
        return out

    def _clip_qpos(self, arm: str, qpos: np.ndarray) -> np.ndarray:
        entity = self.robot.left_entity if arm == "left" else self.robot.right_entity
        try:
            limits = np.asarray(entity.get_qlimits(), dtype=np.float64)
        except Exception:
            return qpos
        n = min(qpos.shape[1], limits.shape[0])
        clipped = qpos.copy()
        clipped[:, :n] = np.clip(clipped[:, :n], limits[:n, 0], limits[:n, 1])
        return clipped

    @staticmethod
    def _recompute_velocity(qpos: np.ndarray, vel, dt: float = 1.0 / 250.0) -> np.ndarray:
        if qpos.shape[0] < 2:
            if vel is None:
                return np.zeros_like(qpos)
            return np.asarray(vel, dtype=np.float64)
        dq = np.diff(qpos, axis=0) / dt
        dq = np.vstack((dq, dq[-1]))
        return dq
