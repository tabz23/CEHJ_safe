"""CuRobo IK / motion controller for a RoboTwin Env.

Call `CuroboIKController.install()` (or `ResidualController.install()`) before
`Env(...)` so `Robot.set_planner` loads CuRobo (mplib RRT is a plan fallback).
The default eval controller is `ResidualController`.

Switching / latency probes:
  vanilla_play_once       — one plan per Action, play the full path
  plan_play_once_everyk   — receding horizon: plan, execute K steps, replan
"""

from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np

from main.timing import EpisodeClock, cuda_sync, log_time

from .env import Env, prepare
from .record import grab_window_frame, save_video

ResidualFn = Callable[..., np.ndarray]

CONTROLLER_NAMES = (
    "residual",
    "nominal",
    "vanilla_play_once",
    "plan_play_once_everyk",
    "vanilla_tick",
)


class CuroboIKController:
    """Nominal controller: CuRobo IK + motion gen to a gripper pose.

    Subclass and override `plan()` to change the joint path. Call `attach()`
    so `task.play_once()` uses this controller instead of the robot planners
    directly.
    """

    _curobo_cache: dict = {}
    _session_warmup_s = 0.0
    _session_reuse_s = 0.0
    _session_warmup_n = 0
    _session_reuse_n = 0

    @classmethod
    def reset_session_timers(cls) -> None:
        cls._session_warmup_s = 0.0
        cls._session_reuse_s = 0.0
        cls._session_warmup_n = 0
        cls._session_reuse_n = 0

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

    controller_name = "nominal"

    def __init__(self, env: Env):
        self.env = env
        self.robot = env.robot
        self._left_plan = env.robot.left_plan_path
        self._right_plan = env.robot.right_plan_path
        self._n_plan = 0
        self.clock: EpisodeClock | None = None
        self._count_next_plan_as_replan = False

    @staticmethod
    def install() -> None:
        prepare()
        from envs.robot.planner import CuroboPlanner, MplibPlanner
        from envs.robot.robot import Robot

        from .obstacle import patch_curobo_collision_cache

        patch_curobo_collision_cache()
        if getattr(Robot.set_planner, "_cehj_installed", False):
            return

        orig_set = Robot.set_planner
        orig_curobo_init = CuroboPlanner.__init__

        def _cached_curobo_init(self, robot_origion_pose, active_joints_name, all_joints, yml_path=None):
            t_init = time.perf_counter()
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
                dt = time.perf_counter() - t_init
                CuroboIKController._session_reuse_s += dt
                CuroboIKController._session_reuse_n += 1
                log_time(
                    f"curobo REUSE {Path(str(yml_path)).name}",
                    dt,
                    n_reuse=CuroboIKController._session_reuse_n,
                )
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
            dt = time.perf_counter() - t_init
            CuroboIKController._session_warmup_s += dt
            CuroboIKController._session_warmup_n += 1
            log_time(
                f"curobo WARMUP {Path(str(yml_path)).name}",
                dt,
                n_warmup=CuroboIKController._session_warmup_n,
            )

        CuroboPlanner.__init__ = _cached_curobo_init

        def _mplib_plan_batch(planner, curr_joint_pos, target_gripper_pose_list,
                              constraint_pose=None, arms_tag=None):
            """plan_batch shim for the mplib fallback: CuroboPlanner has it,
            MplibPlanner does not (upstream RoboTwin never added it), so the
            fallback crashed any multi-pose call (choose_best_pose) with
            AttributeError. Loops plan_path and assembles the same
            {status, position, velocity} arrays."""
            import numpy as np

            n = len(target_gripper_pose_list)
            status = np.array(["Failure"] * n, dtype=object)
            positions, velocities = [], []
            for i, pose in enumerate(target_gripper_pose_list):
                try:
                    r = planner.plan_path(curr_joint_pos, pose,
                                          arms_tag=arms_tag, log=False)
                except Exception:
                    r = {"status": "Fail"}
                if r.get("status") == "Success":
                    status[i] = "Success"
                    positions.append(np.asarray(r["position"]))
                    velocities.append(np.asarray(
                        r.get("velocity", np.zeros_like(np.asarray(r["position"])))
                    ))
            out = {"status": status}
            if positions:
                # mplib trajectories are variable-length; pad to the longest
                # (repeat the final waypoint) to match cuRobo's (n, m, l)
                positions = [
                    np.concatenate(
                        [p, np.repeat(p[-1:], max_len - len(p), axis=0)]
                    ) if len(p) < max_len else p
                    for p in positions
                ] if (max_len := max(len(p) for p in positions)) else positions
                velocities = [
                    np.concatenate(
                        [v, np.repeat(v[-1:], max_len - len(v), axis=0)]
                    ) if len(v) < max_len else v
                    for v in velocities
                ]
                out["position"] = np.stack(positions)
                out["velocity"] = np.stack(velocities)
            return out

        def _mplib_pair(robot, scene=None):
            import types

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
            for p in (robot.left_mplib_planner, robot.right_mplib_planner):
                p.plan_batch = types.MethodType(_mplib_plan_batch, p)
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
        _install_skill_tracker(self.env.task)
        _install_plan_end_logger(self)

    def plan(self, arm: str, target_pose, **kwargs):
        """IK + collision-aware path to a gripper pose `[x,y,z, qw,qx,qy,qz]`."""
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        is_replan = bool(self._count_next_plan_as_replan)
        clock = self.clock
        if clock is not None:
            cuda_sync()
        t0 = time.perf_counter()
        try:
            result = self._plan_body(arm, target_pose, **kwargs)
        finally:
            if clock is not None:
                cuda_sync()
        dt = time.perf_counter() - t0
        ok = bool(result) and result.get("status") == "Success"
        n_wp = 0 if not ok else len(np.asarray(result.get("position", [])))
        ee_err = _ee_err(self.robot, arm, target_pose)
        tgt = _as_pose_list(target_pose)
        targets = getattr(self.env.task, "_cehj_ee_target", None)
        if not isinstance(targets, dict):
            targets = {}
            self.env.task._cehj_ee_target = targets
        targets[arm] = tgt
        if clock is not None:
            clock.add_plan(
                dt,
                arm=arm,
                ok=ok,
                n_wp=n_wp,
                is_replan=is_replan,
                status=None if not result else result.get("status"),
                ee_err=ee_err,
                constraint_dropped=bool(kwargs.get("_cehj_constraint_dropped")),
                skill=current_skill(self.env.task),
                constrained=kwargs.get("constraint_pose") is not None
                and not kwargs.get("_cehj_constraint_dropped"),
            )
        if isinstance(result, dict):
            result["_cehj_ee_err"] = ee_err
        return result

    def _plan_body(self, arm: str, target_pose, **kwargs):
        self._n_plan += 1
        stats = self._plan_stats(arm, target_pose, **kwargs)
        nominal = self._left_plan if arm == "left" else self._right_plan
        plan_kwargs = {k: v for k, v in kwargs.items() if not str(k).startswith("_cehj")}
        result = nominal(target_pose, **plan_kwargs)
        ok = bool(result) and result.get("status") == "Success"
        n_wp = 0 if not ok else len(result.get("position", []))
        print(
            f"[controller] plan #{self._n_plan} {arm} "
            f"curobo/rrt={'ok' if ok else result.get('status') if result else None} "
            f"n_wp={n_wp} {stats}"
        )
        if ok:
            return result
        result = self._plan_screw(arm, target_pose, **plan_kwargs)
        if result and result.get("status") == "Success":
            print(f"[controller] plan #{self._n_plan} {arm} screw fallback ok")
            return result
        lifted = list(target_pose)
        lifted[2] += 0.08
        result = self._plan_screw(arm, lifted, **plan_kwargs)
        if result and result.get("status") == "Success":
            print(f"[controller] plan #{self._n_plan} {arm} world-Z fallback ok")
        else:
            print(f"[controller] plan #{self._n_plan} {arm} all fallbacks failed")
        return result

    def _plan_stats(self, arm: str, target_pose, **kwargs) -> str:
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

    controller_name = "residual"

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


def _as_pose_list(pose) -> list | None:
    if pose is None:
        return None
    if hasattr(pose, "p") and hasattr(pose, "q"):
        return list(np.asarray(pose.p, dtype=np.float64)) + list(np.asarray(pose.q, dtype=np.float64))
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    return arr.tolist()


def _ee_err(robot, arm: str, target_pose) -> float | None:
    tgt = _as_pose_list(target_pose)
    if tgt is None or len(tgt) < 3:
        return None
    cur = robot.get_left_ee_pose() if arm == "left" else robot.get_right_ee_pose()
    cur = np.asarray(cur, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(cur[:3] - np.asarray(tgt[:3], dtype=np.float64)))


def _plan_ok(result) -> bool:
    return bool(result) and result.get("status") == "Success" and result.get("position") is not None


def _n_wp(result) -> int:
    if not _plan_ok(result):
        return 0
    return int(np.asarray(result["position"]).shape[0])


def _has_tag(result) -> bool:
    return isinstance(result, dict) and result.get("_cehj_target_pose") is not None


def _tag_path(result, arm: str, pose, constraint_pose):
    if not result:
        return result
    out = dict(result)
    out["_cehj_arm"] = arm
    out["_cehj_target_pose"] = _as_pose_list(pose)
    out["_cehj_constraint_pose"] = constraint_pose
    if out.get("velocity") is None and out.get("position") is not None:
        pos = np.asarray(out["position"], dtype=np.float64)
        out["velocity"] = np.zeros_like(pos)
    return out


def current_skill(task) -> str:
    return str(getattr(task, "_cehj_skill", "") or "")


def _actor_label(task, actor) -> str:
    if actor is None:
        return "obj"
    for key, val in vars(task).items():
        if val is actor and not str(key).startswith("_"):
            return str(key)
    for meth in ("get_name", "get_key"):
        fn = getattr(actor, meth, None)
        if callable(fn):
            try:
                name = fn()
            except Exception:
                name = None
            if name:
                return str(name)
    return "obj"


def _stamp_skill(actions, labels) -> None:
    if not actions:
        return
    for action, label in zip(actions, labels):
        if action is None:
            continue
        args = getattr(action, "args", None)
        if not isinstance(args, dict):
            action.args = {"cehj_skill": label}
        else:
            args["cehj_skill"] = label


def _skill_of(action, side: str) -> str | None:
    if action is None:
        return None
    args = getattr(action, "args", None) or {}
    tagged = args.get("cehj_skill")
    if tagged:
        return str(tagged)
    kind = getattr(action, "action", None) or "act"
    if kind == "move" and args.get("constraint_pose") is not None:
        return f"{side} constrained-move"
    return f"{side} {kind}"


def _format_skill_pair(left, right) -> str:
    parts = [p for p in (_skill_of(left, "L"), _skill_of(right, "R")) if p]
    return " | ".join(parts) if parts else ""


def _move_action_lists(actions_by_arm1, actions_by_arm2):
    def get_actions(pair, arm_tag: str):
        if pair is None:
            return []
        if pair[1] is None:
            return pair[0][1] if pair[0][0] == arm_tag else []
        if pair[0][0] == arm_tag:
            return pair[0][1]
        return pair[1][1]

    actions = [actions_by_arm1, actions_by_arm2]
    left = list(get_actions(actions, "left"))
    right = list(get_actions(actions, "right"))
    n = max(len(left), len(right))
    left += [None] * (n - len(left))
    right += [None] * (n - len(right))
    return list(zip(left, right))


def _install_skill_tracker(task) -> None:
    """Stamp play_once Actions and keep ``task._cehj_skill`` in sync while they run."""
    if getattr(task, "_cehj_skill_attached", False):
        return
    task._cehj_skill = ""
    orig_grasp = task.grasp_actor
    orig_place = task.place_actor
    orig_disp = task.move_by_displacement
    orig_home = task.back_to_origin
    orig_to_pose = task.move_to_pose
    orig_close = task.close_gripper
    orig_open = task.open_gripper
    orig_move = task.move

    def grasp_actor(actor, arm_tag, pre_grasp_dis=0.1, grasp_dis=0, **kw):
        arm, actions = orig_grasp(
            actor, arm_tag, pre_grasp_dis=pre_grasp_dis, grasp_dis=grasp_dis, **kw
        )
        name = _actor_label(task, actor)
        side = str(arm or arm_tag)
        n = len(actions or [])
        if n >= 3:
            labels = [
                f"{side} pre_grasp {name}",
                f"{side} grasp {name}",
                f"{side} close {name}",
            ]
        elif n == 2:
            labels = [f"{side} grasp {name}", f"{side} close {name}"]
        else:
            labels = [f"{side} grasp {name}"] * n
        _stamp_skill(actions, labels)
        return arm, actions

    def place_actor(actor, arm_tag, target_pose, **kw):
        arm, actions = orig_place(actor, arm_tag, target_pose, **kw)
        name = _actor_label(task, actor)
        side = str(arm or arm_tag)
        n = len(actions or [])
        labels = [f"{side} pre_place {name}", f"{side} place {name}", f"{side} open {name}"]
        _stamp_skill(actions, labels[:n])
        return arm, actions

    def move_by_displacement(arm_tag, x=0.0, y=0.0, z=0.0, quat=None, move_axis="world"):
        arm, actions = orig_disp(
            arm_tag, x=x, y=y, z=z, quat=quat, move_axis=move_axis
        )
        side = str(arm or arm_tag)
        parts = []
        if abs(float(x)) > 1e-9:
            parts.append(f"dx={x:g}")
        if abs(float(y)) > 1e-9:
            parts.append(f"dy={y:g}")
        if abs(float(z)) > 1e-9:
            parts.append(f"dz={z:g}")
        delta = " ".join(parts) if parts else "0"
        _stamp_skill(actions, [f"{side} lift {delta}"])
        return arm, actions

    def back_to_origin(arm_tag):
        arm, actions = orig_home(arm_tag)
        _stamp_skill(actions, [f"{arm or arm_tag} home"])
        return arm, actions

    def move_to_pose(arm_tag, target_pose):
        arm, actions = orig_to_pose(arm_tag, target_pose)
        _stamp_skill(actions, [f"{arm or arm_tag} move_to_pose"])
        return arm, actions

    def close_gripper(arm_tag, pos=0.0):
        arm, actions = orig_close(arm_tag, pos=pos)
        _stamp_skill(actions, [f"{arm or arm_tag} close"])
        return arm, actions

    def open_gripper(arm_tag, pos=1.0):
        arm, actions = orig_open(arm_tag, pos=pos)
        _stamp_skill(actions, [f"{arm or arm_tag} open"])
        return arm, actions

    def move(actions_by_arm1, actions_by_arm2=None, save_freq=-1):
        pairs = _move_action_lists(actions_by_arm1, actions_by_arm2)
        idx = {"i": 0}

        orig_together = task.together_move_to_pose
        orig_take = task.take_dense_action
        orig_left = task.left_move_to_pose
        orig_right = task.right_move_to_pose

        def peek_skill():
            if idx["i"] < len(pairs):
                left, right = pairs[idx["i"]]
                task._cehj_skill = _format_skill_pair(left, right)

        def advance_skill():
            peek_skill()
            idx["i"] += 1

        def together(*args, **kwargs):
            advance_skill()
            return orig_together(*args, **kwargs)

        def take_dense(control_seq, save_freq_inner=-1):
            advance_skill()
            return orig_take(control_seq, save_freq_inner)

        def left_move(*args, **kwargs):
            peek_skill()
            return orig_left(*args, **kwargs)

        def right_move(*args, **kwargs):
            peek_skill()
            return orig_right(*args, **kwargs)

        task.together_move_to_pose = together
        task.take_dense_action = take_dense
        task.left_move_to_pose = left_move
        task.right_move_to_pose = right_move
        try:
            return orig_move(actions_by_arm1, actions_by_arm2, save_freq)
        finally:
            task.together_move_to_pose = orig_together
            task.take_dense_action = orig_take
            task.left_move_to_pose = orig_left
            task.right_move_to_pose = orig_right

    task.grasp_actor = grasp_actor
    task.place_actor = place_actor
    task.move_by_displacement = move_by_displacement
    task.back_to_origin = back_to_origin
    task.move_to_pose = move_to_pose
    task.close_gripper = close_gripper
    task.open_gripper = open_gripper
    task.move = move
    task._cehj_skill_attached = True


def _finish_open_arm(ctrl, arm: str, n_exec: int | None = None) -> None:
    clock = ctrl.clock
    if clock is None:
        return
    targets = getattr(ctrl.env.task, "_cehj_ee_target", None) or {}
    err = _ee_err(ctrl.robot, arm, targets.get(arm))
    clock.finish_arm(arm, ee_err_end=err, n_exec=n_exec)


def _finish_open_plans(ctrl, n_exec: int | None = None) -> None:
    clock = ctrl.clock
    if clock is None:
        return
    for arm in list(clock._open_plan):
        _finish_open_arm(ctrl, arm, n_exec=n_exec)


def _install_plan_end_logger(ctrl) -> None:
    """After vanilla plays a full spline, write ee_err_end / n_exec on that plan."""
    task = ctrl.env.task
    if getattr(task, "_cehj_plan_end_attached", False):
        return
    orig_together = task.together_move_to_pose
    orig_take = task.take_dense_action

    def together(*args, **kwargs):
        ret = orig_together(*args, **kwargs)
        _finish_open_plans(ctrl)
        return ret

    def take_dense(control_seq, save_freq=-1):
        ret = orig_take(control_seq, save_freq)
        _finish_open_plans(ctrl)
        return ret

    task.together_move_to_pose = together
    task.take_dense_action = take_dense
    task._cehj_plan_end_attached = True


def _path_arrays(result) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(result["position"], dtype=np.float64)
    vel = result.get("velocity")
    if vel is None:
        return pos, np.zeros_like(pos)
    vel = np.asarray(vel, dtype=np.float64)
    if vel.shape != pos.shape:
        return pos, np.zeros_like(pos)
    return pos, vel


class VanillaPlayOnceController(ResidualController):
    """Current expert: one CuRobo plan per Action, then play the whole path."""

    controller_name = "vanilla_play_once"


class PlanEveryKController(ResidualController):
    """Receding-horizon follower inside each play_once Action.

    Plan to the Action gripper pose, execute ``k`` joint steps, replan from the
    live ``qpos``, repeat until the pose is reached (or planning fails).
    """

    controller_name = "plan_play_once_everyk"

    def __init__(
        self,
        env: Env,
        k: int = 20,
        replan_max: int = 2500,
        ee_tol: float = 0.02,
    ):
        super().__init__(env)
        self.k = max(1, int(k))
        self.replan_max = max(1, int(replan_max))
        self.ee_tol = float(ee_tol)
        self.window_dir: Path | None = None
        self.window_draw_bbox = False
        self.window_fps = 1.0
        self.window_max = 20
        # 2000 physics steps → 10 clips; cap at window_max.
        self.window_stride_steps = 200
        self._window_i = 0
        self._window_frames: list = []
        self._steps_played = 0
        self._next_clip_at = 0
        self._window_cap_logged = False

    def attach(self) -> None:
        super().attach()
        task = self.env.task
        if getattr(task, "_cehj_mpc_attached", False):
            return
        orig_take = task.take_dense_action
        orig_left = task.left_move_to_pose
        orig_right = task.right_move_to_pose
        ctrl = self

        def left_move_to_pose(pose, constraint_pose=None, **kw):
            result = orig_left(pose, constraint_pose=constraint_pose, **kw)
            return _tag_path(result, "left", pose, constraint_pose)

        def right_move_to_pose(pose, constraint_pose=None, **kw):
            result = orig_right(pose, constraint_pose=constraint_pose, **kw)
            return _tag_path(result, "right", pose, constraint_pose)

        def take_dense_action(control_seq, save_freq=-1):
            return ctrl._mpc_take_dense_action(orig_take, control_seq, save_freq)

        def together_move_to_pose(
            left_target_pose,
            right_target_pose,
            left_constraint_pose=None,
            right_constraint_pose=None,
            **kw,
        ):
            return ctrl._mpc_together(
                left_target_pose,
                right_target_pose,
                left_constraint_pose,
                right_constraint_pose,
            )

        task.left_move_to_pose = left_move_to_pose
        task.right_move_to_pose = right_move_to_pose
        task.take_dense_action = take_dense_action
        task.together_move_to_pose = together_move_to_pose
        task._cehj_mpc_attached = True
        print(f"[controller] plan_play_once_everyk k={self.k} replan_max={self.replan_max}")

    def _after_chunk(self, arm: str, result, n_arm: int, n_chunk: int):
        if result is None:
            return None
        target = result.get("_cehj_target_pose")
        constraint = result.get("_cehj_constraint_pose")
        if target is None:
            return None
        err = _ee_err(self.robot, arm, target)
        at_goal = err is not None and err <= self.ee_tol
        close_enough = err is not None and err <= max(self.ee_tol * 2.0, 0.04)
        err_start = result.get("_cehj_ee_err")
        progress = None
        if err is not None and err_start is not None:
            progress = float(err_start) - float(err)
        # Leftover constrained grasp can be ~75 time samples that do not move
        # (~2.42 cm → ~2.43 cm). K=75/80 consume that spline and stop; K=70
        # never does and replans forever. End the Action after one no-progress
        # window once we are already within 4 cm.
        no_progress = progress is not None and progress < 0.002
        if at_goal or (n_arm <= n_chunk and close_enough) or (close_enough and no_progress):
            if close_enough and no_progress and not at_goal and n_arm > n_chunk:
                print(
                    f"[mpc] {arm} no-progress stop err={round(err, 4)} "
                    f"d={round(progress, 5)} n_wp={n_arm} k={n_chunk}"
                )
            return None
        if n_arm == 0:
            return False
        self._count_next_plan_as_replan = True
        new = self.plan(arm, target, constraint_pose=constraint)
        if not _plan_ok(new) and constraint is not None:
            print(f"[mpc] {arm} constrained replan failed; retry unconstrained")
            new = self.plan(arm, target, _cehj_constraint_dropped=True)
        self._count_next_plan_as_replan = False
        if not _plan_ok(new):
            return False
        tagged = _tag_path(new, arm, target, constraint)
        err = _ee_err(self.robot, arm, target)
        print(
            f"[mpc] replan {arm} k={self.k} n_wp={_n_wp(tagged)} "
            f"err={None if err is None else round(err, 4)}"
        )
        return tagged

    def _should_record_window(self) -> bool:
        """Save ~1 clip per ``window_stride_steps`` (2000 steps → 10 clips), cap at ``window_max``."""
        if self.window_dir is None:
            return False
        if self.window_max > 0 and self._window_i >= self.window_max:
            if not self._window_cap_logged:
                print(f"[mpc] window clips capped at {self.window_max}")
                self._window_cap_logged = True
            return False
        return int(self._steps_played) >= int(self._next_clip_at)

    def _play_seq(self, task, left_arm, right_arm, left_g, right_g, n_chunk: int, grip_i: int) -> None:
        record = self._should_record_window() and (left_arm is not None or right_arm is not None)
        frames: list = []
        for i in range(int(n_chunk)):
            if left_arm is not None and i < _n_wp(left_arm):
                pos, vel = _path_arrays(left_arm)
                task.robot.set_arm_joints(pos[i], vel[i], "left")
            if right_arm is not None and i < _n_wp(right_arm):
                pos, vel = _path_arrays(right_arm)
                task.robot.set_arm_joints(pos[i], vel[i], "right")
            gi = grip_i + i
            if left_g is not None and gi < int(left_g["num_step"]):
                task.robot.set_gripper(left_g["result"][gi], "left", left_g["per_step"])
            if right_g is not None and gi < int(right_g["num_step"]):
                task.robot.set_gripper(right_g["result"][gi], "right", right_g["per_step"])
            task.scene.step()
            if task.render_freq and i % task.render_freq == 0:
                task._update_render()
            if record:
                err_l = _ee_err(self.robot, "left", (left_arm or {}).get("_cehj_target_pose"))
                err_r = _ee_err(self.robot, "right", (right_arm or {}).get("_cehj_target_pose"))
                skill = current_skill(task) or "skill=?"
                t_grab = time.perf_counter()
                frames.append(
                    grab_window_frame(
                        self.env,
                        [
                            skill,
                            f"w={self._window_i + 1} step={i + 1}/{int(n_chunk)} k={self.k}",
                            f"errL={None if err_l is None else round(err_l, 4)} "
                            f"errR={None if err_r is None else round(err_r, 4)}",
                        ],
                        draw_bbox=self.window_draw_bbox,
                    )
                )
                if self.clock is not None:
                    self.clock.add("window_grab", time.perf_counter() - t_grab)
        if left_arm is not None:
            _finish_open_arm(self, "left", n_exec=min(int(n_chunk), _n_wp(left_arm)))
        if right_arm is not None:
            _finish_open_arm(self, "right", n_exec=min(int(n_chunk), _n_wp(right_arm)))
        self._steps_played += int(n_chunk)
        if record:
            self._save_window_clip(frames, n_chunk)
            self._next_clip_at += max(1, int(self.window_stride_steps))

    def _save_window_clip(self, frames: list, n_chunk: int) -> None:
        if not frames or self.window_dir is None:
            return
        self._window_i += 1
        path = (
            self.window_dir
            / f"w{self._window_i:04d}_step{self._steps_played}_k{self.k}_n{int(n_chunk)}.mp4"
        )
        t0 = time.perf_counter()
        try:
            save_video(frames, path, self.window_fps)
        except Exception as exc:
            print(f"[mpc] window clip failed: {exc}")
            return
        dt = time.perf_counter() - t0
        if self.clock is not None:
            self.clock.add("window_encode", dt)
            self.clock.n_window_clips += 1
            log_time(
                f"window_encode w{self._window_i:04d}",
                dt,
                n_frames=len(frames),
                k=self.k,
                cum_encode_s=round(self.clock.t_window_encode, 3),
                cum_grab_s=round(self.clock.t_window_grab, 3),
            )

    def _mpc_take_dense_action(self, orig, control_seq, save_freq=-1):
        left_arm = control_seq.get("left_arm")
        right_arm = control_seq.get("right_arm")
        left_g = control_seq.get("left_gripper")
        right_g = control_seq.get("right_gripper")
        if not _has_tag(left_arm) and not _has_tag(right_arm):
            return orig(control_seq, save_freq)

        task = self.env.task
        grip_i = 0
        windows = 0
        while task.plan_success:
            windows += 1
            if windows > self.replan_max:
                print("[mpc] take_dense_action hit replan_max")
                task.plan_success = False
                return False
            n_l = _n_wp(left_arm) if _has_tag(left_arm) else 0
            n_r = _n_wp(right_arm) if _has_tag(right_arm) else 0
            n_lg = 0 if left_g is None else max(0, int(left_g["num_step"]) - grip_i)
            n_rg = 0 if right_g is None else max(0, int(right_g["num_step"]) - grip_i)
            if n_l == 0 and n_r == 0:
                n_chunk = max(n_lg, n_rg)
                if n_chunk <= 0:
                    break
            else:
                n_chunk = min(self.k, max(n_l, n_r, 1))
            self._play_seq(task, left_arm if n_l else None, right_arm if n_r else None, left_g, right_g, n_chunk, grip_i)
            grip_i += n_chunk
            if self.clock is not None:
                self.clock.n_mpc_windows += 1
            if n_l == 0 and n_r == 0:
                continue
            if _has_tag(left_arm) and n_l > 0:
                nxt = self._after_chunk("left", left_arm, n_l, n_chunk)
                if nxt is False:
                    task.plan_success = False
                    return False
                left_arm = nxt
            else:
                left_arm = None
            if _has_tag(right_arm) and n_r > 0:
                nxt = self._after_chunk("right", right_arm, n_r, n_chunk)
                if nxt is False:
                    task.plan_success = False
                    return False
                right_arm = nxt
            else:
                right_arm = None
        return True

    def _mpc_together(
        self,
        left_target_pose,
        right_target_pose,
        left_constraint_pose=None,
        right_constraint_pose=None,
    ):
        task = self.env.task
        if not task.plan_success:
            return
        left_target = _as_pose_list(left_target_pose)
        right_target = _as_pose_list(right_target_pose)
        if left_target is None or right_target is None:
            task.plan_success = False
            return
        self._count_next_plan_as_replan = False
        left = _tag_path(
            self.plan("left", left_target, constraint_pose=left_constraint_pose),
            "left",
            left_target,
            left_constraint_pose,
        )
        right = _tag_path(
            self.plan("right", right_target, constraint_pose=right_constraint_pose),
            "right",
            right_target,
            right_constraint_pose,
        )
        if not _plan_ok(left) or not _plan_ok(right):
            task.plan_success = False
            return
        task.left_joint_path.append(deepcopy(left))
        task.right_joint_path.append(deepcopy(right))

        windows = 0
        while task.plan_success and (left is not None or right is not None):
            windows += 1
            if windows > self.replan_max:
                print("[mpc] together_move_to_pose hit replan_max")
                task.plan_success = False
                return
            n_l = _n_wp(left)
            n_r = _n_wp(right)
            n_chunk = min(self.k, max(n_l, n_r, 1))
            self._play_seq(task, left, right, None, None, n_chunk, 0)
            if self.clock is not None:
                self.clock.n_mpc_windows += 1
            if left is not None:
                nxt = self._after_chunk("left", left, n_l, n_chunk)
                if nxt is False:
                    task.plan_success = False
                    return
                left = nxt
            if right is not None:
                nxt = self._after_chunk("right", right, n_r, n_chunk)
                if nxt is False:
                    task.plan_success = False
                    return
                right = nxt


class TickChunkedController(ResidualController):
    """Vanilla nominal with per-control-tick chunking and a filter hook.

    Stock take_dense_action plays an entire cuRobo spline in one call. This
    controller walks the plan in fixed tick chunks (PHYSICS_FREQ /
    control_freq physics steps — exactly 10 at 25 Hz) and calls
    self.tick_hook(ctx) at every tick boundary:

      hook returns False -> play the next tick of plan rows
      hook returns True  -> an intervention just drove the arm; the
                            remaining waypoints are stale, so the plan is
                            DISCARDED and replanned from live qpos to the
                            same tagged target (forced _after_chunk
                            semantics). The gripper clock is carried, not
                            advanced, across the intervention.

    ctx carries the live plan dicts, the row cursor, and the gripper
    cursor so the hook can form a_nom = pos[min(i_end, T-1)] - qpos_now.
    """

    controller_name = "vanilla_tick"

    def __init__(self, env: Env, **kw):
        super().__init__(env, **kw)
        self.tick_hook = None  # RolloutController sets this
        self.n_interventions = 0

    @property
    def steps_per_tick(self) -> int:
        return int(round(self.env.PHYSICS_FREQ / self.env.control_freq))

    def attach(self) -> None:
        super().attach()  # plan() routing + skill tracker
        task = self.env.task
        if getattr(task, "_cehj_tick_attached", False):
            return
        orig_take = task.take_dense_action
        orig_left = task.left_move_to_pose
        orig_right = task.right_move_to_pose
        ctrl = self

        def left_move_to_pose(pose, constraint_pose=None, **kw):
            result = orig_left(pose, constraint_pose=constraint_pose, **kw)
            return _tag_path(result, "left", pose, constraint_pose)

        def right_move_to_pose(pose, constraint_pose=None, **kw):
            result = orig_right(pose, constraint_pose=constraint_pose, **kw)
            return _tag_path(result, "right", pose, constraint_pose)

        def take_dense_action(control_seq, save_freq=-1):
            return ctrl._tick_take_dense_action(orig_take, control_seq, save_freq)

        task.left_move_to_pose = left_move_to_pose
        task.right_move_to_pose = right_move_to_pose
        task.take_dense_action = take_dense_action
        task._cehj_tick_attached = True
        print("[controller] vanilla_tick: tick-chunked, replan on intervention")

    def _replan_arm(self, arm: str, plan: dict):
        """Discard the stale plan; replan the same target from live qpos."""
        target = plan.get("_cehj_target_pose")
        constraint = plan.get("_cehj_constraint_pose")
        self._count_next_plan_as_replan = True
        new = self.plan(arm, target, constraint_pose=constraint)
        self._count_next_plan_as_replan = False
        if not _plan_ok(new) and constraint is not None:
            new = self.plan(arm, target)
        if not _plan_ok(new):
            return None
        return _tag_path(new, arm, target, constraint)

    def _tick_take_dense_action(self, orig, control_seq, save_freq=-1):
        task = self.env.task
        left_arm = control_seq.get("left_arm")
        right_arm = control_seq.get("right_arm")
        left_g = control_seq.get("left_gripper")
        right_g = control_seq.get("right_gripper")
        if not _has_tag(left_arm) and not _has_tag(right_arm):
            return orig(control_seq, save_freq)

        spt = self.steps_per_tick
        i = 0    # physics-step cursor into the arm plans
        g_i = 0  # gripper row cursor (frozen during interventions)
        needs_replan = False  # set by interventions; paid only on release
        while task.plan_success:
            n_l = _n_wp(left_arm) if left_arm is not None else 0
            n_r = _n_wp(right_arm) if right_arm is not None else 0
            n_lg = 0 if left_g is None else max(0, int(left_g["num_step"]) - g_i)
            n_rg = 0 if right_g is None else max(0, int(right_g["num_step"]) - g_i)
            if i >= max(n_l, n_r) and max(n_lg, n_rg) <= 0:
                break

            if self.tick_hook is not None and (n_l > 0 or n_r > 0):
                ctx = {
                    "left_arm": left_arm, "right_arm": right_arm, "i": i,
                    "left_g": left_g, "right_g": right_g, "g_i": g_i,
                }
                if self.tick_hook(ctx):
                    # intervention: the actor drove this block. Do NOT replan
                    # yet — the release test next tick scores a_nom from the
                    # (stale) existing plan, and a cuRobo plan is ~100 ms of
                    # wall time; pay it only when control actually hands back.
                    self.n_interventions += 1
                    needs_replan = True
                    continue
                if needs_replan:
                    # release: discard the stale plans, replan from live qpos
                    if left_arm is not None:
                        left_arm = self._replan_arm("left", left_arm)
                    if right_arm is not None:
                        right_arm = self._replan_arm("right", right_arm)
                    if (control_seq.get("left_arm") is not None and left_arm is None) or (
                        control_seq.get("right_arm") is not None and right_arm is None
                    ):
                        task.plan_success = False
                        return False
                    i = 0  # the new plan starts at its own row 0
                    needs_replan = False
                    continue

            pos_l = vel_l = pos_r = vel_r = None
            if left_arm is not None:
                pos_l, vel_l = _path_arrays(left_arm)
            if right_arm is not None:
                pos_r, vel_r = _path_arrays(right_arm)
            for k in range(spt):
                row = i + k
                if pos_l is not None and row < n_l:
                    task.robot.set_arm_joints(pos_l[row], vel_l[row], "left")
                if pos_r is not None and row < n_r:
                    task.robot.set_arm_joints(pos_r[row], vel_r[row], "right")
                if left_g is not None and g_i + k < int(left_g["num_step"]):
                    task.robot.set_gripper(
                        left_g["result"][g_i + k], "left", left_g["per_step"]
                    )
                if right_g is not None and g_i + k < int(right_g["num_step"]):
                    task.robot.set_gripper(
                        right_g["result"][g_i + k], "right", right_g["per_step"]
                    )
                task.scene.step()
                if task.render_freq and row % task.render_freq == 0:
                    task._update_render()
            i += spt
            g_i += spt
        return True


def controller_class(name: str):
    return {
        "residual": ResidualController,
        "nominal": CuroboIKController,
        "vanilla_play_once": VanillaPlayOnceController,
        "plan_play_once_everyk": PlanEveryKController,
        "vanilla_tick": TickChunkedController,
    }[str(name)]


def make_controller(
    name: str,
    env: Env,
    *,
    replan_k: int = 20,
    replan_max: int = 2500,
):
    name = str(name)
    if name == "vanilla_play_once":
        return VanillaPlayOnceController(env)
    if name == "vanilla_tick":
        return TickChunkedController(env)
    if name == "plan_play_once_everyk":
        return PlanEveryKController(env, k=replan_k, replan_max=replan_max)
    if name == "nominal":
        return CuroboIKController(env)
    return ResidualController(env)
