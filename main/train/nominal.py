"""Stepwise obstacle-agnostic nominal controller.

The nominal is the task expert's own path, executed stepwise at control
rate. The expert's path comes from cuRobo planning with a table-only world
— obstacle-agnostic by construction, so the safety filter has something to
do and comparisons downstream aren't confounded.

probe(): run the expert (play_once) on a fresh env with the same seed and
record its drive-target trajectory at control rate. This is done once per
(task, embodiment, seed) and cached to disk.

goal(t): the recorded target qpos [14] ([L6, gripL, R6, gripR], gripper
normalized). Collection composes dtheta = clip(goal - theta_measured,
±dtheta_max) — feedback toward the reference path, which also recovers
from injected perturbations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NominalTracker:
    def __init__(self, traj: np.ndarray, success: bool):
        self.traj = traj                      # [T, 14] targets at control rate
        self.success = bool(success)
        self.T = len(traj)

    def goal(self, t: int) -> np.ndarray:
        return self.traj[min(t, self.T - 1)]

    @classmethod
    def probe(cls, env, control_freq: float = 20.0) -> "NominalTracker":
        """Run the expert on `env`, recording drive targets per control step."""
        physics_freq = env.PHYSICS_FREQ
        traj = []
        orig_step = env.task.scene.step
        state = {"n": 0, "acc": 0.0}

        def step():
            orig_step()
            state["acc"] += physics_freq / control_freq
            state["n"] += 1
            if state["acc"] >= 1.0:
                state["acc"] -= 1.0
            # record at control boundaries: every time accumulated steps
            # cross the control interval
            if state["n"] >= round((len(traj) + 1) * physics_freq / control_freq):
                traj.append(
                    np.array(
                        env.robot.get_left_arm_jointState()
                        + env.robot.get_right_arm_jointState(),
                        dtype=np.float32,
                    )
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
        return cls(np.asarray(traj, dtype=np.float32), success)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.traj)
        path.with_suffix(".json").write_text(
            json.dumps({"success": self.success, "T": self.T})
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NominalTracker":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(np.load(path.with_suffix(".npy")), meta["success"])

    @classmethod
    def cached(cls, path: str | Path, env, control_freq: float = 20.0) -> "NominalTracker":
        path = Path(path)
        if path.with_suffix(".npy").exists():
            return cls.load(path)
        tracker = cls.probe(env, control_freq)
        tracker.save(path)
        env.close()
        return tracker
