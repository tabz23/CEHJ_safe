"""Per-episode wall-clock split: planning vs physics vs render vs other."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class EpisodeClock:
    """Accumulate seconds in named buckets. Start with :meth:`start` around play_once."""

    def __init__(self) -> None:
        self.t_plan = 0.0
        self.t_physics = 0.0
        self.t_render = 0.0
        self.t_metric = 0.0
        self.t0: float | None = None
        self.t1: float | None = None
        self.n_plan = 0
        self.n_plan_ok = 0
        self.n_plan_fail = 0
        self.n_replan = 0
        self.n_mpc_windows = 0
        self.plan_events: list[dict[str, Any]] = []
        self._open_plan: dict[str, int] = {}
        self._scene_wrapped = False
        self._orig_scene_step = None

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def stop(self) -> None:
        self.t1 = time.perf_counter()

    @property
    def t_episode(self) -> float:
        t0 = self.t0
        t1 = self.t1 if self.t1 is not None else time.perf_counter()
        if t0 is None:
            return 0.0
        return max(0.0, t1 - t0)

    @property
    def t_other(self) -> float:
        used = self.t_plan + self.t_physics + self.t_render
        return max(0.0, self.t_episode - used)

    @contextmanager
    def span(self, bucket: str, *, cuda: bool = False):
        if cuda:
            cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                cuda_sync()
            dt = time.perf_counter() - t0
            if bucket == "plan":
                self.t_plan += dt
            elif bucket == "physics":
                self.t_physics += dt
            elif bucket == "render":
                self.t_render += dt
            elif bucket == "metric":
                self.t_metric += dt

    def wrap_scene(self, scene) -> None:
        if self._scene_wrapped:
            return
        orig = scene.step
        clock = self

        def step():
            t0 = time.perf_counter()
            try:
                orig()
            finally:
                clock.t_physics += time.perf_counter() - t0

        scene.step = step
        self._orig_scene_step = orig
        self._scene_wrapped = True

    def add_plan(
        self,
        dt_s: float,
        *,
        arm: str,
        ok: bool,
        n_wp: int,
        is_replan: bool,
        status: Any = None,
        ee_err: float | None = None,
        constraint_dropped: bool = False,
        n_exec: int | None = None,
        skill: str | None = None,
        constrained: bool = False,
    ) -> None:
        self.n_plan += 1
        if ok:
            self.n_plan_ok += 1
        else:
            self.n_plan_fail += 1
        if is_replan:
            self.n_replan += 1
        event = {
            "i": self.n_plan,
            "arm": arm,
            "ok": bool(ok),
            "status": None if status is None else str(status),
            "n_wp": int(n_wp),
            "is_replan": bool(is_replan),
            "t_plan_ms": round(float(dt_s) * 1000.0, 3),
            "ee_err": None if ee_err is None else float(ee_err),
            "ee_err_end": None,
            "skill": skill or "",
            "constrained": bool(constrained),
            "constraint_dropped": bool(constraint_dropped),
            "n_exec": n_exec,
        }
        self.plan_events.append(event)
        self._open_plan[str(arm)] = len(self.plan_events) - 1

    def finish_arm(
        self,
        arm: str,
        *,
        ee_err_end: float | None = None,
        n_exec: int | None = None,
    ) -> None:
        idx = self._open_plan.pop(str(arm), None)
        if idx is None or idx >= len(self.plan_events):
            return
        event = self.plan_events[idx]
        event["ee_err_end"] = None if ee_err_end is None else float(ee_err_end)
        if n_exec is None and event.get("n_exec") is None:
            event["n_exec"] = int(event.get("n_wp") or 0)
        elif n_exec is not None:
            event["n_exec"] = int(n_exec)

    def summary(self) -> dict[str, Any]:
        times = [float(e["t_plan_ms"]) / 1000.0 for e in self.plan_events]
        first = times[0] if times else None
        rest = times[1:]
        rest_mean = sum(rest) / len(rest) if rest else None
        rest_sorted = sorted(rest)
        rest_median = None
        if rest_sorted:
            mid = len(rest_sorted) // 2
            if len(rest_sorted) % 2:
                rest_median = rest_sorted[mid]
            else:
                rest_median = 0.5 * (rest_sorted[mid - 1] + rest_sorted[mid])
        tot = self.t_episode
        def _frac(part: float) -> float | None:
            if tot <= 1e-12:
                return None
            return float(part / tot)

        return {
            "t_episode_s": round(self.t_episode, 4),
            "t_plan_s": round(self.t_plan, 4),
            "t_physics_s": round(self.t_physics, 4),
            "t_render_s": round(self.t_render, 4),
            "t_metric_s": round(self.t_metric, 4),
            "t_other_s": round(self.t_other, 4),
            "t_plan_frac": _frac(self.t_plan),
            "t_physics_frac": _frac(self.t_physics),
            "t_render_frac": _frac(self.t_render),
            "t_other_frac": _frac(self.t_other),
            "n_plan": self.n_plan,
            "n_plan_ok": self.n_plan_ok,
            "n_plan_fail": self.n_plan_fail,
            "n_replan": self.n_replan,
            "n_mpc_windows": self.n_mpc_windows,
            "t_plan_first_s": None if first is None else round(first, 4),
            "t_plan_rest_mean_s": None if rest_mean is None else round(rest_mean, 4),
            "t_plan_rest_median_s": None if rest_median is None else round(rest_median, 4),
        }
