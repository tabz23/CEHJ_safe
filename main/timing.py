"""Per-episode wall-clock split with stdout [time] lines for every cost."""

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


def log_time(label: str, dt_s: float, **kv: Any) -> None:
    extra = "  ".join(f"{k}={v}" for k, v in kv.items() if v is not None and v != "")
    line = f"[time] {label}  {dt_s * 1000.0:.1f}ms"
    if extra:
        line = f"{line}  {extra}"
    print(line, flush=True)


class EpisodeClock:
    """Accumulate seconds in named buckets. Start with :meth:`start` around play_once."""

    _PLAY_BUCKETS = (
        "plan",
        "physics",
        "render",
        "metric",
        "window_grab",
        "window_encode",
    )

    def __init__(self) -> None:
        self.t_plan = 0.0
        self.t_plan_initial = 0.0
        self.t_plan_replan = 0.0
        self.t_physics = 0.0
        self.t_render = 0.0
        self.t_metric = 0.0
        self.t_window_grab = 0.0
        self.t_window_encode = 0.0
        self.t_video_encode = 0.0
        self.t_setup_env = 0.0
        self.t_curobo_warmup = 0.0
        self.t_curobo_reuse = 0.0
        self.t_spawn = 0.0
        self.t_world = 0.0
        self.t_check_success = 0.0
        self.t_hold_trace = 0.0
        self.t0: float | None = None
        self.t1: float | None = None
        self.n_plan = 0
        self.n_plan_ok = 0
        self.n_plan_fail = 0
        self.n_replan = 0
        self.n_plan_initial = 0
        self.n_mpc_windows = 0
        self.n_window_clips = 0
        self.n_curobo_warmup = 0
        self.n_curobo_reuse = 0
        self.plan_events: list[dict[str, Any]] = []
        self._open_plan: dict[str, int] = {}
        self._scene_wrapped = False
        self._orig_scene_step = None

    def start(self) -> None:
        self.t0 = time.perf_counter()
        print("[time] play_once start", flush=True)

    def stop(self) -> None:
        self.t1 = time.perf_counter()
        print(
            f"[time] play_once done  {self.t_episode:.3f}s  "
            f"plan={self.t_plan:.3f}s (first={self.t_plan_initial:.3f}s "
            f"n={self.n_plan_initial}  replan={self.t_plan_replan:.3f}s n={self.n_replan})  "
            f"phys={self.t_physics:.3f}s  cam={self.t_render:.3f}s  "
            f"dist={self.t_metric:.3f}s  win_grab={self.t_window_grab:.3f}s  "
            f"win_encode={self.t_window_encode:.3f}s  other={self.t_other:.3f}s",
            flush=True,
        )

    @property
    def t_episode(self) -> float:
        t0 = self.t0
        t1 = self.t1 if self.t1 is not None else time.perf_counter()
        if t0 is None:
            return 0.0
        return max(0.0, t1 - t0)

    @property
    def t_other(self) -> float:
        used = sum(getattr(self, f"t_{name}") for name in self._PLAY_BUCKETS)
        return max(0.0, self.t_episode - used)

    def add(self, bucket: str, dt_s: float) -> None:
        attr = f"t_{bucket}"
        setattr(self, attr, float(getattr(self, attr, 0.0)) + float(dt_s))

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
            self.add(bucket, time.perf_counter() - t0)

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
        dt = float(dt_s)
        self.t_plan += dt
        if is_replan:
            self.n_replan += 1
            self.t_plan_replan += dt
            kind = "REPLAN"
        else:
            self.n_plan_initial += 1
            self.t_plan_initial += dt
            kind = "FIRST "
        mean_ms = (self.t_plan / self.n_plan) * 1000.0 if self.n_plan else 0.0
        err_s = "" if ee_err is None else f"{ee_err:.4f}"
        log_time(
            f"plan #{self.n_plan} {kind} {arm}",
            dt,
            n_wp=n_wp,
            ok=ok,
            status=None if status is None else str(status),
            err=err_s,
            constrained=bool(constrained),
            skill=skill or "",
            cum_plan_s=round(self.t_plan, 3),
            cum_first_s=round(self.t_plan_initial, 3),
            cum_replan_s=round(self.t_plan_replan, 3),
            n_first=self.n_plan_initial,
            n_replan=self.n_replan,
            mean_ms=round(mean_ms, 1),
        )
        event = {
            "i": self.n_plan,
            "arm": arm,
            "ok": bool(ok),
            "status": None if status is None else str(status),
            "n_wp": int(n_wp),
            "is_replan": bool(is_replan),
            "t_plan_ms": round(dt * 1000.0, 3),
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

    def dump(self, extra: dict[str, Any] | None = None) -> None:
        """Print a full split. Setup/video_encode sit outside play_once."""
        print("[time] ---- split ----", flush=True)
        rows = [
            ("setup env+robot", self.t_setup_env),
            ("CuRobo MotionGen warmup (cold init)", self.t_curobo_warmup),
            ("CuRobo MotionGen reuse (cache hit)", self.t_curobo_reuse),
            ("spawn obstacle", self.t_spawn),
            ("update CuRobo world", self.t_world),
            ("play_once wall", self.t_episode),
            ("  CuRobo FIRST plans", self.t_plan_initial),
            ("  CuRobo REPLAN", self.t_plan_replan),
            ("  CuRobo all plans", self.t_plan),
            ("  PhysX scene.step", self.t_physics),
            ("  episode cameras (agent/wrist/bbox)", self.t_render),
            ("  distance_info every step", self.t_metric),
            ("  MPC window frame grab", self.t_window_grab),
            ("  MPC window mp4 encode", self.t_window_encode),
            ("  other inside play_once", self.t_other),
            ("episode mp4 encode (after play_once)", self.t_video_encode),
            ("check_success", self.t_check_success),
            ("hold_trace.csv", self.t_hold_trace),
        ]
        for name, val in rows:
            print(f"[time]   {name:<42} {val:8.3f}s", flush=True)
        if extra:
            for key, val in extra.items():
                print(f"[time]   {key:<42} {val}", flush=True)
        print("[time] ---- end ----", flush=True)

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
        init_mean = (
            self.t_plan_initial / self.n_plan_initial if self.n_plan_initial else None
        )
        replan_mean = self.t_plan_replan / self.n_replan if self.n_replan else None
        tot = self.t_episode

        def _frac(part: float) -> float | None:
            if tot <= 1e-12:
                return None
            return float(part / tot)

        return {
            "t_episode_s": round(self.t_episode, 4),
            "t_plan_s": round(self.t_plan, 4),
            "t_plan_initial_s": round(self.t_plan_initial, 4),
            "t_plan_replan_s": round(self.t_plan_replan, 4),
            "t_plan_initial_mean_s": None if init_mean is None else round(init_mean, 4),
            "t_plan_replan_mean_s": None if replan_mean is None else round(replan_mean, 4),
            "t_physics_s": round(self.t_physics, 4),
            "t_render_s": round(self.t_render, 4),
            "t_metric_s": round(self.t_metric, 4),
            "t_window_grab_s": round(self.t_window_grab, 4),
            "t_window_encode_s": round(self.t_window_encode, 4),
            "t_video_encode_s": round(self.t_video_encode, 4),
            "t_setup_env_s": round(self.t_setup_env, 4),
            "t_curobo_warmup_s": round(self.t_curobo_warmup, 4),
            "t_curobo_reuse_s": round(self.t_curobo_reuse, 4),
            "t_spawn_s": round(self.t_spawn, 4),
            "t_world_s": round(self.t_world, 4),
            "t_check_success_s": round(self.t_check_success, 4),
            "t_hold_trace_s": round(self.t_hold_trace, 4),
            "t_other_s": round(self.t_other, 4),
            "t_plan_frac": _frac(self.t_plan),
            "t_physics_frac": _frac(self.t_physics),
            "t_render_frac": _frac(self.t_render),
            "t_metric_frac": _frac(self.t_metric),
            "t_other_frac": _frac(self.t_other),
            "n_plan": self.n_plan,
            "n_plan_ok": self.n_plan_ok,
            "n_plan_fail": self.n_plan_fail,
            "n_plan_initial": self.n_plan_initial,
            "n_replan": self.n_replan,
            "n_mpc_windows": self.n_mpc_windows,
            "n_window_clips": self.n_window_clips,
            "n_curobo_warmup": self.n_curobo_warmup,
            "n_curobo_reuse": self.n_curobo_reuse,
            "t_plan_first_s": None if first is None else round(first, 4),
            "t_plan_rest_mean_s": None if rest_mean is None else round(rest_mean, 4),
            "t_plan_rest_median_s": None if rest_median is None else round(rest_median, 4),
        }
