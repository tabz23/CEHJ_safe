"""Eval utilities for HJ-SAC: episode runner, h/V figure, panel video.

The per-eval figure is the central diagnostic: h(t) and V(t) on shared
axes, y=0 marked, the h - V gap shaded (that gap is the contribution over
a myopic distance function), intervention spans shaded. V <= h pointwise
is reported as a metric (a crossing means the Bellman/discount handling is
wrong — enforced as an assert only for trained runs, not smoke).

The video composites the observer frame with a white PIL panel showing
t / h / V / h-V / control mode / |dtheta| and a scrolling h-V trace —
watching the filter engage as h drops is the point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sys

CEHJ_ROOT = Path(__file__).resolve().parents[2]


def compute_trace_metrics(trace: dict, h_scale: float,
                          contact_force_threshold: float = 20.0) -> dict:
    """Per-episode metrics from a RolloutController trace (collection or
    eval). h/V are stored in h_scale (training) units; distances are
    reported in cm."""
    h_arr = np.array(trace["h"])
    V_arr = np.array(trace["V"])
    Vt_arr = np.array(trace["V_t"])
    cm = 100.0 / float(h_scale)
    if len(V_arr) and len(h_arr):
        h_at_v = np.interp(Vt_arr, np.array(trace["t"]), h_arr)
        v_le_h = float((V_arr <= h_at_v + 1e-6).mean())
        mean_gap = float((h_at_v - V_arr).mean()) * cm
    else:
        v_le_h, mean_gap = float("nan"), float("nan")
    force = np.array(trace.get("contact_force", []))
    if len(force) and len(force) == len(h_arr):
        f_viol = force >= contact_force_threshold
        violation_force = float(f_viol.mean())
        violation_any = float(((h_arr < 0) | f_viol).mean())
    else:
        violation_force = violation_any = float("nan")
    return {
        "violation_rate": float((h_arr < 0).mean()) if len(h_arr) else float("nan"),
        "violation_force": violation_force,
        "violation_any": violation_any,
        "contact_force_max": float(force.max()) if len(force) else float("nan"),
        "contact_force_mean_nz": (
            float(force[force > 0].mean())
            if len(force) and (force > 0).any() else 0.0
        ),
        "intervention_rate": float(trace.get("intervention_rate", 0.0)),
        "mode_switches": int(trace.get("mode_switches", 0)),
        "task_success": bool(trace["success"]),
        "realized_cmd_ratio": float(trace.get("mean_realized_ratio", float("nan"))),
        "min_h": float(h_arr.min()) * cm if len(h_arr) else float("nan"),
        "v_le_h_frac": v_le_h,
        "mean_gap": mean_gap,
    }


def run_eval_episode(cfg, seed, task, embodiment, encoder, policy_enc, actor,
                     critics, record_video=True, tag=""):
    """Run one filtered eval episode (nominal + safety filter, the Q(s,
    a_nom) trigger) on an explicit (task, embodiment, seed) scene via
    RolloutController; return trace dict.

    Note: NOT globally no_grad — cuRobo planning needs autograd. Model
    calls are wrapped in no_grad locally inside RolloutController.
    """
    import dataclasses

    from main.train.collect import CKPT_DIR, make_kinematics
    from main.train.rollout import RolloutController

    cfg_ep = dataclasses.replace(cfg, task=task, embodiment=embodiment)
    # same on/off-path draw as collection (deterministic per eval seed)
    if cfg_ep.randomize_scenes:
        rng = np.random.RandomState(seed)
        mode = "off_path" if rng.rand() < cfg_ep.off_path_frac else "on_path"
        cfg_ep = dataclasses.replace(cfg_ep, obstacle_mode=mode)
    kin = make_kinematics(CKPT_DIR, cfg_ep.embodiment)
    print(f"[eval] {cfg_ep.embodiment} / {cfg_ep.task} / "
          f"{cfg_ep.obstacle_model} / {cfg_ep.obstacle_mode} seed={seed}")

    video = None
    vid_dir = CEHJ_ROOT / "runs" / "videos"
    vid_name = (f"eval_filtered_{cfg_ep.embodiment}_{cfg_ep.task}"
                f"_s{seed}{tag}.mp4")
    if record_video:
        vid_dir.mkdir(parents=True, exist_ok=True)
        video = PanelVideoWriter(
            vid_dir / vid_name, fps=25.0,
            h_scale=float(getattr(cfg_ep, "h_scale", 1.0)),
        )  # 10 fps sim-time
    # total spawn guarantee (same as collect): with h = d_block only, an
    # obstacle-free episode yields h = +inf — corrupting the eval figure and
    # violation_rate — so re-draw the scene seed rather than run it
    ro = None
    used_seed = seed
    for retry in range(5):
        candidate = RolloutController(
            cfg_ep, seed + retry * 1000, mode="filtered", encoder=encoder,
            kin=kin, policy_enc=policy_enc, actor=actor, critics=critics,
            video=video if retry == 0 else video,
            max_steps=int(cfg_ep.max_steps_per_episode * 250.0 / 25.0),
        )
        if candidate.env.obstacle is not None:
            ro = candidate
            used_seed = seed + retry * 1000
            break
        print(f"[eval] spawn failed (seed {seed + retry * 1000}); re-drawing")
        candidate.env.close()
    if ro is None:
        video.close() if video is not None else None
        raise RuntimeError(f"eval spawn failed 5x for {task}/{embodiment}")
    trace = ro.run()
    trace["h_scale"] = float(getattr(cfg_ep, "h_scale", 1.0))
    trace["scene"] = {
        "task": cfg_ep.task, "embodiment": cfg_ep.embodiment,
        "obstacle": cfg_ep.obstacle_model, "seed": int(used_seed),
    }
    if video is not None:
        video.close()
        trace["video_path"] = str(vid_dir / vid_name)

    h_arr = np.array(trace["h"])
    trace["metrics"] = compute_trace_metrics(trace, trace["h_scale"])
    return trace


def save_eval_figure(trace, out_dir: Path, mode: str, step: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.array(trace["t"])
    # stored in h_scale (training) units; display in cm
    cm = 100.0 / float(trace.get("h_scale", 1.0))
    h = np.array(trace["h"]) * cm
    # V is evaluated at window boundaries (filter cadence), not per tick
    vt = np.array(trace.get("V_t", trace["t"][: len(trace["V"])]))
    V = np.array(trace["V"]) * cm
    interv = np.array(trace["intervened"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, h, label="h (clearance)", color="tab:blue", lw=1.5)
    ax.plot(vt, V, label="V (learned)", color="tab:red", lw=1.5, marker=".", ms=3)
    ax.axhline(0, color="k", lw=0.8)
    if len(V) and len(t):
        V_on_t = np.interp(t, vt, V)
        ax.fill_between(t, V_on_t, h, color="tab:green", alpha=0.15, label="h - V gap")
    if interv.any():
        ax.fill_between(vt, h.min(), h.max(), where=interv,
                        color="tab:orange", alpha=0.15, label="filter active")
    m = trace["metrics"]
    scene = trace.get("scene", {})
    scene_str = (f"{scene.get('embodiment', '?')}/{scene.get('task', '?')}"
                 f"/{scene.get('obstacle', '?')}")
    ax.set_title(
        f"{mode} {scene_str}  viol={m['violation_rate']:.2f}  "
        f"int={m['intervention_rate']:.2f}  "
        f"success={m['task_success']}  V<=h {m['v_le_h_frac']:.2f}",
        fontsize=9,
    )
    ax.set_xlabel("t (s)")
    ax.set_ylabel("h / V (cm)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"hv_{mode}_step{step}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


class PanelVideoWriter:
    """Observer frame + 320px white PIL panel -> 960x480 mp4."""

    def __init__(self, path: Path, fps: float = 5.0, h_scale: float = 1.0):
        import imageio.v2 as imageio

        # stored h/V are in h_scale (training) units; display in cm
        self.cm = 100.0 / h_scale

        self.writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", format="FFMPEG",
            pixelformat="yuv420p", macro_block_size=1,
        )
        self.hist_h, self.hist_V, self.hist_int = [], [], []

    def _panel(self, t, step, n_steps, h, V, control, ratio, extras=None):
        from PIL import Image, ImageDraw

        cm = self.cm
        h_cm = h * cm

        def _cm(val):
            if val is None or not np.isfinite(val):
                return "  inf "
            return f"{val * 100:5.1f}"

        W, H = 320, 480
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        color = (200, 60, 40) if control == "FILTER" else (60, 60, 60)
        lines = [
            f"t = {t:5.2f} s      step {step}/{n_steps}",
            f"h        = {h_cm:+.1f} cm",
        ]
        if V is not None and np.isfinite(V):
            lines += [
                f"Q(s,a_nom)= {V * cm:+.1f} cm",
                f"h - Q    = {(h - V) * cm:+.1f}",
            ]
        else:
            # warmup / nominal-only: no critic, nothing scored
            lines += ["Q(s,a_nom)=   -", "h - Q    =   -"]
        y = 16
        for line in lines:
            d.text((12, y), line, fill=(0, 0, 0))
            y += 22
        d.text((12, y), f"control  =  {control}", fill=color)
        y += 22
        # ratio: nominal's demanded step vs the actor bound on NOMINAL
        # ticks, the actor's own action on FILTER ticks
        d.text((12, y), f"|a| =  {ratio:.2f} x max", fill=(0, 0, 0))
        y += 22
        # cumulative intervention count (0 ticks on pure-nominal episodes)
        if extras:
            iv = extras.get("intervened")
            if iv is not None:
                d.text(
                    (12, y),
                    f"intervened: {iv[0]} ticks ({iv[1]:.2f} s)",
                    fill=(200, 60, 40) if iv[0] else (60, 60, 60),
                )
                y += 22
        block = extras.get("block") if extras else None
        if block is not None:
            d.text(
                (12, y),
                f"block {block[0]}/{block[1]}   engaged {block[2]:.2f} s",
                fill=(200, 60, 40),
            )
            y += 22
        if extras:
            d.text(
                (12, y),
                f"dL {_cm(extras.get('d_left'))}  dR {_cm(extras.get('d_right'))} cm",
                fill=(0, 0, 0),
            )
            y += 18
            d.text(
                (12, y),
                f"dLh {_cm(extras.get('d_left_held'))} dRh {_cm(extras.get('d_right_held'))} cm",
                fill=(0, 0, 0),
            )
            y += 18
            argmin = extras.get("true_argmin") or "-"
            d.text((12, y), f"argmin: {argmin}", fill=(0, 0, 0))
            y += 18
            # contact force to the obstacle (N); red above the collision
            # threshold (20 N — real contacts in sim are >= 79 N)
            force = extras.get("contact_force", 0.0) or 0.0
            f_color = (200, 60, 40) if force >= 20.0 else (60, 60, 60)
            d.text((12, y), f"force = {force:5.1f} N", fill=f_color)
            y += 18
            # HOLD line with provenance, same format as record.py's recorder
            dbg = extras.get("hold_debug") or {}
            hl = extras.get("holding_left") or "none"
            hr = extras.get("holding_right") or "none"
            sl = dbg.get("L_source") or "none"
            sr = dbg.get("R_source") or "none"
            hold_line = f"HOLD L={hl}({sl})  R={hr}({sr})"
            d.text((12, y), hold_line[:40], fill=(0, 0, 0))
            y += 18
            skill = extras.get("skill") or "-"
            d.text((12, y), skill[:38], fill=(80, 80, 80))
            y += 18

        # scrolling mini-plot of h and V (last ~5 s), in cm like the readout
        self.hist_h.append(h_cm)
        self.hist_V.append(V * cm if np.isfinite(V) else V)
        hist = self.hist_h[-100:]
        histV = self.hist_V[-100:]
        if len(hist) > 1:
            x0, y0, pw, ph = 12, y + 20, W - 24, 160
            d.rectangle([x0, y0, x0 + pw, y0 + ph], outline=(200, 200, 200))
            finV = [v for v in histV if np.isfinite(v)]
            lo = min([min(hist), -0.5] + ([min(finV)] if finV else []))
            hi = max([max(hist), 0.5] + ([max(finV)] if finV else []))

            def sy(val):
                return y0 + ph - (val - lo) / (hi - lo) * ph

            zero_y = sy(0.0)
            d.line([x0, zero_y, x0 + pw, zero_y], fill=(0, 0, 0), width=1)
            n = len(hist)
            pts_h = [(x0 + i / (n - 1) * pw, sy(v)) for i, v in enumerate(hist)]
            d.line(pts_h, fill=(30, 100, 200), width=2)
            # V may contain NaN (warmup: no critic) — draw finite segments
            seg = []
            for i, v in enumerate(histV):
                if np.isfinite(v):
                    seg.append((x0 + i / (n - 1) * pw, sy(v)))
                else:
                    if len(seg) > 1:
                        d.line(seg, fill=(220, 60, 50), width=2)
                    seg = []
            if len(seg) > 1:
                d.line(seg, fill=(220, 60, 50), width=2)
            d.text((x0 + 4, y0 + 4), "h", fill=(30, 100, 200))
            d.text((x0 + 20, y0 + 4), "V", fill=(220, 60, 50))
        return img

    def add(self, frame_rgb, t, step, n_steps, h, V, control, ratio, extras=None):
        panel = self._panel(t, step, n_steps, h, V, control, ratio, extras=extras)
        combo = np.hstack([frame_rgb, np.asarray(panel)])
        self.writer.append_data(combo)

    def close(self):
        self.writer.close()
