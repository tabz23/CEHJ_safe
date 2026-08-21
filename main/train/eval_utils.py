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
import torch

import sys

CEHJ_ROOT = Path(__file__).resolve().parents[2]

from main.network.body_features import BodyTokenExtractor  # noqa: E402
from main.envs.env import Env  # noqa: E402
from main.train.filter import SafetyFilter  # noqa: E402
from main.train.hfunc import compute_h  # noqa: E402
from main.train.nominal import NominalTracker  # noqa: E402
from main.envs.obstacle import choose_and_spawn  # noqa: E402


def run_eval_episode(cfg, seed, mode, encoder, kin, policy_enc, actor, critics,
                     record_video=True):
    """Run one eval episode ('nominal' or 'filtered'); return trace dict.

    Note: NOT globally no_grad — cuRobo planning (probe / nominal) needs
    autograd. Model calls are wrapped in no_grad locally instead.
    """
    probe_env = Env(cfg.task, cfg.embodiment, seed=seed, control_freq=20.0)
    tracker = NominalTracker.cached(
        CEHJ_ROOT / "nominal_cache" / f"{cfg.task}_{cfg.embodiment}_s{seed}",
        probe_env, control_freq=20.0,
    )
    env = Env(cfg.task, cfg.embodiment, seed=seed, control_freq=20.0)
    actor_ob, xyz, half, _ = choose_and_spawn(env, cfg.obstacle_mode, "geometric", 0.6, "auto")
    env.obstacle, env.obstacle_xyz, env.obstacle_half = actor_ob, xyz, half
    extractor = BodyTokenExtractor(env)
    dtheta_max = extractor.delta_theta_max.astype(np.float32)
    flt = SafetyFilter(
        actor, critics,
        body_joint_index(extractor).cuda(),
        torch.from_numpy(dtheta_max).cuda(),
        margin_on=cfg.margin_on, margin_off=cfg.margin_off,
    )

    trace = {"t": [], "h": [], "V": [], "intervened": [], "dtheta_ratio": [],
             "control": [], "frames": []}
    T = min(tracker.T, cfg.max_steps_per_episode)
    video = None
    if record_video:
        vid_dir = CEHJ_ROOT / "runs" / "videos"
        vid_dir.mkdir(parents=True, exist_ok=True)
        video = PanelVideoWriter(vid_dir / f"eval_{mode}_s{seed}.mp4", fps=10.0)
    for t in range(T):
        obs = env.get_encoder_obs()
        tokens, pos_mean, _ = encoder.encode_visual_tokens(
            obs["imgs"], obs["depths"], obs["image_wh"], obs["projection_mat"]
        )
        state = encoder.encode_joint_angles(obs["joint_state"], kin).squeeze(2)
        body = extractor.extract()

        goal = tracker.goal(t)
        theta = obs["joint_state"][0, 0].numpy()
        dtheta = np.zeros(16, dtype=np.float64)
        dtheta[0:6] = np.clip(goal[0:6] - theta[0:6], -dtheta_max[0:6], dtheta_max[0:6])
        dtheta[8:14] = np.clip(goal[7:13] - theta[7:13], -dtheta_max[8:14], dtheta_max[8:14])
        a_nom = torch.from_numpy(dtheta[None].astype(np.float32)).cuda()

        with torch.no_grad():
            enc = policy_enc(
                state, body["body_tokens"][None], body["link_pos"][None],
                body["body_mask"][None], body["joint_index"][None],
                tokens, pos_mean,
            )
            Jlin = body["Jlin"][None]
            Jang = body["Jang"][None]

            if mode == "filtered":
                action, V, intervened = flt.action(enc, Jlin, Jang, a_nom)
            else:
                dtheta_pi, _, _ = actor(enc.body, flt.joint_index, flt.dtheta_max,
                                        deterministic=True)
                V = critics.qmin(enc, dtheta_pi, Jlin, Jang, flt.joint_index)
                action, intervened = a_nom, False

        env.step_dtheta(action[0].cpu().numpy(), grip_left=goal[6], grip_right=goal[13])
        h, _ = compute_h(env, cfg.table_margin)

        v_float = float(V.item()) if hasattr(V, "item") else float(V)
        trace["t"].append(t * cfg.control_dt)
        trace["h"].append(float(h))
        trace["V"].append(v_float)
        trace["intervened"].append(bool(intervened))
        ratio = float(np.abs(action[0].cpu().numpy()).max() / dtheta_max.max())
        trace["dtheta_ratio"].append(ratio)
        trace["control"].append("FILTER" if intervened else "NOMINAL")
        if record_video and t % 2 == 0:
            cam = env.task.cameras.observer_camera
            cam.take_picture()
            rgba = cam.get_picture("Color")
            frame = (rgba * 255).clip(0, 255).astype(np.uint8)[:, :, :3]
            video.add(frame, t * cfg.control_dt, t, T, float(h), v_float,
                      trace["control"][-1], ratio)

    h_arr = np.array(trace["h"])
    succ = False
    try:
        succ = bool(env.task.check_success())
    except Exception:
        pass
    trace["metrics"] = {
        "violation_rate": float((h_arr < 0).mean()),
        "intervention_rate": flt.intervention_rate if mode == "filtered" else 0.0,
        "task_success": succ,
        "min_h": float(h_arr.min()),
        "v_le_h_frac": float((np.array(trace["V"]) <= h_arr + 1e-6).mean()),
        "mean_gap": float((h_arr - np.array(trace["V"])).mean()),
    }
    if video is not None:
        video.close()
        trace["video_path"] = str(vid_dir / f"eval_{mode}_s{seed}.mp4")
    env.close()
    return trace


def body_joint_index(extractor):
    import torch as _t

    return _t.from_numpy(extractor.extract()["joint_index"].numpy()[None])


def save_eval_figure(trace, out_dir: Path, mode: str, step: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.array(trace["t"])
    h = np.array(trace["h"])
    V = np.array(trace["V"])
    interv = np.array(trace["intervened"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, h, label="h (clearance)", color="tab:blue", lw=1.5)
    ax.plot(t, V, label="V (learned)", color="tab:red", lw=1.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.fill_between(t, V, h, color="tab:green", alpha=0.15, label="h - V gap")
    if interv.any():
        ax.fill_between(t, h.min(), h.max(), where=interv,
                        color="tab:orange", alpha=0.15, label="filter active")
    m = trace["metrics"]
    ax.set_title(
        f"{mode}  viol={m['violation_rate']:.2f}  int={m['intervention_rate']:.2f}  "
        f"success={m['task_success']}  V<=h {m['v_le_h_frac']:.2f}"
    )
    ax.set_xlabel("t (s)")
    ax.set_ylabel("metres")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"hv_{mode}_step{step}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


class PanelVideoWriter:
    """Observer frame + 320px white PIL panel -> 960x480 mp4."""

    def __init__(self, path: Path, fps: float = 10.0):
        import imageio.v2 as imageio

        self.writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", format="FFMPEG",
            pixelformat="yuv420p", macro_block_size=1,
        )
        self.hist_h, self.hist_V, self.hist_int = [], [], []

    def _panel(self, t, step, n_steps, h, V, control, ratio):
        from PIL import Image, ImageDraw

        W, H = 320, 480
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        color = (200, 60, 40) if control == "FILTER" else (60, 60, 60)
        lines = [
            f"t = {t:5.2f} s      step {step}/{n_steps}",
            f"h        = {h:+.3f} m",
            f"V        = {V:+.3f} m",
            f"h - V    = {h - V:+.3f}",
        ]
        y = 16
        for line in lines:
            d.text((12, y), line, fill=(0, 0, 0))
            y += 22
        d.text((12, y), f"control  =  {control}", fill=color)
        y += 22
        d.text((12, y), f"|dtheta| =  {ratio:.2f} x max", fill=(0, 0, 0))

        # scrolling mini-plot of h and V (last ~5 s)
        self.hist_h.append(h)
        self.hist_V.append(V)
        hist = self.hist_h[-100:]
        histV = self.hist_V[-100:]
        if len(hist) > 1:
            x0, y0, pw, ph = 12, y + 20, W - 24, 160
            d.rectangle([x0, y0, x0 + pw, y0 + ph], outline=(200, 200, 200))
            lo = min(min(hist), min(histV), -0.05)
            hi = max(max(hist), max(histV), 0.05)

            def sy(val):
                return y0 + ph - (val - lo) / (hi - lo) * ph

            zero_y = sy(0.0)
            d.line([x0, zero_y, x0 + pw, zero_y], fill=(0, 0, 0), width=1)
            n = len(hist)
            pts_h = [(x0 + i / (n - 1) * pw, sy(v)) for i, v in enumerate(hist)]
            pts_V = [(x0 + i / (n - 1) * pw, sy(v)) for i, v in enumerate(histV)]
            d.line(pts_h, fill=(30, 100, 200), width=2)
            d.line(pts_V, fill=(220, 60, 50), width=2)
            d.text((x0 + 4, y0 + 4), "h", fill=(30, 100, 200))
            d.text((x0 + 20, y0 + 4), "V", fill=(220, 60, 50))
        return img

    def add(self, frame_rgb, t, step, n_steps, h, V, control, ratio):
        panel = self._panel(t, step, n_steps, h, V, control, ratio)
        combo = np.hstack([frame_rgb, np.asarray(panel)])
        self.writer.append_data(combo)

    def close(self):
        self.writer.close()
