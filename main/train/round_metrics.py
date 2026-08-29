"""Round-level filter metrics, aggregated from COLLECTION episodes.

Replaces the standalone eval sweep during training: collection rounds
already run the deployment stack, so each filter-active episode's trace is
recorded here (nominal-only episodes are excluded — they say nothing about
the filter). Perturbation ticks are rare (perturb_prob) and included.

Outputs, per render:
  <run>/eval/<metric>/<tag>.png   one 5x5 (task x embodiment) heatmap per
                                  metric — a folder per metric, not 200
                                  loose files
  <run>/eval/round_metrics.jsonl  every filter episode's raw record
  wandb: round_overview/<metric>  overall mean over all combos
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METRICS = (
    "task_success", "violation_rate", "intervention_rate", "mode_switches",
    "min_h", "mean_gap", "v_le_h_frac", "realized_cmd_ratio",
)


class RoundMetrics:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, round_index: int, task: str, embodiment: str,
               metrics: dict) -> None:
        rec = {"round": int(round_index), "task": task,
               "embodiment": embodiment, **{k: metrics.get(k) for k in METRICS}}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def render(self, out_dir: Path, tag: str, tasks, embodiments,
               wandb_run=None, step: int = 0) -> dict:
        """Per-metric heatmaps (combo mean over all rounds so far) + overall
        means. Returns the overall dict (also logged to wandb)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        recs = self.load()
        # per-combo mean over every filter episode recorded so far
        combo: dict = {}
        # per-round overall mean (trend over training, all combos pooled)
        by_round: dict[int, list[dict]] = {}
        for r in recs:
            combo.setdefault((r["task"], r["embodiment"]), []).append(r)
            by_round.setdefault(int(r["round"]), []).append(r)

        overall = {}
        for metric in METRICS:
            grid = np.full((len(tasks), len(embodiments)), np.nan)
            for i, t in enumerate(tasks):
                for j, e in enumerate(embodiments):
                    vals = [r[metric] for r in combo.get((t, e), [])
                            if r.get(metric) is not None]
                    vals = [float(v) for v in vals if np.isfinite(float(v))]
                    if vals:
                        grid[i, j] = float(np.mean(vals))
            overall[metric] = (
                float(np.nanmean(grid)) if np.isfinite(grid).any()
                else float("nan")
            )

            mdir = out_dir / metric
            mdir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(7, 5.5))
            finite = grid[np.isfinite(grid)]
            vmin, vmax = (float(finite.min()), float(finite.max())) \
                if finite.size else (0.0, 1.0)
            if vmin == vmax:
                vmin, vmax = vmin - 0.5, vmax + 0.5
            im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax,
                           aspect="auto")
            ax.set_xticks(range(len(embodiments)), embodiments, rotation=30,
                          ha="right", fontsize=8)
            ax.set_yticks(range(len(tasks)), tasks, fontsize=8)
            for i in range(len(tasks)):
                for j in range(len(embodiments)):
                    if np.isfinite(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center",
                                va="center", fontsize=7, color="w")
            n_eps = sum(len(v) for v in combo.values())
            ax.set_title(f"{metric}  ({tag}, {n_eps} filter episodes; "
                         f"overall {overall[metric]:.3f})", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(mdir / f"{tag}.png", dpi=110)
            plt.close(fig)

            # trend over training: per-round overall mean, all combos pooled
            # (answers "is X getting better as rounds progress" at a glance)
            rs, ys = [], []
            for r_idx in sorted(by_round):
                vals = [rec.get(metric) for rec in by_round[r_idx]]
                vals = [float(v) for v in vals
                        if v is not None and np.isfinite(float(v))]
                if vals:
                    rs.append(r_idx)
                    ys.append(float(np.mean(vals)))
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(rs, ys, marker="o", ms=4)
            ax.set_xlabel("collection round")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} over rounds (overall, all combos)",
                         fontsize=10)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(mdir / "trend.png", dpi=110)
            plt.close(fig)

        if wandb_run is not None:
            import wandb

            from main.train.train import _safe_wandb_log

            # heatmaps + trends as images so they are visible in the run
            imgs = {}
            for metric in METRICS:
                mdir = out_dir / metric
                heat = mdir / f"{tag}.png"
                if heat.exists():
                    imgs[f"heatmap/{metric}"] = wandb.Image(str(heat))
                trend = mdir / "trend.png"
                if trend.exists():
                    imgs[f"trend/{metric}"] = wandb.Image(str(trend))
            _safe_wandb_log(wandb_run, imgs, step)
            _safe_wandb_log(
                wandb_run,
                {f"round_overview/{k}": v for k, v in overall.items()},
                step,
            )
        return overall
