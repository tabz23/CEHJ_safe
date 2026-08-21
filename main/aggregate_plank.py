#!/usr/bin/env python3
"""Build a same-seed latency table across vanilla vs plan_everyk_* trees.

    python aggregate_plank.py /path/to/safe_unsafe_bimanual_plank
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _load_summaries(root: Path) -> list[dict]:
    rows = []
    for summary in root.glob("*/*/*/summary.json"):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        data["_path"] = str(summary)
        rows.append(data)
    return rows


def _timing(data: dict, key: str):
    timing = data.get("timing") or {}
    if key in data and data[key] is not None:
        return data[key]
    return timing.get(key)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser().resolve()
    rows = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if child.name == "logs":
            continue
        for data in _load_summaries(child):
            rows.append(
                {
                    "policy_dir": child.name,
                    "controller": data.get("controller"),
                    "controller_tag": data.get("controller_tag") or child.name,
                    "replan_k": data.get("replan_k"),
                    "task": data.get("task"),
                    "embodiment": data.get("embodiment"),
                    "seed": data.get("seed"),
                    "obstacle_mode": data.get("obstacle_mode"),
                    "place_mode": data.get("place_mode"),
                    "plan_mode": data.get("plan_mode"),
                    "success": data.get("success"),
                    "plan_success": data.get("plan_success"),
                    "contact": data.get("contact"),
                    "d_min": data.get("d_min"),
                    "n_steps": data.get("n_steps"),
                    "n_frames": data.get("n_frames"),
                    "n_plans": data.get("n_plans"),
                    "n_plan_ok": data.get("n_plan_ok"),
                    "n_plan_fail": data.get("n_plan_fail"),
                    "n_replans": data.get("n_replans"),
                    "n_mpc_windows": data.get("n_mpc_windows"),
                    "t_episode_s": _timing(data, "t_episode_s"),
                    "t_plan_s": _timing(data, "t_plan_s"),
                    "t_physics_s": _timing(data, "t_physics_s"),
                    "t_render_s": _timing(data, "t_render_s"),
                    "t_metric_s": _timing(data, "t_metric_s"),
                    "t_other_s": _timing(data, "t_other_s"),
                    "t_plan_frac": _timing(data, "t_plan_frac"),
                    "t_physics_frac": _timing(data, "t_physics_frac"),
                    "t_render_frac": _timing(data, "t_render_frac"),
                    "t_plan_first_s": _timing(data, "t_plan_first_s"),
                    "t_plan_rest_mean_s": _timing(data, "t_plan_rest_mean_s"),
                    "play_error": data.get("play_error"),
                    "summary": data.get("_path"),
                }
            )
    out_csv = root / "latency_compare.csv"
    out_json = root / "latency_compare.json"
    if rows:
        fields = list(rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("", encoding="utf-8")
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
