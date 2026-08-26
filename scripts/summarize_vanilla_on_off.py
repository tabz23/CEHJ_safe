#!/usr/bin/env python3
"""Side-by-side on_path / off_path metrics for vanilla_play_once summaries.

  python CEHJ/scripts/summarize_vanilla_on_off.py
  python CEHJ/scripts/summarize_vanilla_on_off.py /path/to/vanilla_play_once
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path(
    "/storage1/fs1/sibai/Active/yuxuan/cross_embodiment/CEHJ/outputs/ihabnew/"
    "safe_unsafe_bimanual_drinkobstacle_testing_trajs/vanilla_play_once"
)

TASK_ORDER = [
    "place_container_plate",
    "place_burger_fries",
    "place_cans_plasticbox",
    "stack_blocks_two",
    "place_can_basket",
    "place_bread_basket",
    "pick_dual_bottles",
    "stack_bowls_two",
]
EMB_ORDER = ["aloha-agilex", "piper", "ARX-X5", "franka-panda", "ur5-wsg"]
MODES = ("on_path", "off_path")


def _load(root: Path):
    # task -> emb -> mode -> list of dicts
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for summary in root.glob("*/*/*/summary.json"):
        try:
            d = json.loads(summary.read_text())
        except Exception:
            continue
        task = d.get("task") or summary.parts[-4]
        emb = d.get("embodiment") or summary.parts[-3]
        mode = d.get("obstacle_mode") or "unknown"
        if mode not in MODES:
            continue
        data[task][emb][mode].append(
            {
                "plan_success": bool(d.get("plan_success")),
                "success": bool(d.get("success")),
                "contact": bool(d.get("contact")),
                "n_steps": int(d.get("n_steps") or 0),
            }
        )
    return data


def _fmt(rows: list[dict]) -> str:
    n = len(rows)
    if n == 0:
        return "—".ljust(48)
    ps = sum(r["plan_success"] for r in rows)
    ss = sum(r["success"] for r in rows)
    cc = sum(r["contact"] for r in rows)
    steps = [r["n_steps"] for r in rows if r["n_steps"] > 0]
    avg = (sum(steps) / len(steps)) if steps else float("nan")
    avg_s = f"{avg:.0f}" if steps else "—"
    return f"n={n:2d} plan={ps}/{n} sim={ss}/{n} hit={cc}/{n} steps≈{avg_s}".ljust(48)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT).expanduser()
    data = _load(root)
    tasks = [t for t in TASK_ORDER if t in data] + sorted(set(data) - set(TASK_ORDER))
    total = sum(len(v) for t in data.values() for e in t.values() for v in e.values())
    print(f"root={root}")
    print(f"episodes={total}")
    print("plan=plan_success  sim=simulation/task success  hit=contact  steps≈mean n_steps\n")
    header = f"{'task':24s} {'emb':14s} | {'on_path':48s} | {'off_path':48s}"
    print(header)
    print("-" * len(header))
    for task in tasks:
        embs = [e for e in EMB_ORDER if e in data[task]] + sorted(
            set(data[task]) - set(EMB_ORDER)
        )
        for i, emb in enumerate(embs):
            on_s = _fmt(data[task][emb].get("on_path", []))
            off_s = _fmt(data[task][emb].get("off_path", []))
            task_col = task if i == 0 else ""
            print(f"{task_col:24s} {emb:14s} | {on_s} | {off_s}")
        print("-" * len(header))


if __name__ == "__main__":
    main()
