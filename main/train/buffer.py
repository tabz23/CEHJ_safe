"""Flat per-step memmap replay buffer for HJ-SAC.

Per-step fields (~630 KB/step, dominated by fp16 scene tokens). Stored per
step: scene_tokens, scene_pos, state_tokens, body_tokens, link_pos, qpos
(policy layout), qpos_raw (per-arm raw joints), dtheta, h, per_link_h,
d_arm_arm, body_mask, joint_index, Jlin/Jang, dtheta_max, embodiment_id,
episode_id, done — with episode_id and done written from step zero, so
sampling s' across an episode boundary is rejected at sample time.

CROSS-EMBODIMENT: the buffer mixes embodiments, so variable-width fields
are ZERO-PADDED to fixed maxima at append time (validity via body_mask /
joint_index): state_tokens [16, 256] (14 for 6-dof arms), qpos [16],
qpos_raw/dtheta/dtheta_max [18] (16 for 6-dof arms), Jlin/Jang [20, 3, 18].
Jlin/Jang/dtheta_max are STORED (not recomputed at sample time) because the
recompute would need the right embodiment's URDF per sample.

Header (JSON, once): config dict incl. h_scale, softmin_T, dt, the
task/embodiment/obstacle choice lists, field shapes — without these the
stored values are uninterpretable later. Reopening reads capacity from the
header; a fresh buffer requires an explicit capacity (an invented capacity
would try to allocate petabytes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# cross-embodiment maxima (6-dof arms: 14/16 wide; franka 7-dof: 16/18)
MAX_STATE_TOKENS = 16
MAX_ACTION = 18

# name -> (dtype, per-step shape)
FIELDS = {
    "scene_tokens": (np.float16, (1200, 256)),
    "scene_pos": (np.float32, (1200, 3)),
    "state_tokens": (np.float16, (MAX_STATE_TOKENS, 256)),
    "body_tokens": (np.float32, (20, 16)),
    "link_pos": (np.float32, (20, 3)),
    "Jlin": (np.float16, (20, 3, MAX_ACTION)),
    "Jang": (np.float16, (20, 3, MAX_ACTION)),
    "qpos": (np.float32, (16,)),
    "qpos_raw": (np.float32, (MAX_ACTION,)),
    "dtheta": (np.float32, (MAX_ACTION,)),
    "dtheta_max": (np.float32, (MAX_ACTION,)),
    "h": (np.float32, ()),
    "per_link_h": (np.float32, (20,)),
    "d_arm_arm": (np.float32, ()),
    "body_mask": (np.bool_, (20,)),
    "joint_index": (np.int8, (20,)),
    "embodiment_id": (np.int8, ()),
    "task_id": (np.int8, ()),
    "action_source": (np.int8, ()),  # 0 planner / 1 actor / 2 perturbation
    "episode_id": (np.int32, ()),
    "done": (np.bool_, ()),
}

# Fields read at sample time. `_next` needs only the observation side;
# splitting avoids reading every field twice (the throughput ceiling).
NEXT_FIELDS = (
    "state_tokens", "body_tokens", "link_pos", "body_mask",
    "joint_index", "scene_tokens", "scene_pos",
    "Jlin", "Jang", "dtheta_max",
)
CUR_FIELDS = NEXT_FIELDS + ("h", "dtheta", "embodiment_id", "task_id",
                            "action_source")


class StepBuffer:
    def __init__(self, root: str | Path, capacity: int | None = None,
                 header: dict | None = None):
        # resolve NOW: RoboTwin chdirs during env setup, so a relative root
        # would silently point somewhere else by the first append
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        header_path = self.root / "header.json"
        if header is not None:
            if capacity is None:
                raise ValueError("capacity is required when writing a header")
            header = dict(header)
            header["shapes"] = {k: list(v[1]) for k, v in FIELDS.items()}
            header["capacity"] = int(capacity)
            header_path.write_text(json.dumps(header, indent=2))
        self.header = json.loads(header_path.read_text()) if header_path.exists() else {}
        if capacity is None:
            capacity = self.header.get("capacity")
            if capacity is None:
                raise ValueError(
                    f"{self.root}: no capacity given and no header.json found"
                )
        self.capacity = int(capacity)
        self.arrays = {}
        # arrays are created lazily on first append with the shapes the
        # pipeline actually produces (scene token count follows camera
        # resolution / feature_level — never hardcode it)
        for name, (dtype, shape) in FIELDS.items():
            p = self.root / f"{name}.npy"
            if p.exists():
                self.arrays[name] = np.load(p, mmap_mode="r+")
        if self.arrays:
            self._shapes = {n: a.shape[1:] for n, a in self.arrays.items()}
        else:
            self._shapes = dict(self.header.get("shapes", {}))
        self.n = int(self.header.get("n", 0))
        self._valid_t: np.ndarray | None = None

    def _create(self, name: str, value: np.ndarray) -> np.ndarray:
        dtype = FIELDS[name][0]
        arr = np.lib.format.open_memmap(
            self.root / f"{name}.npy", mode="w+",
            dtype=dtype, shape=(self.capacity, *value.shape),
        )
        self.arrays[name] = arr
        self._shapes[name] = value.shape
        self.header["shapes"] = {k: list(v) for k, v in self._shapes.items()}
        return arr

    def append(self, step: dict) -> None:
        assert self.n < self.capacity, "buffer full"
        for name, value in step.items():
            value = np.asarray(value)
            arr = self.arrays.get(name)
            if arr is not None and tuple(arr.shape[1:]) != tuple(value.shape):
                if self.n > 0:
                    raise ValueError(
                        f"{name}: shape changed {tuple(arr.shape[1:])} -> "
                        f"{tuple(value.shape)} after {self.n} steps; refusing "
                        f"to silently invalidate the buffer"
                    )
                print(f"[buffer] {name}: recreating with new shape {tuple(value.shape)}")
                (self.root / f"{name}.npy").unlink()
                arr = None
            if arr is None:
                arr = self._create(name, value)
            arr[self.n] = value
        self.n += 1
        self.header["n"] = self.n
        self._valid_t = None  # invalidate the sample cache

    def flush(self) -> None:
        for arr in self.arrays.values():
            arr.flush()
        (self.root / "header.json").write_text(json.dumps(self.header, indent=2))

    def __len__(self) -> int:
        return self.n

    def _valid_transitions(self) -> np.ndarray:
        """Cached list of t where (t, t+1) stays inside one episode."""
        if self._valid_t is None:
            ep = self.arrays["episode_id"][: self.n]
            done = self.arrays["done"][: self.n]
            self._valid_t = np.nonzero((ep[:-1] == ep[1:]) & ~done[:-1])[0]
        return self._valid_t

    def sample(self, batch: int, rng: np.random.RandomState | None = None) -> dict:
        """Sample (t, t+1) pairs; pairs never cross an episode boundary.

        Reads CUR_FIELDS at t and NEXT_FIELDS at t+1 (not every field
        twice).
        """
        rng = rng or np.random.RandomState()
        valid_t = self._valid_transitions()
        if len(valid_t) == 0:
            raise RuntimeError("no valid transitions yet")
        idx = rng.choice(valid_t, size=min(batch, len(valid_t)), replace=len(valid_t) < batch)
        out = {}
        for name in CUR_FIELDS:
            out[name] = np.asarray(self.arrays[name][idx])
        for name in NEXT_FIELDS:
            out[name + "_next"] = np.asarray(self.arrays[name][idx + 1])
        return out
