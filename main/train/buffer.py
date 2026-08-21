"""Flat per-step memmap replay buffer for HJ-SAC.

Per-step fields (~630 KB/step, dominated by fp16 scene tokens). Stored per
step: scene_tokens, scene_pos, state_tokens, body_tokens, link_pos, qpos
(policy layout, 14), qpos_raw (per-arm raw 8+8, for exact Jacobian
recompute), dtheta, h, body_mask, joint_index, episode_id, done — with
episode_id and done written from step zero, so sampling s' across an
episode boundary is rejected at sample time.

Header (JSON, once): l_reach, joint_vel_limits, link_names, kappa,
control_dt, T, embodiment, urdf_hash, field shapes — without these the
stored values are uninterpretable later.

Not stored: Jlin/Jang (recomputed from qpos_raw via
BodyTokenExtractor.jacobian_batch — sub-ms batched, and a URDF fix does
not invalidate the buffer), dtheta_max, T_cam_world.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# name -> (dtype, per-step shape)
FIELDS = {
    "scene_tokens": (np.float16, (1200, 256)),
    "scene_pos": (np.float32, (1200, 3)),
    "state_tokens": (np.float16, (14, 256)),
    "body_tokens": (np.float32, (20, 16)),
    "link_pos": (np.float32, (20, 3)),
    "qpos": (np.float32, (14,)),
    "qpos_raw": (np.float32, (16,)),
    "dtheta": (np.float32, (16,)),
    "h": (np.float32, ()),
    "per_link_h": (np.float32, (20,)),
    "d_arm_arm": (np.float32, ()),
    "body_mask": (np.bool_, (20,)),
    "joint_index": (np.int8, (20,)),
    "episode_id": (np.int32, ()),
    "done": (np.bool_, ()),
}


class StepBuffer:
    def __init__(self, root: str | Path, capacity: int, header: dict | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        header_path = self.root / "header.json"
        if header is not None:
            header = dict(header)
            header["shapes"] = {k: list(v[1]) for k, v in FIELDS.items()}
            header["capacity"] = self.capacity
            header_path.write_text(json.dumps(header, indent=2))
        self.header = json.loads(header_path.read_text()) if header_path.exists() else {}
        self.arrays = {}
        for name, (dtype, shape) in FIELDS.items():
            p = self.root / f"{name}.npy"
            if p.exists():
                arr = np.load(p, mmap_mode="r+")
            else:
                arr = np.lib.format.open_memmap(
                    p, mode="w+", dtype=dtype, shape=(self.capacity, *shape)
                )
            self.arrays[name] = arr
        self.n = int(self.header.get("n", 0))

    def append(self, step: dict) -> None:
        assert self.n < self.capacity, "buffer full"
        for name, arr in self.arrays.items():
            arr[self.n] = step[name]
        self.n += 1
        self.header["n"] = self.n

    def flush(self) -> None:
        for arr in self.arrays.values():
            arr.flush()
        (self.root / "header.json").write_text(json.dumps(self.header, indent=2))

    def __len__(self) -> int:
        return self.n

    def sample(self, batch: int, rng: np.random.RandomState | None = None) -> dict:
        """Sample (t, t+1) pairs; reject pairs crossing an episode boundary."""
        rng = rng or np.random.RandomState()
        ep = self.arrays["episode_id"][: self.n]
        done = self.arrays["done"][: self.n]
        valid_t = np.nonzero((ep[:-1] == ep[1:]) & ~done[:-1])[0]
        if len(valid_t) == 0:
            raise RuntimeError("no valid transitions yet")
        idx = rng.choice(valid_t, size=min(batch, len(valid_t)), replace=len(valid_t) < batch)
        out = {}
        for name, arr in self.arrays.items():
            out[name] = np.asarray(arr[idx])
            out[name + "_next"] = np.asarray(arr[idx + 1])
        return out
