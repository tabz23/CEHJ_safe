"""Flat per-step memmap replay buffer for the X-VLA EE6D HJ-SAC baseline.

Slim EE6D schema (~400 KB/step, dominated by fp16 scene tokens). Stored per
step: scene_tokens [200, 1024] fp16 (PRE-adapter frozen X-VLA VLM tokens —
the adapter and proprio encoder train at train time, like the parent's
injection/trunk), proprio [20] (raw X-VLA EE6D layout: per arm
[xyz, rot6d, grip] with grip = 1-2g), action [18] (the COMMANDED EE6D delta
of the tick that led to the NEXT state — planner a_nom / actor /
perturbation, see action_source), h, embodiment_id, task_id, episode_id,
done. No contact fields, no Jacobians, no padding — EE6D is
embodiment-agnostic.

Header (JSON, once): config dict incl. h_scale, softmin_T, dt, the
task/embodiment/obstacle choice lists, field shapes — without these the
stored values are uninterpretable later. Reopening reads capacity from the
header; a fresh buffer requires an explicit capacity (an invented capacity
would try to allocate petabytes).

Ring semantics identical to the joint-space parent: at capacity the oldest
transitions are overwritten; sampling excludes pairs crossing the write
head / wrap seam; discard_to (success-only rollback) works only pre-wrap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCENE_TOKENS = 200
TOKEN_DIM = 1024  # pre-adapter X-VLA feature width
PROPRIO_DIM = 20  # X-VLA EE6D proprio x 2 arms (input encoding, with gripper)
N_ACTION = 18     # action: dxyz + drot6d per arm, NO gripper (filter never drives it)

# name -> (dtype, per-step shape)
FIELDS = {
    "scene_tokens": (np.float16, (SCENE_TOKENS, TOKEN_DIM)),
    "proprio": (np.float32, (PROPRIO_DIM,)),
    "action": (np.float32, (N_ACTION,)),
    "h": (np.float32, ()),
    "embodiment_id": (np.int8, ()),
    "task_id": (np.int8, ()),
    "action_source": (np.int8, ()),  # 0 planner / 1 actor / 2 perturbation
    "episode_id": (np.int32, ()),
    "done": (np.bool_, ()),
}

# Fields read at sample time. `_next` needs only the observation side;
# splitting avoids reading every field twice (the throughput ceiling).
# `h` is read at both t (Bellman backup) and t+1 (collision precision/recall
# ground truth) — adding it here only affects sampling, not the memmap layout.
NEXT_FIELDS = ("scene_tokens", "proprio", "h")
CUR_FIELDS = NEXT_FIELDS + ("action", "embodiment_id", "task_id",
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
        # pipeline actually produces (scene token count follows the encoder —
        # never hardcode it)
        for name, (dtype, shape) in FIELDS.items():
            p = self.root / f"{name}.npy"
            if p.exists():
                self.arrays[name] = np.load(p, mmap_mode="r+")
        if self.arrays:
            self._shapes = {n: a.shape[1:] for n, a in self.arrays.items()}
        else:
            self._shapes = dict(self.header.get("shapes", {}))
        self.n = int(self.header.get("n", 0))
        self.head = int(self.header.get("head", self.n))  # next write position
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
        """Ring semantics: at capacity the OLDEST transition is overwritten
        (warmup data ages out first). Never raises 'buffer full'."""
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
            arr[self.head] = value
        self.head = (self.head + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)
        self.header["n"] = self.n
        self.header["head"] = self.head
        self._valid_t = None  # invalidate the sample cache

    def flush(self) -> None:
        for arr in self.arrays.values():
            arr.flush()
        # atomic header write: a concurrent reader (async trainer) must
        # never see a partially-written n
        tmp = self.root / "header.json.tmp"
        tmp.write_text(json.dumps(self.header, indent=2))
        tmp.replace(self.root / "header.json")

    def refresh(self) -> int:
        """Re-read n/head from the header and invalidate the sample cache.

        For async collection: the collector process appends to the same
        memmaps and flushes per episode; the trainer calls refresh() every
        N grad steps to see the new rows. Rows below the flushed n are
        complete (append bumps n only after all fields of a row are
        written, and flush publishes n after the arrays are flushed).
        """
        header_path = self.root / "header.json"
        try:
            header = json.loads(header_path.read_text())
        except (OSError, json.JSONDecodeError):
            return self.n  # mid-rename or partially written; keep last n
        n = int(header.get("n", self.n))
        head = int(header.get("head", self.head))
        if n != self.n or head != self.head:
            self.n = n
            self.head = head
            self._valid_t = None
        return self.n

    def discard_to(self, n: int) -> None:
        """Roll back to n rows (drop the tail in place). Success-only
        collection uses this to drop a failed episode after the fact.
        Only valid before the ring has ever wrapped."""
        n = int(n)
        assert 0 <= n <= self.n, f"discard_to({n}) with only {self.n} rows"
        assert self.head == self.n, "discard_to after the ring wrapped"
        self.n = n
        self.head = n
        self.header["n"] = n
        self.header["head"] = n
        self._valid_t = None

    def __len__(self) -> int:
        return self.n

    def set_embodiment_filter(self, allowed_ids: set[int] | None) -> None:
        """LOO support: restrict sampling to these embodiment_ids (None = all).

        One shared warmup buffer holds all embodiments; each leave-one-out
        run just filters the held-out embodiment out of the sampled batches.
        """
        self._emb_filter = None if allowed_ids is None else np.array(
            sorted(allowed_ids), dtype=np.int8
        )
        self._valid_t = None  # invalidate the sample cache

    def _valid_transitions(self) -> np.ndarray:
        """Cached list of t where (t, t+1) stays inside one episode and,
        if an embodiment filter is set, belongs to an allowed embodiment
        (t and t+1 share an episode, hence an embodiment — masking t
        suffices). Ring-aware: pairs crossing the write head (where the
        oldest slot meets the newest) are excluded."""
        if self._valid_t is None:
            if "episode_id" not in self.arrays or self.n == 0:
                # fresh buffer: arrays are created lazily on first append
                self._valid_t = np.zeros(0, dtype=np.int64)
                return self._valid_t
            wrapped = self.n == self.capacity
            hi = self.capacity if wrapped else self.n
            ep = self.arrays["episode_id"][:hi]
            done = self.arrays["done"][:hi]
            ep_next = np.roll(ep, -1)
            mask = (ep == ep_next) & ~done
            mask[-1] = False                    # never pair last -> first
            if wrapped:
                mask[self.head - 1] = False     # seam: newest -> oldest
            else:
                mask[self.n - 1:] = False
            emb_filter = getattr(self, "_emb_filter", None)
            if emb_filter is not None:
                emb = self.arrays["embodiment_id"][:hi]
                mask &= np.isin(emb, emb_filter)
            self._valid_t = np.nonzero(mask)[0]
        return self._valid_t

    def seed_from(self, src_dir) -> None:
        """Copy every row from another buffer into this one — seeds the
        training ring from a reusable, separately-collected warmup buffer.
        The warmup buffer is never written to; when the ring fills, warmup
        rows are the oldest and get evicted first."""
        src = StepBuffer(src_dir)
        assert self.n == 0, "seed_from into a non-empty buffer"
        for name, arr in src.arrays.items():
            dst = self._create(name, np.asarray(arr[0]))
            dst[: src.n] = arr[: src.n]
            dst.flush()
        self.n = self.head = src.n
        self.header["n"] = src.n
        self.header["head"] = src.n
        self.flush()

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
        idx_next = (idx + 1) % self.capacity  # ring wrap
        out = {}
        for name in CUR_FIELDS:
            out[name] = np.asarray(self.arrays[name][idx])
        for name in NEXT_FIELDS:
            out[name + "_next"] = np.asarray(self.arrays[name][idx_next])
        return out
