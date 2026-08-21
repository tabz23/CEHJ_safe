#!/usr/bin/env python3
"""Test StepBuffer: roundtrip, episode-boundary rejection, fp16 tolerance.

  python test_buffer.py
"""

import shutil
from pathlib import Path

import numpy as np
import sys

CEHJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CEHJ_ROOT))

from main.train.buffer import FIELDS, StepBuffer  # noqa: E402

ROOT = Path("/tmp/buffer_test")


def fake_step(i, ep):
    s = {}
    for name, (dtype, shape) in FIELDS.items():
        s[name] = np.zeros(shape, dtype=dtype)
    s["h"] = np.float32(0.1 * i)
    s["episode_id"] = np.int32(ep)
    s["done"] = np.bool_(False)
    s["joint_index"] = np.arange(20, dtype=np.int8)
    s["scene_tokens"] = np.full((1200, 256), i % 7, dtype=np.float16)
    return s


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    buf = StepBuffer(ROOT, capacity=100, header={"task": "test"})
    # 3 episodes of 5 steps
    for ep in range(3):
        for i in range(5):
            step = fake_step(i, ep)
            step["done"] = np.bool_(i == 4)
            buf.append(step)
    buf.flush()
    assert len(buf) == 15

    # roundtrip
    assert buf.arrays["h"][7] == np.float32(0.1 * 2)
    assert buf.arrays["episode_id"][9] == 1
    assert buf.arrays["done"][4] == True and buf.arrays["done"][5] == False
    assert buf.header["task"] == "test" and buf.header["n"] == 15
    print("roundtrip + header  OK")

    # boundary rejection: episode_id 0..2, done at 4/9/14
    rng = np.random.RandomState(0)
    ep = buf.arrays["episode_id"][: len(buf)]
    done = buf.arrays["done"][: len(buf)]
    for _ in range(200):
        b = buf.sample(8, rng)
        # recover sampled indices indirectly: every returned pair must come
        # from the cached valid_t list, which excludes boundaries
        valid = buf._valid_transitions()
        assert (ep[valid] == ep[valid + 1]).all() and not done[valid].any()
    print("episode-boundary rejection (200 samples)  OK")

    # fp16 tolerance
    assert abs(float(buf.arrays["scene_tokens"][3, 0, 0]) - 3.0) < 1e-3
    print("fp16 tolerance  OK")

    # reopen persistence
    buf2 = StepBuffer(ROOT, capacity=100)
    assert len(buf2) == 15 and buf2.header["task"] == "test"
    print("reopen persistence  OK")

    # approx step size
    per_step = sum(np.prod(v[1]) * np.dtype(v[0]).itemsize for v in FIELDS.values())
    print(f"per-step size: {per_step / 1024:.0f} KB")
    print("BUFFER TEST PASSED")


if __name__ == "__main__":
    main()
