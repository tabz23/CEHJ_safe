"""End-to-end async online test: warmup collect, then trainer + watcher
concurrently; verify the buffer grows during training and the collector
reloads weights when the trainer publishes a checkpoint."""
import subprocess
import sys
import time
from pathlib import Path

CEHJ = Path("/root/autodl-tmp/CEHJ_safe")
PY = "/root/miniconda3/envs/RoboTwin/bin/python"
DATA = CEHJ / "data" / "smoke_async"
RUN = CEHJ_ROOT = CEHJ / "runs" / "smoke_async"


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=CEHJ, **kw)


def bg(cmd, log):
    f = open(log, "w")
    return subprocess.Popen(cmd, cwd=CEHJ, stdout=f, stderr=subprocess.STDOUT)


def main():
    sh(["rm", "-rf", str(DATA), str(RUN)])
    print("== warmup: 2 episodes ==", flush=True)
    r = sh([PY, "main/train/collect.py", "--episodes", "2",
            "--capacity", "4520", "--out", str(DATA)])
    assert r.returncode == 0, "warmup collect failed"

    import json
    n0 = json.loads((DATA / "buffer" / "header.json").read_text())["n"]
    print(f"warmup buffer: {n0} steps", flush=True)

    print("== trainer + watcher ==", flush=True)
    trainer = bg([PY, "main/train/train.py", "--data", str(DATA),
                  "--run", str(RUN), "--grad-steps", "60",
                  "--eval-every", "20", "--no-wandb"], "/tmp/async_train.log")
    time.sleep(10)  # let the trainer construct before the watcher starts
    watcher = bg([PY, "main/train/collect.py", "--episodes", "2",
                  "--capacity", "4520", "--watch",
                  "--follow", str(RUN / "checkpoint.pt"),
                  "--out", str(DATA)], "/tmp/async_watch.log")

    trainer.wait()
    time.sleep(20)          # let the watcher notice the final checkpoint
    watcher.terminate()
    watcher.wait()

    n1 = json.loads((DATA / "buffer" / "header.json").read_text())["n"]
    print(f"final buffer: {n1} steps (warmup had {n0})", flush=True)
    watch_log = Path("/tmp/async_watch.log").read_text()
    train_log = Path("/tmp/async_train.log").read_text()
    reloaded = "follow: reloaded" in watch_log
    refreshed = "[train] waiting" not in train_log or True
    print(f"collector reloaded weights: {reloaded}", flush=True)
    assert n1 > n0, "buffer did not grow while training"
    assert reloaded, "collector never reloaded the checkpoint"
    assert "TRAIN DONE" in train_log
    print("ASYNC ONLINE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
