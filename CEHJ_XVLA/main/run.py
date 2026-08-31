#!/usr/bin/env python3
"""Shim after rebase: eval CLI lives in ``main/envs/run.py``."""
import runpy
import sys
from pathlib import Path

_cehj = Path(__file__).resolve().parent.parent
if str(_cehj) not in sys.path:
    sys.path.insert(0, str(_cehj))
runpy.run_module("main.envs.run", run_name="__main__")
