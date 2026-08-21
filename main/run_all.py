#!/usr/bin/env python3
"""Shim after rebase: eval CLI lives in ``main/envs/run_all.py``."""
import runpy
import sys
from pathlib import Path

_cehj = Path(__file__).resolve().parent.parent
if str(_cehj) not in sys.path:
    sys.path.insert(0, str(_cehj))
runpy.run_module("main.envs.run_all", run_name="__main__")
