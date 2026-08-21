"""Minimal config-driven object builder (vendored from robo_orchard_lab.utils.build).

Instantiates an object from a config dict of the form:

    {"type": "my.module:MyClass", "arg1": ..., ...}

where "type" (or "class_type") is either an import string
("pkg.module:Class" or "pkg.module") or a callable. Remaining keys are
passed as constructor keyword arguments. Nested config dicts are passed
through as-is — constructors that accept nested configs build them
themselves. Non-dict inputs are returned unchanged, so `build` is safe
to call on already-constructed objects.
"""

import copy
import importlib
from typing import Any

__all__ = ["build", "import_string"]


def import_string(path: str) -> Any:
    """Import 'pkg.module:Class' or 'pkg.module' and return the object."""
    if ":" in path:
        module_name, cls_name = path.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, cls_name)
    return importlib.import_module(path)


def build(obj: Any, *args) -> Any:
    """Instantiate `obj` if it is a config dict, otherwise return it as-is."""
    if isinstance(obj, dict):
        cfg = copy.deepcopy(obj)
        if "class_type" in cfg:
            cls = cfg.pop("class_type")
        elif "type" in cfg:
            cls = cfg.pop("type")
        else:
            raise KeyError("Missing type key `class_type` or `type`")
        if isinstance(cls, str):
            cls = import_string(cls)
        import torch

        if cls is torch.nn.GroupNorm and len(args) == 1:
            return cls(num_channels=args[0], **cfg)
        return cls(*args, **cfg)
    return obj
