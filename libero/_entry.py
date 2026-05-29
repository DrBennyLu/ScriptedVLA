"""Bootstrap `python libero/<script>.py` → `python -m libero.<script>`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def maybe_reroute_main(name: str, package: str | None, file: str) -> None:
    if package or name != "__main__":
        return
    root = Path(file).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    module = f"libero.{Path(file).stem}"
    runpy.run_module(module, run_name="__main__")
    raise SystemExit(0)
