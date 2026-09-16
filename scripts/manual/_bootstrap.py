"""Shared path bootstrap for the repository's standalone diagnostic scripts."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_project_paths() -> None:
    for directory in (PROJECT_ROOT, PROJECT_ROOT / "utils", PROJECT_ROOT / "simulator", PROJECT_ROOT / "training"):
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)
