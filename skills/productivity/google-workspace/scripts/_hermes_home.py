"""Resolve GPUCLOUD_HOME for standalone skill scripts.

Skill scripts may run outside the GPUCLOUD process (e.g. system Python,
nix env, CI) where ``gpucloud_constants`` is not importable.  This module
provides the same ``get_gpucloud_home()`` and ``display_gpucloud_home()``
contracts as ``gpucloud_constants`` without requiring it on ``sys.path``.

When ``gpucloud_constants`` IS available it is used directly so that any
future enhancements (profile resolution, Docker detection, etc.) are
picked up automatically.  The fallback path replicates the core logic
from ``gpucloud_constants.py`` using only the stdlib.

All scripts under ``google-workspace/scripts/`` should import from here
instead of duplicating the ``GPUCLOUD_HOME = Path(os.getenv(...))`` pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from gpucloud_constants import display_gpucloud_home as display_gpucloud_home
    from gpucloud_constants import get_gpucloud_home as get_gpucloud_home
except (ModuleNotFoundError, ImportError):

    def get_gpucloud_home() -> Path:
        """Return the GPUCLOUD home directory (default: ~/.gpucloud).

        Mirrors ``gpucloud_constants.get_gpucloud_home()``."""
        val = os.environ.get("GPUCLOUD_HOME", "").strip()
        return Path(val) if val else Path.home() / ".gpucloud"

    def display_gpucloud_home() -> str:
        """Return a user-friendly ``~/``-shortened display string.

        Mirrors ``gpucloud_constants.display_gpucloud_home()``."""
        home = get_gpucloud_home()
        try:
            return "~/" + str(home.relative_to(Path.home()))
        except ValueError:
            return str(home)
