"""Deprecated compatibility package for ``gpucloud_cli``.

New code should import from ``gpucloud_cli``.  This package keeps older
plugins and scripts importable during the GPUCLOUD rename.
"""

from __future__ import annotations

import importlib
import sys

_gpucloud_cli = importlib.import_module("gpucloud_cli")
__path__ = _gpucloud_cli.__path__
__all__ = getattr(_gpucloud_cli, "__all__", [])

sys.modules[__name__] = _gpucloud_cli
