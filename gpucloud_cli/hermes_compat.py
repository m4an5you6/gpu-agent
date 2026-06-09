"""Deprecated ``hermes`` command compatibility entry point."""

from __future__ import annotations

import sys


def main() -> None:
    """Warn once, then dispatch to the GPUCLOUD CLI."""
    print(
        "Warning: 'hermes' is deprecated and will be removed in a future "
        "GPUCLOUD release. Use 'gpucloud' instead.",
        file=sys.stderr,
    )
    from gpucloud_cli.main import main as gpucloud_main

    gpucloud_main()
