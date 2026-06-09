"""Regression tests for _apply_profile_override GPUCLOUD_HOME guard (issue #22502).

When GPUCLOUD_HOME is set to the gpucloud root (e.g. systemd hardcodes
GPUCLOUD_HOME=/root/.gpucloud), _apply_profile_override must still read
active_profile and update GPUCLOUD_HOME to the profile directory.

When GPUCLOUD_HOME is already a profile directory (.../profiles/<name>),
_apply_profile_override must trust it and return without re-reading
active_profile (child-process inheritance contract).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path



def _run_apply_profile_override(
    tmp_path, monkeypatch, *, gpucloud_home: str | None, active_profile: str | None,
    argv: list[str] | None = None,
):
    """Run _apply_profile_override in isolation.

    Returns the value of os.environ["GPUCLOUD_HOME"] after the call,
    or None if unset.
    """
    gpucloud_root = tmp_path / ".gpucloud"
    gpucloud_root.mkdir(parents=True, exist_ok=True)

    if active_profile is not None:
        (gpucloud_root / "active_profile").write_text(active_profile)

    if active_profile and active_profile != "default":
        (gpucloud_root / "profiles" / active_profile).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    if gpucloud_home is not None:
        monkeypatch.setenv("GPUCLOUD_HOME", gpucloud_home)
    else:
        monkeypatch.delenv("GPUCLOUD_HOME", raising=False)

    monkeypatch.setattr(sys, "argv", argv or ["gpucloud", "gateway", "start"])

    from gpucloud_cli.main import _apply_profile_override
    _apply_profile_override()

    return os.environ.get("GPUCLOUD_HOME")


class TestApplyProfileOverrideGPUCLOUDHomeGuard:
    """Regression guard for issue #22502.

    Verifies that GPUCLOUD_HOME pointing to the gpucloud root does NOT suppress
    the active_profile check, while GPUCLOUD_HOME already pointing to a
    profile directory IS trusted as-is.
    """

    def test_gpucloud_home_at_root_with_active_profile_is_redirected(
        self, tmp_path, monkeypatch
    ):
        """GPUCLOUD_HOME=/root/.gpucloud + active_profile=coder must redirect
        GPUCLOUD_HOME to .../profiles/coder.

        Bug scenario from #22502: systemd sets GPUCLOUD_HOME to the gpucloud root
        and the user switches to a profile via `gpucloud profile use`.
        Before the fix, the guard returned early and active_profile was ignored.
        """
        gpucloud_root = tmp_path / ".gpucloud"
        gpucloud_root.mkdir(parents=True, exist_ok=True)

        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            gpucloud_home=str(gpucloud_root),
            active_profile="coder",
        )

        assert result is not None, "GPUCLOUD_HOME must be set after profile redirect"
        assert "profiles" in result, (
            f"Expected GPUCLOUD_HOME to point into profiles/ dir, got: {result!r}"
        )
        assert result.endswith("coder"), (
            f"Expected GPUCLOUD_HOME to end with 'coder', got: {result!r}"
        )

    def test_gpucloud_home_already_profile_dir_is_trusted(self, tmp_path, monkeypatch):
        """GPUCLOUD_HOME=.../profiles/coder must not be overridden even when
        active_profile says something different.

        Preserves the child-process inheritance contract: a subprocess spawned
        with GPUCLOUD_HOME already set to a specific profile must stay in that
        profile.
        """
        gpucloud_root = tmp_path / ".gpucloud"
        profile_dir = gpucloud_root / "profiles" / "coder"
        profile_dir.mkdir(parents=True, exist_ok=True)

        (gpucloud_root / "active_profile").write_text("other")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("GPUCLOUD_HOME", str(profile_dir))
        monkeypatch.setattr(sys, "argv", ["gpucloud", "gateway", "start"])

        from gpucloud_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("GPUCLOUD_HOME") == str(profile_dir), (
            "GPUCLOUD_HOME must remain unchanged when already pointing to a profile dir"
        )

    def test_gpucloud_home_unset_reads_active_profile(self, tmp_path, monkeypatch):
        """Classic case: GPUCLOUD_HOME unset + active_profile=coder must set
        GPUCLOUD_HOME to the profile directory (existing behaviour must not regress).
        """
        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            gpucloud_home=None,
            active_profile="coder",
        )

        assert result is not None
        assert "coder" in result

    def test_gpucloud_home_unset_default_profile_no_redirect(self, tmp_path, monkeypatch):
        """active_profile=default must not redirect GPUCLOUD_HOME."""
        gpucloud_root = tmp_path / ".gpucloud"
        gpucloud_root.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("GPUCLOUD_HOME", raising=False)
        monkeypatch.setattr(sys, "argv", ["gpucloud", "gateway", "start"])
        (gpucloud_root / "active_profile").write_text("default")

        from gpucloud_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("GPUCLOUD_HOME") is None
