"""Shared constants for GPUCLOUD Agent.

Import-safe module with no dependencies — can be imported from anywhere
without risk of circular imports.
"""

import os
import shutil
import sys
import sysconfig
from contextvars import ContextVar, Token
from pathlib import Path


_profile_fallback_warned: bool = False
_legacy_home_warned: bool = False
_UNSET = object()
_GPUCLOUD_HOME_OVERRIDE: ContextVar[str | object] = ContextVar(
    "_GPUCLOUD_HOME_OVERRIDE", default=_UNSET
)


def set_gpucloud_home_override(path: str | Path | None) -> Token:
    """Set a context-local GPUCLOUD home override and return its reset token.

    This is for in-process, per-task scoping.  It deliberately does not mutate
    ``os.environ`` because that is shared by every thread in the process.
    """
    value: str | object = _UNSET if path is None else str(path)
    return _GPUCLOUD_HOME_OVERRIDE.set(value)


def reset_gpucloud_home_override(token: Token) -> None:
    """Restore the previous context-local GPUCLOUD home override."""
    _GPUCLOUD_HOME_OVERRIDE.reset(token)


def get_gpucloud_home_override() -> str | None:
    """Return the active context-local GPUCLOUD home override, if any."""
    override = _GPUCLOUD_HOME_OVERRIDE.get()
    if override is _UNSET or not override:
        return None
    return str(override)


def _warn_legacy_home_once(path: str) -> None:
    """Warn once when falling back to deprecated HERMES_HOME."""
    global _legacy_home_warned
    if _legacy_home_warned:
        return
    _legacy_home_warned = True
    try:
        sys.stderr.write(
            "[GPUCLOUD migration] HERMES_HOME is deprecated. "
            f"Using legacy path {path!r} because GPUCLOUD_HOME is unset. "
            "Set GPUCLOUD_HOME or run `gpucloud config migrate-home` when ready.\n"
        )
        sys.stderr.flush()
    except Exception:
        pass


def _get_env_gpucloud_home() -> str:
    """Return GPUCLOUD_HOME, falling back to deprecated HERMES_HOME."""
    val = os.environ.get("GPUCLOUD_HOME", "").strip()
    if val:
        return val
    legacy = os.environ.get("HERMES_HOME", "").strip()
    if legacy:
        _warn_legacy_home_once(legacy)
        return legacy
    return ""


def get_gpucloud_home() -> Path:
    """Return the GPUCLOUD home directory (default: ~/.gpucloud).

    Reads GPUCLOUD_HOME env var, falls back to deprecated HERMES_HOME only
    when GPUCLOUD_HOME is unset, then falls back to ~/.gpucloud.
    This is the single source of truth — all other copies should import this.

    When ``GPUCLOUD_HOME`` is unset but an ``active_profile`` file indicates
    a non-default profile is active, logs a loud one-shot warning to
    ``errors.log`` so cross-profile data corruption is diagnosable instead
    of silent.  Behavior is unchanged otherwise — we still return
    ``~/.gpucloud`` — because raising here would brick 30+ module-level
    callers that import this at load time.  Subprocess spawners are
    expected to propagate ``GPUCLOUD_HOME`` explicitly (see the systemd
    template in ``gpucloud_cli/gateway.py`` and the kanban dispatcher in
    ``gpucloud_cli/kanban_db.py``).  See https://github.com/NousResearch/gpucloud-agent/issues/18594.
    """
    override = get_gpucloud_home_override()
    if override:
        return Path(override)

    val = _get_env_gpucloud_home()
    if val:
        return Path(val)

    # Guard: if a non-default profile is sticky-active, warn once that
    # the fallback to the default profile is almost certainly wrong.
    global _profile_fallback_warned
    if not _profile_fallback_warned:
        try:
            # Inline the default-root resolution from get_default_gpucloud_root()
            # to stay import-safe (this function is called from module scope
            # in 30+ files; we cannot afford to trigger logging setup here).
            active_path = (Path.home() / ".gpucloud" / "active_profile")
            active = active_path.read_text().strip() if active_path.exists() else ""
        except (UnicodeDecodeError, OSError):
            active = ""
        if active and active != "default":
            _profile_fallback_warned = True
            # Write directly to stderr.  We intentionally do NOT route this
            # through ``logging`` because (a) this function is called at
            # module-import time from 30+ sites, often before logging is
            # configured, and (b) root-logger propagation would double-emit
            # on consoles where a StreamHandler is already attached.
            msg = (
                f"[GPUCLOUD_HOME fallback] GPUCLOUD_HOME is unset but active "
                f"profile is {active!r}. Falling back to ~/.gpucloud, which "
                f"is the DEFAULT profile — not {active!r}. Any data this "
                f"process writes will land in the wrong profile. The "
                f"subprocess spawner should pass GPUCLOUD_HOME explicitly "
                f"(see issue #18594)."
            )
            try:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            except Exception:
                pass

    return Path.home() / ".gpucloud"


def get_default_gpucloud_root() -> Path:
    """Return the root GPUCLOUD directory for profile-level operations.

    In standard deployments this is ``~/.gpucloud``.

    In Docker or custom deployments where ``GPUCLOUD_HOME`` points outside
    ``~/.gpucloud`` (e.g. ``/opt/data``), returns ``GPUCLOUD_HOME`` directly
    — that IS the root.

    In profile mode where ``GPUCLOUD_HOME`` is ``<root>/profiles/<name>``,
    returns ``<root>`` so that ``profile list`` can see all profiles.
    Works both for standard (``~/.gpucloud/profiles/coder``) and Docker
    (``/opt/data/profiles/coder``) layouts.

    Import-safe — no dependencies beyond stdlib.
    """
    native_home = Path.home() / ".gpucloud"
    env_home = os.environ.get("GPUCLOUD_HOME", "")
    if not env_home:
        env_home = _get_env_gpucloud_home()
    if not env_home:
        return native_home
    env_path = Path(env_home)
    try:
        env_path.resolve().relative_to(native_home.resolve())
        # GPUCLOUD_HOME is under ~/.gpucloud (normal or profile mode)
        return native_home
    except ValueError:
        pass

    # Docker / custom deployment.
    # Check if this is a profile path: <root>/profiles/<name>
    # If the immediate parent dir is named "profiles", the root is
    # the grandparent — this covers Docker profiles correctly.
    if env_path.parent.name == "profiles":
        return env_path.parent.parent

    # Not a profile path — GPUCLOUD_HOME itself is the root
    return env_path


def _get_packaged_data_dir(name: str) -> Path | None:
    """Return an installed data-files directory if one exists.

    Used to discover bundled skills/optional-skills when GPUCLOUD is installed
    from a wheel that emitted them via setuptools data_files.
    """
    candidates = []
    for scheme in ("data", "purelib", "platlib"):
        raw = sysconfig.get_path(scheme)
        if raw:
            candidates.append(Path(raw) / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_optional_skills_dir(default: Path | None = None) -> Path:
    """Return the optional-skills directory, honoring package-manager wrappers.

    Packaged installs may ship ``optional-skills`` outside the Python package
    tree and expose it via ``GPUCLOUD_OPTIONAL_SKILLS``.
    """
    override = os.getenv("GPUCLOUD_OPTIONAL_SKILLS", "").strip()
    if override:
        return Path(override)
    packaged = _get_packaged_data_dir("optional-skills")
    if packaged is not None:
        return packaged
    if default is not None:
        return default
    return get_gpucloud_home() / "optional-skills"


def get_optional_mcps_dir(default: Path | None = None) -> Path:
    """Return the optional-mcps directory, honoring package-manager wrappers.

    Mirrors :func:`get_optional_skills_dir` for the MCP catalog (Nous-approved
    Model Context Protocol servers shipped with the repo but disabled by
    default). Packaged installs may ship ``optional-mcps`` outside the Python
    package tree and expose it via ``GPUCLOUD_OPTIONAL_MCPS``.
    """
    override = os.getenv("GPUCLOUD_OPTIONAL_MCPS", "").strip()
    if override:
        return Path(override)
    packaged = _get_packaged_data_dir("optional-mcps")
    if packaged is not None:
        return packaged
    if default is not None:
        return default
    return get_gpucloud_home() / "optional-mcps"


def get_bundled_skills_dir(default: Path | None = None) -> Path:
    """Return the bundled skills directory for source and packaged installs.

    Resolution order:
        1. ``GPUCLOUD_BUNDLED_SKILLS`` env var (Nix wrapper / explicit override)
        2. Wheel-installed ``<sysconfig data>/skills`` (pip install path)
        3. Caller-supplied ``default`` (typically the source-checkout path)
        4. ``<GPUCLOUD_HOME>/skills`` last-resort
    """
    override = os.getenv("GPUCLOUD_BUNDLED_SKILLS", "").strip()
    if override:
        return Path(override)
    packaged = _get_packaged_data_dir("skills")
    if packaged is not None:
        return packaged
    if default is not None:
        return default
    return get_gpucloud_home() / "skills"


def get_gpucloud_dir(new_subpath: str, old_name: str) -> Path:
    """Resolve a GPUCLOUD subdirectory with backward compatibility.

    New installs get the consolidated layout (e.g. ``cache/images``).
    Existing installs that already have the old path (e.g. ``image_cache``)
    keep using it — no migration required.

    Args:
        new_subpath: Preferred path relative to GPUCLOUD_HOME (e.g. ``"cache/images"``).
        old_name: Legacy path relative to GPUCLOUD_HOME (e.g. ``"image_cache"``).

    Returns:
        Absolute ``Path`` — old location if it exists on disk, otherwise the new one.
    """
    home = get_gpucloud_home()
    old_path = home / old_name
    if old_path.exists():
        return old_path
    return home / new_subpath


def migrate_hermes_home_to_gpucloud_home(
    src: str | Path | None = None,
    dst: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Copy legacy ``~/.hermes`` data into ``~/.gpucloud`` without deleting it.

    Existing destination files are left untouched.  This helper is intentionally
    conservative so users can run it before switching ``GPUCLOUD_HOME`` and
    still keep the old directory available for rollback.
    """
    source = Path(src) if src is not None else Path.home() / ".hermes"
    target_root = Path(dst) if dst is not None else Path.home() / ".gpucloud"
    result: dict[str, object] = {
        "source": str(source),
        "target": str(target_root),
        "dry_run": dry_run,
        "copied": [],
        "skipped": [],
        "missing_source": False,
    }
    copied = result["copied"]
    skipped = result["skipped"]
    assert isinstance(copied, list)
    assert isinstance(skipped, list)

    if not source.exists():
        result["missing_source"] = True
        return result

    if not dry_run:
        target_root.mkdir(parents=True, exist_ok=True)

    for child in source.iterdir():
        destination = target_root / child.name
        if destination.exists():
            skipped.append(str(destination))
            continue
        copied.append(str(destination))
        if dry_run:
            continue
        if child.is_dir():
            shutil.copytree(child, destination, symlinks=True)
        else:
            shutil.copy2(child, destination, follow_symlinks=False)

    return result


def display_gpucloud_home() -> str:
    """Return a user-friendly display string for the current GPUCLOUD_HOME.

    Uses ``~/`` shorthand for readability::

        default:  ``~/.gpucloud``
        profile:  ``~/.gpucloud/profiles/coder``
        custom:   ``/opt/gpucloud-custom``

    Use this in **user-facing** print/log messages instead of hardcoding
    ``~/.gpucloud``.  For code that needs a real ``Path``, use
    :func:`get_gpucloud_home` instead.
    """
    home = get_gpucloud_home()
    try:
        return "~/" + str(home.relative_to(Path.home()))
    except ValueError:
        return str(home)


def secure_parent_dir(path: Path) -> None:
    """Chmod ``0o700`` on the parent directory of *path*, but only if safe.

    Refuses to chmod ``/`` or any top-level directory (resolved parent with
    fewer than 3 parts, i.e. ``/`` or any direct child like ``/usr``) to
    prevent catastrophic host bricking when ``GPUCLOUD_HOME`` or other path
    env vars resolve to an unexpected location.

    See https://github.com/NousResearch/gpucloud-agent/issues/25821.
    """
    parent = path.parent.resolve()
    # Refuse root and its direct children (/usr, /home, /var, /tmp, …).
    if parent == Path("/") or len(parent.parts) < 3:
        return
    try:
        os.chmod(parent, 0o700)
    except OSError:
        pass


def get_subprocess_home() -> str | None:
    """Return a per-profile HOME directory for subprocesses, or None.

    When ``{GPUCLOUD_HOME}/home/`` exists on disk, subprocesses should use it
    as ``HOME`` so system tools (git, ssh, gh, npm …) write their configs
    inside the GPUCLOUD data directory instead of the OS-level ``/root`` or
    ``~/``.  This provides:

    * **Docker persistence** — tool configs land inside the persistent volume.
    * **Profile isolation** — each profile gets its own git identity, SSH
      keys, gh tokens, etc.

    The Python process's own ``os.environ["HOME"]`` and ``Path.home()`` are
    **never** modified — only subprocess environments should inject this value.
    Activation is directory-based: if the ``home/`` subdirectory doesn't
    exist, returns ``None`` and behavior is unchanged.
    """
    gpucloud_home = get_gpucloud_home_override() or _get_env_gpucloud_home()
    if not gpucloud_home:
        return None
    profile_home = os.path.join(gpucloud_home, "home")
    if os.path.isdir(profile_home):
        return profile_home
    return None


VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def parse_reasoning_effort(effort: str) -> dict | None:
    """Parse a reasoning effort level into a config dict.

    Valid levels: "none", "minimal", "low", "medium", "high", "xhigh".
    Returns None when the input is empty or unrecognized (caller uses default).
    Returns {"enabled": False} for "none".
    Returns {"enabled": True, "effort": <level>} for valid effort levels.
    """
    if not effort or not effort.strip():
        return None
    effort = effort.strip().lower()
    if effort == "none":
        return {"enabled": False}
    if effort in VALID_REASONING_EFFORTS:
        return {"enabled": True, "effort": effort}
    return None


def is_termux() -> bool:
    """Return True when running inside a Termux (Android) environment.

    Checks ``TERMUX_VERSION`` (set by Termux) or the Termux-specific
    ``PREFIX`` path.  Import-safe — no heavy deps.
    """
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


_wsl_detected: bool | None = None


def is_wsl() -> bool:
    """Return True when running inside WSL (Windows Subsystem for Linux).

    Checks ``/proc/version`` for the ``microsoft`` marker that both WSL1
    and WSL2 inject.  Result is cached for the process lifetime.
    Import-safe — no heavy deps.
    """
    global _wsl_detected
    if _wsl_detected is not None:
        return _wsl_detected
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            _wsl_detected = "microsoft" in f.read().lower()
    except Exception:
        _wsl_detected = False
    return _wsl_detected


_container_detected: bool | None = None


def is_container() -> bool:
    """Return True when running inside a Docker/Podman container.

    Checks ``/.dockerenv`` (Docker), ``/run/.containerenv`` (Podman),
    and ``/proc/1/cgroup`` for container runtime markers.  Result is
    cached for the process lifetime.  Import-safe — no heavy deps.
    """
    global _container_detected
    if _container_detected is not None:
        return _container_detected
    if os.path.exists("/.dockerenv"):
        _container_detected = True
        return True
    if os.path.exists("/run/.containerenv"):
        _container_detected = True
        return True
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8") as f:
            cgroup = f.read()
            if "docker" in cgroup or "podman" in cgroup or "/lxc/" in cgroup:
                _container_detected = True
                return True
    except OSError:
        pass
    _container_detected = False
    return False


# ─── Well-Known Paths ─────────────────────────────────────────────────────────


def get_config_path() -> Path:
    """Return the path to ``config.yaml`` under GPUCLOUD_HOME.

    Replaces the ``get_gpucloud_home() / "config.yaml"`` pattern repeated
    in 7+ files (skill_utils.py, gpucloud_logging.py, gpucloud_time.py, etc.).
    """
    return get_gpucloud_home() / "config.yaml"


def get_skills_dir() -> Path:
    """Return the path to the skills directory under GPUCLOUD_HOME."""
    return get_gpucloud_home() / "skills"



def get_env_path() -> Path:
    """Return the path to the ``.env`` file under GPUCLOUD_HOME."""
    return get_gpucloud_home() / ".env"


# ─── Network Preferences ─────────────────────────────────────────────────────


def apply_ipv4_preference(force: bool = False) -> None:
    """Monkey-patch ``socket.getaddrinfo`` to prefer IPv4 connections.

    On servers with broken or unreachable IPv6, Python tries AAAA records
    first and hangs for the full TCP timeout before falling back to IPv4.
    This affects httpx, requests, urllib, the OpenAI SDK — everything that
    uses ``socket.getaddrinfo``.

    When *force* is True, patches ``getaddrinfo`` so that calls with
    ``family=AF_UNSPEC`` (the default) resolve as ``AF_INET`` instead,
    skipping IPv6 entirely.  If no A record exists, falls back to the
    original unfiltered resolution so pure-IPv6 hosts still work.

    Safe to call multiple times — only patches once.
    Set ``network.force_ipv4: true`` in ``config.yaml`` to enable.
    """
    if not force:
        return

    import socket

    # Guard against double-patching
    if getattr(socket.getaddrinfo, "_hermes_ipv4_patched", False):
        return

    _original_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if family == 0:  # AF_UNSPEC — caller didn't request a specific family
            try:
                return _original_getaddrinfo(
                    host, port, socket.AF_INET, type, proto, flags
                )
            except socket.gaierror:
                # No A record — fall back to full resolution (pure-IPv6 hosts)
                return _original_getaddrinfo(host, port, family, type, proto, flags)
        return _original_getaddrinfo(host, port, family, type, proto, flags)

    _ipv4_getaddrinfo._hermes_ipv4_patched = True  # type: ignore[attr-defined]
    socket.getaddrinfo = _ipv4_getaddrinfo  # type: ignore[assignment]


# ─── Streaming Response Constants ────────────────────────────────────────────

# Response ID for partial stream stubs used during error recovery
PARTIAL_STREAM_STUB_ID = "partial-stream-stub"

FINISH_REASON_LENGTH = "length"


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"


# Deprecated Hermes compatibility aliases.  Keep these import-only shims small:
# new code should use the GPUCLOUD names above.
set_hermes_home_override = set_gpucloud_home_override
reset_hermes_home_override = reset_gpucloud_home_override
get_hermes_home_override = get_gpucloud_home_override
get_hermes_home = get_gpucloud_home
display_hermes_home = display_gpucloud_home
get_default_hermes_root = get_default_gpucloud_root
get_hermes_dir = get_gpucloud_dir
