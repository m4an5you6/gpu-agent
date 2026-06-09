"""Lightweight read-only SSH helpers for GPUCLOUD phase 5."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_OUTPUT_LIMIT = 8000


@dataclass
class SSHResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool = False
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "truncated": self.truncated,
            "error": self.error,
        }


def ssh_available() -> bool:
    return bool(shutil.which("ssh"))


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n... [truncated at {limit} chars]", True


def quote_remote_path(path: str) -> str:
    """Quote a remote shell path while preserving leading ~/ expansion."""
    text = str(path or "").strip()
    if text == "~":
        return "$HOME"
    if text.startswith("~/"):
        rest = text[2:]
        return "$HOME/" + shlex.quote(rest)
    return shlex.quote(text)


def build_ssh_command(
    *,
    host: str,
    user: str,
    port: int,
    key_path: str,
    remote_command: str,
) -> List[str]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        "-p",
        str(int(port)),
    ]
    if key_path:
        cmd.extend(["-i", key_path])
    target = f"{user}@{host}"
    cmd.append(target)
    cmd.append(remote_command)
    return cmd


def run_ssh_command(
    *,
    host: str,
    user: str,
    port: int,
    key_path: str,
    remote_command: str,
    timeout_sec: int = 60,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
) -> SSHResult:
    if not ssh_available():
        return SSHResult(
            ok=False,
            exit_code=-1,
            stdout="",
            stderr="",
            error="ssh client not found in PATH",
        )

    cmd = build_ssh_command(
        host=host,
        user=user,
        port=port,
        key_path=key_path,
        remote_command=remote_command,
    )
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
        )
    except subprocess.TimeoutExpired:
        return SSHResult(
            ok=False,
            exit_code=-1,
            stdout="",
            stderr="",
            error=f"SSH command timed out after {timeout_sec}s",
        )
    except OSError as exc:
        return SSHResult(
            ok=False,
            exit_code=-1,
            stdout="",
            stderr="",
            error=str(exc),
        )

    stdout, trunc_out = _truncate(proc.stdout or "", output_limit)
    stderr, trunc_err = _truncate(proc.stderr or "", output_limit)
    ok = proc.returncode == 0
    return SSHResult(
        ok=ok,
        exit_code=int(proc.returncode),
        stdout=stdout,
        stderr=stderr,
        truncated=trunc_out or trunc_err,
        error=None if ok else (stderr.strip() or stdout.strip() or f"exit {proc.returncode}"),
    )


def ssh_connect_check(
    *,
    host: str,
    user: str,
    port: int,
    key_path: str,
    timeout_sec: int = 30,
) -> SSHResult:
    return run_ssh_command(
        host=host,
        user=user,
        port=port,
        key_path=key_path,
        remote_command="echo gpucloud-ssh-ok",
        timeout_sec=timeout_sec,
        output_limit=256,
    )


def command_allowed(command: str, allowed_prefixes: List[str]) -> bool:
    text = (command or "").strip()
    if not text:
        return False
    first = text.split(None, 1)[0]
    return any(first == p or text.startswith(p + " ") for p in allowed_prefixes)
