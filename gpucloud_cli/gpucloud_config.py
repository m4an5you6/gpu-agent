"""GPUCLOUD cluster YAML — load, validate, default merge (phase 4).

Loaded only on explicit CLI (``gpucloud config validate``) or when ``/goal``
activates (see ``GoalManager.set``). Never at CLI/Agent startup.
"""

from __future__ import annotations

import os
import shlex
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml

# ── Errors ────────────────────────────────────────────────────────────


class GpucloudConfigError(Exception):
    """Configuration load or validation failure."""

    def __init__(self, message: str, *, errors: Optional[Sequence[str]] = None):
        super().__init__(message)
        self.errors: Tuple[str, ...] = tuple(errors or ())


# ── Discovery ─────────────────────────────────────────────────────────

DEFAULT_FILENAMES: Tuple[str, ...] = ("gpucloud.yaml",)
USER_CONFIG_REL = Path(".gpucloud") / "config.yaml"


def discover_config_paths(*, start_dir: Optional[Path] = None) -> List[Path]:
    """Return candidate paths in search order (first existing wins in resolve)."""
    env = os.environ.get("GPUCLOUD_CONFIG", "").strip()
    if env:
        return [Path(env).expanduser()]

    cwd = (start_dir or Path.cwd()).resolve()
    home = Path.home()
    return [
        cwd / "gpucloud.yaml",
        cwd / USER_CONFIG_REL,
        home / USER_CONFIG_REL,
    ]


def resolve_config_path(
    explicit: Optional[Union[str, Path]] = None,
    *,
    start_dir: Optional[Path] = None,
    required: bool = True,
) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise GpucloudConfigError(f"config file not found: {path}")
        return path.resolve()

    for candidate in discover_config_paths(start_dir=start_dir):
        if candidate.is_file():
            return candidate.resolve()

    if required:
        searched = ", ".join(str(p) for p in discover_config_paths(start_dir=start_dir))
        raise GpucloudConfigError(
            f"no gpucloud.yaml found (searched: {searched})"
        )
    raise GpucloudConfigError("no gpucloud config file")


# ── Validation (required fields only) ─────────────────────────────────


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _looks_like_inline_ssh_key(value: str) -> bool:
    text = value.strip()
    if "BEGIN" in text and "PRIVATE KEY" in text.upper():
        return True
    if "\n" in text and len(text) > 80:
        return True
    return False


def validate_required(data: Any) -> List[str]:
    """Return list of missing required field paths (empty if ok)."""
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["(root): must be a mapping"]

    if not isinstance(data.get("clusters"), list) or len(data["clusters"]) == 0:
        errors.append("clusters")

    for field in ("dataset_name", "model_name"):
        if _is_blank(data.get(field)):
            errors.append(field)

    clusters = data.get("clusters")
    if isinstance(clusters, list):
        for ci, cluster in enumerate(clusters):
            prefix = f"clusters[{ci}]"
            if not isinstance(cluster, dict):
                errors.append(f"{prefix}")
                continue
            if _is_blank(cluster.get("name")):
                errors.append(f"{prefix}.name")
            nodes = cluster.get("nodes")
            if not isinstance(nodes, list) or len(nodes) == 0:
                errors.append(f"{prefix}.nodes")
                continue
            for ni, node in enumerate(nodes):
                np = f"{prefix}.nodes[{ni}]"
                if not isinstance(node, dict):
                    errors.append(np)
                    continue
                for key in ("host", "user", "ssh_key"):
                    if _is_blank(node.get(key)):
                        errors.append(f"{np}.{key}")
                if "port" not in node or node.get("port") is None:
                    errors.append(f"{np}.port")
                else:
                    try:
                        int(node["port"])
                    except (TypeError, ValueError):
                        errors.append(f"{np}.port")
                ssh_key = node.get("ssh_key")
                if isinstance(ssh_key, str) and _looks_like_inline_ssh_key(ssh_key):
                    errors.append(f"{np}.ssh_key")

    return errors


# ── Defaults & resolution ─────────────────────────────────────────────


def _first_node_workdir(data: Dict[str, Any]) -> str:
    for cluster in data.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        for node in cluster.get("nodes") or []:
            if isinstance(node, dict) and node.get("workdir"):
                return str(node["workdir"])
    return "~/gpucloud"


def resolve_effective_dataset_model(data: Dict[str, Any]) -> Tuple[str, str]:
    training = data.get("training") if isinstance(data.get("training"), dict) else {}
    inference = data.get("inference") if isinstance(data.get("inference"), dict) else {}

    effective_dataset = training.get("dataset_name") or data.get("dataset_name") or ""
    effective_model = training.get("model_name") or data.get("model_name") or ""
    return str(effective_dataset), str(effective_model)


def merge_gpucloud_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy with §6.4.2 defaults applied."""
    out = deepcopy(data)
    workdir_default = "~/gpucloud"
    effective_dataset, effective_model = resolve_effective_dataset_model(out)
    primary_workdir = _first_node_workdir(out) or workdir_default

    clusters = out.get("clusters")
    if isinstance(clusters, list):
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            nodes = cluster.get("nodes")
            if not isinstance(nodes, list):
                continue
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                node.setdefault("role", "worker")
                node.setdefault("workdir", workdir_default)
                if "gpu_count" not in node:
                    node["gpu_count"] = None
                primary_workdir = str(node.get("workdir") or primary_workdir)

    training = out.setdefault("training", {})
    if not isinstance(training, dict):
        training = {}
        out["training"] = training
    training.setdefault("framework", "megatron-lm")
    training.setdefault("env", {})
    training.setdefault(
        "log_dir",
        f"{primary_workdir}/logs/{effective_dataset}",
    )
    training.setdefault(
        "checkpoint_dir",
        f"{primary_workdir}/checkpoints/{effective_model}",
    )

    inference = out.setdefault("inference", {})
    if not isinstance(inference, dict):
        inference = {}
        out["inference"] = inference
    inf_model = inference.get("model_name") or out.get("model_name") or effective_model
    inference.setdefault("engine", "vllm")
    inference.setdefault("port", 8000)
    inference.setdefault("tensor_parallel", 1)
    inference.setdefault(
        "model_path",
        f"{primary_workdir}/models/{inf_model}",
    )

    security = out.setdefault("security", {})
    if not isinstance(security, dict):
        security = {}
        out["security"] = security
    security.setdefault("dry_run_required", True)
    security.setdefault("max_concurrent_ssh", 4)
    security.setdefault("command_timeout_sec", 3600)
    security.setdefault(
        "allowed_remote_prefixes",
        ["python", "torchrun", "vllm", "bash"],
    )

    return out


def _gpu_count_for_command(data: Dict[str, Any]) -> int:
    for cluster in data.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        for node in cluster.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            gc = node.get("gpu_count")
            if isinstance(gc, int) and gc > 0:
                return gc
    return 1


def _quote_remote_arg(value: str) -> str:
    text = str(value or "").strip()
    if text == "~":
        return "$HOME"
    if text.startswith("~/"):
        return "$HOME/" + shlex.quote(text[2:])
    return shlex.quote(text)


def generate_training_command(data: Dict[str, Any]) -> str:
    training = data.get("training") or {}
    if isinstance(training, dict) and training.get("command"):
        return str(training["command"]).strip()

    effective_dataset, effective_model = resolve_effective_dataset_model(data)
    checkpoint_dir = ""
    if isinstance(training, dict):
        checkpoint_dir = str(training.get("checkpoint_dir") or "")
    if not checkpoint_dir:
        workdir = _first_node_workdir(data)
        checkpoint_dir = f"{workdir}/checkpoints/{effective_model}"

    nproc = _gpu_count_for_command(data)
    megatron_script = "${MEGATRON_LM_DIR:-./Megatron-LM}/pretrain_gpt.py"
    return (
        f"torchrun --nproc_per_node={nproc} {megatron_script} "
        f"--data-path {_quote_remote_arg(effective_dataset)} "
        f"--save {_quote_remote_arg(checkpoint_dir)} "
        f"--load {_quote_remote_arg(checkpoint_dir)}"
    )


def ensure_training_command(data: Dict[str, Any]) -> Dict[str, Any]:
    """Set training.command when absent (after merge)."""
    out = deepcopy(data)
    training = out.setdefault("training", {})
    if not isinstance(training, dict):
        training = {}
        out["training"] = training
    if not training.get("command"):
        training["command"] = generate_training_command(out)
    return out


# ── Pipeline ──────────────────────────────────────────────────────────


@dataclass
class GpucloudPreparedConfig:
    path: Path
    raw: Dict[str, Any]
    merged: Dict[str, Any]
    effective_dataset: str
    effective_model: str
    training_command: str

    def summary_lines(self) -> List[str]:
        """Human-readable summary; never includes ssh key file contents."""
        lines = [
            f"config: {self.path}",
            f"dataset: {self.effective_dataset}",
            f"model: {self.effective_model}",
            f"training.command: {self.training_command}",
        ]
        clusters = self.merged.get("clusters") or []
        for ci, cluster in enumerate(clusters):
            if not isinstance(cluster, dict):
                continue
            cname = cluster.get("name", f"cluster-{ci}")
            for ni, node in enumerate(cluster.get("nodes") or []):
                if not isinstance(node, dict):
                    continue
                host = node.get("host", "?")
                port = node.get("port", "?")
                user = node.get("user", "?")
                key_path = node.get("ssh_key", "?")
                workdir = node.get("workdir", "?")
                lines.append(
                    f"  [{cname}] node[{ni}] {user}@{host}:{port} "
                    f"workdir={workdir} ssh_key={key_path}"
                )
        inf = self.merged.get("inference") or {}
        if isinstance(inf, dict):
            lines.append(
                f"inference: engine={inf.get('engine')} port={inf.get('port')} "
                f"path={inf.get('model_path')}"
            )
        sec = self.merged.get("security") or {}
        if isinstance(sec, dict):
            lines.append(
                f"security: dry_run_required={sec.get('dry_run_required')} "
                f"timeout_sec={sec.get('command_timeout_sec')}"
            )
        return lines

    def context_block_for_goal(self, goal: str = "") -> str:
        """Injected into /goal user messages only."""
        from gpucloud_cli.gpucloud_goal import build_goal_context_block

        return build_goal_context_block(self, goal=goal)


def load_raw_yaml(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GpucloudConfigError(f"cannot read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise GpucloudConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise GpucloudConfigError(f"{path}: root must be a mapping")
    return data


def prepare_gpucloud_config(
    explicit: Optional[Union[str, Path]] = None,
    *,
    start_dir: Optional[Path] = None,
) -> GpucloudPreparedConfig:
    """Full pipeline: resolve → validate → merge → command."""
    path = resolve_config_path(explicit, start_dir=start_dir, required=True)
    raw = load_raw_yaml(path)
    errors = validate_required(raw)
    if errors:
        raise GpucloudConfigError(
            "required fields missing",
            errors=errors,
        )
    merged = merge_gpucloud_defaults(raw)
    merged = ensure_training_command(merged)
    effective_dataset, effective_model = resolve_effective_dataset_model(merged)
    cmd = generate_training_command(merged)
    return GpucloudPreparedConfig(
        path=path,
        raw=raw,
        merged=merged,
        effective_dataset=effective_dataset,
        effective_model=effective_model,
        training_command=cmd,
    )


def load_gpucloud_for_goal(
    explicit: Optional[Union[str, Path]] = None,
) -> GpucloudPreparedConfig:
    """Entry point for ``/goal`` — same as prepare, clearer name."""
    return prepare_gpucloud_config(explicit)


def run_config_validate(
    explicit: Optional[Union[str, Path]] = None,
    *,
    start_dir: Optional[Path] = None,
) -> int:
    """CLI handler for ``gpucloud config validate``. Returns exit code."""
    try:
        prepared = prepare_gpucloud_config(explicit, start_dir=start_dir)
    except GpucloudConfigError as exc:
        print("GPUCLOUD config validation failed.")
        if exc.errors:
            for err in exc.errors:
                print(f"  - {err}")
        else:
            print(f"  {exc}")
        return 1

    print("GPUCLOUD config OK")
    for line in prepared.summary_lines():
        print(f"  {line}")
    return 0


__all__ = [
    "GpucloudConfigError",
    "GpucloudPreparedConfig",
    "discover_config_paths",
    "resolve_config_path",
    "validate_required",
    "merge_gpucloud_defaults",
    "resolve_effective_dataset_model",
    "generate_training_command",
    "prepare_gpucloud_config",
    "load_gpucloud_for_goal",
    "run_config_validate",
]
