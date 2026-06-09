"""Read-only cluster / GPU probe logic (phase 5)."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from gpucloud_cli.gpucloud_context import (
    iter_cluster_nodes,
    node_label,
    node_ssh_key_path,
    resolve_config_for_tool,
)
from gpucloud_cli.gpucloud_config import GpucloudConfigError
from gpucloud_cli.gpucloud_ssh import (
    command_allowed,
    run_ssh_command,
    ssh_connect_check,
    ssh_available,
)

_NVIDIA_SMI_QUERY = (
    "nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used "
    "--format=csv,noheader,nounits 2>/dev/null || nvidia-smi -L 2>/dev/null || echo NO_GPU"
)


def _security(merged: Dict[str, Any]) -> Dict[str, Any]:
    sec = merged.get("security")
    return sec if isinstance(sec, dict) else {}


def _timeout(merged: Dict[str, Any]) -> int:
    return int(_security(merged).get("command_timeout_sec") or 3600)


def _output_limit(merged: Dict[str, Any]) -> int:
    return int(_security(merged).get("max_output_chars") or 8000)


def _allowed_prefixes(merged: Dict[str, Any]) -> List[str]:
    raw = _security(merged).get("allowed_remote_prefixes")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]
    return ["python", "torchrun", "vllm", "bash", "echo", "test", "nvidia-smi", "ls", "cat", "nvcc"]


def parse_nvidia_smi_output(text: str) -> Dict[str, Any]:
    """Best-effort parse of nvidia-smi CSV or -L listing."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines or lines == ["NO_GPU"] or "not found" in text.lower():
        return {
            "available": False,
            "gpus": [],
            "gpu_count": 0,
            "driver_version": None,
            "cuda_note": None,
        }

    gpus: List[Dict[str, Any]] = []
    driver_version = None
    for line in lines:
        if line.startswith("GPU "):
            m = re.match(r"GPU (\d+):\s+(.+)", line)
            if m:
                gpus.append({"index": int(m.group(1)), "name": m.group(2).strip()})
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "driver_version": parts[2],
                        "memory_total_mib": parts[3],
                        "memory_used_mib": parts[4],
                    }
                )
                driver_version = driver_version or parts[2]
            except ValueError:
                gpus.append({"raw": line})
        else:
            gpus.append({"raw": line})

    return {
        "available": bool(gpus),
        "gpus": gpus,
        "gpu_count": len(gpus),
        "driver_version": driver_version,
        "cuda_note": "use remote nvcc --version via gpucloud_ssh_exec if needed",
    }


def probe_local_gpu() -> Dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {
            "target": "local",
            "available": False,
            "skipped": True,
            "reason": "nvidia-smi not in PATH",
            "gpus": [],
        }
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        parsed = parse_nvidia_smi_output(proc.stdout or proc.stderr or "")
        parsed["target"] = "local"
        parsed["skipped"] = False
        parsed["exit_code"] = proc.returncode
        return parsed
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {
            "target": "local",
            "available": False,
            "skipped": True,
            "reason": str(exc),
            "gpus": [],
        }


def probe_remote_gpu(node: Dict[str, Any], merged: Dict[str, Any]) -> Dict[str, Any]:
    key_path = node_ssh_key_path(node)
    result = run_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=key_path,
        remote_command=_NVIDIA_SMI_QUERY,
        timeout_sec=min(120, _timeout(merged)),
        output_limit=_output_limit(merged),
    )
    parsed = parse_nvidia_smi_output(result.stdout)
    parsed["target"] = "remote"
    parsed["ssh_ok"] = result.ok
    if result.error:
        parsed["error"] = result.error
    if not result.ok and not parsed.get("available"):
        parsed["skipped"] = True
        parsed["reason"] = result.error or f"exit {result.exit_code}"
    else:
        parsed["skipped"] = False
    return parsed


def check_node(
    cluster: str,
    index: int,
    node: Dict[str, Any],
    merged: Dict[str, Any],
    *,
    cluster_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    label = node_label(cluster, index, node)
    key_path = node_ssh_key_path(node)
    timeout = min(120, _timeout(merged))
    workdir = str(node.get("workdir") or "~/gpucloud")

    entry: Dict[str, Any] = {
        "label": label,
        "cluster": cluster,
        "node_index": index,
        "host": node.get("host"),
        "port": node.get("port"),
        "user": node.get("user"),
        "role": node.get("role", "worker"),
        "workdir": workdir,
        "ssh_key_path": str(node.get("ssh_key", "")),
    }

    if not ssh_available():
        entry["ssh"] = {"ok": False, "error": "ssh not in PATH"}
        entry["status"] = "error"
        return entry

    conn = ssh_connect_check(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=key_path,
        timeout_sec=timeout,
    )
    entry["ssh"] = conn.as_dict()

    if not conn.ok:
        entry["workdir_accessible"] = False
        entry["gpu"] = {
            "available": False,
            "skipped": True,
            "reason": "skipped: SSH not connected",
            "gpus": [],
        }
        entry["detected_gpu_count"] = "unknown"
        entry["status"] = "error"
        return entry

    wd_cmd = f"test -d {shlex.quote(workdir)} && echo workdir-ok || echo workdir-missing"
    wd = run_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=key_path,
        remote_command=wd_cmd,
        timeout_sec=timeout,
        output_limit=512,
    )
    entry["workdir_accessible"] = wd.ok and "workdir-ok" in (wd.stdout or "")

    shared_dirs = []
    if cluster_meta and isinstance(cluster_meta.get("shared_dirs"), list):
        shared_dirs = [str(p) for p in cluster_meta["shared_dirs"] if str(p).strip()]
    shared_results = []
    for path in shared_dirs:
        sd_cmd = f"test -d {shlex.quote(path)} && echo exists || echo missing"
        sd = run_ssh_command(
            host=str(node["host"]),
            user=str(node["user"]),
            port=int(node["port"]),
            key_path=key_path,
            remote_command=sd_cmd,
            timeout_sec=timeout,
            output_limit=256,
        )
        shared_results.append(
            {"path": path, "exists": sd.ok and "exists" in (sd.stdout or "")}
        )
    if shared_results:
        entry["shared_dirs"] = shared_results

    gpu = probe_remote_gpu(node, merged)
    entry["gpu"] = gpu
    if isinstance(node.get("gpu_count"), int):
        entry["configured_gpu_count"] = node["gpu_count"]
    elif gpu.get("gpu_count"):
        entry["detected_gpu_count"] = gpu["gpu_count"]
    else:
        entry["detected_gpu_count"] = "unknown"

    if conn.ok and entry.get("workdir_accessible"):
        entry["status"] = "ok"
    elif conn.ok:
        entry["status"] = "degraded"
    else:
        entry["status"] = "error"

    return entry


def run_cluster_check(
    *,
    config_file: Optional[str] = None,
    cluster_name: Optional[str] = None,
    allow_discover_without_goal: bool = False,
) -> Dict[str, Any]:
    try:
        prepared = resolve_config_for_tool(
            config_file,
            allow_discover_without_goal=allow_discover_without_goal,
        )
    except GpucloudConfigError as exc:
        return {"ok": False, "error": str(exc), "nodes": []}

    merged = prepared.merged
    nodes_out: List[Dict[str, Any]] = []
    clusters = merged.get("clusters") or []
    cluster_by_name = {
        str(c.get("name")): c for c in clusters if isinstance(c, dict) and c.get("name")
    }

    for cname, idx, node in iter_cluster_nodes(merged, cluster_name=cluster_name):
        meta = cluster_by_name.get(cname)
        nodes_out.append(check_node(cname, idx, node, merged, cluster_meta=meta))

    ok_count = sum(1 for n in nodes_out if n.get("status") == "ok")
    return {
        "ok": ok_count == len(nodes_out) and len(nodes_out) > 0,
        "config_path": str(prepared.path),
        "dataset": prepared.effective_dataset,
        "model": prepared.effective_model,
        "nodes_checked": len(nodes_out),
        "nodes_ok": ok_count,
        "nodes": nodes_out,
    }


def run_ssh_exec(
    *,
    command: str,
    config_file: Optional[str] = None,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    try:
        prepared = resolve_config_for_tool(config_file)
    except GpucloudConfigError as exc:
        return {"ok": False, "error": str(exc)}

    merged = prepared.merged
    allowed = _allowed_prefixes(merged)
    if not command_allowed(command, allowed):
        return {
            "ok": False,
            "error": f"command not allowed; must start with one of: {allowed}",
        }

    node_found = None
    c_found = None
    for cname, idx, node in iter_cluster_nodes(merged, cluster_name=cluster_name):
        if idx == node_index:
            node_found = node
            c_found = cname
            break

    if node_found is None:
        return {"ok": False, "error": f"node index {node_index} not found"}

    sec = _security(merged)
    do_dry = sec.get("dry_run_required", True) if dry_run is None else bool(dry_run)
    if do_dry:
        return {
            "ok": True,
            "dry_run": True,
            "cluster": c_found,
            "node_index": node_index,
            "command": command,
            "ssh_command": build_ssh_display(node_found, command),
        }

    result = run_ssh_command(
        host=str(node_found["host"]),
        user=str(node_found["user"]),
        port=int(node_found["port"]),
        key_path=node_ssh_key_path(node_found),
        remote_command=command,
        timeout_sec=_timeout(merged),
        output_limit=_output_limit(merged),
    )
    out = result.as_dict()
    out["dry_run"] = False
    out["cluster"] = c_found
    out["node_index"] = node_index
    return out


def build_ssh_display(node: Dict[str, Any], command: str) -> str:
    from gpucloud_cli.gpucloud_ssh import build_ssh_command
    parts = build_ssh_command(
        host=str(node["host"]),
        user=str(node["user"]),
        port=int(node["port"]),
        key_path=node_ssh_key_path(node),
        remote_command=command,
    )
    return " ".join(parts)


def run_gpu_info(
    *,
    config_file: Optional[str] = None,
    cluster_name: Optional[str] = None,
    node_index: Optional[int] = None,
    target: str = "remote",
) -> Dict[str, Any]:
    if target == "local":
        return probe_local_gpu()

    try:
        prepared = resolve_config_for_tool(config_file)
    except GpucloudConfigError as exc:
        return {"ok": False, "error": str(exc)}

    merged = prepared.merged
    for cname, idx, node in iter_cluster_nodes(merged, cluster_name=cluster_name):
        if idx == node_index:
            data = probe_remote_gpu(node, merged)
            data["cluster"] = cname
            data["node_index"] = node_index
            data["ok"] = data.get("available") or data.get("skipped")
            return data
    return {"ok": False, "error": f"node index {node_index} not found"}


def cluster_check_json(**kwargs) -> str:
    return json.dumps(run_cluster_check(**kwargs), ensure_ascii=False, indent=2)
