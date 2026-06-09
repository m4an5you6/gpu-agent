"""GPUCLOUD /goal workflow planning (phase 9)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from gpucloud_cli.gpucloud_config import GpucloudConfigError
from gpucloud_cli.gpucloud_context import resolve_config_for_tool
from gpucloud_cli.gpucloud_inference import run_infer_start
from gpucloud_cli.gpucloud_probe import run_cluster_check
from gpucloud_cli.gpucloud_train import run_train_start

if TYPE_CHECKING:
    from gpucloud_cli.gpucloud_config import GpucloudPreparedConfig

_TRAIN_KEYWORDS = (
    "train",
    "training",
    "finetune",
    "fine-tune",
    "fine tune",
    "megatron",
    "checkpoint",
    "resume",
    "训练",
    "微调",
    "续训",
    "恢复训练",
)

_INFER_KEYWORDS = (
    "infer",
    "inference",
    "serve",
    "serving",
    "deploy",
    "endpoint",
    "api",
    "vllm",
    "推理",
    "部署",
    "服务",
    "接口",
)


def infer_goal_intent(goal: str = "", mode: Optional[str] = None) -> str:
    """Return train, infer, train_and_infer, or auto-resolved train."""
    requested = (mode or "").strip().lower().replace("-", "_")
    if requested in {"train", "training"}:
        return "train"
    if requested in {"infer", "inference", "serve", "serving"}:
        return "infer"
    if requested in {"both", "train_and_infer", "train_infer", "end_to_end"}:
        return "train_and_infer"

    text = (goal or "").lower()
    wants_train = any(token in text for token in _TRAIN_KEYWORDS)
    wants_infer = any(token in text for token in _INFER_KEYWORDS)
    if wants_train and wants_infer:
        return "train_and_infer"
    if wants_infer:
        return "infer"
    return "train"


def is_gpucloud_goal(goal: str) -> bool:
    text = (goal or "").lower()
    return any(token in text for token in (*_TRAIN_KEYWORDS, *_INFER_KEYWORDS))


def _intent_label(intent: str) -> str:
    if intent == "infer":
        return "vLLM inference service"
    if intent == "train_and_infer":
        return "Megatron-LM training plus vLLM inference"
    return "Megatron-LM training"


def build_goal_context_block(
    prepared: "GpucloudPreparedConfig",
    *,
    goal: str = "",
) -> str:
    """Context injected only into GPUCLOUD /goal turns."""
    intent = infer_goal_intent(goal)
    summary = "\n".join(prepared.summary_lines())
    return (
        "[GPUCLOUD goal context — use only for this /goal ML workflow]\n"
        f"{summary}\n\n"
        f"Goal workflow intent: {_intent_label(intent)}.\n"
        "Mandatory workflow:\n"
        "1. Call gpucloud_goal_prepare first. It performs cluster check and returns "
        "only dry-run train/infer plans.\n"
        "2. If cluster_check.ok is false, stop at the probe stage and report the "
        "failed node details plus next diagnostic steps. Do not start training or inference.\n"
        "3. If dry-run succeeds, present the launch command(s), log path(s), "
        "checkpoint/model path(s), and plan_summary. Explain Megatron communication "
        "scope before asking for explicit user confirmation.\n"
        "4. Do not set confirm_execute=true, confirm_stop=true, or confirm_delete=true "
        "unless the user explicitly confirms in the current conversation.\n"
        "5. After a confirmed start, use gpucloud_train_status/gpucloud_train_logs/"
        "gpucloud_checkpoint_latest for training and gpucloud_infer_status/"
        "gpucloud_infer_health for inference monitoring.\n"
        "Megatron note: the generated default command is single-node torchrun. "
        "Multi-node or heterogeneous GPU training must use an explicit external "
        "launcher in training.command, such as K8s/Slurm/Ray, that owns ranks, "
        "MASTER_ADDR/MASTER_PORT, WORLD_SIZE, and NCCL networking.\n"
        "Do not read or print SSH private key contents."
    )


def _diagnostic_steps(stage: str) -> List[str]:
    if stage == "cluster_check":
        return [
            "Verify SSH host, port, user, and ssh_key path in gpucloud.yaml.",
            "Confirm the remote workdir exists and is accessible.",
            "Run gpucloud cluster check after fixing connectivity.",
        ]
    if stage == "dry_run":
        return [
            "Inspect the dry-run error and generated command inputs.",
            "Check training.command or inference.model_path overrides.",
            "Re-run gpucloud_goal_prepare before any confirmed execution.",
        ]
    return [
        "Fix the reported gpucloud.yaml validation error.",
        "Re-run /goal after the required fields are present.",
    ]


def _cluster_line(cluster: Dict[str, Any]) -> str:
    checked = int(cluster.get("nodes_checked") or 0)
    ok = int(cluster.get("nodes_ok") or 0)
    return f"Cluster check: {ok}/{checked} node(s) OK"


def megatron_communication_notes() -> Dict[str, Any]:
    return {
        "default_scope": "single_node_torchrun",
        "default_communication": (
            "torchrun starts local ranks on one host; ranks communicate through "
            "PyTorch distributed/NCCL inside that node."
        ),
        "multi_node_requirement": (
            "Multi-node Megatron-LM must be launched by an external launcher or "
            "explicit training.command that sets nnodes, node_rank, MASTER_ADDR, "
            "MASTER_PORT, world size, rank mapping, and NCCL network settings."
        ),
        "heterogeneous_gpu_warning": (
            "Heterogeneous GPUs are not automatically balanced by GPUCLOUD; use "
            "K8s/Slurm/Ray or a custom launcher to partition workloads and avoid "
            "mixing incompatible memory or performance profiles."
        ),
    }


def build_plan_summary(
    *,
    intent: str,
    prepared: "GpucloudPreparedConfig",
    cluster: Dict[str, Any],
    dry_runs: Optional[Dict[str, Any]] = None,
    stopped_stage: Optional[str] = None,
    error: str = "",
) -> str:
    lines = [
        "GPUCLOUD Goal Plan",
        f"- Intent: {_intent_label(intent)}",
        f"- Config: {prepared.path}",
        f"- Dataset: {prepared.effective_dataset}",
        f"- Model: {prepared.effective_model}",
        f"- {_cluster_line(cluster)}",
    ]
    if stopped_stage:
        lines.extend(
            [
                f"- Stopped at: {stopped_stage}",
                f"- Reason: {error or 'not available'}",
                "- No training or inference command was started.",
            ]
        )
        return "\n".join(lines)

    dry_runs = dry_runs or {}
    train = dry_runs.get("train") or {}
    if train:
        lines.extend(
            [
                "- Train dry-run:",
                f"  command: {train.get('launch_command')}",
                f"  log: {train.get('log_path')}",
                f"  checkpoint: {train.get('checkpoint_path')}",
            ]
        )
        comm = megatron_communication_notes()
        lines.extend(
            [
                "- Megatron communication:",
                f"  default: {comm['default_communication']}",
                f"  multi-node: {comm['multi_node_requirement']}",
                f"  hetero GPU: {comm['heterogeneous_gpu_warning']}",
            ]
        )
    infer = dry_runs.get("infer") or {}
    if infer:
        lines.extend(
            [
                "- Inference dry-run:",
                f"  command: {infer.get('launch_command')}",
                f"  model_path: {infer.get('model_path')}",
                f"  service_url: {infer.get('service_url')}",
                f"  log: {infer.get('log_path')}",
            ]
        )
    lines.extend(
        [
            "- Execution: dry-run only; no remote training or inference has started.",
            "- Next: ask the user for explicit confirmation before using confirm_execute=true.",
        ]
    )
    return "\n".join(lines)


def run_goal_prepare(
    *,
    goal: str = "",
    mode: Optional[str] = None,
    config_file: Optional[str] = None,
    cluster_name: Optional[str] = None,
    node_index: int = 0,
    allow_discover_without_goal: bool = False,
) -> Dict[str, Any]:
    """Run the phase-9 prep flow through cluster check and dry-run only."""
    try:
        prepared = resolve_config_for_tool(
            config_file,
            allow_discover_without_goal=allow_discover_without_goal,
        )
    except GpucloudConfigError as exc:
        return {
            "ok": False,
            "stage": "config",
            "error": str(exc),
            "next_steps": _diagnostic_steps("config"),
        }

    intent = infer_goal_intent(goal, mode)
    cluster = run_cluster_check(
        config_file=config_file,
        cluster_name=cluster_name,
        allow_discover_without_goal=allow_discover_without_goal,
    )
    if not cluster.get("ok"):
        error = "cluster check failed; stopped before training/inference dry-run"
        return {
            "ok": False,
            "stage": "cluster_check",
            "intent": intent,
            "config_path": str(prepared.path),
            "cluster_check": cluster,
            "error": error,
            "next_steps": _diagnostic_steps("cluster_check"),
            "plan_summary": build_plan_summary(
                intent=intent,
                prepared=prepared,
                cluster=cluster,
                stopped_stage="cluster_check",
                error=error,
            ),
        }

    dry_runs: Dict[str, Any] = {}
    if intent in {"train", "train_and_infer"}:
        dry_runs["train"] = run_train_start(
            config_file=config_file,
            cluster_name=cluster_name,
            node_index=node_index,
            dry_run=True,
            confirm_execute=False,
            allow_discover_without_goal=allow_discover_without_goal,
        )
    if intent in {"infer", "train_and_infer"}:
        dry_runs["infer"] = run_infer_start(
            config_file=config_file,
            cluster_name=cluster_name,
            node_index=node_index,
            dry_run=True,
            confirm_execute=False,
            allow_discover_without_goal=allow_discover_without_goal,
        )

    failed = {name: data for name, data in dry_runs.items() if not data.get("ok")}
    if failed:
        error = "one or more dry-run plans failed"
        return {
            "ok": False,
            "stage": "dry_run",
            "intent": intent,
            "config_path": str(prepared.path),
            "cluster_check": cluster,
            "dry_runs": dry_runs,
            "error": error,
            "next_steps": _diagnostic_steps("dry_run"),
            "communication": megatron_communication_notes(),
            "plan_summary": build_plan_summary(
                intent=intent,
                prepared=prepared,
                cluster=cluster,
                dry_runs=dry_runs,
                stopped_stage="dry_run",
                error=error,
            ),
        }

    return {
        "ok": True,
        "stage": "dry_run",
        "intent": intent,
        "config_path": str(prepared.path),
        "dataset": prepared.effective_dataset,
        "model": prepared.effective_model,
        "cluster_check": cluster,
        "dry_runs": dry_runs,
        "communication": megatron_communication_notes(),
        "plan_summary": build_plan_summary(
            intent=intent,
            prepared=prepared,
            cluster=cluster,
            dry_runs=dry_runs,
        ),
        "message": (
            "Goal preparation reached dry-run only. Review commands and ask the "
            "user for explicit confirmation before starting remote work."
        ),
    }
