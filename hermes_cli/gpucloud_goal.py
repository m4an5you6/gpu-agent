"""GPUCLOUD /goal workflow planning (phase 9)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from hermes_cli.gpucloud_config import GpucloudConfigError
from hermes_cli.gpucloud_context import resolve_config_for_tool
from hermes_cli.gpucloud_inference import run_infer_start
from hermes_cli.gpucloud_probe import run_cluster_check
from hermes_cli.gpucloud_train import run_train_start

if TYPE_CHECKING:
    from hermes_cli.gpucloud_config import GpucloudPreparedConfig

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
        "checkpoint/model path(s), and ask for explicit user confirmation before execution.\n"
        "4. Do not set confirm_execute=true, confirm_stop=true, or confirm_delete=true "
        "unless the user explicitly confirms in the current conversation.\n"
        "5. After a confirmed start, use gpucloud_train_status/gpucloud_train_logs/"
        "gpucloud_checkpoint_latest for training and gpucloud_infer_status/"
        "gpucloud_infer_health for inference monitoring.\n"
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
        return {
            "ok": False,
            "stage": "cluster_check",
            "intent": intent,
            "config_path": str(prepared.path),
            "cluster_check": cluster,
            "error": "cluster check failed; stopped before training/inference dry-run",
            "next_steps": _diagnostic_steps("cluster_check"),
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
        return {
            "ok": False,
            "stage": "dry_run",
            "intent": intent,
            "config_path": str(prepared.path),
            "cluster_check": cluster,
            "dry_runs": dry_runs,
            "error": "one or more dry-run plans failed",
            "next_steps": _diagnostic_steps("dry_run"),
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
        "message": (
            "Goal preparation reached dry-run only. Review commands and ask the "
            "user for explicit confirmation before starting remote work."
        ),
    }
