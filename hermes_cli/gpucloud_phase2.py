"""GPUCLOUD phase 2: peripheral entry and registration cutover.

Stage 2 removes user-visible references to messaging, TUI, IDE, dashboard,
and entertainment peripherals without deleting implementation directories.
Stage 3 deletes the unused code after references are gone.
"""

from __future__ import annotations

from fnmatch import fnmatch

# Top-level CLI subcommands hidden from argparse registration.
DISABLED_CLI_COMMANDS: frozenset[str] = frozenset(
    {
        "gateway",
        "dashboard",
        "webhook",
        "kanban",
        "whatsapp",
        "slack",
        "send",
        "lsp",
        "acp",
        "computer-use",
        "pairing",
        "portal",
    }
)

# Slash commands removed from COMMAND_REGISTRY exposure (CLI + gateway menus).
DISABLED_SLASH_COMMANDS: frozenset[str] = frozenset(
    {
        "handoff",
        "kanban",
        "browser",
        "voice",
        "image",
        "paste",
        "platforms",
        "platform",
        "start",
        "topic",
        "approve",
        "deny",
        "sethome",
        "set-home",
        "restart",
        "commands",
        "footer",
        "indicator",
        "plugins",
    }
)

# Built-in tool modules skipped during registry discovery (files remain on disk).
DISABLED_TOOL_MODULES: frozenset[str] = frozenset(
    {
        "browser_tool.py",
        "browser_cdp_tool.py",
        "browser_dialog_tool.py",
        "browser_camofox.py",
        "image_generation_tool.py",
        "video_generation_tool.py",
        "vision_tools.py",
        "tts_tool.py",
        "transcription_tools.py",
        "voice_mode.py",
        "send_message_tool.py",
        "homeassistant_tool.py",
        "kanban_tools.py",
        "discord_tool.py",
        "feishu_doc_tool.py",
        "feishu_drive_tool.py",
        "yuanbao_tools.py",
        "x_search_tool.py",
        "computer_use_tool.py",
    }
)

# Default CLI/cron tool names (hermes-cli / hermes-cron core lists).
GPUCLOUD_CORE_TOOLS: list[str] = [
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "terminal",
    "process",
    "skills_list",
    "skill_view",
    "skill_manage",
    "todo",
    "memory",
    "session_search",
    "clarify",
    "execute_code",
    "delegate_task",
    "cronjob",
    "gpucloud_cluster_check",
    "gpucloud_ssh_exec",
    "gpucloud_gpu_probe",
    "gpucloud_train_start",
    "gpucloud_train_status",
    "gpucloud_train_logs",
    "gpucloud_infer_start",
    "gpucloud_infer_status",
    "gpucloud_infer_health",
    "gpucloud_infer_stop",
    "gpucloud_goal_prepare",
    "gpucloud_checkpoint_list",
    "gpucloud_checkpoint_latest",
    "gpucloud_checkpoint_validate",
    "gpucloud_train_resume",
    "gpucloud_checkpoint_cleanup",
]

# Bundled plugin keys (path-derived) not auto-loaded in phase 2.
DISABLED_BUNDLED_PLUGIN_KEYS: frozenset[str] = frozenset(
    {
        "spotify",
        "google_meet",
        "teams_pipeline",
        "disk-cleanup",
        "dashboard_auth/nous",
    }
)

# Glob patterns for bundled plugin keys under category directories.
DISABLED_BUNDLED_PLUGIN_GLOBS: tuple[str, ...] = (
    "platforms/*",
    "browser/*",
    "image_gen/*",
    "video_gen/*",
)


def cli_command_enabled(name: str) -> bool:
    return name not in DISABLED_CLI_COMMANDS


def slash_command_enabled(name: str) -> bool:
    return name.lstrip("/").lower() not in DISABLED_SLASH_COMMANDS


def tool_module_enabled(filename: str) -> bool:
    return filename not in DISABLED_TOOL_MODULES


def bundled_plugin_enabled(lookup_key: str) -> bool:
    key = (lookup_key or "").strip()
    if not key:
        return True
    if key in DISABLED_BUNDLED_PLUGIN_KEYS:
        return False
    return not any(fnmatch(key, pattern) for pattern in DISABLED_BUNDLED_PLUGIN_GLOBS)


def filter_command_registry(commands: list) -> list:
    """Return CommandDef entries allowed in GPUCLOUD phase 2."""
    return [cmd for cmd in commands if slash_command_enabled(cmd.name)]


def prune_disabled_cli_subcommands(
    parser: "argparse.ArgumentParser",
    subparsers: "argparse._SubParsersAction",
) -> None:
    """Remove disabled subcommands from routing and from ``--help`` output."""
    import argparse

    for name in DISABLED_CLI_COMMANDS:
        if hasattr(subparsers, "choices") and subparsers.choices is not None:
            subparsers.choices.pop(name, None)

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            action._choices_actions = [
                choice_action
                for choice_action in action._choices_actions
                if getattr(choice_action, "dest", None) not in DISABLED_CLI_COMMANDS
            ]
            break
