"""Non-interactive AutoGoal loop for training, inference, and deployment.

``/autogoal`` is intentionally separate from the plain ``/goal`` Ralph loop.
It owns long-running ML/service automation where the agent should continue
autonomously after the first user sentence. Missing information is handled by
inspection, conservative defaults, internal self-audit, or a blocked state,
never by asking the user follow-up questions.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpucloud_cli.goals import DEFAULT_MAX_TURNS, judge_goal

logger = logging.getLogger(__name__)

DEFAULT_AUTOGOAL_MAX_TURNS = 200

AUTO_GOAL_KICKOFF_TEMPLATE = """[AutoGoal: non-interactive autonomous ML/service loop]
Objective:
{goal}

Operating contract:
- You are running under /autogoal, not /goal. This is an autonomous training,
  inference, or deployment service loop for multi-GPU, multi-node, and multi-IP
  scenarios.
- Do not ask the user questions. Do not call clarify. Do not wait for user
  confirmation. Consume only the initial objective and continue autonomously.
- If information is missing, inspect the host/repo/config, probe available
  infrastructure, or choose conservative defaults and record your rationale.
- Before starting training, inference, deployment, or any high-risk remote
  action, write an internal `decision_record` in your response covering:
  inputs, assumptions, risks, rollback/stop plan, and why proceeding is safe.
- If you cannot safely proceed after self-audit, enter a blocked state by
  explicitly writing `AUTO_GOAL_BLOCKED:` followed by the reason and next safe
  action. Do not ask the user to decide.
- Prefer dry-run, preflight, health checks, reversible steps, and clear logs.

Optional gpucloud.yaml context:
{config_context}

Start by discovering the current repo, runtime, cluster/GPU/SSH state, conda
or venv environments, data/checkpoint/scratch paths, and any existing training
or deployment configuration. Then make the next concrete autonomous step.
"""

AUTO_GOAL_CONTINUATION_TEMPLATE = """[Continuing AutoGoal]
Objective:
{goal}

Last audit/judge reason:
{reason}

Continue autonomously toward the objective. Do not ask the user questions, do
not call clarify, and do not wait for confirmation. Inspect, infer, choose a
conservative default, run self-audit, proceed if safe, or explicitly block with
`AUTO_GOAL_BLOCKED:` if no safe path remains.
"""

AUTO_GOAL_JUDGE_GOAL_TEMPLATE = """Autonomous /autogoal objective:
{goal}

Completion criteria:
- The training, inference, or deployment objective is complete; OR
- The agent explicitly entered AUTO_GOAL_BLOCKED with a concrete safety reason.

The agent must not ask the user questions or wait for confirmation."""


@dataclass
class AutoGoalState:
    """Serializable AutoGoal state stored independently from /goal."""

    goal: str
    status: str = "active"  # active | paused | done | cleared | blocked
    turns_used: int = 0
    max_turns: int = DEFAULT_AUTOGOAL_MAX_TURNS
    created_at: float = 0.0
    last_turn_at: float = 0.0
    last_verdict: Optional[str] = None
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None
    config_path: str = ""
    config_warnings: List[str] = field(default_factory=list)
    decision_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "AutoGoalState":
        data = json.loads(raw)
        return cls(
            goal=str(data.get("goal") or ""),
            status=str(data.get("status") or "active"),
            turns_used=int(data.get("turns_used", 0) or 0),
            max_turns=int(data.get("max_turns", DEFAULT_AUTOGOAL_MAX_TURNS) or DEFAULT_AUTOGOAL_MAX_TURNS),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_turn_at=float(data.get("last_turn_at", 0.0) or 0.0),
            last_verdict=data.get("last_verdict"),
            last_reason=data.get("last_reason"),
            paused_reason=data.get("paused_reason"),
            config_path=str(data.get("config_path") or ""),
            config_warnings=[
                str(item) for item in (data.get("config_warnings") or []) if str(item).strip()
            ],
            decision_records=[
                item for item in (data.get("decision_records") or []) if isinstance(item, dict)
            ],
        )


def _meta_key(session_id: str) -> str:
    return f"autogoal:{session_id}"


def _get_session_db() -> Optional[Any]:
    try:
        from gpucloud_cli import goals

        return goals._get_session_db()  # Reuse the profile-aware DB cache.
    except Exception as exc:  # pragma: no cover
        logger.debug("AutoGoalManager: SessionDB bootstrap failed (%s)", exc)
        return None


def load_autogoal(session_id: str) -> Optional[AutoGoalState]:
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("AutoGoalManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        return AutoGoalState.from_json(raw)
    except Exception as exc:
        logger.warning("AutoGoalManager: could not parse stored autogoal for %s: %s", session_id, exc)
        return None


def save_autogoal(session_id: str, state: AutoGoalState) -> None:
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("AutoGoalManager: set_meta failed: %s", exc)


def _discover_gpucloud_yaml() -> Optional[Path]:
    explicit = os.environ.get("GPUCLOUD_CONFIG", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    for path in (Path.cwd() / "gpucloud.yaml", Path.cwd() / ".gpucloud" / "config.yaml"):
        if path.is_file():
            return path
    return None


def _load_optional_gpucloud_context() -> tuple[str, str, List[str]]:
    """Return ``(path, context, warnings)`` for optional gpucloud.yaml."""
    path = _discover_gpucloud_yaml()
    if path is None:
        return "", "No gpucloud.yaml found. Continue with automatic discovery and conservative defaults.", [
            "gpucloud.yaml not found; autogoal will auto-discover configuration"
        ]

    warnings: List[str] = []
    try:
        import yaml
    except Exception:
        return str(path), f"Found {path}, but PyYAML is unavailable; treat it as unreadable.", [
            "PyYAML unavailable; gpucloud.yaml could not be parsed"
        ]

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return str(path), f"Found {path}, but parsing failed: {type(exc).__name__}: {exc}", [
            f"gpucloud.yaml parse failed: {type(exc).__name__}"
        ]

    if not isinstance(data, dict):
        return str(path), f"Found {path}, but the root is not a mapping.", [
            "gpucloud.yaml root is not a mapping"
        ]

    missing: List[str] = []
    clusters = data.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        missing.append("clusters")
    for field_name in ("dataset_name", "model_name"):
        value = data.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_name)
    if missing:
        warnings.append(f"gpucloud.yaml missing recommended fields: {', '.join(missing)}")

    summary = {
        "clusters": len(clusters) if isinstance(clusters, list) else 0,
        "dataset_name": data.get("dataset_name") or "",
        "model_name": data.get("model_name") or "",
        "has_training": isinstance(data.get("training"), dict),
        "has_inference": isinstance(data.get("inference"), dict),
        "warnings": warnings,
    }
    return str(path), json.dumps(summary, ensure_ascii=False, indent=2), warnings


def _looks_blocked(text: str) -> bool:
    lowered = (text or "").lower()
    return "auto_goal_blocked:" in lowered or "autogoal_blocked:" in lowered


class AutoGoalManager:
    """Per-session non-interactive autogoal state and continuation logic."""

    def __init__(self, session_id: str, *, default_max_turns: int = DEFAULT_AUTOGOAL_MAX_TURNS):
        self.session_id = session_id
        self.default_max_turns = int(default_max_turns or DEFAULT_AUTOGOAL_MAX_TURNS)
        self._state: Optional[AutoGoalState] = load_autogoal(session_id)

    @property
    def state(self) -> Optional[AutoGoalState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_autogoal(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused", "blocked"}

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status == "cleared":
            return "No active autogoal. Set one with /autogoal <objective>."
        turns = f"{s.turns_used}/{s.max_turns} turns"
        warn = f", {len(s.config_warnings)} warning(s)" if s.config_warnings else ""
        if s.status == "active":
            return f"⊙ AutoGoal (active, {turns}{warn}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ AutoGoal (paused, {turns}{warn}{extra}): {s.goal}"
        if s.status == "blocked":
            reason = f" — {s.last_reason}" if s.last_reason else ""
            return f"■ AutoGoal blocked ({turns}{warn}{reason}): {s.goal}"
        if s.status == "done":
            return f"✓ AutoGoal done ({turns}{warn}): {s.goal}"
        return f"AutoGoal ({s.status}, {turns}{warn}): {s.goal}"

    def set(self, goal: str, *, max_turns: Optional[int] = None) -> AutoGoalState:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("autogoal text is empty")
        config_path, _context, warnings = _load_optional_gpucloud_context()
        state = AutoGoalState(
            goal=goal,
            status="active",
            turns_used=0,
            max_turns=int(max_turns) if max_turns else self.default_max_turns,
            created_at=time.time(),
            last_turn_at=0.0,
            config_path=config_path,
            config_warnings=warnings,
        )
        self._state = state
        save_autogoal(self.session_id, state)
        return state

    def pause(self, reason: str = "user-paused") -> Optional[AutoGoalState]:
        if not self._state:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        save_autogoal(self.session_id, self._state)
        return self._state

    def resume(self, *, reset_budget: bool = True) -> Optional[AutoGoalState]:
        if not self._state:
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        if reset_budget:
            self._state.turns_used = 0
        save_autogoal(self.session_id, self._state)
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._state.status = "cleared"
        save_autogoal(self.session_id, self._state)
        self._state = None

    def kickoff_prompt(self) -> str:
        config_path, config_context, warnings = _load_optional_gpucloud_context()
        if self._state:
            self._state.config_path = config_path
            self._state.config_warnings = warnings
            save_autogoal(self.session_id, self._state)
        return AUTO_GOAL_KICKOFF_TEMPLATE.format(
            goal=self._state.goal if self._state else "",
            config_context=config_context,
        )

    def next_continuation_prompt(self, reason: str = "") -> Optional[str]:
        if not self._state or self._state.status != "active":
            return None
        return AUTO_GOAL_CONTINUATION_TEMPLATE.format(
            goal=self._state.goal,
            reason=reason or self._state.last_reason or "continue",
        )

    def evaluate_after_turn(self, last_response: str, *, user_initiated: bool = True) -> Dict[str, Any]:
        state = self._state
        if state is None or state.status != "active":
            return {
                "status": state.status if state else None,
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active autogoal",
                "message": "",
            }

        state.turns_used += 1
        state.last_turn_at = time.time()

        if _looks_blocked(last_response):
            state.status = "blocked"
            state.last_verdict = "blocked"
            state.last_reason = "agent entered AUTO_GOAL_BLOCKED"
            save_autogoal(self.session_id, state)
            return {
                "status": "blocked",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "blocked",
                "reason": state.last_reason,
                "message": f"■ AutoGoal blocked: {state.last_reason}",
            }

        judge_goal_text = AUTO_GOAL_JUDGE_GOAL_TEMPLATE.format(goal=state.goal)
        verdict, reason, _parse_failed = judge_goal(judge_goal_text, last_response)
        state.last_verdict = verdict
        state.last_reason = reason

        if verdict == "done":
            state.status = "done"
            save_autogoal(self.session_id, state)
            return {
                "status": "done",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "done",
                "reason": reason,
                "message": f"✓ AutoGoal achieved: {reason}",
            }

        if state.turns_used >= state.max_turns:
            state.status = "paused"
            state.paused_reason = f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
            save_autogoal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ AutoGoal paused — {state.turns_used}/{state.max_turns} turns used. "
                    "Use /autogoal resume to continue, or /autogoal clear to stop."
                ),
            }

        save_autogoal(self.session_id, state)
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": self.next_continuation_prompt(reason),
            "verdict": "continue",
            "reason": reason,
            "message": (
                f"↻ Continuing autogoal ({state.turns_used}/{state.max_turns}): {reason}"
            ),
        }
