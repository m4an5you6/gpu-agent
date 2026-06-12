"""Inbound event routing: record / queue / guide / interrupt / execute_direct."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from plugins.cluster.config import ClusterConfig, RouteMode
from plugins.cluster.models import ClusterEvent, new_id
from plugins.cluster.store import ClusterStore

_log = logging.getLogger(__name__)

SteerFn = Callable[[str], bool]
InterruptFn = Callable[[str], bool]
ExecuteFn = Callable[[Dict[str, Any]], Any]


@dataclass
class EventBridgeCallbacks:
    on_queue: Optional[Callable[[str, ClusterEvent], None]] = None
    on_guide: Optional[SteerFn] = None
    on_interrupt: Optional[InterruptFn] = None
    on_execute_direct: Optional[ExecuteFn] = None


@dataclass
class ClusterEventBridge:
    cfg: ClusterConfig
    store: ClusterStore
    callbacks: EventBridgeCallbacks = field(default_factory=EventBridgeCallbacks)
    _queued: Deque[ClusterEvent] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def route_mode_for(self, event_type: str) -> RouteMode:
        routing = self.cfg.event_routing or {}
        return routing.get(event_type) or routing.get("default") or "record"  # type: ignore[return-value]

    def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        job_id: str = "",
        node_id: str = "",
        request_id: str = "",
        route_mode: Optional[RouteMode] = None,
    ) -> ClusterEvent:
        mode = route_mode or self.route_mode_for(event_type)
        event = ClusterEvent(
            event_id=new_id("ev-"),
            event_type=event_type,
            payload=payload,
            route_mode=mode,
            job_id=job_id,
            node_id=node_id,
            request_id=request_id,
        )
        self.store.append_event(event)
        self._dispatch(event)
        return event

    def _dispatch(self, event: ClusterEvent) -> None:
        mode = event.route_mode
        text = self._format_message(event)

        if mode == "record":
            return

        if mode == "execute_direct":
            if self.callbacks.on_execute_direct:
                try:
                    self.callbacks.on_execute_direct(event.payload)
                except Exception as exc:
                    _log.warning("execute_direct handler failed: %s", exc)
            return

        if mode == "guide":
            if self.callbacks.on_guide:
                self.callbacks.on_guide(text)
            else:
                with self._lock:
                    self._queued.append(event)
            return

        if mode == "interrupt":
            if self.callbacks.on_interrupt:
                self.callbacks.on_interrupt(text)
            else:
                with self._lock:
                    self._queued.append(event)
            return

        # queue (default for non-record modes without handler)
        if self.callbacks.on_queue:
            self.callbacks.on_queue(text, event)
        else:
            with self._lock:
                self._queued.append(event)

    @staticmethod
    def _format_message(event: ClusterEvent) -> str:
        summary = event.payload.get("summary") or event.payload.get("message") or ""
        return (
            f"[cluster:{event.event_type}] job={event.job_id or '-'} "
            f"node={event.node_id or '-'} {summary}".strip()
        )

    def drain_queue(self) -> List[ClusterEvent]:
        with self._lock:
            items = list(self._queued)
            self._queued.clear()
            return items

    def wire_gateway_agent(self, agent: Any) -> None:
        """Attach steer/interrupt callbacks to a running AIAgent instance."""

        def on_guide(text: str) -> bool:
            steer = getattr(agent, "steer", None)
            if callable(steer):
                return bool(steer(text))
            return False

        def on_interrupt(text: str) -> bool:
            interrupt = getattr(agent, "interrupt", None)
            if callable(interrupt):
                interrupt(text)
                return True
            return False

        self.callbacks.on_guide = on_guide
        self.callbacks.on_interrupt = on_interrupt

    def wire_cli_inject(self, ctx: Any) -> None:
        """Use plugin context inject_message for queue delivery."""

        def on_queue(text: str, _event: ClusterEvent) -> None:
            inject = getattr(ctx, "inject_message", None)
            if callable(inject):
                inject(text)

        self.callbacks.on_queue = on_queue
