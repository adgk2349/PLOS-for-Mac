from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import time
from typing import Any

from .db import Database
from .models import (
    MemoryClearResponse,
    MemoryClearScope,
    MemoryEventRequest,
    MemoryEventResponse,
    PinnedMemoryItem,
    RelevantMemoryBundle,
    UserPreferenceItem,
    WorkspaceIdentity,
    WorkspaceMemoryMode,
)


@dataclass(slots=True)
class ResolvedMemoryPreferences:
    response_length: str = "medium"
    show_citations: bool = True
    confirm_external_calls: bool = False
    prefer_action_suggestions: bool = True
    default_action_order: list[str] = None  # type: ignore[assignment]
    default_mode: str | None = None
    workspace_weights: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.default_action_order is None:
            self.default_action_order = []
        if self.workspace_weights is None:
            self.workspace_weights = {}


class MemoryService:
    def __init__(self, db: Database):
        self._db = db
        self._last_inferred_refresh_by_workspace: dict[str, float] = {}

    def get_workspace_identity(self) -> WorkspaceIdentity:
        workspace = self._db.get_workspace()
        return self._db.get_workspace_identity(workspace)

    def getRelevantSessionMemory(self, session_id: str) -> list:
        settings = self._db.get_settings()
        if not settings.session_memory_enabled:
            return []
        return self._db.get_relevant_session_memory(session_id=session_id, limit=40)

    def getLastConversationalContext(self, session_id: str) -> dict[str, Any] | None:
        for item in self.getRelevantSessionMemory(session_id):
            if item.key != "last_conversational_context":
                continue
            payload = item.value_json
            if isinstance(payload, dict):
                return payload
        return None

    def getLastCandidateSet(self, session_id: str) -> list[str]:
        context = self.getLastConversationalContext(session_id)
        output: list[str] = []
        if context:
            candidates = context.get("top_candidates")
            if isinstance(candidates, list):
                for item in candidates:
                    if isinstance(item, str) and item.strip():
                        output.append(item.strip())
        if output:
            return output[:8]
        for item in self.getRelevantSessionMemory(session_id):
            if item.key != "recent_file_ids":
                continue
            file_ids = item.value_json.get("file_ids")
            if isinstance(file_ids, list):
                return [str(v).strip() for v in file_ids if str(v).strip()][:8]
        return []

    def getLastSelectedFile(self, session_id: str) -> str | None:
        context = self.getLastConversationalContext(session_id)
        if context:
            value = context.get("selected_file")
            if isinstance(value, str) and value.strip():
                return value.strip()
        for item in self.getRelevantSessionMemory(session_id):
            if item.key != "recent_file_ids":
                continue
            file_ids = item.value_json.get("file_ids")
            if isinstance(file_ids, list) and file_ids:
                first = file_ids[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
        return None

    def getLastShownActions(self, session_id: str) -> list[str]:
        context = self.getLastConversationalContext(session_id)
        if not context:
            return []
        actions = context.get("shown_actions")
        if not isinstance(actions, list):
            return []
        output: list[str] = []
        for item in actions:
            if isinstance(item, str) and item.strip():
                output.append(item.strip())
        return output[:8]

    def write_conversational_context(
        self,
        *,
        session_id: str,
        context: dict[str, Any],
    ) -> None:
        settings = self._db.get_settings()
        if not settings.session_memory_enabled:
            return
        safe: dict[str, Any] = dict(context)
        safe["updated_at"] = time.time()
        self._db.write_session_memory(
            session_id=session_id,
            key="last_conversational_context",
            value_json=safe,
            ttl_hours=24,
            keep_recent=40,
        )

    def getRelevantWorkspaceMemory(self, workspace_id: str, intent: str | None) -> list:
        settings = self._db.get_settings()
        if not settings.workspace_memory_enabled:
            return []
        if settings.workspace_memory_mode == WorkspaceMemoryMode.DISABLED:
            return []
        if settings.workspace_memory_mode == WorkspaceMemoryMode.PINNED_ONLY:
            return []
        return self._db.get_relevant_workspace_memory(workspace_id=workspace_id, intent=intent, limit=30)

    def getUserPreferences(self) -> list[UserPreferenceItem]:
        settings = self._db.get_settings()
        items = self._db.get_user_preferences()
        if settings.adaptive_personalization_enabled:
            return items
        return [item for item in items if item.source == "explicit"]

    def getRelevantEpisodicMemory(self, workspace_id: str | None, intent: str | None, relatedFiles: list[str]) -> list:
        settings = self._db.get_settings()
        if not settings.workspace_memory_enabled:
            return []
        if settings.workspace_memory_mode == WorkspaceMemoryMode.DISABLED:
            return []
        if settings.workspace_memory_mode == WorkspaceMemoryMode.PINNED_ONLY:
            return []
        return self._db.get_relevant_episodic_memory(
            workspace_id=workspace_id,
            intent=intent,
            related_file_ids=relatedFiles,
            limit=20,
        )

    def get_relevant_memory_bundle(
        self,
        *,
        session_id: str,
        workspace_id: str,
        intent: str,
        related_file_ids: list[str],
    ) -> RelevantMemoryBundle:
        settings = self._db.get_settings()
        session_items = self.getRelevantSessionMemory(session_id)
        workspace_items = self.getRelevantWorkspaceMemory(workspace_id, intent)
        pref_items = self.getUserPreferences()
        episodic_items = self.getRelevantEpisodicMemory(workspace_id, intent, related_file_ids)
        pinned_items = self._db.list_pinned_memory(workspace_id=workspace_id, limit=30)
        if settings.workspace_memory_mode == WorkspaceMemoryMode.PINNED_ONLY:
            workspace_items = []
            episodic_items = []
        identity = self.get_workspace_identity()
        return RelevantMemoryBundle(
            workspace_identity=identity,
            session_items=session_items,
            workspace_items=workspace_items,
            preference_items=pref_items,
            episodic_items=episodic_items,
            pinned_items=pinned_items,
        )

    def writeMemoryEvent(self, event: MemoryEventRequest) -> MemoryEventResponse:
        settings = self._db.get_settings()
        event_id = ""

        if settings.session_memory_enabled and event.session_id:
            self._db.write_session_memory(
                session_id=event.session_id,
                key="recent_event",
                value_json={
                    "event_type": event.event_type.value,
                    "summary": event.summary,
                    "metadata": event.metadata_json,
                },
                ttl_hours=24,
                keep_recent=40,
            )
            if event.event_type.value == "query":
                self._db.write_session_memory(
                    session_id=event.session_id,
                    key="recent_query",
                    value_json={
                        "summary": event.summary,
                        "mode": event.metadata_json.get("mode"),
                        "workspace_id": event.workspace_id,
                    },
                    ttl_hours=24,
                    keep_recent=40,
                )
            if event.related_file_ids:
                self._db.write_session_memory(
                    session_id=event.session_id,
                    key="recent_file_ids",
                    value_json={"file_ids": event.related_file_ids[:8]},
                    ttl_hours=24,
                    keep_recent=40,
                )
            if event.related_action_ids:
                self._db.write_session_memory(
                    session_id=event.session_id,
                    key="recent_action",
                    value_json={"action_ids": event.related_action_ids[:4]},
                    ttl_hours=24,
                    keep_recent=40,
                )

        if (
            settings.workspace_memory_enabled
            and event.workspace_id
            and settings.workspace_memory_mode != WorkspaceMemoryMode.DISABLED
            and settings.workspace_memory_mode != WorkspaceMemoryMode.PINNED_ONLY
        ):
            # Explicit workspace rules from user-initiated setting changes.
            if event.event_type.value == "manual_override":
                if "default_mode" in event.metadata_json:
                    self._db.upsert_workspace_memory(
                        workspace_id=event.workspace_id,
                        memory_type="default_mode",
                        key="default_mode",
                        value_json={"value": event.metadata_json.get("default_mode")},
                        confidence=1.0,
                        source="explicit",
                    )
                if "privacy_rule" in event.metadata_json:
                    self._db.upsert_workspace_memory(
                        workspace_id=event.workspace_id,
                        memory_type="privacy_rule",
                        key="privacy_rule",
                        value_json={"value": event.metadata_json.get("privacy_rule")},
                        confidence=1.0,
                        source="explicit",
                    )

        episodic_enabled = (
            settings.workspace_memory_mode != WorkspaceMemoryMode.DISABLED
            and settings.workspace_memory_mode != WorkspaceMemoryMode.PINNED_ONLY
        )
        if settings.workspace_memory_enabled and episodic_enabled:
            record = self._db.insert_episodic_memory(
                workspace_id=event.workspace_id,
                event_type=event.event_type.value,
                summary=event.summary,
                related_file_ids=event.related_file_ids,
                related_action_ids=event.related_action_ids,
                metadata_json=event.metadata_json,
                importance=event.importance,
            )
            event_id = record.id
            if (
                event.workspace_id
                and settings.adaptive_personalization_enabled
                and self._should_refresh_inferred(event.event_type)
            ):
                self._refresh_inferred_with_throttle(event.workspace_id, min_interval_seconds=30.0)

        return MemoryEventResponse(event_id=event_id or "session-only", accepted=True)

    def clearMemory(
        self,
        *,
        scope: MemoryClearScope,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryClearResponse:
        count = self._db.clear_memory(scope=scope, workspace_id=workspace_id, session_id=session_id)
        return MemoryClearResponse(cleared_rows=count, scope=scope)

    def pinMemory(
        self,
        *,
        memory_id: str | None,
        scope: str,
        workspace_id: str | None,
        title: str | None,
        content: str | None,
    ) -> PinnedMemoryItem:
        if memory_id:
            item = self._db.create_pin_from_memory(memory_id=memory_id, scope=scope, workspace_id=workspace_id)
            if item is not None:
                return item

        safe_title = (title or "Pinned Memory").strip() or "Pinned Memory"
        safe_content = (content or "").strip()
        if not safe_content:
            safe_content = "No content"
        return self._db.create_pinned_memory(
            scope=scope,
            workspace_id=workspace_id,
            title=safe_title,
            content=safe_content,
        )

    def unpinMemory(self, memory_id: str) -> bool:
        return self._db.delete_pinned_memory(memory_id=memory_id)

    def listPinnedMemory(self, *, scope: str | None = None, workspace_id: str | None = None) -> list[PinnedMemoryItem]:
        return self._db.list_pinned_memory(scope=scope, workspace_id=workspace_id, limit=120)

    def resolve_preferences(self, bundle: RelevantMemoryBundle) -> ResolvedMemoryPreferences:
        explicit_map: dict[str, UserPreferenceItem] = {}
        inferred_map: dict[str, UserPreferenceItem] = {}
        for item in bundle.preference_items:
            if item.source == "explicit":
                explicit_map[item.key] = item
            else:
                existing = inferred_map.get(item.key)
                if existing is None or item.confidence > existing.confidence:
                    inferred_map[item.key] = item

        merged = dict(inferred_map)
        merged.update(explicit_map)

        resolved = ResolvedMemoryPreferences()
        if "response_length" in merged:
            resolved.response_length = str(merged["response_length"].value_json.get("value") or "medium")
        if "show_citations" in merged:
            resolved.show_citations = bool(merged["show_citations"].value_json.get("value", True))
        if "confirm_external_calls" in merged:
            resolved.confirm_external_calls = bool(merged["confirm_external_calls"].value_json.get("value", False))
        if "prefer_action_suggestions" in merged:
            resolved.prefer_action_suggestions = bool(merged["prefer_action_suggestions"].value_json.get("value", True))
        action_order: list[str] = []
        explicit_action = explicit_map.get("default_action_order")
        inferred_action = inferred_map.get("default_action_order")
        if explicit_action:
            action_order = self._as_str_list(explicit_action.value_json.get("value"))
        if not action_order:
            action_order = self._action_order_from_pins(bundle.pinned_items)
        if not action_order:
            action_order = self._action_order_from_workspace(bundle.workspace_items)
        if not action_order:
            action_order = self._action_order_from_episodic(bundle.episodic_items)
        if not action_order and inferred_action:
            action_order = self._as_str_list(inferred_action.value_json.get("value"))
        resolved.default_action_order = action_order

        default_mode: str | None = None
        explicit_mode = explicit_map.get("default_mode")
        inferred_mode = inferred_map.get("default_mode")
        if explicit_mode:
            value = explicit_mode.value_json.get("value")
            default_mode = str(value).strip() if value else None
        if not default_mode:
            default_mode = self._default_mode_from_pins(bundle.pinned_items)
        if not default_mode:
            default_mode = self._default_mode_from_workspace(bundle.workspace_items)
        if not default_mode:
            default_mode = self._default_mode_from_episodic(bundle.episodic_items)
        if not default_mode and inferred_mode:
            value = inferred_mode.value_json.get("value")
            default_mode = str(value).strip() if value else None
        resolved.default_mode = default_mode

        weights: dict[str, float] = {}
        for item in bundle.workspace_items:
            if item.memory_type != "retrieval_weight":
                continue
            weight = item.value_json.get("weight")
            try:
                weights[item.key] = max(0.5, min(float(weight), 1.8))
            except Exception:
                continue
        resolved.workspace_weights = weights
        return resolved

    def _refresh_inferred_workspace_memory(self, workspace_id: str) -> None:
        events = self._db.list_recent_episodic_memory(workspace_id=workspace_id, days=7, limit=240)
        if not events:
            return
        mode_counter: Counter[str] = Counter()
        action_counter: Counter[str] = Counter()
        for event in events:
            mode = str(event.metadata_json.get("mode") or "").strip()
            if mode:
                mode_counter[mode] += 1
            action = str(event.metadata_json.get("action_kind") or "").strip()
            if action:
                action_counter[action] += 1

        if mode_counter:
            mode, count = mode_counter.most_common(1)[0]
            if count >= 3:
                self._db.upsert_workspace_memory(
                    workspace_id=workspace_id,
                    memory_type="default_mode",
                    key="default_mode",
                    value_json={"value": mode},
                    confidence=0.62,
                    source="inferred",
                )

        if action_counter:
            ordered = [item for item, count in action_counter.most_common(5) if count >= 3]
            if ordered:
                self._db.upsert_workspace_memory(
                    workspace_id=workspace_id,
                    memory_type="preferred_actions",
                    key="preferred_actions",
                    value_json={"actions": ordered},
                    confidence=0.62,
                    source="inferred",
                )

    def _refresh_inferred_user_preferences(self, workspace_id: str) -> None:
        events = self._db.list_recent_episodic_memory(workspace_id=workspace_id, days=7, limit=240)
        if not events:
            return
        action_counter: Counter[str] = Counter()
        for event in events:
            action = str(event.metadata_json.get("action_kind") or "").strip()
            if action:
                action_counter[action] += 1
        ordered = [item for item, count in action_counter.most_common(5) if count >= 3]
        if ordered:
            self._db.upsert_user_preference(
                key="default_action_order",
                value_json={"value": ordered},
                source="inferred",
                confidence=0.62,
            )

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        output: list[str] = []
        for item in value:
            raw = str(item).strip()
            if raw:
                output.append(raw)
        return output

    def _action_order_from_workspace(self, items: list) -> list[str]:
        for item in items:
            if item.memory_type != "preferred_actions":
                continue
            actions = self._as_str_list(item.value_json.get("actions"))
            if actions:
                return actions
        return []

    @staticmethod
    def _default_mode_from_workspace(items: list) -> str | None:
        for item in items:
            if item.memory_type != "default_mode":
                continue
            value = str(item.value_json.get("value") or "").strip()
            if value:
                return value
        return None

    @staticmethod
    def _default_mode_from_episodic(items: list) -> str | None:
        counter: Counter[str] = Counter()
        for item in items:
            mode = str(item.metadata_json.get("mode") or "").strip()
            if mode:
                counter[mode] += 1
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def _action_order_from_episodic(self, items: list) -> list[str]:
        counter: Counter[str] = Counter()
        for item in items:
            action = str(item.metadata_json.get("action_kind") or "").strip()
            if action:
                counter[action] += 1
        return [action for action, count in counter.most_common(5) if count >= 2]

    def _action_order_from_pins(self, pins: list[PinnedMemoryItem]) -> list[str]:
        for pin in pins:
            payload = self._parse_pin_content(pin.content)
            for key in ("actions", "value"):
                actions = self._as_str_list(payload.get(key))
                if actions:
                    return actions
        return []

    def _default_mode_from_pins(self, pins: list[PinnedMemoryItem]) -> str | None:
        for pin in pins:
            payload = self._parse_pin_content(pin.content)
            for key in ("default_mode", "value", "mode"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _parse_pin_content(content: str) -> dict[str, Any]:
        raw = (content or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    @staticmethod
    def _should_refresh_inferred(event_type) -> bool:
        return event_type.value in {
            "action_executed",
            "manual_override",
            "external_analysis",
            "comparison",
            "summary_created",
            "draft_created",
        }

    def _refresh_inferred_with_throttle(self, workspace_id: str, *, min_interval_seconds: float) -> None:
        now = time.monotonic()
        last = self._last_inferred_refresh_by_workspace.get(workspace_id, 0.0)
        if (now - last) < min_interval_seconds:
            return
        self._last_inferred_refresh_by_workspace[workspace_id] = now
        self._refresh_inferred_workspace_memory(workspace_id)
        self._refresh_inferred_user_preferences(workspace_id)
