from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
from typing import Any
import os
import json
import re
import time
from datetime import datetime, timezone
from collections import Counter, deque

from ....nlu.clarification_budget import ClarificationBudget, ClarificationBudgetState
from ....db import Database
from ....embedding import EmbeddingService
from ....executor import LocalExecutor
from ....nlu.followup_resolver import FollowUpResolution, FollowUpResolver
from ....nlu.intent_parser import IntentParser
from ....language_utils import insufficient_evidence_message, resolve_response_language
from ....local_planner import LocalPlanner
from ....memory_service import MemoryService
from ....models import *
from ....composition.composer import ResponseComposer
from ....retrieval import extract_query_hints, merge_filters, retrieve_bundle
from ....vector_store import VectorStore
from ....verifier import ResultVerifier

class SettingsSysHelpers:
    def __init__(self, dependencies: dict[str, Any]):
        self._db = dependencies.get('db')
        self._memory = dependencies.get('memory')
        self._embedding = dependencies.get('embedding_service')
        self._vector_store = dependencies.get('vector_store')
        self._composer = dependencies.get('composer')
        self._executor = dependencies.get('executor')
        self._intent_parser = dependencies.get('intent_parser')
        self._followup = dependencies.get('followup_resolver')
        self._reranker = getattr(dependencies.get('embedding_service'), '_reranker', None)
        self._clarification_budget = dependencies.get('clarification_budget')
        self._capabilities = dependencies.get('capabilities')

    @staticmethod
    def system_memory_gb() -> int:
        override = str(os.getenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "")).strip()
        if override:
            try:
                parsed = int(override)
                if parsed > 0:
                    return parsed
            except Exception:
                pass
        # macOS fast path
        try:
            import subprocess

            proc = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                mem_bytes = int((proc.stdout or "").strip() or "0")
                if mem_bytes > 0:
                    return max(1, int(mem_bytes / (1024**3)))
        except Exception:
            pass
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            total_bytes = page_size * phys_pages
            return max(1, int(total_bytes / (1024**3)))
        except Exception:
            return 16

    @classmethod
    def get_performance_config(cls) -> dict[str, Any]:
        memory_gb = cls.system_memory_gb()
        if memory_gb <= 16:
            return {
                "max_tokens": 2048,
                "rerank_top_k": 5,
                "retrieval_limit": 20,
                "reflection_depth": "standard",
                "tier": "Efficient",
                "label": "Eco"
            }
        if memory_gb <= 32:
            return {
                "max_tokens": 4096,
                "rerank_top_k": 10,
                "retrieval_limit": 50,
                "reflection_depth": "enhanced",
                "tier": "Balanced",
                "label": "Standard"
            }
        if memory_gb <= 64:
            return {
                "max_tokens": 8192,
                "rerank_top_k": 20,
                "retrieval_limit": 100,
                "reflection_depth": "deep",
                "tier": "Pro",
                "label": "High Performance"
            }
        if memory_gb <= 128:
            return {
                "max_tokens": 16384,
                "rerank_top_k": 30,
                "retrieval_limit": 200,
                "reflection_depth": "ultra",
                "tier": "Ultimate",
                "label": "Studio Elite"
            }
        return {
            "max_tokens": 32768,
            "rerank_top_k": 50,
            "retrieval_limit": 400,
            "reflection_depth": "ultra",
            "tier": "God Mode",
            "label": "Infinite Performance"
        }

    @classmethod
    def memory_capped_conversation_max_tokens(cls) -> int:
        return cls.get_performance_config()["max_tokens"]

    def effective_behavior_policy(
        self,
        *,
        req: LocalChatRequestV2,
        memory_bundle,
        default_action_order: list[str],
        default_mode: str | None,
        workspace_weights: dict[str, float],
    ) -> BehaviorPolicy:
        stored = self._db.get_behavior_policy()
        overrides = req.behavior_overrides
        preferred_action_order = list(stored.preferred_action_order)
        if default_action_order:
            parsed_order: list[SuggestedActionKind] = []
            for raw in default_action_order:
                try:
                    parsed_order.append(SuggestedActionKind(raw))
                except Exception:
                    continue
            if parsed_order:
                preferred_action_order = parsed_order

        preferred_mode = stored.preferred_mode
        if preferred_mode is None and default_mode:
            try:
                preferred_mode = WorkMode(default_mode)
            except Exception:
                preferred_mode = None
        if preferred_mode is None:
            for item in memory_bundle.workspace_items:
                if item.memory_type == "default_mode":
                    mode_value = str(item.value_json.get("value") or "").strip()
                    if mode_value:
                        try:
                            preferred_mode = WorkMode(mode_value)
                        except Exception:
                            preferred_mode = None
                        break

        merged = BehaviorPolicy(
            workspace_weights=dict(workspace_weights or stored.workspace_weights),
            preferred_mode=preferred_mode,
            preferred_action_order=preferred_action_order,
            preferred_response_length=stored.preferred_response_length,
        )
        for item in memory_bundle.workspace_items:
            if item.memory_type == "retrieval_weight":
                try:
                    merged.workspace_weights[item.key] = float(item.value_json.get("weight"))
                except Exception:
                    continue
            if item.memory_type == "preferred_actions":
                actions = item.value_json.get("actions")
                if isinstance(actions, list):
                    parsed_actions: list[SuggestedActionKind] = []
                    for raw in actions:
                        try:
                            parsed_actions.append(SuggestedActionKind(str(raw)))
                        except Exception:
                            continue
                    if parsed_actions:
                        merged.preferred_action_order = parsed_actions

        if overrides is not None:
            if overrides.workspace_weights is not None:
                merged.workspace_weights = dict(overrides.workspace_weights)
            if overrides.preferred_mode is not None:
                merged.preferred_mode = overrides.preferred_mode
            if overrides.preferred_action_order is not None:
                merged.preferred_action_order = list(overrides.preferred_action_order)
            if overrides.preferred_response_length is not None:
                merged.preferred_response_length = overrides.preferred_response_length
            # Keep local personalization policy synced with explicit override input.
            self._db.update_behavior_policy(merged)
        return merged

    @classmethod
    def conversation_max_tokens(
        cls,
        response_length: str,
        model_profile: str = "recommended",
        query: str = "",
        adaptive_scale_override: float | None = None,
    ) -> int:
        """Map response_length preference to a conversation max_tokens cap tuned for interactive chat latency."""
        profile = str(model_profile or "recommended").lower()
        query_text = str(query or "").strip()
        lowered = query_text.lower()

        # Conversation-first defaults: keep outputs responsive on local hardware.
        mapping = {
            "short": 256,
            "medium": 512,
            "long": 1024,
        }
        base = mapping.get(str(response_length).lower(), 512)

        # Scale by profile but avoid runaway default budgets.
        if profile == "deep" or profile == "advanced":
            multiplier = 1.5
        elif profile == "fast":
            multiplier = 0.8
        else:
            multiplier = 1.0

        # Casual / phatic chat should stay concise regardless of preferred length.
        casual_tokens = ("안녕", "hi", "hello", "ㅎㅇ", "뭐해", "how are you", "날씨 어때", "점심 뭐", "저녁 뭐")
        if query_text and any(token in lowered for token in casual_tokens):
            base = min(base, 256)

        # Explicit detail request may increase budget moderately.
        if query_text and cls.is_detailed_explanation_requested(query_text):
            multiplier *= 1.25

        if adaptive_scale_override is not None:
            try:
                multiplier *= max(0.5, min(1.5, float(adaptive_scale_override)))
            except Exception:
                pass

        scaled = int(base * multiplier)
        capped = min(scaled, cls.memory_capped_conversation_max_tokens())
        return max(160, capped)

    @staticmethod
    def is_detailed_explanation_requested(query: str) -> bool:
        # Placeholder for detailed explanation detection logic
        return any(keyword in query.lower() for keyword in ["detailed", "explain in depth", "핵심", "자세히"])

    @staticmethod
    def model_size_b(model_name: str) -> int | None:
        if not model_name:
            return None
        match = re.search(r"(\d+)([bB])", model_name)
        if match:
            return int(match.group(1))
        return None

    @classmethod
    def is_16gb_tier_model(cls, settings) -> bool:
        model_name = getattr(settings, "llama_model_path", "") or getattr(settings, "mlx_model_path", "")
        size = cls.model_size_b(str(model_name))
        if size is None:
            return True # Fallback to optimistic
        return size <= 16
