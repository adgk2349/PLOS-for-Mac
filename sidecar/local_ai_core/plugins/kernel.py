from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar

from ..models import (
    ExtensionCapabilitiesResponse,
    ExtensionCapability,
    ExtensionCapabilityState,
    PluginPrivacyMode,
    PluginCapabilitySource,
    PluginErrorCode,
    PluginManifestV1,
    PrivacyMode,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)

SUPPORTED_CAPABILITIES: tuple[ExtensionCapability, ...] = (
    ExtensionCapability.RETRIEVER_SEARCH,
    ExtensionCapability.RERANKER_RANK,
    ExtensionCapability.SUMMARIZER_GENERATE,
    ExtensionCapability.RETRIEVAL_QUERY_TRANSFORM,
    ExtensionCapability.RETRIEVAL_POST_FILTER,
    ExtensionCapability.CHUNKING_STRATEGY,
    ExtensionCapability.EMBEDDING_PROVIDER,
    ExtensionCapability.INDEXING_PREPROCESS,
    ExtensionCapability.FINETUNE_JOB_SUBMIT,
    ExtensionCapability.FINETUNE_JOB_STATUS,
    ExtensionCapability.FINETUNE_MODEL_PUBLISH,
)


@dataclass(frozen=True)
class CapabilityInvocation(Generic[T]):
    value: T
    source: PluginCapabilitySource
    plugin_id: str | None = None
    error_code: PluginErrorCode | None = None
    error_message: str | None = None


class PluginRegistry(Protocol):
    def list_registered_plugins(self) -> list[PluginManifestV1]: ...

    def get_enabled_plugin(self, capability: ExtensionCapability) -> PluginManifestV1 | None: ...

    def get_app_privacy_mode(self) -> PrivacyMode: ...


class SQLitePluginRegistry:
    def __init__(self, db):
        self._db = db

    def _rows(self) -> list[dict]:
        return self._db.list_plugin_registry_entries()

    def list_registered_plugins(self) -> list[PluginManifestV1]:
        output: list[PluginManifestV1] = []
        for row in self._rows():
            raw = row.get("manifest_json")
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                manifest = PluginManifestV1.model_validate(payload)
            except Exception:
                continue
            output.append(manifest)
        return output

    def get_enabled_plugin(self, capability: ExtensionCapability) -> PluginManifestV1 | None:
        for row in self._rows():
            if not bool(row.get("enabled")):
                continue
            state = str(row.get("state") or "").lower()
            if state not in {"enabled", "active"}:
                continue
            raw = row.get("manifest_json")
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                manifest = PluginManifestV1.model_validate(payload)
            except Exception:
                continue
            if capability in manifest.capabilities:
                return manifest
        return None

    def get_app_privacy_mode(self) -> PrivacyMode:
        try:
            return self._db.get_settings().privacy_mode
        except Exception:
            return PrivacyMode.HYBRID


class BuiltInCapabilityProvider:
    def retriever_search(self, *, query: str, bundle):
        _ = query
        return bundle

    def reranker_rank(self, *, query: str, chunk_candidates):
        _ = query
        return chunk_candidates

    def summarizer_generate(self, *, fallback: Callable[[], T]) -> T:
        return fallback()

    def retrieval_query_transform(self, *, query: str, fallback: Callable[[], str]) -> str:
        _ = query
        return fallback()

    def retrieval_post_filter(self, *, query: str, chunk_candidates, fallback: Callable[[], list]):
        _ = query
        _ = chunk_candidates
        return fallback()

    def chunking_strategy(self, *, text: str, fallback: Callable[[], list[str]]) -> list[str]:
        _ = text
        return fallback()

    def embedding_provider(self, *, texts: list[str], fallback: Callable[[], list[list[float]]]) -> list[list[float]]:
        _ = texts
        return fallback()

    def indexing_preprocess(self, *, file_path: str, text: str, fallback: Callable[[], str]) -> str:
        _ = file_path
        _ = text
        return fallback()

    def finetune_job_submit(self, *, payload: dict, fallback: Callable[[], dict]) -> dict:
        _ = payload
        return fallback()

    def finetune_job_status(self, *, payload: dict, fallback: Callable[[], dict]) -> dict:
        _ = payload
        return fallback()

    def finetune_model_publish(self, *, payload: dict, fallback: Callable[[], dict]) -> dict:
        _ = payload
        return fallback()


class CapabilityRouter:
    def __init__(self, registry: PluginRegistry, built_in: BuiltInCapabilityProvider | None = None):
        self._registry = registry
        self._built_in = built_in or BuiltInCapabilityProvider()

    @staticmethod
    def _privacy_rank(value: PluginPrivacyMode) -> int:
        if value == PluginPrivacyMode.LOCAL_ONLY:
            return 0
        if value == PluginPrivacyMode.HYBRID:
            return 1
        return 2

    @staticmethod
    def _plugin_mode_from_app(mode: PrivacyMode) -> PluginPrivacyMode:
        if mode == PrivacyMode.LOCAL_ONLY:
            return PluginPrivacyMode.LOCAL_ONLY
        if mode == PrivacyMode.HYBRID:
            return PluginPrivacyMode.HYBRID
        return PluginPrivacyMode.EXTERNAL_ALLOWED

    @classmethod
    def _effective_privacy_mode(
        cls,
        *,
        app_mode: PrivacyMode,
        plugin_mode: PluginPrivacyMode,
    ) -> PluginPrivacyMode:
        app_bound = cls._plugin_mode_from_app(app_mode)
        if cls._privacy_rank(plugin_mode) <= cls._privacy_rank(app_bound):
            return plugin_mode
        return app_bound

    @staticmethod
    def _requires_external(mode: PluginPrivacyMode) -> bool:
        return mode == PluginPrivacyMode.EXTERNAL_ALLOWED

    @staticmethod
    def _requires_hybrid(mode: PluginPrivacyMode) -> bool:
        return mode in {PluginPrivacyMode.HYBRID, PluginPrivacyMode.EXTERNAL_ALLOWED}

    def invoke(
        self,
        *,
        capability: str,
        fallback: Callable[[], T],
    ) -> CapabilityInvocation[T]:
        trace_id = str(uuid.uuid4())
        supported_names = {item.value: item for item in SUPPORTED_CAPABILITIES}
        parsed = supported_names.get(capability)
        if parsed is None:
            logger.warning("plugin_capability:unsupported capability=%s trace_id=%s", capability, trace_id)
            return CapabilityInvocation(
                value=fallback(),
                source=PluginCapabilitySource.DISABLED,
                error_code=PluginErrorCode.PLUGIN_VALIDATION_ERROR,
                error_message=f"Unsupported capability: {capability}; trace_id={trace_id}",
            )

        plugin = self._registry.get_enabled_plugin(parsed)
        if plugin is not None:
            app_mode = self._registry.get_app_privacy_mode()
            effective_mode = self._effective_privacy_mode(
                app_mode=app_mode,
                plugin_mode=plugin.privacy_mode,
            )
            blocked_reason: str | None = None
            if self._requires_external(plugin.privacy_mode) and not self._requires_external(effective_mode):
                blocked_reason = "blocked_by_app_privacy_mode"
            elif self._requires_hybrid(plugin.privacy_mode) and not self._requires_hybrid(effective_mode):
                blocked_reason = "blocked_by_app_privacy_mode"
            if blocked_reason:
                logger.info(
                    "plugin_privacy_blocked:capability=%s plugin_id=%s plugin_mode=%s effective_mode=%s trace_id=%s",
                    parsed.value,
                    plugin.plugin_id,
                    plugin.privacy_mode.value,
                    effective_mode.value,
                    trace_id,
                )
            else:
                logger.info(
                    "plugin_capability:disabled_runtime capability=%s plugin_id=%s trace_id=%s",
                    parsed.value,
                    plugin.plugin_id,
                    trace_id,
                )
            return CapabilityInvocation(
                value=fallback(),
                source=PluginCapabilitySource.DISABLED,
                plugin_id=plugin.plugin_id,
                error_code=PluginErrorCode.PLUGIN_PERMISSION_DENIED if blocked_reason else PluginErrorCode.PLUGIN_UNAVAILABLE,
                error_message=(
                    f"{blocked_reason}; effective_privacy_mode={effective_mode.value}; plugin_privacy_mode={plugin.privacy_mode.value}; trace_id={trace_id}"
                    if blocked_reason
                    else f"Out-of-process plugin runtime is not enabled in community build; trace_id={trace_id}"
                ),
            )

        logger.info("plugin_capability:built_in capability=%s trace_id=%s", parsed.value, trace_id)
        return CapabilityInvocation(value=fallback(), source=PluginCapabilitySource.BUILT_IN)

    def process_retriever_search(self, *, query: str, bundle) -> CapabilityInvocation:
        return self.invoke(
            capability=ExtensionCapability.RETRIEVER_SEARCH.value,
            fallback=lambda: self._built_in.retriever_search(query=query, bundle=bundle),
        )

    def process_reranker_rank(self, *, query: str, chunk_candidates) -> CapabilityInvocation:
        return self.invoke(
            capability=ExtensionCapability.RERANKER_RANK.value,
            fallback=lambda: self._built_in.reranker_rank(query=query, chunk_candidates=chunk_candidates),
        )

    def process_summarizer_generate(self, *, fallback: Callable[[], T]) -> CapabilityInvocation[T]:
        return self.invoke(
            capability=ExtensionCapability.SUMMARIZER_GENERATE.value,
            fallback=lambda: self._built_in.summarizer_generate(fallback=fallback),
        )

    def process_retrieval_query_transform(self, *, query: str) -> CapabilityInvocation[str]:
        return self.invoke(
            capability=ExtensionCapability.RETRIEVAL_QUERY_TRANSFORM.value,
            fallback=lambda: self._built_in.retrieval_query_transform(query=query, fallback=lambda: query),
        )

    def process_retrieval_post_filter(self, *, query: str, chunk_candidates: list) -> CapabilityInvocation[list]:
        return self.invoke(
            capability=ExtensionCapability.RETRIEVAL_POST_FILTER.value,
            fallback=lambda: self._built_in.retrieval_post_filter(
                query=query,
                chunk_candidates=chunk_candidates,
                fallback=lambda: chunk_candidates,
            ),
        )

    def process_chunking_strategy(self, *, text: str, fallback: Callable[[], list[str]]) -> CapabilityInvocation[list[str]]:
        return self.invoke(
            capability=ExtensionCapability.CHUNKING_STRATEGY.value,
            fallback=lambda: self._built_in.chunking_strategy(text=text, fallback=fallback),
        )

    def process_embedding_provider(
        self,
        *,
        texts: list[str],
        fallback: Callable[[], list[list[float]]],
    ) -> CapabilityInvocation[list[list[float]]]:
        return self.invoke(
            capability=ExtensionCapability.EMBEDDING_PROVIDER.value,
            fallback=lambda: self._built_in.embedding_provider(texts=texts, fallback=fallback),
        )

    def process_indexing_preprocess(
        self,
        *,
        file_path: str,
        text: str,
        fallback: Callable[[], str],
    ) -> CapabilityInvocation[str]:
        return self.invoke(
            capability=ExtensionCapability.INDEXING_PREPROCESS.value,
            fallback=lambda: self._built_in.indexing_preprocess(file_path=file_path, text=text, fallback=fallback),
        )

    def process_finetune_job_submit(self, *, payload: dict, fallback: Callable[[], dict]) -> CapabilityInvocation[dict]:
        return self.invoke(
            capability=ExtensionCapability.FINETUNE_JOB_SUBMIT.value,
            fallback=lambda: self._built_in.finetune_job_submit(payload=payload, fallback=fallback),
        )

    def process_finetune_job_status(self, *, payload: dict, fallback: Callable[[], dict]) -> CapabilityInvocation[dict]:
        return self.invoke(
            capability=ExtensionCapability.FINETUNE_JOB_STATUS.value,
            fallback=lambda: self._built_in.finetune_job_status(payload=payload, fallback=fallback),
        )

    def process_finetune_model_publish(self, *, payload: dict, fallback: Callable[[], dict]) -> CapabilityInvocation[dict]:
        return self.invoke(
            capability=ExtensionCapability.FINETUNE_MODEL_PUBLISH.value,
            fallback=lambda: self._built_in.finetune_model_publish(payload=payload, fallback=fallback),
        )

    def capability_states(self) -> list[ExtensionCapabilityState]:
        states: list[ExtensionCapabilityState] = []
        app_mode = self._registry.get_app_privacy_mode()
        for capability in SUPPORTED_CAPABILITIES:
            plugin = self._registry.get_enabled_plugin(capability)
            if plugin is None:
                states.append(
                    ExtensionCapabilityState(
                        capability=capability,
                        source=PluginCapabilitySource.BUILT_IN,
                        plugin_enabled=False,
                        plugin_id=None,
                        error_code=None,
                        effective_privacy_mode=self._plugin_mode_from_app(app_mode),
                    )
                )
                continue
            effective_mode = self._effective_privacy_mode(app_mode=app_mode, plugin_mode=plugin.privacy_mode)
            blocked_reason = None
            error_code = PluginErrorCode.PLUGIN_UNAVAILABLE
            if self._requires_external(plugin.privacy_mode) and not self._requires_external(effective_mode):
                blocked_reason = "blocked_by_app_privacy_mode"
                error_code = PluginErrorCode.PLUGIN_PERMISSION_DENIED
            elif self._requires_hybrid(plugin.privacy_mode) and not self._requires_hybrid(effective_mode):
                blocked_reason = "blocked_by_app_privacy_mode"
                error_code = PluginErrorCode.PLUGIN_PERMISSION_DENIED
            states.append(
                ExtensionCapabilityState(
                    capability=capability,
                    source=PluginCapabilitySource.DISABLED,
                    plugin_enabled=True,
                    plugin_id=plugin.plugin_id,
                    error_code=error_code,
                    plugin_privacy_mode=plugin.privacy_mode,
                    effective_privacy_mode=effective_mode,
                    blocked_reason=blocked_reason,
                )
            )
        return states


class ExtensionKernel:
    def __init__(self, db, registry: PluginRegistry | None = None, router: CapabilityRouter | None = None):
        self._registry = registry or SQLitePluginRegistry(db)
        self._router = router or CapabilityRouter(self._registry)

    @property
    def router(self) -> CapabilityRouter:
        return self._router

    def capabilities_snapshot(self) -> ExtensionCapabilitiesResponse:
        return ExtensionCapabilitiesResponse(version=1, capabilities=self._router.capability_states())
