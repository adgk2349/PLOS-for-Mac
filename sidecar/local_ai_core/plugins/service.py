from __future__ import annotations

import json
from datetime import datetime, timezone
import logging
import uuid

from ..db import Database
from ..models import (
    ExtensionCapabilityState,
    FinetuneJobState,
    FinetuneJobStatusResponse,
    FinetuneJobSubmitRequest,
    FinetuneJobSubmitResponse,
    FinetuneModelPublishRequest,
    FinetuneModelPublishResponse,
    PluginManifestV1,
    PluginPrivacyMode,
    PluginRegisterRequest,
    PluginRegistryEntry,
    PluginRegistryResponse,
)
from .kernel import SUPPORTED_CAPABILITIES
from .validator import PluginManifestValidator

logger = logging.getLogger(__name__)

BUILTIN_PLUGIN_ID = "builtin.core"


class PluginRegistryService:
    def __init__(self, db: Database, *, extension_kernel=None, validator: PluginManifestValidator | None = None):
        self._db = db
        self._extensions = extension_kernel
        self._validator = validator or PluginManifestValidator()

    def list_plugins(self) -> PluginRegistryResponse:
        rows = self._db.list_plugin_registry_entries()
        entries = [self._row_to_entry(row) for row in rows if str(row.get("plugin_id") or "") != BUILTIN_PLUGIN_ID]
        entries.append(self._built_in_entry())
        return PluginRegistryResponse(entries=entries)

    def register_plugin(self, payload: PluginRegisterRequest) -> PluginRegistryEntry:
        if payload.manifest.plugin_id == BUILTIN_PLUGIN_ID:
            raise ValueError(f"{BUILTIN_PLUGIN_ID} is reserved for bundled capabilities")
        app_privacy_mode = self._db.get_settings().privacy_mode
        validated_manifest = self._validator.validate(
            manifest=payload.manifest,
            app_privacy_mode=app_privacy_mode,
        )
        logger.info(
            "plugin_register:accepted plugin_id=%s enabled=%s privacy_mode=%s",
            validated_manifest.plugin_id,
            payload.enabled,
            validated_manifest.privacy_mode.value,
        )
        state = "enabled" if payload.enabled else "disabled"
        self._db.upsert_plugin_registry_entry(
            plugin_id=validated_manifest.plugin_id,
            manifest_json=validated_manifest.model_dump_json(),
            enabled=payload.enabled,
            state=state,
        )
        row = self._db.get_plugin_registry_entry(validated_manifest.plugin_id)
        if row is None:
            raise RuntimeError("Failed to persist plugin manifest")
        return self._row_to_entry(row)

    def enable_plugin(self, plugin_id: str) -> PluginRegistryEntry:
        if plugin_id == BUILTIN_PLUGIN_ID:
            raise KeyError(f"Plugin not found: {plugin_id}")
        row = self._db.get_plugin_registry_entry(plugin_id)
        if row is None:
            raise KeyError(f"Plugin not found: {plugin_id}")
        self._db.upsert_plugin_registry_entry(
            plugin_id=plugin_id,
            manifest_json=row["manifest_json"],
            enabled=True,
            state="enabled",
        )
        updated = self._db.get_plugin_registry_entry(plugin_id)
        if updated is None:
            raise RuntimeError("Failed to enable plugin")
        logger.info("plugin_register:enabled plugin_id=%s", plugin_id)
        return self._row_to_entry(updated)

    def disable_plugin(self, plugin_id: str) -> PluginRegistryEntry:
        if plugin_id == BUILTIN_PLUGIN_ID:
            raise KeyError(f"Plugin not found: {plugin_id}")
        row = self._db.get_plugin_registry_entry(plugin_id)
        if row is None:
            raise KeyError(f"Plugin not found: {plugin_id}")
        self._db.upsert_plugin_registry_entry(
            plugin_id=plugin_id,
            manifest_json=row["manifest_json"],
            enabled=False,
            state="disabled",
        )
        updated = self._db.get_plugin_registry_entry(plugin_id)
        if updated is None:
            raise RuntimeError("Failed to disable plugin")
        logger.info("plugin_register:disabled plugin_id=%s", plugin_id)
        return self._row_to_entry(updated)

    def delete_plugin(self, plugin_id: str) -> bool:
        if plugin_id == BUILTIN_PLUGIN_ID:
            raise KeyError(f"Plugin not found: {plugin_id}")
        removed = self._db.delete_plugin_registry_entry(plugin_id)
        logger.info("plugin_register:deleted plugin_id=%s removed=%s", plugin_id, removed)
        return removed

    def current_capability_states(self) -> list[ExtensionCapabilityState]:
        if self._extensions is None:
            return []
        snapshot = self._extensions.capabilities_snapshot()
        return snapshot.capabilities

    def _row_to_entry(self, row: dict) -> PluginRegistryEntry:
        manifest_json = row.get("manifest_json")
        validation_error = None
        manifest: PluginManifestV1
        try:
            payload = json.loads(manifest_json)
            manifest = PluginManifestV1.model_validate(payload)
        except Exception as exc:
            # Keep row visible even if malformed to aid repair workflows.
            validation_error = str(exc)
            plugin_id = str(row.get("plugin_id") or "unknown.plugin")
            manifest = PluginManifestV1(
                plugin_id=plugin_id,
                version="0.0.0",
                api_version="v1",
                capabilities=[],
                privacy_mode=PluginPrivacyMode.LOCAL_ONLY,
                permissions=[],
                entrypoint="disabled",
                build_target="community",
            )

        updated_raw = row.get("updated_at")
        try:
            updated_at = datetime.fromisoformat(str(updated_raw))
        except Exception:
            updated_at = datetime.fromtimestamp(0, tz=timezone.utc)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        return PluginRegistryEntry(
            plugin_id=str(row.get("plugin_id") or manifest.plugin_id),
            manifest=manifest,
            enabled=bool(row.get("enabled")),
            state=str(row.get("state") or "disabled"),
            updated_at=updated_at,
            validation_error=validation_error,
            is_builtin=False,
        )

    def _built_in_entry(self) -> PluginRegistryEntry:
        built_in_manifest = PluginManifestV1(
            plugin_id=BUILTIN_PLUGIN_ID,
            version="1.0.0",
            api_version="v1",
            capabilities=list(SUPPORTED_CAPABILITIES),
            privacy_mode=PluginPrivacyMode.LOCAL_ONLY,
            permissions=["builtin"],
            entrypoint="builtin://core",
            build_target="community",
        )
        return PluginRegistryEntry(
            plugin_id=BUILTIN_PLUGIN_ID,
            manifest=built_in_manifest,
            enabled=True,
            state="built_in",
            updated_at=datetime.now(timezone.utc),
            validation_error=None,
            is_builtin=True,
        )


class FinetuneJobService:
    def __init__(self, *, extension_kernel):
        self._extensions = extension_kernel
        self._jobs: dict[str, dict] = {}

    def submit_job(self, payload: FinetuneJobSubmitRequest) -> FinetuneJobSubmitResponse:
        now = datetime.now()
        request_payload = payload.model_dump(mode="json")
        capability = self._extensions.router.process_finetune_job_submit(
            payload=request_payload,
            fallback=lambda: {
                "accepted": True,
                "state": FinetuneJobState.QUEUED.value,
            },
        )
        _ = capability
        job_id = str(uuid.uuid4())
        record = {
            "job_id": job_id,
            "plugin_id": payload.plugin_id,
            "state": FinetuneJobState.QUEUED.value,
            "detail": "finetune_job:submit_accepted",
            "updated_at": now,
            "metrics": {},
            "base_model": payload.base_model,
            "dataset_uri": payload.dataset_uri,
        }
        self._jobs[job_id] = record
        return FinetuneJobSubmitResponse(
            job_id=job_id,
            plugin_id=payload.plugin_id,
            state=FinetuneJobState.QUEUED,
            created_at=now,
        )

    def get_job_status(self, *, job_id: str) -> FinetuneJobStatusResponse:
        row = self._jobs.get(job_id)
        if row is None:
            raise KeyError(f"Finetune job not found: {job_id}")
        payload = {"job_id": job_id, "plugin_id": row["plugin_id"]}
        capability = self._extensions.router.process_finetune_job_status(
            payload=payload,
            fallback=lambda: {
                "state": row["state"],
                "detail": row["detail"],
                "metrics": row["metrics"],
            },
        )
        response_payload = capability.value if isinstance(capability.value, dict) else {}
        state_value = str(response_payload.get("state") or row["state"])
        detail = str(response_payload.get("detail") or row["detail"])
        if not detail.startswith("finetune_job:"):
            detail = f"finetune_job:{detail}"
        metrics = response_payload.get("metrics") if isinstance(response_payload.get("metrics"), dict) else row["metrics"]
        return FinetuneJobStatusResponse(
            job_id=job_id,
            plugin_id=row["plugin_id"],
            state=FinetuneJobState(state_value),
            detail=detail,
            updated_at=row["updated_at"],
            metrics=metrics,
        )

    def publish_model(self, payload: FinetuneModelPublishRequest) -> FinetuneModelPublishResponse:
        row = self._jobs.get(payload.job_id)
        if row is None:
            raise KeyError(f"Finetune job not found: {payload.job_id}")
        request_payload = payload.model_dump(mode="json")
        capability = self._extensions.router.process_finetune_model_publish(
            payload=request_payload,
            fallback=lambda: {"ok": True},
        )
        _ = capability
        now = datetime.now()
        row["state"] = FinetuneJobState.COMPLETED.value
        row["detail"] = "finetune_job:model_published"
        row["updated_at"] = now
        return FinetuneModelPublishResponse(
            ok=True,
            job_id=payload.job_id,
            plugin_id=row["plugin_id"],
            target_model_id=payload.target_model_id,
            published_at=now,
        )
