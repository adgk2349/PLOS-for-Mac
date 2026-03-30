from __future__ import annotations

import json
import sys
import types

import pytest

from local_ai_core.db import Database
from local_ai_core.plugins.kernel import CapabilityRouter, SQLitePluginRegistry
from local_ai_core.models import (
    ExtensionCapability,
    PluginCapabilitySource,
    PluginErrorCode,
    PluginManifestV1,
    PluginPrivacyMode,
    PrivacyMode,
)
from local_ai_core.platform.contracts import PlatformServices
from local_ai_core.platform.defaults import load_platform_services


def test_plugin_manifest_rejects_duplicate_capabilities():
    with pytest.raises(ValueError):
        PluginManifestV1(
            plugin_id="demo.plugin",
            version="0.1.0",
            api_version="v1",
            capabilities=[
                ExtensionCapability.RETRIEVER_SEARCH,
                ExtensionCapability.RETRIEVER_SEARCH,
            ],
            privacy_mode=PluginPrivacyMode.LOCAL_ONLY,
            permissions=["read_workspace"],
            entrypoint="python -m demo",
            build_target="community",
        )


def test_capability_router_without_plugins_uses_built_in(tmp_path):
    db = Database(tmp_path / "local_ai_core.sqlite3")
    router = CapabilityRouter(SQLitePluginRegistry(db))

    invocation = router.process_retriever_search(query="hello", bundle={"value": 1})
    assert invocation.source == PluginCapabilitySource.BUILT_IN
    assert invocation.error_code is None
    assert invocation.value == {"value": 1}


def test_capability_router_unknown_capability_returns_validation_error(tmp_path):
    db = Database(tmp_path / "local_ai_core.sqlite3")
    router = CapabilityRouter(SQLitePluginRegistry(db))

    invocation = router.invoke(capability="unknown.capability", fallback=lambda: {"ok": True})
    assert invocation.source == PluginCapabilitySource.DISABLED
    assert invocation.error_code == PluginErrorCode.PLUGIN_VALIDATION_ERROR
    assert invocation.value == {"ok": True}


def test_capability_router_enabled_plugin_is_marked_unavailable(tmp_path):
    db = Database(tmp_path / "local_ai_core.sqlite3")
    manifest = {
        "plugin_id": "acme.retriever",
        "version": "0.1.0",
        "api_version": "v1",
        "capabilities": ["retriever.search"],
        "privacy_mode": "HYBRID",
        "permissions": ["read_workspace"],
        "entrypoint": "python -m acme.plugin",
        "build_target": "enterprise",
    }
    db.upsert_plugin_registry_entry(
        plugin_id="acme.retriever",
        manifest_json=json.dumps(manifest),
        enabled=True,
        state="enabled",
    )

    router = CapabilityRouter(SQLitePluginRegistry(db))
    invocation = router.process_retriever_search(query="hello", bundle={"value": 2})
    assert invocation.source == PluginCapabilitySource.DISABLED
    assert invocation.error_code == PluginErrorCode.PLUGIN_UNAVAILABLE
    assert invocation.plugin_id == "acme.retriever"
    assert invocation.value == {"value": 2}


def test_extensions_capabilities_endpoint_defaults_to_built_in(client, auth_headers):
    response = client.get("/v1/extensions/capabilities", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    caps = payload["capabilities"]
    names = {item["capability"] for item in caps}
    assert names == {
        "retriever.search",
        "reranker.rank",
        "summarizer.generate",
        "retrieval.query_transform",
        "retrieval.post_filter",
        "chunking.strategy",
        "embedding.provider",
        "indexing.preprocess",
        "finetune.job_submit",
        "finetune.job_status",
        "finetune.model_publish",
    }
    assert all(item["source"] == "built_in" for item in caps)


def test_extension_plugin_registry_includes_single_builtin_bundle(client, auth_headers):
    response = client.get("/v1/extensions/plugins", headers=auth_headers)
    assert response.status_code == 200
    entries = response.json()["entries"]
    builtin = [item for item in entries if item["plugin_id"] == "builtin.core"]
    assert len(builtin) == 1
    assert builtin[0]["enabled"] is True
    assert builtin[0]["state"] == "built_in"
    assert builtin[0]["is_builtin"] is True
    assert "+" in str(builtin[0]["updated_at"]) or str(builtin[0]["updated_at"]).endswith("Z")
    capability_names = set(builtin[0]["manifest"]["capabilities"])
    assert capability_names == {
        "retriever.search",
        "reranker.rank",
        "summarizer.generate",
        "retrieval.query_transform",
        "retrieval.post_filter",
        "chunking.strategy",
        "embedding.provider",
        "indexing.preprocess",
        "finetune.job_submit",
        "finetune.job_status",
        "finetune.model_publish",
    }


def test_extension_plugin_register_enable_disable_delete_flow(client, auth_headers):
    manifest = {
        "plugin_id": "acme.summarizer",
        "version": "0.1.0",
        "api_version": "v1",
        "capabilities": ["summarizer.generate"],
        "privacy_mode": "LOCAL_ONLY",
        "permissions": ["read_workspace"],
        "entrypoint": "python -m acme.plugin",
        "build_target": "enterprise",
    }
    registered = client.post(
        "/v1/extensions/plugins/register",
        headers=auth_headers,
        json={"manifest": manifest, "enabled": False},
    )
    assert registered.status_code == 200
    entries = registered.json()["entries"]
    assert any(item["plugin_id"] == "acme.summarizer" and item["enabled"] is False for item in entries)

    enabled = client.post("/v1/extensions/plugins/acme.summarizer/enable", headers=auth_headers)
    assert enabled.status_code == 200
    payload = enabled.json()
    assert payload["plugin"]["enabled"] is True
    cap = next(item for item in payload["capabilities"] if item["capability"] == "summarizer.generate")
    assert cap["source"] == "disabled"
    assert cap["plugin_id"] == "acme.summarizer"

    disabled = client.post("/v1/extensions/plugins/acme.summarizer/disable", headers=auth_headers)
    assert disabled.status_code == 200
    assert disabled.json()["plugin"]["enabled"] is False

    removed = client.delete("/v1/extensions/plugins/acme.summarizer", headers=auth_headers)
    assert removed.status_code == 200
    assert removed.json()["removed"] is True

    listed = client.get("/v1/extensions/plugins", headers=auth_headers)
    assert listed.status_code == 200
    assert all(item["plugin_id"] != "acme.summarizer" for item in listed.json()["entries"])


def test_extension_plugin_enable_unknown_returns_404(client, auth_headers):
    response = client.post("/v1/extensions/plugins/missing.plugin/enable", headers=auth_headers)
    assert response.status_code == 404


def test_builtin_plugin_mutation_routes_return_404(client, auth_headers):
    disable = client.post("/v1/extensions/plugins/builtin.core/disable", headers=auth_headers)
    assert disable.status_code == 404

    delete = client.delete("/v1/extensions/plugins/builtin.core", headers=auth_headers)
    assert delete.status_code == 404


def test_extension_plugin_register_missing_privacy_mode_rejected(client, auth_headers):
    manifest = {
        "plugin_id": "acme.invalid",
        "version": "0.1.0",
        "api_version": "v1",
        "capabilities": ["summarizer.generate"],
        "permissions": ["read_workspace"],
        "entrypoint": "python -m acme.invalid",
        "build_target": "community",
    }
    response = client.post(
        "/v1/extensions/plugins/register",
        headers=auth_headers,
        json={"manifest": manifest, "enabled": False},
    )
    assert response.status_code == 422


def test_capability_router_blocks_external_plugin_when_app_local_only(tmp_path):
    db = Database(tmp_path / "local_ai_core.sqlite3")
    settings = db.get_settings().model_copy(update={"privacy_mode": PrivacyMode.LOCAL_ONLY})
    db.update_settings(settings)
    manifest = {
        "plugin_id": "acme.ext",
        "version": "0.1.0",
        "api_version": "v1",
        "capabilities": ["retriever.search"],
        "privacy_mode": "EXTERNAL_ALLOWED",
        "permissions": ["network"],
        "entrypoint": "python -m acme.ext",
        "build_target": "community",
    }
    db.upsert_plugin_registry_entry(
        plugin_id="acme.ext",
        manifest_json=json.dumps(manifest),
        enabled=True,
        state="enabled",
    )
    router = CapabilityRouter(SQLitePluginRegistry(db))
    invocation = router.process_retriever_search(query="hello", bundle={"value": 3})
    assert invocation.source == PluginCapabilitySource.DISABLED
    assert invocation.error_code == PluginErrorCode.PLUGIN_PERMISSION_DENIED
    assert "blocked_by_app_privacy_mode" in str(invocation.error_message or "")


def test_capability_router_allows_external_scope_when_app_external_allowed(tmp_path):
    db = Database(tmp_path / "local_ai_core.sqlite3")
    settings = db.get_settings().model_copy(update={"privacy_mode": PrivacyMode.EXTERNAL_ALLOWED})
    db.update_settings(settings)
    manifest = {
        "plugin_id": "acme.ext.allowed",
        "version": "0.1.0",
        "api_version": "v1",
        "capabilities": ["retriever.search"],
        "privacy_mode": "EXTERNAL_ALLOWED",
        "permissions": ["network"],
        "entrypoint": "python -m acme.ext.allowed",
        "build_target": "community",
    }
    db.upsert_plugin_registry_entry(
        plugin_id="acme.ext.allowed",
        manifest_json=json.dumps(manifest),
        enabled=True,
        state="enabled",
    )
    router = CapabilityRouter(SQLitePluginRegistry(db))
    invocation = router.process_retriever_search(query="hello", bundle={"value": 7})
    assert invocation.source == PluginCapabilitySource.DISABLED
    assert invocation.error_code == PluginErrorCode.PLUGIN_UNAVAILABLE


def test_finetune_job_endpoints_contract(client, auth_headers):
    register = client.post(
        "/v1/extensions/plugins/register",
        headers=auth_headers,
        json={
            "enabled": True,
            "manifest": {
                "plugin_id": "acme.finetune",
                "version": "0.1.0",
                "api_version": "v1",
                "capabilities": ["finetune.job_submit", "finetune.job_status", "finetune.model_publish"],
                "privacy_mode": "HYBRID",
                "permissions": ["network"],
                "entrypoint": "python -m acme.finetune",
                "build_target": "community",
            },
        },
    )
    assert register.status_code == 200

    submit = client.post(
        "/v1/extensions/finetune/jobs/submit",
        headers=auth_headers,
        json={
            "plugin_id": "acme.finetune",
            "job_name": "sample",
            "dataset_uri": "file:///tmp/data.jsonl",
            "base_model": "qwen3.5-9b",
            "params": {"epochs": 1},
        },
    )
    assert submit.status_code == 200
    payload = submit.json()
    job_id = payload["job_id"]
    assert payload["state"] == "queued"

    status = client.get(f"/v1/extensions/finetune/jobs/{job_id}/status", headers=auth_headers)
    assert status.status_code == 200
    assert status.json()["job_id"] == job_id

    publish = client.post(
        "/v1/extensions/finetune/jobs/publish",
        headers=auth_headers,
        json={
            "job_id": job_id,
            "target_model_id": "acme/adapter-v1",
            "artifact_uri": "s3://bucket/adapter",
        },
    )
    assert publish.status_code == 200
    assert publish.json()["ok"] is True


def test_load_platform_services_supports_factory_override(monkeypatch):
    module = types.ModuleType("test_platform_factory_mod")

    class DummyAuth:
        def verify_request(self, request) -> None:  # pragma: no cover - simple contract stub
            _ = request

    class DummyLicense:
        def is_allowed(self, *, feature, context=None) -> bool:  # pragma: no cover
            _ = feature
            _ = context
            return True

    class DummyAudit:
        def emit(self, *, event, payload) -> None:  # pragma: no cover
            _ = event
            _ = payload

    class DummyPolicy:
        def get_policy(self):  # pragma: no cover
            return {"mode": "test"}

    def build(*, session_token, defaults):
        _ = session_token
        _ = defaults
        return PlatformServices(
            auth_provider=DummyAuth(),
            license_gate=DummyLicense(),
            audit_sink=DummyAudit(),
            policy_provider=DummyPolicy(),
        )

    module.build = build
    sys.modules[module.__name__] = module

    monkeypatch.setenv("LOCAL_AI_PLATFORM_FACTORY", f"{module.__name__}:build")
    services = load_platform_services("session-token")
    assert services.policy_provider.get_policy() == {"mode": "test"}
