from __future__ import annotations


def _payload(privacy_mode: str) -> dict:
    return {
        "manifest": {
            "plugin_id": "sample.plugin",
            "version": "0.1.0",
            "api_version": "v1",
            "capabilities": ["retriever.search"],
            "privacy_mode": privacy_mode,
            "permissions": ["fs.read"],
            "entrypoint": "python -m sample",
            "build_target": "community",
        },
        "enabled": False,
    }


def test_register_plugin_rejects_privacy_violation(client, auth_headers):
    settings_resp = client.get("/v1/settings", headers=auth_headers)
    assert settings_resp.status_code == 200
    body = settings_resp.json()
    body["privacy_mode"] = "HYBRID"
    put_resp = client.put("/v1/settings", headers=auth_headers, json=body)
    assert put_resp.status_code == 200

    resp = client.post("/v1/extensions/plugins/register", headers=auth_headers, json=_payload("EXTERNAL_ALLOWED"))
    assert resp.status_code == 400
    assert "PLUGIN_PERMISSION_DENIED" in str(resp.json())


def test_register_plugin_accepts_compatible_privacy(client, auth_headers):
    resp = client.post("/v1/extensions/plugins/register", headers=auth_headers, json=_payload("LOCAL_ONLY"))
    assert resp.status_code == 200
    entries = resp.json().get("entries", [])
    assert any(item.get("plugin_id") == "sample.plugin" for item in entries)


def test_register_plugin_rejects_builtin_plugin_id(client, auth_headers):
    payload = _payload("LOCAL_ONLY")
    payload["manifest"]["plugin_id"] = "builtin.core"
    resp = client.post("/v1/extensions/plugins/register", headers=auth_headers, json=payload)
    assert resp.status_code == 400
    assert "reserved" in str(resp.json()).lower()
