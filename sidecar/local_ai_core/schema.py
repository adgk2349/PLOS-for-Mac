from __future__ import annotations

import sqlite3

TABLES_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS workspace (
    id INTEGER PRIMARY KEY CHECK (id=1),
    included_paths TEXT NOT NULL,
    excluded_paths TEXT NOT NULL,
    startup_profile TEXT NOT NULL,
    default_mode TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY CHECK (id=1),
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    file_type TEXT NOT NULL,
    modified_at REAL NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_order INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    reason TEXT NOT NULL,
    last_attempt_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS external_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    sent_chars INTEGER NOT NULL,
    approved_by_user INTEGER NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS behavior_policies (
    id INTEGER PRIMARY KEY CHECK (id=1),
    preferred_mode TEXT,
    preferred_action_order TEXT NOT NULL DEFAULT '[]',
    preferred_response_length TEXT NOT NULL DEFAULT 'long',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_weights (
    path TEXT PRIMARY KEY,
    weight REAL NOT NULL DEFAULT 1.0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_memory (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS workspace_memory (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.62,
    source TEXT NOT NULL DEFAULT 'inferred',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_preferences (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.62,
    source TEXT NOT NULL DEFAULT 'inferred',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodic_memory (
    id TEXT PRIMARY KEY,
    workspace_id TEXT,
    event_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    related_file_ids TEXT NOT NULL,
    related_action_ids TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pinned_memory (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    workspace_id TEXT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plugin_registry (
    plugin_id TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'disabled',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS platform_adapters (
    adapter_key TEXT PRIMARY KEY,
    adapter_class TEXT NOT NULL,
    health TEXT NOT NULL DEFAULT 'unknown',
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_memory_session_updated
ON session_memory(session_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_workspace_memory_ws_type
ON workspace_memory(workspace_id, memory_type, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_episodic_memory_ws_created
ON episodic_memory(workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_plugin_registry_enabled
ON plugin_registry(enabled, updated_at DESC);
"""

DOCUMENT_COLUMN_ADDITIONS = [
    ("summary", "TEXT NOT NULL DEFAULT ''"),
    ("category", "TEXT NOT NULL DEFAULT '참고자료'"),
    ("subcategory", "TEXT NOT NULL DEFAULT ''"),
    ("document_type", "TEXT NOT NULL DEFAULT ''"),
    ("tags", "TEXT NOT NULL DEFAULT '[]'"),
    ("year", "INTEGER"),
    ("project", "TEXT"),
    ("importance", "REAL NOT NULL DEFAULT 0.5"),
    ("excluded", "INTEGER NOT NULL DEFAULT 0"),
    ("user_category", "TEXT"),
    ("user_subcategory", "TEXT"),
    ("user_document_type", "TEXT"),
    ("user_tags", "TEXT"),
    ("user_year", "INTEGER"),
    ("user_project", "TEXT"),
    ("user_importance", "REAL"),
    ("user_excluded", "INTEGER"),
]

def migrate(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(TABLES_SCHEMA)
    
    # Ensure document columns
    cur.execute("PRAGMA table_info(documents)")
    columns = {row[1] for row in cur.fetchall()}
    for name, ddl in DOCUMENT_COLUMN_ADDITIONS:
        if name not in columns:
            cur.execute(f"ALTER TABLE documents ADD COLUMN {name} {ddl}")
    
    conn.commit()
