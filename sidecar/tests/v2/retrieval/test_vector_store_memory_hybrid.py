from __future__ import annotations

from pathlib import Path

from local_ai_core.vector_store import VectorHit, VectorStore


def test_get_memories_schema_uses_runtime_embedding_dim(monkeypatch):
    import local_ai_core.vector_store as vector_store_module

    class _PAStub:
        @staticmethod
        def float32():
            return "float32"

        @staticmethod
        def string():
            return "string"

        @staticmethod
        def list_(dtype, length):
            return ("list", dtype, length)

        @staticmethod
        def field(name, dtype):
            return {"name": name, "dtype": dtype}

        @staticmethod
        def schema(fields):
            return {"fields": fields}

    monkeypatch.setattr(vector_store_module, "pa", _PAStub)
    store = VectorStore.__new__(VectorStore)
    store._dim = 1536

    schema = store._get_memories_schema()
    fields = schema["fields"]
    vector_field = next(item for item in fields if item["name"] == "vector")
    assert vector_field["dtype"] == ("list", "float32", 1536)


def test_search_memories_hybrid_filters_sparse_hits_by_session_id(tmp_path: Path, monkeypatch):
    store = VectorStore(tmp_path / "lancedb", dim=4)
    dense = [
        VectorHit(
            chunk_id="webmem:sess-a:1",
            doc_id="sess-a",
            file_path="",
            text="dense-a",
            score=0.9,
            modified_at=0.0,
        )
    ]
    sparse = [
        VectorHit(
            chunk_id="webmem:sess-b:1",
            doc_id="sess-b",
            file_path="",
            text="sparse-b",
            score=1.0,
            modified_at=0.0,
        )
    ]

    monkeypatch.setattr(store, "search_memories", lambda *args, **kwargs: dense)
    monkeypatch.setattr(store, "search_memories_fts", lambda *args, **kwargs: sparse)

    hits = store.search_memories_hybrid(
        query_text="query",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        session_id="sess-a",
        limit=4,
    )
    assert hits
    assert all(hit.doc_id == "sess-a" for hit in hits)


def test_search_memories_hybrid_with_dense_empty_does_not_leak_other_session_sparse(tmp_path: Path, monkeypatch):
    store = VectorStore(tmp_path / "lancedb", dim=4)
    monkeypatch.setattr(store, "search_memories", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        store,
        "search_memories_fts",
        lambda *args, **kwargs: [
            VectorHit(
                chunk_id="webmem:sess-b:2",
                doc_id="sess-b",
                file_path="",
                text="sparse-b",
                score=0.8,
                modified_at=0.0,
            )
        ],
    )

    hits = store.search_memories_hybrid(
        query_text="query",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        session_id="sess-a",
        limit=4,
    )
    assert hits == []
