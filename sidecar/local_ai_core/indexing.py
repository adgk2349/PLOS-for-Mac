from __future__ import annotations

import hashlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .classification import DocumentClassifier
from .chunker import chunk_text
from .db import Database
from .embedding import EmbeddingService
from .models import DocumentMetadata, IndexJobStatus, StartupProfile, WorkspaceResponse
from .parsers import ParseError, SUPPORTED_EXTENSIONS, parse_file
from .vector_store import VectorStore


@dataclass(slots=True)
class IndexDocument:
    path: Path
    modified_at: float


class IndexingService:
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        classifier: DocumentClassifier,
    ):
        self._db = db
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._classifier = classifier
        self._jobs: dict[str, IndexJobStatus] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._watcher = IncrementalWatcher(self)

    def start_watcher(self) -> None:
        self._watcher.start()

    def stop_watcher(self) -> None:
        self._watcher.stop()

    def start_job(self, scope: str, workspace: WorkspaceResponse) -> IndexJobStatus:
        job_id = str(uuid.uuid4())
        status = IndexJobStatus(job_id=job_id, scope=scope, status="queued", stage="queued")
        with self._lock:
            self._jobs[job_id] = status
        self._executor.submit(self._run_job, job_id, scope, workspace)
        return status

    def get_job(self, job_id: str) -> IndexJobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_failures(self):
        return self._db.list_failures()

    def reclassify_document(self, doc_id: str) -> DocumentMetadata:
        record = self._db.get_document_record(doc_id)
        if record is None:
            raise FileNotFoundError(f"document not found: {doc_id}")
        path = Path(record["path"])
        parse_result = parse_file(path)
        metadata = self._safe_classification(path, parse_result.text)
        updated = self._db.update_document_auto_metadata(doc_id, metadata)
        if updated is None:
            raise FileNotFoundError(f"document not found after reclassify: {doc_id}")
        return updated

    def _run_job(self, job_id: str, scope: str, workspace: WorkspaceResponse) -> None:
        self._update(job_id, status="running", stage="scan", progress=0.01)
        try:
            if scope == "full":
                # Avoid showing stale failures from previous runs before this full rescan.
                self._db.clear_all_failures()
            snapshot = self._snapshot_workspace(workspace)
            self._prune_deleted_documents(snapshot)
            docs = self._documents_to_index(snapshot, scope=scope)
            total = len(docs)
            processed = 0
            failures = 0

            if total == 0:
                self._update(
                    job_id,
                    status="completed",
                    stage="done",
                    progress=1.0,
                    processed_files=0,
                    failed_files=0,
                )
                return

            for doc in docs:
                processed += 1
                self._update(job_id, stage="parse")
                try:
                    parse_result = parse_file(doc.path)
                    chunks = chunk_text(parse_result.text)
                    if not chunks:
                        raise ParseError("Parsed text is empty")

                    doc_id = self._stable_doc_id(doc.path)
                    self._update(job_id, stage="classify")
                    metadata = self._safe_classification(doc.path, parse_result.text)
                    self._db.upsert_document(
                        doc_id,
                        str(doc.path),
                        doc.path.suffix.lower(),
                        doc.modified_at,
                        metadata=metadata,
                    )
                    self._vector_store.delete_doc(doc_id)

                    self._update(job_id, stage="embed")
                    vectors = self._embedding_service.embed_documents(chunks)

                    self._update(job_id, stage="store")
                    db_chunks = []
                    rows = []
                    for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                        chunk_id = f"{doc_id}:{idx}"
                        db_chunks.append((chunk_id, doc_id, idx, chunk))
                        rows.append(
                            {
                                "chunk_id": chunk_id,
                                "doc_id": doc_id,
                                "file_path": str(doc.path),
                                "text": chunk,
                                "modified_at": doc.modified_at,
                                "vector": vector,
                            }
                        )

                    self._db.insert_chunks(doc_id, db_chunks)
                    self._vector_store.upsert_rows(rows)
                    if metadata.get("_classification_warning"):
                        self._db.record_failure(str(doc.path), metadata["_classification_warning"])
                    else:
                        self._db.clear_failure(str(doc.path))
                except Exception as exc:
                    failures += 1
                    self._db.record_failure(str(doc.path), str(exc))

                progress = processed / total
                self._update(
                    job_id,
                    progress=progress,
                    processed_files=processed,
                    failed_files=failures,
                )

            self._update(job_id, status="completed", stage="done", progress=1.0)
        except Exception as exc:
            self._update(job_id, status="failed", stage="failed", error=str(exc), progress=1.0)

    def _safe_classification(self, path: Path, text: str) -> dict:
        try:
            result = self._classifier.classify(path, text)
            return result.model_dump()
        except Exception as exc:
            compact = " ".join(text.split())[:240]
            return {
                "summary": compact,
                "category": "참고자료",
                "subcategory": "",
                "document_type": "",
                "tags": [],
                "year": None,
                "project": None,
                "importance": 0.5,
                "_classification_warning": f"classification fallback: {exc}",
            }

    def _snapshot_workspace(self, workspace: WorkspaceResponse) -> dict[str, float]:
        snapshot: dict[str, float] = {}
        excluded = [Path(path).resolve() for path in workspace.excluded_paths]

        for included_path in workspace.included_paths:
            root = Path(included_path).expanduser().resolve()
            if not root.exists():
                continue

            if root.is_file():
                if self._is_supported(root) and not self._is_excluded(root, excluded):
                    snapshot[str(root)] = root.stat().st_mtime
                continue

            for file in root.rglob("*"):
                if not file.is_file() or not self._is_supported(file):
                    continue
                if self._is_excluded(file, excluded):
                    continue
                snapshot[str(file)] = file.stat().st_mtime

        return snapshot

    def _documents_to_index(self, snapshot: dict[str, float], *, scope: str) -> list[IndexDocument]:
        if scope == "full":
            return [IndexDocument(path=Path(path), modified_at=mtime) for path, mtime in snapshot.items()]

        indexed = self._db.get_indexed_documents()
        docs: list[IndexDocument] = []
        for path, mtime in snapshot.items():
            previous = indexed.get(path)
            if previous is None or mtime > previous:
                docs.append(IndexDocument(path=Path(path), modified_at=mtime))
        return docs

    def _prune_deleted_documents(self, snapshot: dict[str, float]) -> None:
        indexed_paths = set(self._db.get_indexed_documents().keys())
        existing_paths = set(snapshot.keys())
        removed_paths = sorted(indexed_paths - existing_paths)
        if not removed_paths:
            return

        removed_doc_ids = self._db.delete_documents_by_paths(removed_paths)
        for doc_id in removed_doc_ids:
            self._vector_store.delete_doc(doc_id)

    @staticmethod
    def _is_supported(path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    @staticmethod
    def _is_excluded(path: Path, excluded_roots: list[Path]) -> bool:
        for root in excluded_roots:
            if root == path:
                return True
            if root in path.parents:
                return True
        return False

    @staticmethod
    def _stable_doc_id(path: Path) -> str:
        digest = hashlib.sha1(str(path).encode("utf-8"), usedforsecurity=False).hexdigest()
        return digest

    def _update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            current = self._jobs[job_id]
            payload = current.model_dump()
            payload.update(kwargs)
            self._jobs[job_id] = IndexJobStatus(**payload)


class IncrementalWatcher:
    """Simple polling-based watcher used as a portable fallback for filewatch incrementals."""

    def __init__(self, indexing_service: IndexingService, interval_sec: float = 5.0):
        self._indexing_service = indexing_service
        self._interval = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_snapshot: dict[str, float] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                workspace = self._indexing_service._db.get_workspace()
                snapshot = self._snapshot(workspace)
                if self._has_delta(snapshot):
                    self._indexing_service.start_job("incremental", workspace)
                self._last_snapshot = snapshot
            except Exception:
                # Watcher should not bring down sidecar.
                pass
            self._stop.wait(self._interval)

    def _snapshot(self, workspace: WorkspaceResponse) -> dict[str, float]:
        state: dict[str, float] = {}
        excluded = [Path(path).resolve() for path in workspace.excluded_paths]
        for included in workspace.included_paths:
            root = Path(included).expanduser().resolve()
            if not root.exists():
                continue
            if root.is_file() and root.suffix.lower() in SUPPORTED_EXTENSIONS:
                state[str(root)] = root.stat().st_mtime
                continue
            for file in root.rglob("*"):
                if not file.is_file() or file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                if IndexingService._is_excluded(file, excluded):
                    continue
                state[str(file)] = file.stat().st_mtime
        return state

    def _has_delta(self, snapshot: dict[str, float]) -> bool:
        if not self._last_snapshot:
            return False
        if snapshot.keys() != self._last_snapshot.keys():
            return True
        for key, mtime in snapshot.items():
            if self._last_snapshot.get(key) != mtime:
                return True
        return False
