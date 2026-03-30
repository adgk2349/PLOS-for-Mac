from __future__ import annotations

import hashlib
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .classification import DocumentClassifier
from .chunker import chunk_text, semantic_chunk_text
from ..db import Database
from ..embedding import EmbeddingService
from ..models import DocumentMetadata, IndexJobStatus, StartupProfile, WorkspaceResponse
from .parsers import ParseError, SUPPORTED_EXTENSIONS, parse_file
from ..vector_store import VectorStore

logger = logging.getLogger(__name__)


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
        capability_router=None,
        settings_loader=None,
        max_workers: int | None = None,
    ):
        self._db = db
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._classifier = classifier
        self._capabilities = capability_router
        self._settings_loader = settings_loader
        self._jobs: dict[str, IndexJobStatus] = {}
        self._lock = threading.RLock()
        # Keep default workers conservative for 16GB-class machines to reduce RAM spikes
        # during parse/embedding/vector upserts. Override via LOCAL_AI_INDEX_MAX_WORKERS.
        configured_workers: int | None = None
        try:
            raw = str(os.getenv("LOCAL_AI_INDEX_MAX_WORKERS", "") or "").strip()
            if raw:
                configured_workers = max(1, int(raw))
        except Exception:
            configured_workers = None
        workers = max_workers or configured_workers or min(os.cpu_count() or 2, 2)
        self._executor = ThreadPoolExecutor(max_workers=workers)
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

    def has_running_job(self) -> bool:
        """Return True if any job is currently in a running/queued state."""
        with self._lock:
            return any(
                j.status in ("running", "queued")
                for j in self._jobs.values()
            )

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
            docs = list(self._scan_documents(workspace, scope=scope))
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
                    parse_text = parse_result.text
                    if self._capabilities is not None:
                        preprocess_hook = self._capabilities.process_indexing_preprocess(
                            file_path=str(doc.path),
                            text=parse_text,
                            fallback=lambda: parse_text,
                        )
                        parse_text = str(preprocess_hook.value or parse_text)
                    # Phase 17: Semantic Chunking
                    # Use semantic breakpoints for better topic-aware retrieval
                    if self._capabilities is not None:
                        chunk_hook = self._capabilities.process_chunking_strategy(
                            text=parse_text,
                            fallback=lambda: semantic_chunk_text(parse_text, self._embedding_service),
                        )
                        chunks = list(chunk_hook.value or [])
                    else:
                        chunks = semantic_chunk_text(parse_text, self._embedding_service)
                    if not chunks:
                        raise ParseError("Parsed text is empty")

                    doc_id = self._stable_doc_id(doc.path)
                    self._update(job_id, stage="classify")
                    metadata = self._safe_classification(doc.path, parse_text)

                    # --- Atomicity fix ---
                    # Upsert document record first (this also clears DB chunks via
                    # DocumentRepository.upsert_document which does DELETE FROM chunks
                    # inside the same SQLite transaction before committing).
                    self._db.upsert_document(
                        doc_id,
                        str(doc.path),
                        doc.path.suffix.lower(),
                        doc.modified_at,
                        metadata=metadata,
                    )

                    self._update(job_id, stage="embed")
                    # Phase 16: Contextual RAG
                    # Prepend document-level summary and metadata to every chunk.
                    # This dramatically improves retrieval precision and local model reasoning.
                    doc_summary = metadata.get("summary", "")
                    doc_ctx = f"File: {doc.path.name}"
                    if doc_summary:
                        doc_ctx += f" | {doc_summary}"
                    
                    # Store original chunks for DB (clean display) but embed contextualized ones
                    contextualized_chunks = [f"[{doc_ctx}] {chunk}" for chunk in chunks]
                    if self._capabilities is not None:
                        embed_hook = self._capabilities.process_embedding_provider(
                            texts=contextualized_chunks,
                            fallback=lambda: self._embedding_service.embed_documents(contextualized_chunks),
                        )
                        vectors = list(embed_hook.value or [])
                    else:
                        vectors = self._embedding_service.embed_documents(contextualized_chunks)
                    if len(vectors) != len(contextualized_chunks):
                        vectors = self._embedding_service.embed_documents(contextualized_chunks)

                    self._update(job_id, stage="store")
                    db_chunks = []
                    rows = []
                    for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                        chunk_id = f"{doc_id}:{idx}"
                        # DB stores original chunk
                        db_chunks.append((chunk_id, doc_id, idx, chunk))
                        rows.append(
                            {
                                "chunk_id": chunk_id,
                                "doc_id": doc_id,
                                "file_path": str(doc.path),
                                # Vector store stores contextualized text for retrieval/reasoning
                                "text": contextualized_chunks[idx],
                                "modified_at": doc.modified_at,
                                "vector": vector,
                            }
                        )

                    self._db.insert_chunks(doc_id, db_chunks)
                    # Vector store: delete old entries then add new ones.
                    # If upsert_rows fails the SQLite record still exists and the
                    # next incremental scan will retry (mtime unchanged → re-index).
                    self._vector_store.delete_doc(doc_id)
                    self._vector_store.upsert_rows(rows)

                    if metadata.get("_classification_warning"):
                        self._db.record_failure(str(doc.path), metadata["_classification_warning"])
                    else:
                        self._db.clear_failure(str(doc.path))
                except Exception as exc:
                    failures += 1
                    logger.warning("Indexing failure for %s: %s", doc.path, exc)
                    self._db.record_failure(str(doc.path), str(exc))

                progress = processed / total
                self._update(
                    job_id,
                    progress=progress,
                    processed_files=processed,
                    failed_files=failures,
                )
            
            # Level 2: Update Full-Text Search index after inserting new data
            try:
                self._vector_store.create_fts_index()
            except Exception as e:
                logger.warning("FTS index update failed during indexing job: %s", e)

            self._update(job_id, status="completed", stage="done", progress=1.0)
        except Exception as exc:
            logger.error("Indexing job %s failed: %s", job_id, exc)
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

    def _scan_documents(self, workspace: WorkspaceResponse, *, scope: str) -> Iterable[IndexDocument]:
        excluded = [Path(path).resolve() for path in workspace.excluded_paths]
        indexed = self._db.get_indexed_documents() if scope == "incremental" else {}

        for included_path in workspace.included_paths:
            root = Path(included_path).expanduser().resolve()
            if not root.exists():
                continue

            if root.is_file():
                if self._is_supported(root) and not self._is_excluded(root, excluded):
                    if self._needs_indexing(root, indexed):
                        yield IndexDocument(path=root, modified_at=root.stat().st_mtime)
                continue

            for file in root.rglob("*"):
                if not file.is_file() or not self._is_supported(file):
                    continue
                if self._is_excluded(file, excluded):
                    continue
                if self._needs_indexing(file, indexed):
                    yield IndexDocument(path=file, modified_at=file.stat().st_mtime)

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

    @staticmethod
    def _needs_indexing(path: Path, indexed_map: dict[str, float]) -> bool:
        if not indexed_map:
            return True
        key = str(path)
        prev = indexed_map.get(key)
        if prev is None:
            return True
        return path.stat().st_mtime > prev

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
                    # Skip triggering a new job if one is already running to prevent
                    # duplicate concurrent incremental indexing jobs.
                    if not self._indexing_service.has_running_job():
                        self._indexing_service.start_job("incremental", workspace)
                self._last_snapshot = snapshot
            except Exception:
                # Watcher should not bring down sidecar.
                pass
            self._stop.wait(self._interval)

    def _snapshot(self, workspace: WorkspaceResponse) -> dict[str, float]:
        state: dict[str, float] = {}
        excluded = [Path(path).expanduser().resolve() for path in workspace.excluded_paths]
        
        # Common skip-list for performance
        skip_dirs = {".git", "node_modules", "build", "dist", "bin", "obj", "__pycache__", ".venv"}

        for included in workspace.included_paths:
            root = Path(included).expanduser().resolve()
            if not root.exists():
                continue
            if root.is_file() and root.suffix.lower() in SUPPORTED_EXTENSIONS:
                state[str(root)] = root.stat().st_mtime
                continue
            
            # Efficient walk skipping ignored directories early
            import os
            for dirpath, dirnames, filenames in os.walk(str(root)):
                # Prune dirnames in-place to avoid walking into skipped folders
                dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
                
                # Check for explicit exclusion
                current_p = Path(dirpath).resolve()
                if any(current_p == exc or exc in current_p.parents for exc in excluded):
                    dirnames[:] = []  # Stop recursion
                    continue

                for filename in filenames:
                    if not filename.lower().endswith(tuple(SUPPORTED_EXTENSIONS)):
                        continue
                    
                    full_path = os.path.join(dirpath, filename)
                    try:
                        state[full_path] = os.path.getmtime(full_path)
                    except (OSError, FileNotFoundError):
                        continue
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
