from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
from .model_manager_modules import (
    parse_content_length,
    progress_ratio,
    read_progress_int,
    read_progress_percent,
    system_memory_gb,
)

from .models import (
    DownloadProgressItem,
    DownloadProgressKind,
    DownloadProgressStatus,
    LocalEngine,
    ModelCatalogActivateResponse,
    ModelCatalogDeleteResponse,
    ModelCatalogInstallResponse,
    ModelCatalogItem,
    ModelCatalogItemState,
    ModelCatalogManifest,
    ModelCatalogResponse,
    ModelDownloadResponse,
    ModelInstallStatus,
    ModelListItem,
)


class _HFProgressAggregator:
    def __init__(self, manager: "ModelManager", download_id: str):
        self._manager = manager
        self._download_id = download_id
        self._lock = threading.RLock()
        self._bar_downloaded: dict[int, int] = {}
        self._bar_total: dict[int, int] = {}

    def register(self, bar_id: int, *, current: int, total: float | int | None) -> None:
        with self._lock:
            self._bar_downloaded[bar_id] = max(0, int(current))
            parsed_total = self._parse_total(total)
            if parsed_total is not None:
                self._bar_total[bar_id] = parsed_total
            self._emit_locked()

    def advance(self, bar_id: int, *, current: int, total: float | int | None) -> None:
        with self._lock:
            self._bar_downloaded[bar_id] = max(0, int(current))
            parsed_total = self._parse_total(total)
            if parsed_total is not None:
                self._bar_total[bar_id] = parsed_total
            self._emit_locked()

    def close(self, bar_id: int) -> None:
        with self._lock:
            # Keep historical bytes for completed bars so total progress stays monotonic.
            if bar_id not in self._bar_downloaded:
                return
            self._emit_locked()

    @staticmethod
    def _parse_total(value: float | int | None) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None

    def _emit_locked(self) -> None:
        downloaded = sum(self._bar_downloaded.values())
        total = sum(self._bar_total.values()) if self._bar_total else None
        self._manager._update_progress_bytes(self._download_id, downloaded_bytes=downloaded, total_bytes=total)


class ModelManager:
    def __init__(self, data_dir: Path, models_dir: Path | None = None):
        self._models_dir = (models_dir or (data_dir / "models")).expanduser().resolve()
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._catalog_path = Path(__file__).with_name("model_catalog.json")
        self._state_path = self._models_dir / "catalog_state.json"
        self._progress_lock = threading.RLock()
        self._download_progress: dict[str, dict[str, Any]] = {}
        self._runtime_controller = None
        self._residency_meta: dict[str, dict[str, Any]] = {}

    def set_runtime_controller(self, controller) -> None:
        self._runtime_controller = controller

    def load(self, *, engine: LocalEngine, model_ref: str | None, profile: str = "recommended") -> tuple[bool, str]:
        if self._runtime_controller is None:
            return False, "model_load=unavailable;reason=no_runtime_controller"
        ok, detail = self._runtime_controller.load(engine=engine, model_ref=model_ref, profile=profile)
        key = f"{engine.value}:{str(model_ref or '').strip()}"
        self._residency_meta[key] = {
            "ok": bool(ok),
            "detail": detail,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        return bool(ok), str(detail or "")

    def unload(self, target: str | LocalEngine = "all") -> str:
        if self._runtime_controller is None:
            return "model_unload=unavailable;reason=no_runtime_controller"
        return str(self._runtime_controller.unload(target))

    def switch(self, *, engine: LocalEngine, model_ref: str | None, profile: str = "recommended") -> tuple[bool, str]:
        if self._runtime_controller is None:
            return False, "model_switch=unavailable;reason=no_runtime_controller"
        ok, detail = self._runtime_controller.switch(engine=engine, model_ref=model_ref, profile=profile)
        return bool(ok), str(detail or "")

    def health(self) -> dict[str, Any]:
        if self._runtime_controller is None:
            return {"loaded": [], "resident_engine": None, "policy": {}, "load_failures": {}, "residency_meta": dict(self._residency_meta)}
        payload = self._runtime_controller.health()
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["residency_meta"] = dict(self._residency_meta)
            return payload
        return {"loaded": [], "resident_engine": None, "policy": {}, "load_failures": {}, "residency_meta": dict(self._residency_meta)}

    def download_model(self, *, url: str, engine: LocalEngine, filename: str | None = None) -> ModelDownloadResponse:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("모델 다운로드 URL은 http/https만 지원합니다.")

        target_dir = self._engine_dir(engine)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = self._resolve_filename(url=url, engine=engine, explicit=filename)
        target = target_dir / file_name
        download_id = f"direct:{engine.value}:{file_name}:{uuid.uuid4().hex[:8]}"
        self._start_progress(
            download_id,
            kind=DownloadProgressKind.DIRECT,
            engine=engine,
            file_name=file_name,
            detail="직접 URL 다운로드 시작",
        )

        bytes_written = 0
        total_bytes: int | None = None
        try:
            with httpx.stream("GET", url, timeout=None, follow_redirects=True) as response:
                response.raise_for_status()
                total_bytes = parse_content_length(response.headers.get("Content-Length"))
                self._update_progress_bytes(download_id, downloaded_bytes=0, total_bytes=total_bytes)
                with target.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bytes_written += len(chunk)
                        self._update_progress_bytes(
                            download_id,
                            downloaded_bytes=bytes_written,
                            total_bytes=total_bytes,
                        )
            self._mark_progress_completed(download_id, detail="직접 URL 다운로드 완료")
        except Exception as exc:
            self._mark_progress_failed(download_id, error=str(exc), detail="직접 URL 다운로드 실패")
            raise

        progress_snapshot = self._progress_snapshot(download_id)
        return ModelDownloadResponse(
            file_name=file_name,
            saved_path=str(target),
            engine=engine,
            bytes_written=bytes_written,
            total_bytes=total_bytes,
            progress_percent=read_progress_percent(progress_snapshot),
            download_id=download_id,
        )

    def get_download_progress(self) -> list[DownloadProgressItem]:
        with self._progress_lock:
            items = [
                DownloadProgressItem.model_validate(payload)
                for payload in self._download_progress.values()
            ]
        items.sort(key=lambda item: item.updated_at, reverse=True)
        return items

    def list_models(self, *, runtime_health: dict[str, Any] | None = None) -> list[ModelListItem]:
        resident_map: dict[str, dict[str, Any]] = {}
        if isinstance(runtime_health, dict):
            for item in runtime_health.get("loaded") or []:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "").strip()
                if not path:
                    continue
                resident_map[str(Path(path).expanduser().resolve())] = item
        output: list[ModelListItem] = []
        for engine in (LocalEngine.MLX, LocalEngine.LLAMA_CPP):
            root = self._engine_dir(engine)
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                stat = path.stat()
                resolved_path = str(path.expanduser().resolve())
                resident_info = resident_map.get(resolved_path) or resident_map.get(str(path))
                last_used_at = None
                if isinstance(resident_info, dict):
                    raw_last = resident_info.get("last_used_at")
                    if raw_last:
                        try:
                            last_used_at = datetime.fromisoformat(str(raw_last).replace("Z", "+00:00"))
                        except Exception:
                            last_used_at = None
                output.append(
                    ModelListItem(
                        file_name=path.name,
                        path=str(path),
                        engine=engine,
                        size_bytes=int(stat.st_size),
                        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                        loaded=bool(resident_info),
                        resident_engine=engine if resident_info else None,
                        last_used_at=last_used_at,
                    )
                )
        output.sort(key=lambda item: item.modified_at, reverse=True)
        return output

    def catalog_with_status(self, settings) -> ModelCatalogResponse:
        manifest = self._load_catalog_manifest()
        state = self._load_state()

        active_engine = settings.local_engine
        active_path = settings.mlx_model_path if active_engine == LocalEngine.MLX else settings.llama_model_path
        active_path = str(Path(active_path).expanduser()) if active_path else None

        items: list[ModelCatalogItemState] = []
        for model in manifest.models:
            record = state.get(model.id, {})
            status = ModelInstallStatus(record.get("status", ModelInstallStatus.NOT_INSTALLED.value))
            installed_path = record.get("installed_path")
            failure_reason = record.get("failure_reason")
            progress = self._progress_snapshot(self._catalog_download_id(model.id))

            if installed_path and not Path(installed_path).exists():
                installed_path = None
                if status in {ModelInstallStatus.INSTALLED, ModelInstallStatus.ACTIVE}:
                    status = ModelInstallStatus.NOT_INSTALLED

            # Prevent stale install records when a catalog item changes engine/path strategy.
            # Example: previous llama_cpp artifact for the same model_id should not be treated
            # as installed when the current catalog entry is mlx.
            if installed_path and not self._is_catalog_installed_path_compatible(model=model, installed_path=installed_path):
                installed_path = None
                if status in {ModelInstallStatus.INSTALLED, ModelInstallStatus.ACTIVE}:
                    status = ModelInstallStatus.NOT_INSTALLED
                failure_reason = None

            if installed_path and status not in {ModelInstallStatus.DOWNLOADING, ModelInstallStatus.FAILED}:
                status = ModelInstallStatus.INSTALLED

            is_active = bool(
                installed_path
                and active_path
                and model.engine == active_engine
                and Path(installed_path).expanduser() == Path(active_path).expanduser()
            )
            if is_active:
                status = ModelInstallStatus.ACTIVE

            items.append(
                ModelCatalogItemState(
                    **model.model_dump(),
                    status=status,
                    installed_path=installed_path,
                    active=is_active,
                    failure_reason=failure_reason,
                    progress_percent=read_progress_percent(progress),
                    downloaded_bytes=read_progress_int(progress, "downloaded_bytes"),
                    total_bytes=read_progress_int(progress, "total_bytes"),
                )
            )

        return ModelCatalogResponse(
            version=manifest.version,
            default_profile=manifest.default_profile,
            models=items,
        )

    def install_catalog_model(self, model_id: str) -> ModelCatalogInstallResponse:
        model = self._find_catalog_model(model_id)
        download_id = self._catalog_download_id(model.id)
        avail_memory_gb = system_memory_gb()
        if avail_memory_gb < int(model.recommended_memory_gb):
            raise ValueError(
                f"현재 시스템 메모리({avail_memory_gb}GB)로는 "
                f"{model.name} 다운로드를 권장하지 않습니다. "
                f"최소 권장 사양은 {model.recommended_memory_gb}GB입니다."
            )
        self._start_progress(
            download_id,
            kind=DownloadProgressKind.CATALOG,
            model_id=model.id,
            engine=model.engine,
            file_name=model.filename,
            detail="카탈로그 다운로드 시작",
        )
        state = self._load_state()
        self._set_state_record(
            state,
            model.id,
            status=ModelInstallStatus.DOWNLOADING,
            installed_path=state.get(model.id, {}).get("installed_path"),
            failure_reason=None,
        )
        self._save_state(state)

        try:
            if model.distribution_type.value == "huggingface_repo":
                saved_path = self._download_huggingface_repo(model, download_id=download_id)
            elif model.distribution_type.value == "huggingface_file":
                saved_path = self._download_huggingface_file(model, download_id=download_id)
            else:
                raise ValueError(f"unsupported distribution_type: {model.distribution_type}")

            self._set_state_record(
                state,
                model.id,
                status=ModelInstallStatus.INSTALLED,
                installed_path=saved_path,
                failure_reason=None,
            )
            self._save_state(state)
            self._mark_progress_completed(download_id, detail="카탈로그 다운로드 완료")
            return ModelCatalogInstallResponse(
                model_id=model.id,
                status=ModelInstallStatus.INSTALLED,
                engine=model.engine,
                saved_path=saved_path,
                detail="모델 다운로드 완료",
            )
        except Exception as exc:
            self._set_state_record(
                state,
                model.id,
                status=ModelInstallStatus.FAILED,
                installed_path=state.get(model.id, {}).get("installed_path"),
                failure_reason=str(exc),
            )
            self._save_state(state)
            self._mark_progress_failed(download_id, error=str(exc), detail="카탈로그 다운로드 실패")
            raise

    def activate_catalog_model(self, model_id: str) -> ModelCatalogActivateResponse:
        model = self._find_catalog_model(model_id)
        state = self._load_state()
        record = state.get(model.id, {})
        installed_path = record.get("installed_path")
        if not installed_path:
            raise FileNotFoundError("먼저 모델을 다운로드해 주세요.")

        if not self._is_catalog_installed_path_compatible(model=model, installed_path=str(installed_path)):
            raise FileNotFoundError(
                "설치된 모델 경로가 현재 카탈로그 엔진과 호환되지 않습니다. 모델을 삭제 후 다시 다운로드해 주세요."
            )

        path = Path(installed_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"설치된 모델 경로를 찾지 못했습니다: {path}")

        activated_path = self._normalize_activated_model_path(engine=model.engine, installed_path=path)

        self._set_state_record(
            state,
            model.id,
            status=ModelInstallStatus.ACTIVE,
            installed_path=str(activated_path),
            failure_reason=None,
        )
        self._save_state(state)

        return ModelCatalogActivateResponse(
            model_id=model.id,
            engine=model.engine,
            model_path=str(activated_path),
            profile=model.profile,
        )

    def delete_catalog_model(self, model_id: str) -> ModelCatalogDeleteResponse:
        model = self._find_catalog_model(model_id)
        state = self._load_state()
        record = state.get(model.id, {})
        installed_path = record.get("installed_path")

        removed = False
        if installed_path:
            target = Path(installed_path).expanduser()
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                    removed = True
                else:
                    try:
                        target.unlink(missing_ok=True)
                        removed = True
                    except Exception:
                        removed = False
                    parent = target.parent
                    if parent.exists() and parent.is_dir() and not any(parent.iterdir()):
                        parent.rmdir()

        self._set_state_record(
            state,
            model.id,
            status=ModelInstallStatus.NOT_INSTALLED,
            installed_path=None,
            failure_reason=None,
        )
        self._save_state(state)

        return ModelCatalogDeleteResponse(model_id=model.id, removed=removed)

    def _download_huggingface_repo(self, model: ModelCatalogItem, *, download_id: str) -> str:
        from huggingface_hub import snapshot_download

        target_dir = self._engine_dir(model.engine) / model.id
        target_dir.mkdir(parents=True, exist_ok=True)
        allow_patterns = model.allow_patterns or None
        tqdm_class = self._hf_tqdm_class(download_id)
        snapshot_download(
            repo_id=model.repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=allow_patterns,
            tqdm_class=tqdm_class,
        )
        return str(target_dir)

    def _download_huggingface_file(self, model: ModelCatalogItem, *, download_id: str) -> str:
        if not model.filename:
            raise ValueError("huggingface_file 모델은 filename이 필요합니다.")

        from huggingface_hub import hf_hub_download

        target_dir = self._engine_dir(model.engine) / model.id
        target_dir.mkdir(parents=True, exist_ok=True)
        tqdm_class = self._hf_tqdm_class(download_id)
        path = hf_hub_download(
            repo_id=model.repo_id,
            filename=model.filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            tqdm_class=tqdm_class,
        )
        return str(Path(path).expanduser())

    def _find_catalog_model(self, model_id: str) -> ModelCatalogItem:
        manifest = self._load_catalog_manifest()
        for model in manifest.models:
            if model.id == model_id:
                return model
        raise ValueError(f"지원하지 않는 모델 ID: {model_id}")

    def _load_catalog_manifest(self) -> ModelCatalogManifest:
        if not self._catalog_path.exists():
            raise FileNotFoundError(f"model catalog not found: {self._catalog_path}")
        payload = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        return ModelCatalogManifest.model_validate(payload)

    def _load_state(self) -> dict:
        if not self._state_path.exists():
            return {}
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
        return {}

    def _save_state(self, state: dict) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _set_state_record(
        state: dict,
        model_id: str,
        *,
        status: ModelInstallStatus,
        installed_path: str | None,
        failure_reason: str | None,
    ) -> None:
        state[model_id] = {
            "status": status.value,
            "installed_path": installed_path,
            "failure_reason": failure_reason,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    def _engine_dir(self, engine: LocalEngine) -> Path:
        return self._models_dir / engine.value

    def _normalize_activated_model_path(self, *, engine: LocalEngine, installed_path: Path) -> Path:
        if engine == LocalEngine.LLAMA_CPP:
            if installed_path.is_file():
                if installed_path.suffix.lower() != ".gguf":
                    raise FileNotFoundError(f"llama.cpp 모델 파일은 .gguf여야 합니다: {installed_path}")
                return installed_path

            if installed_path.is_dir():
                candidates = [item for item in installed_path.rglob("*.gguf") if item.is_file()]
                if not candidates:
                    raise FileNotFoundError(f"llama.cpp용 GGUF 파일을 찾지 못했습니다: {installed_path}")
                return max(candidates, key=lambda item: item.stat().st_mtime).resolve()

            raise FileNotFoundError(f"설치된 llama.cpp 모델 경로가 유효하지 않습니다: {installed_path}")

        # MLX는 디렉터리(허브 스냅샷) 기준으로 활성화한다.
        if installed_path.is_dir():
            return installed_path
        if installed_path.is_file():
            return installed_path.parent.resolve()
        raise FileNotFoundError(f"설치된 MLX 모델 경로가 유효하지 않습니다: {installed_path}")

    def _is_catalog_installed_path_compatible(self, *, model: ModelCatalogItem, installed_path: str) -> bool:
        try:
            expected_root = (self._engine_dir(model.engine) / model.id).expanduser().resolve()
            resolved_installed = Path(installed_path).expanduser().resolve()
            resolved_installed.relative_to(expected_root)
            return True
        except Exception:
            return False

    @staticmethod
    def _catalog_download_id(model_id: str) -> str:
        return f"catalog:{model_id}"

    def _start_progress(
        self,
        download_id: str,
        *,
        kind: DownloadProgressKind,
        model_id: str | None = None,
        engine: LocalEngine | None = None,
        file_name: str | None = None,
        total_bytes: int | None = None,
        detail: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "download_id": download_id,
            "kind": kind.value,
            "status": DownloadProgressStatus.RUNNING.value,
            "model_id": model_id,
            "engine": engine.value if engine else None,
            "file_name": file_name,
            "downloaded_bytes": 0,
            "total_bytes": total_bytes,
            "progress_percent": progress_ratio(0, total_bytes),
            "detail": detail,
            "error": None,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        with self._progress_lock:
            self._download_progress[download_id] = payload

    def _update_progress_bytes(
        self,
        download_id: str,
        *,
        downloaded_bytes: int,
        total_bytes: int | None = None,
        detail: str | None = None,
    ) -> None:
        with self._progress_lock:
            payload = self._download_progress.get(download_id)
            if payload is None:
                return
            if total_bytes is not None:
                payload["total_bytes"] = total_bytes
            payload["downloaded_bytes"] = max(0, int(downloaded_bytes))
            payload["progress_percent"] = progress_ratio(payload["downloaded_bytes"], payload.get("total_bytes"))
            if detail is not None:
                payload["detail"] = detail
            payload["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

    def _mark_progress_completed(self, download_id: str, *, detail: str | None = None) -> None:
        with self._progress_lock:
            payload = self._download_progress.get(download_id)
            if payload is None:
                return
            payload["status"] = DownloadProgressStatus.COMPLETED.value
            total_bytes = payload.get("total_bytes")
            downloaded_bytes = int(payload.get("downloaded_bytes") or 0)
            if isinstance(total_bytes, int) and total_bytes > 0:
                payload["downloaded_bytes"] = max(downloaded_bytes, total_bytes)
                payload["progress_percent"] = 100.0
            else:
                payload["progress_percent"] = None
            if detail is not None:
                payload["detail"] = detail
            payload["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

    def _mark_progress_failed(self, download_id: str, *, error: str, detail: str | None = None) -> None:
        with self._progress_lock:
            payload = self._download_progress.get(download_id)
            if payload is None:
                return
            payload["status"] = DownloadProgressStatus.FAILED.value
            payload["error"] = error
            if detail is not None:
                payload["detail"] = detail
            payload["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

    def _progress_snapshot(self, download_id: str) -> dict[str, Any] | None:
        with self._progress_lock:
            payload = self._download_progress.get(download_id)
            if payload is None:
                return None
            return dict(payload)

    def _hf_tqdm_class(self, download_id: str):
        from tqdm.auto import tqdm

        aggregator = _HFProgressAggregator(self, download_id)

        class _TrackedTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                aggregator.register(id(self), current=int(getattr(self, "n", 0)), total=getattr(self, "total", None))

            def update(self, n=1):
                super().update(n)
                aggregator.advance(id(self), current=int(getattr(self, "n", 0)), total=getattr(self, "total", None))

            def close(self):
                aggregator.advance(id(self), current=int(getattr(self, "n", 0)), total=getattr(self, "total", None))
                aggregator.close(id(self))
                super().close()

        return _TrackedTqdm

    @staticmethod
    def _resolve_filename(*, url: str, engine: LocalEngine, explicit: str | None) -> str:
        if explicit and explicit.strip():
            candidate = explicit.strip()
        else:
            parsed = urlparse(url)
            candidate = unquote(Path(parsed.path).name)
            if not candidate:
                candidate = "model"

        candidate = candidate.replace("/", "_").replace("\\", "_")
        if engine == LocalEngine.LLAMA_CPP and "." not in candidate:
            candidate += ".gguf"
        return candidate
