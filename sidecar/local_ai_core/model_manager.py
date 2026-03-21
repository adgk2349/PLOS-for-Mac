from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from .models import (
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


class ModelManager:
    def __init__(self, data_dir: Path):
        self._models_dir = data_dir / "models"
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._catalog_path = Path(__file__).with_name("model_catalog.json")
        self._state_path = self._models_dir / "catalog_state.json"

    def download_model(self, *, url: str, engine: LocalEngine, filename: str | None = None) -> ModelDownloadResponse:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("모델 다운로드 URL은 http/https만 지원합니다.")

        target_dir = self._engine_dir(engine)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = self._resolve_filename(url=url, engine=engine, explicit=filename)
        target = target_dir / file_name

        bytes_written = 0
        with httpx.stream("GET", url, timeout=None, follow_redirects=True) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    handle.write(chunk)
                    bytes_written += len(chunk)

        return ModelDownloadResponse(
            file_name=file_name,
            saved_path=str(target),
            engine=engine,
            bytes_written=bytes_written,
        )

    def list_models(self) -> list[ModelListItem]:
        output: list[ModelListItem] = []
        for engine in (LocalEngine.MLX, LocalEngine.LLAMA_CPP):
            root = self._engine_dir(engine)
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                stat = path.stat()
                output.append(
                    ModelListItem(
                        file_name=path.name,
                        path=str(path),
                        engine=engine,
                        size_bytes=int(stat.st_size),
                        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
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

            if installed_path and not Path(installed_path).exists():
                installed_path = None
                if status in {ModelInstallStatus.INSTALLED, ModelInstallStatus.ACTIVE}:
                    status = ModelInstallStatus.NOT_INSTALLED

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
                )
            )

        return ModelCatalogResponse(
            version=manifest.version,
            default_profile=manifest.default_profile,
            models=items,
        )

    def install_catalog_model(self, model_id: str) -> ModelCatalogInstallResponse:
        model = self._find_catalog_model(model_id)
        system_memory_gb = self._system_memory_gb()
        if system_memory_gb < int(model.recommended_memory_gb):
            raise ValueError(
                f"현재 시스템 메모리({system_memory_gb}GB)로는 "
                f"{model.name} 다운로드를 권장하지 않습니다. "
                f"최소 권장 사양은 {model.recommended_memory_gb}GB입니다."
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
                saved_path = self._download_huggingface_repo(model)
            elif model.distribution_type.value == "huggingface_file":
                saved_path = self._download_huggingface_file(model)
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
            raise

    def activate_catalog_model(self, model_id: str) -> ModelCatalogActivateResponse:
        model = self._find_catalog_model(model_id)
        state = self._load_state()
        record = state.get(model.id, {})
        installed_path = record.get("installed_path")
        if not installed_path:
            raise FileNotFoundError("먼저 모델을 다운로드해 주세요.")

        path = Path(installed_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"설치된 모델 경로를 찾지 못했습니다: {path}")

        self._set_state_record(
            state,
            model.id,
            status=ModelInstallStatus.ACTIVE,
            installed_path=str(path),
            failure_reason=None,
        )
        self._save_state(state)

        return ModelCatalogActivateResponse(
            model_id=model.id,
            engine=model.engine,
            model_path=str(path),
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

    def _download_huggingface_repo(self, model: ModelCatalogItem) -> str:
        from huggingface_hub import snapshot_download

        target_dir = self._engine_dir(model.engine) / model.id
        target_dir.mkdir(parents=True, exist_ok=True)
        allow_patterns = model.allow_patterns or None
        snapshot_download(
            repo_id=model.repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=allow_patterns,
        )
        return str(target_dir)

    def _download_huggingface_file(self, model: ModelCatalogItem) -> str:
        if not model.filename:
            raise ValueError("huggingface_file 모델은 filename이 필요합니다.")

        from huggingface_hub import hf_hub_download

        target_dir = self._engine_dir(model.engine) / model.id
        target_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=model.repo_id,
            filename=model.filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return str(Path(path).expanduser())

    def _find_catalog_model(self, model_id: str) -> ModelCatalogItem:
        manifest = self._load_catalog_manifest()
        for model in manifest.models:
            if model.id == model_id:
                return model
        raise ValueError(f"지원하지 않는 모델 ID: {model_id}")

    @staticmethod
    def _system_memory_gb() -> int:
        override = (os.getenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE") or "").strip()
        if override:
            try:
                parsed = int(float(override))
                if parsed > 0:
                    return parsed
            except Exception:
                pass

        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            if isinstance(page_size, int) and isinstance(phys_pages, int) and page_size > 0 and phys_pages > 0:
                total_bytes = page_size * phys_pages
                return max(1, int(total_bytes / (1024**3)))
        except Exception:
            pass
        return 16

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
