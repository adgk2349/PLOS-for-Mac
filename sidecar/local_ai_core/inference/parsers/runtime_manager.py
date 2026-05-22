from __future__ import annotations
import re
import os
import subprocess
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from ..base import BaseDelegate
from ...models import LocalEngine, RuntimePrepareResponse

if TYPE_CHECKING:
    from ...local_inference import LocalInferenceEngine

# Component: runtime_manager.py
class RuntimeManager(BaseDelegate):
    _MODEL_FAMILY_PRESETS: dict[str, dict[str, dict[str, float | int]]] = {
        # Ollama-benchmarked: conversation temperature raised to 0.75-0.80,
        # repeat_penalty lowered to protect Korean morphology, top_k fixed at 40,
        # min_p=0.05 added for smarter token cutoff (Ollama default).
        "gemma": {
            "conversation": {"temperature": 0.80, "top_p": 0.90, "repeat_penalty": 1.08, "top_k": 40, "min_p": 0.05},
            "rewrite":      {"temperature": 0.42, "top_p": 0.90, "repeat_penalty": 1.10, "top_k": 32, "min_p": 0.0},
            "grounded":     {"temperature": 0.24, "top_p": 0.88, "repeat_penalty": 1.12, "top_k": 24, "min_p": 0.0},
        },
        "qwen": {
            "conversation": {"temperature": 0.78, "top_p": 0.90, "repeat_penalty": 1.08, "top_k": 40, "min_p": 0.05},
            "rewrite":      {"temperature": 0.38, "top_p": 0.90, "repeat_penalty": 1.08, "top_k": 30, "min_p": 0.0},
            "grounded":     {"temperature": 0.22, "top_p": 0.88, "repeat_penalty": 1.10, "top_k": 24, "min_p": 0.0},
        },
        "llama": {
            "conversation": {"temperature": 0.80, "top_p": 0.90, "repeat_penalty": 1.10, "top_k": 40, "min_p": 0.05},
            "rewrite":      {"temperature": 0.36, "top_p": 0.88, "repeat_penalty": 1.08, "top_k": 30, "min_p": 0.0},
            "grounded":     {"temperature": 0.22, "top_p": 0.88, "repeat_penalty": 1.12, "top_k": 24, "min_p": 0.0},
        },
        "deepseek": {
            "conversation": {"temperature": 0.75, "top_p": 0.90, "repeat_penalty": 1.08, "top_k": 40, "min_p": 0.05},
            "rewrite":      {"temperature": 0.36, "top_p": 0.88, "repeat_penalty": 1.08, "top_k": 34, "min_p": 0.0},
            "grounded":     {"temperature": 0.20, "top_p": 0.86, "repeat_penalty": 1.12, "top_k": 24, "min_p": 0.0},
        },
    }
    _MODEL_FAMILY_STOPS: dict[str, dict[str, list[str]]] = {
        "gemma": {
            "conversation": ["<|end_of_turn|>", "<|endoftext|>"],
            "rewrite": ["<|end_of_turn|>", "<|endoftext|>"],
            "grounded": ["<|end_of_turn|>", "<|endoftext|>"],
        },
        "qwen": {
            "conversation": ["<|im_end|>", "<|endoftext|>"],
            "rewrite": ["<|im_end|>", "<|endoftext|>"],
            "grounded": ["<|im_end|>", "<|endoftext|>"],
        },
        "llama": {
            "conversation": ["<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
            "rewrite": ["<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
            "grounded": ["<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"],
        },
        "deepseek": {
            "conversation": ["<|end_of_sentence|>", "<|endoftext|>"],
            "rewrite": ["<|end_of_sentence|>", "<|endoftext|>"],
            "grounded": ["<|end_of_sentence|>", "<|endoftext|>"],
        },
    }

    @staticmethod
    def _model_family_presets_enabled() -> bool:
        raw = str(os.getenv("LOCAL_AI_MODEL_FAMILY_PRESET_ENABLED", "1") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _detect_model_family(model_path: str | None) -> str:
        value = str(model_path or "").strip().lower()
        if not value:
            return "default"
        compact = re.sub(r"[^a-z0-9]+", "", value)
        if "deepseek" in compact:
            return "deepseek"
        if "qwen" in compact:
            return "qwen"
        if "gemma" in compact:
            return "gemma"
        if "llama" in compact or "mistral" in compact:
            return "llama"
        return "default"

    def _sampling_preset_for_model(
        self,
        *,
        style: Literal["grounded", "conversation", "rewrite"],
        engine: LocalEngine,
        model_path: str | None,
    ) -> dict[str, float | int]:
        base = dict(self._sampling_preset(style=style, engine=engine))
        if not self._model_family_presets_enabled():
            return base
        family = self._detect_model_family(model_path)
        family_block = self._MODEL_FAMILY_PRESETS.get(family, {})
        override = family_block.get(style, {})
        if override:
            base.update(override)
        return base

    def _stop_sequences_for_model(
        self,
        *,
        style: Literal["grounded", "conversation", "rewrite"],
        model_path: str | None,
        default_sequences: list[str],
    ) -> list[str]:
        if not self._model_family_presets_enabled():
            return list(default_sequences)
        family = self._detect_model_family(model_path)
        family_block = self._MODEL_FAMILY_STOPS.get(family, {})
        custom = family_block.get(style, [])
        if not custom:
            return list(default_sequences)
        return list(custom)

    @staticmethod
    def _system_memory_usage_ratio() -> float:
        try:
            import psutil  # type: ignore
            mem = psutil.virtual_memory()
            total = float(getattr(mem, "total", 0.0) or 0.0)
            used = float(getattr(mem, "used", 0.0) or 0.0)
            if total <= 0:
                return 0.0
            return max(0.0, min(1.0, used / total))
        except Exception:
            return 0.0

    def _recent_oom_like_failure(self) -> bool:
        try:
            recent_errors = " | ".join(str(v or "") for v in self._last_engine_error.values()).lower()
        except Exception:
            recent_errors = ""
        if not recent_errors:
            return False
        markers = (
            "outofmemory",
            "insufficient memory",
            "kio gpu command buffer",
            "mlx_isolated_failed_no_inprocess_fallback",
            "워커 응답 타임아웃",
            "워커 추론 실패",
        )
        return any(token in recent_errors for token in markers)

    def _oom_aware_token_cap(self, *, base_cap: int) -> int:
        cap = max(256, int(base_cap))
        # Skip dynamic capping if explicitly disabled (e.g. on machines with plenty of memory).
        if str(os.getenv("LOCAL_AI_DISABLE_OOM_TOKEN_CAP", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}:
            return cap
        usage = self._system_memory_usage_ratio()
        # Mac unified memory regularly runs at 80-90% — only cap at extreme pressure.
        if usage >= 0.99:
            cap = min(cap, 256)
        elif usage >= 0.97:
            cap = min(cap, 384)
        elif usage >= 0.95:
            cap = min(cap, 512)
        if self._recent_oom_like_failure():
            cap = min(cap, 384)
        # Optional hard clamp knob for low-memory machines.
        try:
            env_cap = int(float(str(os.getenv("LOCAL_AI_DYNAMIC_MAX_TOKENS_CAP", "0") or "0").strip()))
            if env_cap > 0:
                cap = min(cap, max(256, env_cap))
        except Exception:
            pass
        return max(256, cap)

    def _generate_grounded_candidate(
        self,
        *,
        engine: LocalEngine,
        prompt: str,
        response_language: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
    ) -> str | None:
        prompt_variants = [
            prompt,
            self._grounded_repair_prompt(prompt, response_language=response_language),
        ]
        attempts: list[str] = []
        for idx, prompt_variant in enumerate(prompt_variants, start=1):
            answer = self._generate_with_engine(
                engine=engine,
                prompt=prompt_variant,
                profile=profile,
                mlx_model_path=mlx_model_path,
                llama_model_path=llama_model_path,
                max_tokens=max_tokens,
                style="grounded",
            )
            if answer and self._looks_model_answer(answer):
                return answer
            if answer:
                attempts.append(f"attempt{idx}:filtered")
            else:
                err = self._last_engine_error.get(engine, f"{engine.value} engine failed")
                attempts.append(f"attempt{idx}:{err}")
        if attempts:
            self._set_engine_error(engine, f"{engine.value} grounded response invalid ({'; '.join(attempts)})")
        return None

    def _generate_with_engine(
        self,
        *,
        engine: LocalEngine,
        prompt: str,
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
        style: Literal["grounded", "conversation", "rewrite"] = "grounded",
        message_state: list[dict[str, str]] | None = None,
        response_language: str | None = None,
    ) -> str | None:
        if engine == LocalEngine.LLAMA_CPP:
            return self._generate_with_llama(
                prompt,
                llama_model_path,
                max_tokens=max_tokens,
                style=style,
                message_state=message_state,
                response_language=response_language,
            )
        return self._generate_with_mlx(
            prompt,
            profile,
            mlx_model_path,
            max_tokens=max_tokens,
            style=style,
            message_state=message_state,
            response_language=response_language,
        )

    def prepare_runtime(
        self,
        *,
        engine: LocalEngine,
        profile: str,
        mlx_model_path: str | None = None,
        llama_model_path: str | None = None,
    ) -> RuntimePrepareResponse:
        if engine == LocalEngine.LLAMA_CPP:
            resolved_path = self._resolve_llama_model_path(llama_model_path)
            package_ok = self._ensure_runtime_module(
                engine=LocalEngine.LLAMA_CPP,
                module_name="llama_cpp",
                package_spec="llama-cpp-python>=0.3.9",
                allow_install=True,
            )
            model_exists = bool(resolved_path and Path(resolved_path).expanduser().exists())
            if not resolved_path:
                self._set_engine_error(
                    LocalEngine.LLAMA_CPP,
                    "llama.cpp 모델 경로가 비어 있습니다. GGUF 파일 경로를 지정하거나 다운로드된 모델을 경로 적용해 주세요.",
                )
            elif not model_exists:
                self._set_engine_error(
                    LocalEngine.LLAMA_CPP,
                    f"llama.cpp 모델 파일을 찾지 못했습니다: {resolved_path}",
                )

            ready = False
            if package_ok and model_exists:
                ready = self._ensure_llama_loaded(
                    resolved_path,
                    allow_runtime_install=False,
                )
            detail = self._last_engine_error.get(
                LocalEngine.LLAMA_CPP,
                "llama.cpp 런타임 준비 완료" if ready else "llama.cpp 런타임 준비 실패",
            )
            return RuntimePrepareResponse(
                engine=LocalEngine.LLAMA_CPP,
                ready=ready,
                package_available=package_ok,
                model_path=resolved_path,
                model_exists=model_exists,
                accelerator=self._accelerator_hint(LocalEngine.LLAMA_CPP),
                detail=detail,
            )

        resolved_path = self._resolve_mlx_model_path(profile, mlx_model_path)
        package_ok = self._ensure_runtime_module(
            engine=LocalEngine.MLX,
            module_name="mlx_lm",
            package_spec="mlx-lm>=0.26.0",
            allow_install=True,
        )
        model_exists = self._is_mlx_model_reference_valid(resolved_path)
        if not resolved_path:
            self._set_engine_error(
                LocalEngine.MLX,
                "MLX 모델 경로가 비어 있습니다. MLX 모델 경로를 지정하거나 HuggingFace repo-id를 입력해 주세요.",
            )
        elif not model_exists:
            self._set_engine_error(
                LocalEngine.MLX,
                f"MLX 모델 경로를 검증하지 못했습니다: {resolved_path}",
            )

        ready = False
        if package_ok and model_exists:
            ready = self._ensure_mlx_loaded(
                profile,
                resolved_path,
                allow_runtime_install=False,
            )
        detail = self._last_engine_error.get(
            LocalEngine.MLX,
            "MLX 런타임 준비 완료" if ready else "MLX 런타임 준비 실패",
        )
        return RuntimePrepareResponse(
            engine=LocalEngine.MLX,
            ready=ready,
            package_available=package_ok,
            model_path=resolved_path,
            model_exists=model_exists,
            accelerator=self._accelerator_hint(LocalEngine.MLX),
            detail=detail,
        )

    def _sampling_preset(
        self,
        style: Literal["grounded", "conversation", "rewrite"],
        engine: LocalEngine,
    ) -> dict[str, float | int]:
        if style == "conversation":
            if engine == LocalEngine.LLAMA_CPP:
                return {"temperature": 0.80, "top_p": 0.90, "repeat_penalty": 1.10, "top_k": 40, "min_p": 0.05}
            # MLX: top_k=40 (was 0/unlimited) to prevent stray tokens
            return {"temperature": 0.78, "top_p": 0.90, "repeat_penalty": 1.10, "top_k": 40, "min_p": 0.05}
        if style == "rewrite":
            if engine == LocalEngine.LLAMA_CPP:
                return {"temperature": 0.35, "top_p": 0.90, "repeat_penalty": 1.08, "top_k": 40, "min_p": 0.0}
            return {"temperature": 0.34, "top_p": 0.90, "repeat_penalty": 1.08, "top_k": 40, "min_p": 0.0}
        if engine == LocalEngine.LLAMA_CPP:
            return {"temperature": 0.22, "top_p": 0.90, "repeat_penalty": 1.12, "top_k": 32, "min_p": 0.0}
        return {"temperature": 0.20, "top_p": 0.90, "repeat_penalty": 1.12, "top_k": 32, "min_p": 0.0}

    def _profile_to_model(self, profile: str) -> str | None:
        key = profile.lower()
        if key == "fast":
            return os.getenv("LOCAL_AI_MODEL_FAST")
        if key == "deep":
            return os.getenv("LOCAL_AI_MODEL_DEEP")
        return os.getenv("LOCAL_AI_MODEL_RECOMMENDED")

    def _discover_downloaded_model(self, engine: LocalEngine) -> str | None:
        roots: list[Path] = []
        models_dir_env = str(os.getenv("LOCAL_AI_MODELS_DIR", "") or "").strip()
        if models_dir_env:
            try:
                models_root = Path(models_dir_env).expanduser().resolve()
                env_engine_root = models_root / engine.value
                if env_engine_root.exists():
                    roots.append(env_engine_root)
            except Exception:
                pass
        for data_dir in self._candidate_data_dirs():
            root = data_dir / "models" / engine.value
            if root.exists():
                roots.append(root)
        if not roots:
            return None

        if engine == LocalEngine.LLAMA_CPP:
            candidates: list[Path] = []
            for root in roots:
                candidates.extend([path for path in root.rglob("*.gguf") if path.is_file()])
            if not candidates:
                for root in roots:
                    candidates.extend([path for path in root.rglob("*") if path.is_file()])
            if not candidates:
                return None
            return str(max(candidates, key=lambda item: item.stat().st_mtime))

        directories: list[Path] = []
        for root in roots:
            directories.extend([path for path in root.iterdir() if path.is_dir()])
        if directories:
            return str(max(directories, key=lambda item: item.stat().st_mtime))

        files: list[Path] = []
        for root in roots:
            files.extend([path for path in root.rglob("*") if path.is_file()])
        if files:
            return str(max(files, key=lambda item: item.stat().st_mtime).parent)
        return None

    def _candidate_data_dirs(self) -> list[Path]:
        seen: set[str] = set()
        output: list[Path] = []

        def _add(path: str | Path | None) -> None:
            if path is None:
                return
            text = str(path).strip()
            if not text:
                return
            try:
                resolved = Path(text).expanduser().resolve()
            except Exception:
                return
            key = str(resolved)
            if key in seen:
                return
            seen.add(key)
            output.append(resolved)

        env_data = os.getenv("LOCAL_AI_DATA_DIR", "").strip()
        if env_data:
            _add(env_data)
        strict_data_dir = str(os.getenv("LOCAL_AI_STRICT_DATA_DIR", "") or "").strip().lower() in {"1", "true", "yes", "on"}
        if strict_data_dir:
            return output
        _add("./data")
        _add(Path(__file__).resolve().parents[3] / "data")
        _add(Path.home() / "Library" / "Application Support" / "LocalAICore" / "SidecarRuntime" / "data")
        return output

    def _is_mlx_model_reference_valid(self, model_path: str | None) -> bool:
        if not model_path:
            return False
        candidate = model_path.strip()
        if not candidate:
            return False

        if "://" in candidate:
            return True
        if "/" in candidate and not candidate.startswith("/"):
            return True

        return Path(candidate).expanduser().exists()

    def _ensure_runtime_module(
        self,
        *,
        engine: LocalEngine,
        module_name: str,
        package_spec: str,
        allow_install: bool,
    ) -> bool:
        # Python 3.4+ find_spec
        spec = None
        try:
            spec = importlib.util.find_spec(module_name)
        except (AttributeError, ImportError):
            pass

        if spec is not None:
            self._clear_engine_error(engine)
            return True

        if not allow_install:
            self._set_engine_error(
                engine,
                f"{engine.value} 런타임 패키지({package_spec})가 설치되어 있지 않습니다. 설정에서 엔진 준비를 먼저 실행해 주세요.",
            )
            return False

        command = [sys.executable, "-m", "pip", "install", "--upgrade", package_spec]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            log = (proc.stderr or proc.stdout or "").strip()
            tail = "\n".join(log.splitlines()[-8:]) if log else "(로그 없음)"
            self._set_engine_error(engine, f"{package_spec} 설치 실패 (exit {proc.returncode})\n{tail}")
            return False

        importlib.invalidate_caches()
        if importlib.util.find_spec(module_name) is None:
            self._set_engine_error(engine, f"{package_spec} 설치 후 모듈({module_name}) 확인 실패")
            return False

        self._clear_engine_error(engine)
        return True

    def _accelerator_hint(self, engine: LocalEngine) -> str:
        if engine == LocalEngine.LLAMA_CPP:
            try:
                import llama_cpp
                supports = getattr(llama_cpp, "llama_supports_gpu_offload", None)
                if callable(supports) and bool(supports()):
                    return "Metal GPU offload 가능"
            except Exception:
                pass
            return "CPU 또는 GPU offload 미확인"

        try:
            import mlx.core as mx
            return f"MLX device: {mx.default_device()}"
        except Exception:
            return "MLX 장치 정보 미확인"

    def _set_engine_error(self, engine: LocalEngine, message: str) -> None:
        self._last_engine_error[engine] = message

    def _clear_engine_error(self, engine: LocalEngine) -> None:
        self._last_engine_error.pop(engine, None)

    def _resolve_max_tokens(self, max_tokens: int | None, profile: str) -> int:
        key = str(profile or "recommended").lower()
        if key in {"deep", "advanced"}:
            profile_limit = 1536
            profile_default = 1280
        elif key == "fast":
            profile_limit = 640
            profile_default = 448
        else:
            profile_limit = 1024
            profile_default = 896

        try:
            from ...reasoning.helpers.settings_sys_helpers import SettingsSysHelpers
            memory_cap = max(256, int(SettingsSysHelpers.memory_capped_conversation_max_tokens()))
        except Exception:
            memory_cap = 2048

        requested = int(max_tokens) if max_tokens is not None else profile_default
        dynamic_cap = self._oom_aware_token_cap(base_cap=min(profile_limit, memory_cap))
        effective = min(requested, profile_limit, memory_cap, dynamic_cap)
        return max(160, effective)
