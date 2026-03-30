from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import os
import threading
import gc
from datetime import datetime, timezone
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

from .inference.parsers.mlx_handler import MlxHandler
from .inference.parsers.llama_handler import LlamaHandler
from .inference.parsers.result_sanitizer import ResultSanitizer
from .inference.parsers.prompt_constructor import PromptConstructor
from .inference.parsers.agentic_parser import AgenticParser
from .inference.parsers.conversational_logic import ConversationalLogic
from .inference.parsers.runtime_manager import RuntimeManager

from .inference.types import InferenceResult, _ConversationCandidateResult
from .models import LocalEngine, WorkMode, Citation, ModelResidencyPolicy
from .language_utils import resolve_response_language

if TYPE_CHECKING:
    pass

@dataclass(slots=True)
class LocalInferenceConfig:
    model_path: str | None
    max_tokens: int


class LocalInferenceEngine:
    """Local inference router for MLX and llama.cpp with parser-driven delegation."""

    def __init__(self):
        # Shared State
        self._mlx_model = None
        self._mlx_tokenizer = None
        self._mlx_model_path: str | None = None
        self._mlx_prompt_cache = None
        self._last_mlx_prefix: str = ""
        self._last_mlx_cache_hit: bool = False

        self._llama_model = None
        self._llama_model_path: str | None = None
        self._llama_cache: dict[str, object] = {}
        # llama.cpp Python bindings are not reliably thread-safe across concurrent calls.
        # Serialize model load/inference to avoid native crashes (SIGSEGV in kv cache paths).
        self._llama_lock = threading.RLock()
        self._last_engine_error: dict[LocalEngine, str] = {}
        self._resident_last_used: dict[LocalEngine, datetime] = {}
        self._load_failures: dict[str, int] = {}
        self._policy = ModelResidencyPolicy(
            allow_dual_resident=str(os.getenv("LOCAL_AI_ALLOW_DUAL_RESIDENT", "")).strip().lower() in {"1", "true", "yes", "on"},
            max_resident_models=max(1, int(str(os.getenv("LOCAL_AI_MAX_RESIDENT_MODELS", "1")).strip() or "1")),
            memory_guard_threshold=max(
                0.5,
                min(0.99, float(str(os.getenv("LOCAL_AI_MEMORY_GUARD_THRESHOLD", "0.90")).strip() or "0.90")),
            ),
        )

        # Component instances (Parser-driven distribution)
        self.mlx_h = MlxHandler(self)
        self.llama_h = LlamaHandler(self)
        self.sanitizer = ResultSanitizer(self)
        self.prompts = PromptConstructor(self)
        self.agentic_p = AgenticParser(self)
        self.conv_logic = ConversationalLogic(self)
        self.runtime_m = RuntimeManager(self)
        
        self.components = [
            self.mlx_h, self.llama_h, self.sanitizer, 
            self.prompts, self.agentic_p, self.conv_logic, 
            self.runtime_m
        ]
        self._stream_token_callback_var: ContextVar[object | None] = ContextVar(
            "local_ai_stream_token_callback",
            default=None,
        )

    def __getattr__(self, name):
        """Routes method calls to the appropriate modular component."""
        for comp in self.components:
            try:
                attr = object.__getattribute__(comp, name)
            except AttributeError:
                continue
            else:
                return attr
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def clear_mlx_cache(self):
        """Explicitly clear MLX prompt cache."""
        if self._mlx_prompt_cache is not None:
            import gc
            self._mlx_prompt_cache = None
            self._last_mlx_prefix = ""
            self._last_mlx_cache_hit = False
            gc.collect()
        self._last_engine_error.clear()
        self._resident_last_used.pop(LocalEngine.MLX, None)

    def clear_llama_cache(self):
        with self._llama_lock:
            self._llama_model = None
            self._llama_model_path = None
            self._llama_cache = {}
        self._last_engine_error.clear()
        self._resident_last_used.pop(LocalEngine.LLAMA_CPP, None)

    def _touch_resident(self, engine: LocalEngine) -> None:
        self._resident_last_used[engine] = datetime.now(timezone.utc)

    @staticmethod
    def _process_memory_usage_ratio() -> float:
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

    def _memory_guard_check(self) -> tuple[bool, str]:
        if str(os.getenv("MODEL_MEMORY_GUARD", "on")).strip().lower() in {"0", "off", "false", "no"}:
            return True, "memory_guard=disabled"
        ratio = self._process_memory_usage_ratio()
        threshold = float(self._policy.memory_guard_threshold)
        if ratio <= 0.0:
            return True, f"memory_guard:unknown_usage;threshold={threshold:.2f}"
        if ratio < threshold:
            return True, f"memory_guard:ok;usage={ratio:.2f};threshold={threshold:.2f}"
        self.unload("inactive")
        ratio_after = self._process_memory_usage_ratio()
        if ratio_after <= 0.0 or ratio_after < threshold:
            return True, f"memory_guard:recovered;usage={ratio_after:.2f};threshold={threshold:.2f}"
        return False, f"memory_guard:blocked;usage={ratio_after:.2f};threshold={threshold:.2f}"

    def unload(self, target: str | LocalEngine = "all") -> str:
        normalized = str(target.value if isinstance(target, LocalEngine) else target).strip().lower() or "all"
        logs: list[str] = []
        if normalized in {"all", "mlx", "inactive"}:
            if self._mlx_model is not None or self._mlx_prompt_cache is not None:
                self._mlx_model = None
                self._mlx_tokenizer = None
                self._mlx_model_path = None
                self._mlx_prompt_cache = None
                self._last_mlx_prefix = ""
                self._last_mlx_cache_hit = False
                self._resident_last_used.pop(LocalEngine.MLX, None)
                logs.append("model_unload=mlx")
            elif normalized == "mlx":
                logs.append("model_unload=mlx_noop")
        if normalized in {"all", "llama_cpp", "llama", "inactive"}:
            with self._llama_lock:
                if self._llama_model is not None:
                    self._llama_model = None
                    self._llama_model_path = None
                    self._llama_cache = {}
                    self._resident_last_used.pop(LocalEngine.LLAMA_CPP, None)
                    logs.append("model_unload=llama_cpp")
                elif normalized in {"llama_cpp", "llama"}:
                    logs.append("model_unload=llama_cpp_noop")
        gc.collect()
        return ";".join(logs) if logs else "model_unload=noop"

    def _enforce_residency_policy(self, active_engine: LocalEngine) -> str:
        if self._policy.allow_dual_resident:
            return "model_residency=dual_allowed"
        logs: list[str] = []
        if active_engine == LocalEngine.MLX:
            if self._llama_model is not None:
                logs.append(self.unload(LocalEngine.LLAMA_CPP))
                logs.append("model_evict=llama_cpp")
        else:
            if self._mlx_model is not None:
                logs.append(self.unload(LocalEngine.MLX))
                logs.append("model_evict=mlx")
        return ";".join(item for item in logs if item) if logs else "model_residency=single_ok"

    def load(self, *, engine: LocalEngine, model_ref: str | None = None, profile: str = "recommended") -> tuple[bool, str]:
        self._enforce_residency_policy(engine)
        ok = False
        detail = ""
        try:
            if engine == LocalEngine.MLX:
                ok = bool(self._ensure_mlx_loaded(profile, model_ref, allow_runtime_install=False))
                detail = f"model_load=mlx;path={str(model_ref or self._mlx_model_path or '')}"
            else:
                ok = bool(self._ensure_llama_loaded(model_ref, allow_runtime_install=False))
                detail = f"model_load=llama_cpp;path={str(model_ref or self._llama_model_path or '')}"
        except Exception as exc:
            key = f"{engine.value}:{str(model_ref or '').strip()}"
            self._load_failures[key] = int(self._load_failures.get(key, 0)) + 1
            return False, f"model_load={engine.value};error={str(exc)}"
        if ok:
            self._touch_resident(engine)
            return True, detail
        key = f"{engine.value}:{str(model_ref or '').strip()}"
        self._load_failures[key] = int(self._load_failures.get(key, 0)) + 1
        return False, f"{detail};error=load_failed"

    def switch(self, *, engine: LocalEngine, model_ref: str | None = None, profile: str = "recommended") -> tuple[bool, str]:
        unload_detail = self.unload("inactive")
        loaded, detail = self.load(engine=engine, model_ref=model_ref, profile=profile)
        return loaded, f"{unload_detail};{detail}" if detail else unload_detail

    def health(self) -> dict[str, object]:
        loaded: list[dict[str, object]] = []
        if self._mlx_model is not None:
            loaded.append(
                {
                    "engine": LocalEngine.MLX.value,
                    "path": str(self._mlx_model_path or ""),
                    "last_used_at": self._resident_last_used.get(LocalEngine.MLX).isoformat()
                    if self._resident_last_used.get(LocalEngine.MLX)
                    else None,
                }
            )
        if self._llama_model is not None:
            loaded.append(
                {
                    "engine": LocalEngine.LLAMA_CPP.value,
                    "path": str(self._llama_model_path or ""),
                    "last_used_at": self._resident_last_used.get(LocalEngine.LLAMA_CPP).isoformat()
                    if self._resident_last_used.get(LocalEngine.LLAMA_CPP)
                    else None,
                }
            )
        resident_engine = loaded[-1]["engine"] if loaded else None
        return {
            "loaded": loaded,
            "resident_engine": resident_engine,
            "memory_usage_ratio": round(self._process_memory_usage_ratio(), 4),
            "policy": self._policy.model_dump(),
            "load_failures": dict(self._load_failures),
        }

    def set_stream_token_callback(self, callback) -> Token:
        return self._stream_token_callback_var.set(callback)

    def reset_stream_token_callback(self, token: Token) -> None:
        self._stream_token_callback_var.reset(token)

    def _current_stream_token_callback(self):
        return self._stream_token_callback_var.get()

    def _is_streaming_active(self) -> bool:
        return self._current_stream_token_callback() is not None

    def _emit_stream_token(self, text: str) -> None:
        cb = self._current_stream_token_callback()
        if cb is None:
            return
        value = str(text or "")
        if not value:
            return
        try:
            cb(value)
        except Exception:
            return

    @staticmethod
    def _normalize_model_reference(value: str | None) -> str | None:
        text = (value or "").strip()
        return text or None

    # Backward-compatible static helper bridge for tests that call
    # LocalInferenceEngine._method(...) directly.
    @staticmethod
    def _compat_engine() -> "LocalInferenceEngine":
        return LocalInferenceEngine()

    @staticmethod
    def _looks_pathlike_reference(value: str | None) -> bool:
        candidate = (value or "").strip()
        if not candidate:
            return False
        if candidate.startswith("/") or candidate.startswith("~"):
            return True
        if "\\" in candidate:
            return True
        if candidate.lower().endswith(".gguf"):
            return True
        return False

    def _looks_llama_model_reference(self, value: str | None) -> bool:
        candidate = self._normalize_model_reference(value)
        if not candidate: return False
        lowered = candidate.lower()
        if "://" in lowered: return lowered.endswith(".gguf")
        path = Path(candidate).expanduser()
        if path.suffix.lower() == ".gguf": return True
        if path.exists() and path.is_dir():
            try:
                return any(item.is_file() and item.suffix.lower() == ".gguf" for item in path.rglob("*.gguf"))
            except Exception: return False
        return False

    def _looks_mlx_model_reference(self, value: str | None) -> bool:
        candidate = self._normalize_model_reference(value)
        if not candidate: return False
        if self._looks_llama_model_reference(candidate): return False
        return self._is_mlx_model_reference_valid(candidate)

    def _resolve_engine_with_compatibility(
        self,
        *,
        engine: LocalEngine,
        mlx_model_path: str | None,
        llama_model_path: str | None,
    ) -> tuple[LocalEngine, str | None, str | None, str]:
        normalized_mlx = self._normalize_model_reference(mlx_model_path)
        normalized_llama = self._normalize_model_reference(llama_model_path)
        routing_notes: list[str] = []

        if normalized_llama and not self._looks_llama_model_reference(normalized_llama):
            if self._looks_pathlike_reference(normalized_llama):
                if self._looks_mlx_model_reference(normalized_llama):
                    routing_notes.append("llama_path_kept_for_mlx_compat")
                else:
                    routing_notes.append("ignored_invalid_llama_path")
                    normalized_llama = None
        if normalized_mlx and not self._looks_mlx_model_reference(normalized_mlx):
            if self._looks_pathlike_reference(normalized_mlx):
                if self._looks_llama_model_reference(normalized_mlx):
                    routing_notes.append("mlx_path_kept_for_llama_compat")
                else:
                    routing_notes.append("ignored_invalid_mlx_path")
                    normalized_mlx = None

        if normalized_mlx and self._looks_llama_model_reference(normalized_mlx) and not normalized_llama:
            normalized_llama = normalized_mlx
            routing_notes.append("copied_gguf_from_mlx_path")

        # Auto-discover model refs when settings paths are empty.
        if not normalized_llama:
            env_llama = self._normalize_model_reference(os.getenv("LOCAL_AI_MODEL_LLAMA"))
            if env_llama:
                normalized_llama = env_llama
                routing_notes.append("autodetect:llama_from_env")
            else:
                discovered_llama = self._normalize_model_reference(self._discover_downloaded_model(LocalEngine.LLAMA_CPP))
                if discovered_llama:
                    normalized_llama = discovered_llama
                    routing_notes.append("autodetect:llama_from_downloads")
        if not normalized_mlx:
            env_mlx = (
                self._normalize_model_reference(os.getenv("LOCAL_AI_MODEL_RECOMMENDED"))
                or self._normalize_model_reference(os.getenv("LOCAL_AI_MODEL_FAST"))
                or self._normalize_model_reference(os.getenv("LOCAL_AI_MODEL_DEEP"))
            )
            if env_mlx and self._looks_mlx_model_reference(env_mlx):
                normalized_mlx = env_mlx
                routing_notes.append("autodetect:mlx_from_env")
            else:
                discovered_mlx = self._normalize_model_reference(self._discover_downloaded_model(LocalEngine.MLX))
                if discovered_mlx and self._looks_mlx_model_reference(discovered_mlx):
                    normalized_mlx = discovered_mlx
                    routing_notes.append("autodetect:mlx_from_downloads")

        if normalized_llama and self._looks_mlx_model_reference(normalized_llama):
            if not normalized_mlx:
                normalized_mlx = normalized_llama
                routing_notes.append("copied_mlx_reference_from_llama_path")

        effective_engine = engine
        mlx_usable = self._looks_mlx_model_reference(normalized_mlx)
        llama_usable = self._looks_llama_model_reference(normalized_llama)

        if engine == LocalEngine.MLX:
            if normalized_mlx and self._looks_llama_model_reference(normalized_mlx):
                effective_engine = LocalEngine.LLAMA_CPP
                routing_notes.append("auto_switch:mlx_path_is_gguf")
            elif not mlx_usable and llama_usable:
                effective_engine = LocalEngine.LLAMA_CPP
                routing_notes.append("auto_switch:mlx_unavailable->llama")
                self.clear_mlx_cache()
        else:
            if normalized_llama and self._looks_mlx_model_reference(normalized_llama) and not self._looks_llama_model_reference(normalized_llama):
                effective_engine = LocalEngine.MLX
                routing_notes.append("auto_switch:llama_path_is_mlx")
            elif not llama_usable and mlx_usable:
                effective_engine = LocalEngine.MLX
                routing_notes.append("auto_switch:llama_unavailable->mlx")

        return effective_engine, normalized_mlx, normalized_llama, ";".join(routing_notes)

    def generate(
        self,
        query: str,
        mode: WorkMode,
        citations: list[Citation],
        profile: str,
        *,
        engine: LocalEngine = LocalEngine.MLX,
        mlx_model_path: str | None = None,
        llama_model_path: str | None = None,
        language_preference: str | None = None,
        max_tokens: int | None = None,
    ) -> InferenceResult:
        guard_ok, guard_detail = self._memory_guard_check()
        if not guard_ok:
            return InferenceResult(
                answer=f"Error: {guard_detail}",
                engine_used=engine,
                used_fallback=True,
                detail=guard_detail,
            )
        response_language = resolve_response_language(query, language_preference)
        prompt = self._prompt(query, mode, citations, response_language)
        token_budget = self._resolve_max_tokens(max_tokens, profile)
        primary, mlx_model_path, llama_model_path, routing_detail = self._resolve_engine_with_compatibility(
            engine=engine,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
        )

        residency_detail = self._enforce_residency_policy(primary)
        answer = self._generate_grounded_candidate(
            engine=primary,
            prompt=prompt,
            response_language=response_language,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=token_budget,
        )
        if answer:
            self._touch_resident(primary)
            return InferenceResult(
                answer=answer,
                engine_used=primary,
                detail=(
                    f"primary_engine={primary.value}; {guard_detail}; {residency_detail}; route={routing_detail}"
                    if routing_detail
                    else f"primary_engine={primary.value}; {guard_detail}; {residency_detail}"
                ),
            )

        err = self._last_engine_error.get(primary, f"{primary.value} engine failed")
        return InferenceResult(answer=f"Error: {err}", engine_used=primary, used_fallback=True)

    def generate_conversational(
        self,
        *,
        query: str,
        mode: WorkMode,
        profile: str,
        engine: LocalEngine = LocalEngine.MLX,
        mlx_model_path: str | None = None,
        llama_model_path: str | None = None,
        language_preference: str | None = None,
        max_tokens: int | None = None,
        session_summary: str | None = None,
        allow_static_fallback: bool = True,
    ) -> InferenceResult:
        guard_ok, guard_detail = self._memory_guard_check()
        if not guard_ok:
            return InferenceResult(
                answer="",
                engine_used=engine,
                used_fallback=True,
                detail=guard_detail,
            )
        response_language = resolve_response_language(query, language_preference)
        prompt = self._conversational_prompt(
            query=query,
            mode=mode,
            response_language=response_language,
            session_summary=session_summary,
        )
        token_budget = self._resolve_max_tokens(max_tokens, profile)
        primary, mlx_model_path, llama_model_path, routing_detail = self._resolve_engine_with_compatibility(
            engine=engine,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
        )

        residency_detail = self._enforce_residency_policy(primary)
        cand = self._generate_conversation_candidate(
            engine=primary,
            prompt=prompt,
            query=query,
            response_language=response_language,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=token_budget,
            allow_repair_fallbacks=allow_static_fallback,
        )
        if cand.answer:
            self._touch_resident(primary)
            detail = f"conversational_primary={primary.value}"
            if routing_detail:
                detail = f"{detail}; route={routing_detail}"
            detail = f"{detail}; {guard_detail}; {residency_detail}"
            if cand.quality_repair_reason:
                detail = f"{detail}; quality_issues={cand.quality_repair_reason}"
            return InferenceResult(answer=cand.answer, engine_used=primary, detail=detail)

        primary_error = self._last_engine_error.get(primary, f"{primary.value} engine failed")
        detail_parts = [
            "conversational engine failed",
            f"primary_error={primary.value}",
            str(primary_error),
        ]
        if routing_detail:
            detail_parts.append(f"route={routing_detail}")
        if cand.quality_repair_reason:
            detail_parts.append(f"quality_issues={cand.quality_repair_reason}")
        if cand.leak_blocked:
            detail_parts.append("leak_blocked=1")

        if allow_static_fallback:
            if self._is_brief_chat_query(query):
                detail_parts.append("static_fallback_suppressed=1")
                return InferenceResult(
                    answer="",
                    engine_used=primary,
                    used_fallback=True,
                    detail="; ".join(detail_parts),
                )
            safe_answer = self._minimal_safe_conversation_answer(
                query=query,
                response_language=response_language,
            ).strip()
            if safe_answer:
                detail_parts.append("static_fallback=1")
                return InferenceResult(
                    answer=safe_answer,
                    engine_used=primary,
                    used_fallback=True,
                    detail="; ".join(detail_parts),
                )

        return InferenceResult(
            answer="",
            engine_used=primary,
            used_fallback=True,
            detail="; ".join(detail_parts),
        )

    @staticmethod
    def _sanitize_generated_answer(raw: str, *, prompt: str) -> str:
        return LocalInferenceEngine._compat_engine().sanitizer._sanitize_generated_answer(raw, prompt=prompt)

    @staticmethod
    def _looks_model_answer(text: str, *, min_length: int = 12) -> bool:
        return LocalInferenceEngine._compat_engine().sanitizer._looks_model_answer(text, min_length=min_length)

    @staticmethod
    def _prepare_evidence_lines(citations: list[Citation], max_items: int, response_language: str) -> list[str]:
        return LocalInferenceEngine._compat_engine().sanitizer._prepare_evidence_lines(
            citations=citations,
            max_items=max_items,
            response_language=response_language,
        )

    @staticmethod
    def _looks_conversational_answer(text: str, *, response_language: str, query: str) -> bool:
        return LocalInferenceEngine._compat_engine().sanitizer._looks_conversational_answer(
            text,
            response_language=response_language,
            query=query,
        )

    @staticmethod
    def _postprocess_conversational_answer(answer: str, *, query: str, response_language: str) -> str:
        return LocalInferenceEngine._compat_engine().sanitizer._postprocess_conversational_answer(
            answer,
            query=query,
            response_language=response_language,
        )

    @staticmethod
    def _normalize_three_option_recommendation(answer: str, *, response_language: str) -> str:
        return LocalInferenceEngine._compat_engine().sanitizer._normalize_three_option_recommendation(
            answer,
            response_language=response_language,
        )

    @staticmethod
    def _is_recommendation_chat_query(query: str) -> bool:
        return LocalInferenceEngine._compat_engine().sanitizer._is_recommendation_chat_query(query)

    @staticmethod
    def _korean_quality_issues(*, query: str, answer: str, response_language: str) -> list[str]:
        return LocalInferenceEngine._compat_engine().conv_logic._korean_quality_issues(
            query=query,
            answer=answer,
            response_language=response_language,
        )

    @staticmethod
    def _strip_reasoning_leak(text: str) -> str:
        return LocalInferenceEngine._compat_engine().sanitizer._strip_reasoning_leak(text)

    @staticmethod
    def _pick_best_conversation_answer(
        *,
        primary_answer: str | None,
        primary_issues: list[str],
        primary_valid: bool,
        repaired_answer: str | None,
        repaired_issues: list[str],
        repaired_valid: bool,
        query: str,
        is_recommendation_query: bool,
    ) -> tuple[str | None, list[str]]:
        return LocalInferenceEngine._compat_engine().conv_logic._pick_best_conversation_answer(
            primary_answer=primary_answer,
            primary_issues=primary_issues,
            primary_valid=primary_valid,
            repaired_answer=repaired_answer,
            repaired_issues=repaired_issues,
            repaired_valid=repaired_valid,
            query=query,
            is_recommendation_query=is_recommendation_query,
        )

    @staticmethod
    def _sampling_preset(
        style: str,
        engine: LocalEngine,
    ) -> dict[str, float | int]:
        return LocalInferenceEngine._compat_engine().runtime_m._sampling_preset(style=style, engine=engine)  # type: ignore[arg-type]

    @staticmethod
    def _split_conversation_prompt_for_chat(prompt: str) -> tuple[str, str]:
        return LocalInferenceEngine._compat_engine().mlx_h._split_conversation_prompt_for_chat(prompt)

    @staticmethod
    def _is_qwen35_model_reference(model_path: str | None) -> bool:
        return LocalInferenceEngine._compat_engine().mlx_h._is_qwen35_model_reference(model_path)

    def _mlx_prepare_prompt(
        self,
        prompt: str,
        *,
        style: str,
        model_path: str | None,
    ) -> str:
        return self.mlx_h._mlx_prepare_prompt(prompt=prompt, style=style, model_path=model_path)

    @staticmethod
    def _resolve_max_tokens(max_tokens: int | None, profile: str) -> int:
        return LocalInferenceEngine._compat_engine().runtime_m._resolve_max_tokens(max_tokens=max_tokens, profile=profile)

    @staticmethod
    def _system_memory_gb() -> int:
        override = str(os.getenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE", "")).strip()
        if override:
            try:
                parsed = int(override)
                if parsed > 0:
                    return parsed
            except Exception:
                pass
        return 16

    @staticmethod
    def _mlx_context_window_hint(model_path: str | None) -> int:
        value = str(model_path or "").strip()
        if not value:
            return 32768
        config_path = Path(value).expanduser() / "config.json"
        if not config_path.exists():
            return 32768
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            text_cfg = payload.get("text_config") if isinstance(payload, dict) else None
            if isinstance(text_cfg, dict):
                hint = int(text_cfg.get("max_position_embeddings") or 0)
                if hint > 0:
                    return hint
        except Exception:
            pass
        return 32768

    @classmethod
    def _llama_runtime_context_window(cls, profile: str = "recommended") -> int:
        env_ctx = str(os.getenv("LOCAL_AI_LLAMA_N_CTX", "")).strip()
        requested = int(env_ctx) if env_ctx.isdigit() else 0
        memory = cls._system_memory_gb()
        key = str(profile or "recommended").lower()
        if memory <= 16:
            cap = 6144
        elif memory <= 32:
            cap = 8192
        else:
            cap = 12288 if key in {"deep", "advanced"} else 8192
        if requested > 0:
            return max(1024, min(requested, cap))
        return cap

    @classmethod
    def _llama_runtime_context_window_for_model(cls, *, profile: str, model_path: str | None) -> int:
        model_ref = str(model_path or "").lower()
        memory = cls._system_memory_gb()
        base = cls._llama_runtime_context_window(profile=profile)
        if memory <= 16 and "20b" in model_ref:
            return min(base, 3072)
        return base

    def _reload_llama_with_reduced_context(self, *, profile: str, explicit_model_path: str | None) -> bool:
        model_key = str(explicit_model_path or "").strip()
        if not model_key:
            return False
        current = self._llama_runtime_context_window_for_model(profile=profile, model_path=model_key)
        reduced = max(1024, current // 2)
        overrides = getattr(self, "_llama_n_ctx_overrides", {})
        overrides[model_key] = reduced
        self._llama_n_ctx_overrides = overrides
        return True

    @classmethod
    def _llama_gpu_layers_for_model(cls, model_path: str | None) -> int:
        model_ref = str(model_path or "").lower()
        memory = cls._system_memory_gb()
        if memory <= 16 and "20b" in model_ref:
            return 0
        return -1

    @staticmethod
    def _mlx_chunk_to_text(chunk: object) -> str:
        if chunk is None:
            return ""
        if hasattr(chunk, "text"):
            return str(getattr(chunk, "text") or "")
        if isinstance(chunk, dict):
            return str(chunk.get("text") or "")
        return str(chunk)

    def generate_reflection(
        self,
        *,
        engine: LocalEngine,
        query: str,
        context: str,
        answer: str,
        mlx_model_path: str | None = None,
        llama_model_path: str | None = None,
        response_language: str | None = None,
    ) -> tuple[str, str]:
        _ = (engine, mlx_model_path, llama_model_path, response_language)
        context_text = " ".join(str(context or "").split()).lower()
        answer_text = " ".join(str(answer or "").split()).lower()
        query_text = " ".join(str(query or "").split()).lower()

        if not answer_text:
            return ("INSUFFICIENT", "empty_answer")
        if not context_text:
            return ("INSUFFICIENT", "empty_context")

        answer_tokens = {token for token in answer_text.split() if len(token) >= 2}
        context_tokens = {token for token in context_text.split() if len(token) >= 2}
        query_tokens = {token for token in query_text.split() if len(token) >= 2}

        overlap = len(answer_tokens & context_tokens)
        if overlap == 0:
            if query_tokens and len(answer_tokens & query_tokens) == 0:
                return ("IRRELEVANT", "answer_not_aligned_with_query_or_context")
            return ("INSUFFICIENT", "answer_not_grounded_in_context")

        return ("SUPPORTED", f"context_overlap={overlap}")
