from __future__ import annotations

from typing import Any

from ..models import LocalEngine


class InferenceService:
    def __init__(self, *, local_inference, model_manager):
        self._local_inference = local_inference
        self._model_manager = model_manager

    @property
    def local_inference(self):
        return self._local_inference

    @property
    def model_manager(self):
        return self._model_manager

    def load(self, *, engine: LocalEngine, model_ref: str | None = None, profile: str = "recommended") -> tuple[bool, str]:
        return self._local_inference.load(engine=engine, model_ref=model_ref, profile=profile)

    def unload(self, target: str | LocalEngine = "all") -> str:
        return self._local_inference.unload(target)

    def switch(self, *, engine: LocalEngine, model_ref: str | None = None, profile: str = "recommended") -> tuple[bool, str]:
        return self._local_inference.switch(engine=engine, model_ref=model_ref, profile=profile)

    def health(self) -> dict[str, Any]:
        return self._local_inference.health()

