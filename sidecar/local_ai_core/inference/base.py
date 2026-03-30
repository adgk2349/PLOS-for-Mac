from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..local_inference import LocalInferenceEngine

class BaseDelegate:
    """Base class for modular components that delegate back to the main engine."""
    def __init__(self, engine: 'LocalInferenceEngine'):
        self.engine = engine

    def __getattr__(self, name):
        """Seamlessly route attribute and method calls back to the main engine."""
        return getattr(self.engine, name)
