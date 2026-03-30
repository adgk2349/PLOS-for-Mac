"""Backward-compatible bridge for web retrieval.

Canonical path: local_ai_core.retrieval.web_retrieval
"""

from .retrieval.web_retrieval import *  # noqa: F401,F403
from .retrieval.web_retrieval import (  # noqa: F401
    _DiscoveredURL,
    _FetchedPage,
    _SearxngHTMLResultParser,
)
