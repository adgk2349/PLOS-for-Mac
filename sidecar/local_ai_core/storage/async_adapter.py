from __future__ import annotations

import asyncio
from typing import Any, Callable


class AsyncAdapter:
    """Centralized async adapter for legacy blocking calls."""

    async def run(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

