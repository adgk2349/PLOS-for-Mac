import threading
from typing import Callable, Optional

class StreamContext:
    _ctx = threading.local()

    @classmethod
    def set_callback(cls, cb: Optional[Callable[[dict], None]]):
        cls._ctx.cb = cb

    @classmethod
    def push_event(cls, event: dict):
        cb = getattr(cls._ctx, "cb", None)
        if cb:
            cb(event)
