from __future__ import annotations


class ChatFacade:
    """Thin API-facing facade to keep AppState orchestration separate from chat internals."""

    def __init__(self, chat_service):
        self._chat = chat_service

    async def local_chat_v2(self, req):
        return await self._chat.local_chat_v2(req)

    def local_chat_v2_stream(self, req):
        return self._chat.local_chat_v2_stream(req)

    def local_chat(self, req):
        return self._chat.local_chat(req)

    async def deep_analysis(self, req):
        return await self._chat.deep_analysis(req)

