from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class MessageItem:
    role: Role
    content: str


@dataclass(slots=True)
class MessageState:
    items: list[MessageItem] = field(default_factory=list)

    def add(self, role: Role, content: str) -> None:
        text = str(content or "").strip()
        if not text:
            return
        self.items.append(MessageItem(role=role, content=text))

    def add_system(self, content: str) -> None:
        self.add("system", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def to_list(self) -> list[dict[str, str]]:
        return [{"role": item.role, "content": item.content} for item in self.items]

    def render_legacy_query(self) -> str:
        if not self.items:
            return ""
        user_only = [item.content for item in self.items if item.role == "user"]
        if not user_only:
            return ""
        system_only = [item.content for item in self.items if item.role == "system"]
        if not system_only:
            return user_only[-1]
        system_block = "\n\n".join(system_only)
        return f"{system_block}\n\n{user_only[-1]}".strip()

