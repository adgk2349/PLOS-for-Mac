from __future__ import annotations
import re
import json
from typing import TYPE_CHECKING
from ..base import BaseDelegate
from ...models import AgentAction

if TYPE_CHECKING:
    from ...local_inference import LocalInferenceEngine

# Component: agentic_parser.py
class AgenticParser(BaseDelegate):
    def generate_agentic_step(
        self,
        *,
        engine: LocalEngine,
        query: str,
        history: list[dict[str, str]],
        profile: str,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        max_tokens: int,
        response_language: str,
    ) -> AgentAction:
        prompt = self.build_prompt(query, history, response_language=response_language)
        raw = self._generate_with_engine(
            engine=engine,
            prompt=prompt,
            profile=profile,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            max_tokens=max_tokens,
            style="grounded",
        )
        return self.parse_action(raw or "")

    def build_prompt(self, query: str, history: list[dict[str, str]], response_language: str = "ko") -> str:
        system_prompt = self.agentic_system_prompt(response_language=response_language)
        context_block = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])

        return (
            f"{system_prompt}\n\n"
            f"User Request: {query}\n"
            f"History & Observations:\n{context_block}\n\n"
            "Next Step (Format: <thought>...</thought> followed by <action>...</action> or <final_answer>...</final_answer>):"
        )

    def parse_action(self, raw: str) -> AgentAction:
        if not raw:
            return AgentAction(kind="final_answer", params={"answer": "Error: Empty response from model."})

        thought_match = re.search(r"<thought>(.*?)</thought>", raw, re.DOTALL)
        thought = (thought_match.group(1).strip() if thought_match else "").strip()

        # Check for final answer first
        final_match = re.search(r"<final_answer>(.*?)</final_answer>", raw, re.DOTALL)
        if final_match:
            return AgentAction(kind="final_answer", params={"answer": final_match.group(1).strip()}, thought=thought)

        # Check for tool action
        action_match = re.search(r"<action>(.*?)</action>", raw, re.DOTALL)
        if action_match:
            try:
                action_text = action_match.group(1).strip()
                if action_text.startswith("{") and action_text.endswith("}"):
                    data = json.loads(action_text)
                    return AgentAction(
                        kind=data.get("kind", "spotlight_search"),
                        params=data.get("params", {}),
                        thought=thought
                    )
                else:
                    # Fallback for plain text actions like "spotlight_search: query"
                    if ":" in action_text:
                        kind, param = action_text.split(":", 1)
                        return AgentAction(kind=kind.strip(), params={"query": param.strip()}, thought=thought)
            except Exception:
                pass

        # Default fallback if parsing fails but text exists
        return AgentAction(kind="final_answer", params={"answer": raw.strip()}, thought=thought)
