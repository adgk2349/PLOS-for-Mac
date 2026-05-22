import json
import sys
import os
import time
import argparse
import threading
import inspect
import re
from typing import Any, Dict
from local_ai_core.inference.utils import inject_system_instruction_if_needed

try:
    import mlx_lm
    from mlx_lm import load, generate, stream_generate
except ImportError:
    print(json.dumps({"status": "error", "error": "mlx_lm not installed"}))
    sys.exit(1)

# Global model cache to avoid reloading
_MODEL_CACHE = {
    "path": None,
    "model": None,
    "tokenizer": None
}


def _chunk_to_text(chunk: Any) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        for key in ("text", "token", "content", "delta"):
            value = chunk.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("text", "token", "content", "delta"):
        try:
            value = getattr(chunk, key)
            if isinstance(value, str) and value:
                return value
        except Exception:
            continue
    value = str(chunk)
    return value if value else ""


def _sanitize_piece(text: str, strip_whitespace: bool = True) -> str:
    value = str(text or "")
    if not value:
        return ""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    # Drop obvious internal object repr/meta fragments.
    value = re.sub(r"(?is)GenerationResponse\([^)]*\)", "", value)
    value = re.sub(r"(?i)\b(?:prompt|generation)_tokens\s*=\s*\d+\b", "", value)
    value = re.sub(r"(?i)\b(?:prompt|generation)_tps\s*=\s*[0-9.]+\b", "", value)
    value = re.sub(r"(?i)\bfrom_draft\s*=\s*(?:true|false)\b", "", value)
    value = re.sub(r"(?i)\bfinish_reason\s*=\s*['\"]?[a-z_]+['\"]?\)?", "", value)
    value = re.sub(r"(?i)\bpeak_memory\s*=\s*[0-9.]+\b", "", value)
    value = re.sub(r"(?is)<\|channel[^>]*>", "", value)
    value = re.sub(r"(?is)<followup_hint>.*?(?:</followup_hint>|$)", "", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip() if strip_whitespace else value


def _sanitize_output(text: str) -> str:
    value = _sanitize_piece(text)
    if not value:
        return ""
    value = re.sub(r"(?i)\b(?:recent_user|recent_assistant|last_query|nt_user|nt_assistant)\s*:\s*[^\n]{0,180}", "", value)
    value = re.sub(r"(?i)\b(?:answer|response)\s*[:：]\s*", "", value)
    value = re.sub(r"(?:\(\s*\)\s*){6,}", "", value)
    value = re.sub(r"(?:\)\s*){10,}", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"[ \t]{2,}", " ", value).strip()
    return value

def load_or_get_model(model_path: str):
    if _MODEL_CACHE["path"] == model_path and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"]
    
    # Release previous model if any (best effort for Metal)
    _MODEL_CACHE["model"] = None
    _MODEL_CACHE["tokenizer"] = None
    _MODEL_CACHE["path"] = None
    
    model, tokenizer = load(model_path)
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["tokenizer"] = tokenizer
    _MODEL_CACHE["path"] = model_path
    return model, tokenizer


def _split_prompt_for_chat(prompt: str) -> tuple[str, str]:
    text = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "", ""
    
    # 1. Check for explicit conversation marker
    convo_marker = "--- conversation ---"
    marker_pos = text.lower().rfind(convo_marker)
    if marker_pos != -1:
        system_text = text[:marker_pos].strip()
        tail = text[marker_pos + len(convo_marker):].strip()
        tail = re.sub(
            r"(?is)\n?\s*respond with plain answer text only\.?\s*$",
            "",
            tail,
        ).strip()
        return system_text, tail

    # 2. Check for "Input message:"
    last_marker_pos = text.lower().rfind("input message:")
    if last_marker_pos != -1:
        user_part = text[last_marker_pos:].strip()
        match = re.search(r"(?is)Input message:\s*(.+?)(?:\n\s*Answer:\s*)?$", user_part)
        if match:
            user_text = match.group(1).strip()
            system_text = text[:last_marker_pos].strip()
            return system_text, user_text
        
        user_text = re.sub(r"(?is)^Input message:\s*", "", user_part)
        user_text = re.sub(r"(?is)\n\s*Answer:\s*$", "", user_text).strip()
        system_text = text[:last_marker_pos].strip()
        return system_text, user_text

    # 3. Fallback: split by the last non-empty line as user text,
    # and all previous lines as system text
    lines = text.splitlines()
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    if not non_empty_indices:
        return "", text
    
    last_idx = non_empty_indices[-1]
    user_text = lines[last_idx].strip()
    
    system_lines = lines[:last_idx]
    system_text = "\n".join(system_lines).strip()
    return system_text, user_text


def _inject_system_instruction_if_needed(message_state: list[dict[str, str]], response_language: str | None = None) -> list[dict[str, str]]:
    return inject_system_instruction_if_needed(message_state, response_language)


def _prepare_prompt_for_chat_template(prompt: str, tokenizer: Any, style: str, message_state: list[dict[str, str]] | None = None, response_language: str | None = None) -> str:
    if str(style or "").strip().lower() != "conversation":
        return prompt
    if tokenizer is None:
        return prompt
    if not getattr(tokenizer, "has_chat_template", False):
        return prompt
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    if message_state:
        messages = _inject_system_instruction_if_needed(message_state, response_language)
    else:
        # If the prompt is already chat-templated, do not re-apply
        lower_prompt = prompt.lower()
        chat_markers = ["<|im_start|>", "<start_of_turn>", "<|start_header_id|>", "<|begin_of_text|>"]
        if any(marker in lower_prompt for marker in chat_markers):
            return prompt

        system_text, user_text = _split_prompt_for_chat(prompt)
        if not user_text:
            return prompt

        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    except Exception:
        return prompt
    rendered_text = str(rendered or "").strip()
    return rendered_text or prompt

def handle_request(payload: Dict[str, Any]):
    try:
        model_path = payload.get("model_path")
        if not model_path:
            return {"ok": False, "error": "missing_model_path"}
        
        args = payload.get("kwargs", {})
        prompt = args.get("prompt", "")
        max_tokens = args.get("max_tokens", 512)
        temp = args.get("temp", 0.78)
        top_p = args.get("top_p", 0.90)
        top_k = int(args.get("top_k", 40))
        min_p = float(args.get("min_p", 0.05))
        repetition_penalty = args.get("repetition_penalty", 1.10)
        stop_sequences = args.get("stop_sequences", [])
        stream = bool(payload.get("stream"))
        style = str(payload.get("style") or "")
        message_state = args.get("message_state")
        response_language = args.get("response_language")
        
        model, tokenizer = load_or_get_model(model_path)
        prompt = _prepare_prompt_for_chat_template(prompt, tokenizer, style, message_state=message_state, response_language=response_language)
        
        # MLX generate supports 'stop' which can be a string or list of strings
        # We need to ensure we don't return the stop sequence itself in the output
        start_time = time.time()
        
        # mlx_lm.generate signatures vary by version.
        # Build kwargs conservatively and keep compatibility with older/newer APIs.
        kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        gen_fn = stream_generate if stream else generate
        try:
            sig = inspect.signature(gen_fn)
            names = set(sig.parameters.keys())
        except Exception:
            names = set()
        if "top_p" in names:
            kwargs["top_p"] = top_p
        if "top_k" in names and top_k > 0:
            kwargs["top_k"] = top_k
        if "min_p" in names and min_p > 0.0:
            kwargs["min_p"] = min_p
        if "repetition_penalty" in names:
            kwargs["repetition_penalty"] = repetition_penalty
        if "stop" in names:
            kwargs["stop"] = stop_sequences if stop_sequences else None
        if "verbose" in names:
            kwargs["verbose"] = False
        if "temp" in names:
            kwargs["temp"] = temp
        elif "temperature" in names:
            kwargs["temperature"] = temp
        else:
            # Some builds reject both temp/temperature.
            # Omit temperature override in that case.
            pass

        if stream:
            parts = []
            stream_result = stream_generate(model, tokenizer, **kwargs)
            for chunk in stream_result:
                piece = _chunk_to_text(chunk)
                piece = _sanitize_piece(piece, strip_whitespace=False)
                if not piece:
                    continue
                parts.append(piece)
                print(json.dumps({"event": "token", "text": piece}, ensure_ascii=False), flush=True)
            text = "".join(parts)
        else:
            text = generate(model, tokenizer, **kwargs)
        
        # Basic cleanup: remove prompt from output if it leak
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        text = _sanitize_output(text)
            
        elapsed = time.time() - start_time
        return {
            "ok": True, 
            "text": text, 
            "elapsed": elapsed,
            "tokens_limit": max_tokens
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["persistent", "oneshot"], default="persistent")
    args = parser.parse_args()

    # Indicate ready
    print(json.dumps({"status": "ready"}), flush=True)

    if args.mode == "oneshot":
        line = sys.stdin.readline()
        if line:
            payload = json.loads(line)
            result = handle_request(payload)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    # Persistent mode loop
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            result = handle_request(payload)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"worker_loop_error: {str(e)}"}), flush=True)

if __name__ == "__main__":
    main()
