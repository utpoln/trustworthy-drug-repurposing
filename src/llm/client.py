"""MindRouter (OpenAI-compatible) chat client with retries and JSON parsing."""
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

BASE_URL = os.getenv("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")

MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.5-122b",
    "google/gemma-4-31b",
    "moonshotai/kimi-k3",
    "qwen/qwen3.8-27b",
]


def get_client(timeout: float = 300) -> OpenAI:
    key = os.getenv("MINDROUTER_API_KEY")
    if not key:
        raise RuntimeError("MINDROUTER_API_KEY not set (.env)")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=timeout, max_retries=0)


def strip_think(text: str) -> str:
    if text and "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return (text or "").strip()


def parse_json(text: str) -> Optional[dict]:
    """Parse the first JSON object in `text` (tolerates ```json fences and prose)."""
    text = strip_think(text)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def chat(client: OpenAI, model: str, system: str, user: str, *, temperature: float = 0.0,
         max_tokens: int = 4096, retries: int = 4, json_mode: bool = True) -> dict:
    """One chat call. Returns {content, reasoning, usage, latency_s, attempts, error}."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err = None
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            kw = dict(model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
            if json_mode:
                kw["response_format"] = {"type": "json_object"}
            r = client.chat.completions.create(**kw)
            m = r.choices[0].message
            content = m.content or ""
            reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
            return {
                "content": content,
                "reasoning": reasoning,
                "finish_reason": r.choices[0].finish_reason,
                "usage": {"prompt_tokens": r.usage.prompt_tokens,
                          "completion_tokens": r.usage.completion_tokens} if r.usage else {},
                "latency_s": round(time.time() - t0, 2),
                "attempts": attempt,
                "error": None,
            }
        except Exception as e:  # network / 5xx / unsupported response_format
            last_err = f"{type(e).__name__}: {e}"
            if json_mode and "response_format" in str(e):
                json_mode = False  # model does not support JSON mode: retry without it
                continue
            time.sleep(min(60, 2 ** attempt))
    return {"content": "", "reasoning": None, "finish_reason": None, "usage": {},
            "latency_s": None, "attempts": retries, "error": last_err}
