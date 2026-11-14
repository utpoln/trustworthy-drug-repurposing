"""Smoke test: send one simple prompt to MindRouter and print the reply.

Usage:
    python scripts/test_mindrouter.py                  # model from .env
    python scripts/test_mindrouter.py --model qwen2.5:72b
    python scripts/test_mindrouter.py --list-models
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("MINDROUTER_API_KEY")
BASE_URL = os.getenv("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")
DEFAULT_MODEL = os.getenv("MINDROUTER_MODEL", "llama3.3:70b")

PROMPT = "In one sentence, what is drug repurposing?"


def strip_think(text: str) -> str:
    if text and "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return (text or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if not API_KEY:
        print("ERROR: MINDROUTER_API_KEY not set. Copy .env.example to .env and fill it in.")
        return 1

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)
    print(f"Endpoint: {BASE_URL}")

    if args.list_models:
        try:
            for m in sorted(client.models.list().data, key=lambda m: m.id):
                print("  ", m.id)
        except Exception as e:
            print(f"Could not list models: {type(e).__name__}: {e}")
            return 1
        return 0

    print(f"Model:    {args.model}\nPrompt:   {PROMPT}\n")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=256,
            temperature=0,
        )
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return 1

    msg = resp.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", None) or ""
    print(f"Response ({time.time() - t0:.1f}s):\n{strip_think(content)}\n")
    if resp.usage:
        print(f"Tokens: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens}")
    print("OK: MindRouter connection works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
