"""Provenance logging: one JSON record per recommendation.

Every record carries enough to reproduce it: data version and file hashes,
candidate model + checkpoint hash, retrieval settings, LLM model, prompt
version and full prompt text, retrieved paths, raw response, parsed output,
token usage, latency, timestamp and git commit.
"""
import hashlib
import json
import subprocess
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@lru_cache(maxsize=None)
def git_commit() -> str:
    try:
        c = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JsonlLog:
    """Append-only, thread-safe JSONL log with resume support."""

    def __init__(self, path: Path, key_fields: Tuple[str, ...]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_fields = key_fields
        self._lock = threading.Lock()

    def done_keys(self) -> Set[tuple]:
        keys = set()
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("llm", {}).get("error") is None:
                        keys.add(tuple(r[k] for k in self.key_fields))
        return keys

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")

    def read(self) -> Iterable[dict]:
        with open(self.path) as f:
            for line in f:
                yield json.loads(line)
