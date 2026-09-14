"""Reproducibility and accounting helpers for LongMemEval runs."""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


DEFAULT_MODEL = "gpt-4o-mini"
_call_kind = contextvars.ContextVar("hela_call_kind", default="other_calls")
_item_id = contextvars.ContextVar("hela_item_id", default="")
_lock = threading.Lock()
_usage: Counter = Counter()
_per_item: Dict[str, Counter] = {}


def model_for(role: str) -> str:
    names = {
        "generation": "HEBBIAN_GENERATION_MODEL",
        "extraction": "HEBBIAN_EXTRACTION_MODEL",
        "judge": "HEBBIAN_JUDGE_MODEL",
    }
    legacy = os.environ.get("HEBBIAN_MODEL", DEFAULT_MODEL)
    return os.environ.get(names[role], legacy)


def embedding_model() -> str:
    return os.environ.get("HEBBIAN_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def chat_extra_body() -> Dict[str, Any]:
    value = os.environ.get("HEBBIAN_ENABLE_THINKING", "").strip().lower()
    if value not in ("true", "false"):
        return {}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": value == "true"}}}


@contextmanager
def llm_scope(kind: str, item_id: str = "") -> Iterator[None]:
    kind_token = _call_kind.set(kind)
    item_token = _item_id.set(item_id or _item_id.get())
    try:
        yield
    finally:
        _call_kind.reset(kind_token)
        _item_id.reset(item_token)


def _record(values: Counter) -> None:
    item = _item_id.get()
    with _lock:
        _usage.update(values)
        if item:
            _per_item.setdefault(item, Counter()).update(values)


def record_llm_request() -> None:
    kind = _call_kind.get()
    values = Counter(total_llm_calls=1)
    values[kind] += 1
    _record(values)


def record_llm_usage(response: Any) -> None:
    values = Counter()
    usage = getattr(response, "usage", None)
    if usage is not None:
        for source, target in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = getattr(usage, source, None)
            if value is not None:
                values[target] += int(value)
    _record(values)


def usage_snapshot(item_id: Optional[str] = None) -> Dict[str, int]:
    keys = (
        "total_llm_calls", "keyword_extraction_calls", "profile_extraction_calls",
        "profile_merge_calls", "knowledge_extraction_calls",
        "answer_generation_calls", "judge_calls", "consolidation_calls",
        "other_calls", "prompt_tokens", "completion_tokens", "total_tokens",
    )
    with _lock:
        source = _per_item.get(item_id, Counter()) if item_id else _usage
        return {key: int(source.get(key, 0)) for key in keys}


def strip_reasoning(text: Optional[str]) -> str:
    """Remove Qwen-style reasoning blocks without altering answer text."""
    value = (text or "").strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<analysis>.*?</analysis>", "", value, flags=re.IGNORECASE | re.DOTALL)
    return value.strip()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)
    os.replace(temp_path, target)


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with _lock, target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_valid_json(path: str | Path, required: tuple[str, ...] = ()) -> Optional[Dict[str, Any]]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict) or any(key not in value for key in required):
            return None
        return value
    except (OSError, ValueError, TypeError):
        return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_metadata(data_path: str, num_items: int, config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(data_path).resolve()),
        "dataset_sha256": sha256_file(data_path),
        "num_items": num_items,
        "generation_model": model_for("generation"),
        "extraction_model": model_for("extraction"),
        "judge_model": model_for("judge"),
        "embedding_model": embedding_model(),
        "openai_base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "git_commit": git_commit(),
        "config": config,
    }


class Timer:
    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.seconds = time.perf_counter() - self.started
