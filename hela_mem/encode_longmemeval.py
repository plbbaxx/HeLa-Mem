"""Resumable LongMemEval-S encoding for the unchanged HeLa-Mem algorithm."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .hebbian_knowledge_memory import HebbianKnowledgeMemory
from .hebbian_memory import HebbianMemoryGraph
from .hebbian_retriever import HebbianRetriever
from .profile_utils import OpenAIClient, gpt_personality_analysis, gpt_update_profile
from .runtime import append_jsonl, atomic_write_json, config_fingerprint, llm_scope, model_for, read_valid_json, run_metadata, usage_snapshot
from .utils import get_timestamp, load_api_keys

BUFFER_SIZE = int(os.environ.get("HEBBIAN_KNOWLEDGE_BUFFER_SIZE", "10"))


def parse_sessions(item: Dict[str, Any]) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    sessions, dates = item["haystack_sessions"], item["haystack_dates"]
    for session_idx, session in enumerate(sessions):
        timestamp = dates[session_idx] if session_idx < len(dates) else get_timestamp()
        start = 0
        while start < len(session) and session[start]["role"] != "user":
            start += 1
        i = start
        while i + 1 < len(session):
            if session[i]["role"] == "user" and session[i + 1]["role"] == "assistant":
                turns.append({"user_text": session[i]["content"], "ai_text": session[i + 1]["content"], "timestamp": timestamp})
                i += 2
            else:
                i += 1
    return turns


def process_incremental_buffer(buffer: List[Dict[str, str]], knowledge: HebbianKnowledgeMemory, item_id: str, client: Any) -> None:
    dialogs = [{"user_input": t["user_text"], "agent_response": t["ai_text"], "timestamp": t["timestamp"]} for t in buffer]
    result = gpt_personality_analysis(dialogs, client)
    old_profile = knowledge.get_raw_user_profile(item_id)
    updated = gpt_update_profile(old_profile, result["profile"], client) if old_profile else result["profile"]
    knowledge.update_user_profile(item_id, updated)
    for line in (result["private"] or "").splitlines():
        fact = line.strip().lstrip("- ").strip()
        if fact and fact.lower() != "none" and not fact.startswith("【"):
            knowledge.add_knowledge(fact)
    for line in (result["assistant_knowledge"] or "").splitlines():
        fact = line.strip().lstrip("- ").strip()
        if fact and fact.lower() != "none" and not fact.startswith("【"):
            knowledge.add_assistant_knowledge(fact)


def _paths(output_dir: str, item_id: str) -> Dict[str, Path]:
    root = Path(output_dir)
    return {"memory": root / f"{item_id}_hebbian.json", "knowledge": root / f"{item_id}_long_term.json", "knowledge_graph": root / f"{item_id}_long_term_kb_graph.json", "state": root / "state" / f"{item_id}.json"}


def _complete(output_dir: str, item_id: str, fingerprint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    paths = _paths(output_dir, item_id)
    state = read_valid_json(paths["state"], ("question_id", "status", "llm_usage"))
    if not state or state["question_id"] != item_id or state["status"] != "ok":
        return None
    if fingerprint is not None and state.get("config_fingerprint") != fingerprint:
        return None
    if not read_valid_json(paths["memory"], ("nodes", "edges")) or not read_valid_json(paths["knowledge"], ("user_profiles", "assistant_knowledge")) or not read_valid_json(paths["knowledge_graph"], ("nodes", "edges")):
        return None
    return state


def encode_single_item(item: Dict[str, Any], output_dir: str, client: Any, fingerprint: str = "") -> Dict[str, Any]:
    item_id, paths = item["question_id"], _paths(output_dir, item["question_id"])
    # Invalid partial outputs must not be loaded, otherwise turns are duplicated.
    for key in ("memory", "knowledge", "knowledge_graph"):
        paths[key].unlink(missing_ok=True)
    started = time.perf_counter()
    with llm_scope("other_calls", item_id):
        graph = HebbianMemoryGraph(file_path=str(paths["memory"]))
        knowledge = HebbianKnowledgeMemory(file_path=str(paths["knowledge"]))
        retriever = HebbianRetriever(graph, profile_memory=knowledge)
        turns, buffer = parse_sessions(item), []
        for turn in turns:
            retriever.process_conversation_turn(turn["user_text"], turn["ai_text"], timestamp=turn["timestamp"])
            buffer.append(turn)
            if len(buffer) >= BUFFER_SIZE:
                process_incremental_buffer(buffer, knowledge, item_id, client)
                buffer = []
        if buffer:
            process_incremental_buffer(buffer, knowledge, item_id, client)
        graph.save()
        knowledge.save()
    result = {"question_id": item_id, "status": "ok", "config_fingerprint": fingerprint, "history_turns": len(turns), "episodic_memory_count": len(graph.nodes), "semantic_memory_count": len(knowledge.knowledge_graph.nodes), "encode_seconds": time.perf_counter() - started, "llm_usage": usage_snapshot(item_id)}
    atomic_write_json(paths["state"], result)
    return result


def encode_longmemeval(data_path: str, output_dir: str, num_items: Optional[int] = None, start_item: int = 0, workers: int = 4, resume: bool = True) -> str:
    with open(data_path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    items = dataset[start_item:]
    if num_items is not None:
        items = items[:num_items]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_dir = Path(output_dir).parent
    config = {"stage": "encode", "start_item": start_item, "concurrency": workers, "knowledge_buffer_size": BUFFER_SIZE, "learning_rate": os.environ.get("HEBBIAN_LEARNING_RATE", "0.02"), "decay_rate": os.environ.get("HEBBIAN_DECAY_RATE", "0.995"), "activation_alpha": os.environ.get("HEBBIAN_ACTIVATION_ALPHA", "0.1"), "spreading_threshold": os.environ.get("HEBBIAN_SPREADING_THRESHOLD", "0.4"), "max_flipped": os.environ.get("HEBBIAN_MAX_FLIPPED", "3"), "tau": os.environ.get("HEBBIAN_TAU", "1e7"), "resume": resume}
    stable_config = {key: value for key, value in config.items() if key not in ("concurrency", "resume")}
    manifest = run_metadata(data_path, len(items), stable_config)
    fingerprint = config_fingerprint({key: value for key, value in manifest.items() if key != "created_at"})
    config["config_fingerprint"] = fingerprint
    manifest["config"] = config
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "run_config.json", config | {"generation_model": model_for("generation"), "extraction_model": model_for("extraction"), "judge_model": model_for("judge")})
    keys = load_api_keys()
    if not keys:
        raise RuntimeError("Set OPENAI_API_KEY (use EMPTY for a local vLLM server).")
    clients = [OpenAIClient(key, os.environ.get("OPENAI_BASE_URL")) for key in keys]
    completed, pending = [], []
    for item in items:
        state = _complete(output_dir, item["question_id"], fingerprint) if resume else None
        (completed if state else pending).append(state or item)
    print(f"Encoding {len(items)} items: {len(completed)} resumed, {len(pending)} pending")
    errors_path, started = run_dir / "errors.jsonl", time.perf_counter()
    errors_path.touch(exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(encode_single_item, item, output_dir, clients[i % len(clients)], fingerprint): item for i, item in enumerate(pending)}
        for future in as_completed(futures):
            item = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                append_jsonl(errors_path, {"timestamp": datetime.now().isoformat(), "stage": "encoding_failure", "question_id": item["question_id"], "error_type": type(exc).__name__, "message": str(exc)})
                print(f"[ERROR] {item['question_id']}: {exc}")
    elapsed = time.perf_counter() - started
    completed.sort(key=lambda value: value["question_id"])
    summary = {"timestamp": datetime.now().isoformat(), "total_items": len(items), "encoded": len(completed), "failed": len(items) - len(completed), "encode_time": elapsed, "avg_time_per_item": elapsed / max(len(pending), 1), "results": completed}
    atomic_write_json(Path(output_dir) / "encode_summary.json", summary)
    atomic_write_json(run_dir / "timing.json", {key: value for key, value in summary.items() if key != "results"})
    encode_usage = usage_snapshot()
    atomic_write_json(run_dir / "encode_llm_usage.json", encode_usage)
    atomic_write_json(run_dir / "llm_usage.json", {"encode": encode_usage, "eval": {}})
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_items", type=int)
    parser.add_argument("--start_item", type=int, default=0)
    parser.add_argument("--workers", "--concurrency", dest="workers", type=int, default=4)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()
    encode_longmemeval(args.data_path, args.output_dir, args.num_items, args.start_item, args.workers, not args.no_resume)


if __name__ == "__main__":
    main()
