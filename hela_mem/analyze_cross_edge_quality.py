"""Gold-aware quality audit for existing Base-to-Non-Base Hebbian edges.

This module reads frozen LongMemEval data, baseline Base Top-K evidence, and
already encoded graph JSON.  It never instantiates a retriever or mutates a
graph.  In ``--mode llm`` it asks a deterministic *evidence classifier* to
label candidate memories; it does not generate answers or run the benchmark
judge.

The classifier labels a non-Base target once per question and the annotation
is then attached to every Base-to-target edge.  This avoids paying repeatedly
when several Base nodes point to the same candidate, while retaining the
edge-level statistics needed for the graph diagnosis.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .analyze_associative_candidate_expansion import base_top_k_ids, load_jsonl
from .analyze_graph_topology import undirected_edges
from .runtime import chat_extra_body, model_for, strip_reasoning


LABELS = ("supporting", "redundant", "irrelevant", "uncertain")
SYSTEM_PROMPT = """You are auditing memory evidence for a benchmark question.
You must assess only the supplied candidate memory against the supplied
question, reference answer, and already retrieved Base memories. Do not use
outside knowledge and do not infer unstated facts.

Assign exactly one label to every candidate:
- supporting: the candidate explicitly supplies a fact needed to answer the
  reference answer and adds that answer-relevant fact beyond the Base memories.
- redundant: the candidate is relevant, but its answer-relevant fact is already
  supplied by the Base memories.
- irrelevant: it does not explicitly supply answer-relevant evidence.
- uncertain: the supplied text is insufficient to decide reliably.

Return JSON only: {\"labels\":[{\"candidate_id\":str,\"label\":str,
\"rationale\":str,\"confidence\":number}]}. Rationale must be at most 30
words and confidence must be between 0 and 1."""


def _compact(text: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def extract_cross_edges(graph: Dict[str, Any], base_ids: List[str]) -> List[Dict[str, Any]]:
    """Return canonical Base-to-Non-Base edges from a graph snapshot."""
    base = set(base_ids)
    rows = []
    for left, right, weight in undirected_edges(graph.get("edges", {})):
        if (left in base) == (right in base):
            continue
        base_id, nonbase_id = (left, right) if left in base else (right, left)
        rows.append({"base_id": base_id, "nonbase_id": nonbase_id, "edge_weight": weight})
    return sorted(rows, key=lambda row: (row["nonbase_id"], row["base_id"]))


def build_records(item: Dict[str, Any], prediction: Dict[str, Any], graph: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    base_ids = base_top_k_ids(prediction, top_k)
    nodes = graph.get("nodes", {})
    edges = extract_cross_edges(graph, base_ids)
    targets: Dict[str, Dict[str, Any]] = {}
    for edge in edges:
        target = targets.setdefault(edge["nonbase_id"], {
            "candidate_id": edge["nonbase_id"], "candidate_text": _compact(nodes.get(edge["nonbase_id"], {}).get("content", ""), 700),
            "connected_base_ids": [], "edge_weights": [],
        })
        target["connected_base_ids"].append(edge["base_id"])
        target["edge_weights"].append(edge["edge_weight"])
    base_memories = [
        {"memory_id": node_id, "text": _compact(nodes.get(node_id, {}).get("content", ""), 350)}
        for node_id in base_ids
    ]
    return {
        "question_id": item["question_id"],
        "question_type": item.get("question_type", ""),
        "question": item["question"],
        "reference_answer": item.get("answer", ""),
        "base_top_k_ids": base_ids,
        "base_memories": base_memories,
        "cross_edges": edges,
        "candidate_targets": list(targets.values()),
    }


def build_prompt(record: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    payload = {
        "question_id": record["question_id"],
        "question_type": record["question_type"],
        "question": record["question"],
        "reference_answer": record["reference_answer"],
        "base_memories": record["base_memories"],
        "candidate_memories": [{"candidate_id": row["candidate_id"], "text": row["candidate_text"]} for row in candidates],
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_response(text: str, allowed_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    value = strip_reasoning(text).strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    parsed = json.loads(value)
    labels = {}
    for row in parsed.get("labels", []):
        candidate_id = str(row.get("candidate_id", ""))
        label = str(row.get("label", "")).lower()
        if candidate_id in allowed_ids and label in LABELS:
            labels[candidate_id] = {
                "label": label,
                "rationale": str(row.get("rationale", ""))[:300],
                "confidence": max(0.0, min(1.0, float(row.get("confidence", 0.0)))),
            }
    return labels


def _client() -> Any:
    """Import the SDK only for LLM mode so read-only tooling stays testable."""
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("--mode llm requires openai>=1.0") from error
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=os.environ.get("OPENAI_BASE_URL"))


def classify_batch(record: Dict[str, Any], candidates: List[Dict[str, Any]], model: str, retries: int) -> Dict[str, Dict[str, Any]]:
    prompt = build_prompt(record, candidates)
    allowed_ids = {row["candidate_id"] for row in candidates}
    client = _client()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1200,
                **chat_extra_body(),
            )
            text = response.choices[0].message.content if response.choices else ""
            labels = parse_response(text or "", allowed_ids)
            return {
                candidate_id: labels.get(candidate_id, {"label": "uncertain", "rationale": "classifier omitted this candidate", "confidence": 0.0})
                for candidate_id in allowed_ids
            }
        except Exception as error:  # network/server failures must not discard the audit row
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    message = f"classifier failure: {last_error}"
    return {candidate_id: {"label": "uncertain", "rationale": message[:300], "confidence": 0.0} for candidate_id in allowed_ids}


def annotate_record(record: Dict[str, Any], mode: str, model: str, batch_size: int, retries: int) -> Dict[str, Any]:
    candidates = record["candidate_targets"]
    labels: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        if mode == "llm":
            labels.update(classify_batch(record, batch, model, retries))
        else:
            labels.update({row["candidate_id"]: {"label": "uncertain", "rationale": "manual review required", "confidence": 0.0} for row in batch})
    record["candidate_annotations"] = [dict(row, **labels[row["candidate_id"]]) for row in candidates]
    annotations = {row["candidate_id"]: row for row in record["candidate_annotations"]}
    record["annotated_cross_edges"] = [dict(edge, **annotations[edge["nonbase_id"]]) for edge in record["cross_edges"]]
    return record


def summarize(records: List[Dict[str, Any]], dataset_question_count: int | None = None) -> Dict[str, Any]:
    edge_labels = Counter(edge["label"] for record in records for edge in record["annotated_cross_edges"])
    target_labels = Counter(target["label"] for record in records for target in record["candidate_annotations"])
    questions_with_edges = [record for record in records if record["cross_edges"]]
    useful_questions = sum(any(target["label"] == "supporting" for target in record["candidate_annotations"]) for record in questions_with_edges)
    weights_by_label: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        for edge in record["annotated_cross_edges"]:
            weights_by_label[edge["label"]].append(float(edge["edge_weight"]))
    total_edges, total_targets = sum(edge_labels.values()), sum(target_labels.values())
    total_questions = dataset_question_count if dataset_question_count is not None else len(records)
    return {
        "total_questions": total_questions,
        "questions_with_cross_edges": len(questions_with_edges),
        "questions_with_cross_edges_rate": len(questions_with_edges) / total_questions if total_questions else 0.0,
        "questions_with_supporting_target": useful_questions,
        "questions_with_supporting_target_rate_among_cross_edge_questions": useful_questions / len(questions_with_edges) if questions_with_edges else 0.0,
        "questions_with_supporting_target_rate_all_questions": useful_questions / total_questions if total_questions else 0.0,
        "edge_label_counts": dict(edge_labels),
        "edge_label_rates": {label: edge_labels[label] / total_edges if total_edges else 0.0 for label in LABELS},
        "unique_target_label_counts": dict(target_labels),
        "unique_target_label_rates": {label: target_labels[label] / total_targets if total_targets else 0.0 for label in LABELS},
        "edge_weight_by_label": {
            label: {"count": len(weights), "mean": statistics.mean(weights) if weights else 0.0, "median": statistics.median(weights) if weights else 0.0}
            for label, weights in sorted(weights_by_label.items())
        },
        "caveat": "Labels are automated evidence-judge outputs, not ground truth. Validate a stratified manual sample before making a paper claim.",
    }


def manual_review_sample(records: List[Dict[str, Any]], per_label: int) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        for target in record["candidate_annotations"]:
            buckets[target["label"]].append({
                "question_id": record["question_id"], "question": record["question"], "reference_answer": record["reference_answer"],
                "base_memories": record["base_memories"], "candidate_id": target["candidate_id"], "candidate_text": target["candidate_text"],
                "automatic_label": target["label"], "automatic_rationale": target["rationale"], "automatic_confidence": target["confidence"],
                "human_label": "", "human_notes": "",
            })
    return [row for label in LABELS for row in buckets[label][:per_label]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold-aware, read-only cross-edge evidence-quality audit")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("llm", "manual-template"), default="llm")
    parser.add_argument("--model", default=os.environ.get("HEBBIAN_EVIDENCE_MODEL") or model_for("generation"))
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--manual-samples-per-label", type=int, default=30)
    args = parser.parse_args()

    dataset = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    predictions = load_jsonl(Path(args.base_predictions))
    if {item["question_id"] for item in dataset} != set(predictions):
        raise SystemExit("dataset and baseline predictions must contain identical question IDs")
    records = []
    for item in dataset:
        graph_path = Path(args.mem_dir) / f"{item['question_id']}_hebbian.json"
        if not graph_path.exists():
            raise SystemExit(f"missing encoded graph: {graph_path}")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        record = build_records(item, predictions[item["question_id"]], graph, args.top_k)
        if record["cross_edges"]:
            records.append(record)

    if args.mode == "llm" and not os.environ.get("OPENAI_BASE_URL"):
        raise SystemExit("--mode llm requires OPENAI_BASE_URL (for example http://127.0.0.1:18000/v1)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(annotate_record, record, args.mode, args.model, args.batch_size, args.retries) for record in records]
        completed = []
        for index, future in enumerate(futures, start=1):
            completed.append(future.result())
            print(f"Annotated cross-edge questions: {index}/{len(futures)}", flush=True)
    completed.sort(key=lambda row: row["question_id"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "read_only": True, "answer_generation_executed": False, "benchmark_judge_executed": False,
        "mode": args.mode, "evidence_model": args.model if args.mode == "llm" else None,
        "temperature": 0.0 if args.mode == "llm" else None, "data_path": args.data_path, "mem_dir": args.mem_dir,
        "base_predictions": args.base_predictions, "top_k": args.top_k, "summary": summarize(completed, len(dataset)), "records": completed,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    review_path = output.with_name(output.stem + "_manual_review.json")
    review_path.write_text(json.dumps(manual_review_sample(completed, args.manual_samples_per_label), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    print("label\tedges\tedge_rate\tunique_targets\ttarget_rate")
    for label in LABELS:
        print(f"{label}\t{summary['edge_label_counts'].get(label, 0)}\t{summary['edge_label_rates'][label]:.4f}\t{summary['unique_target_label_counts'].get(label, 0)}\t{summary['unique_target_label_rates'][label]:.4f}")
    print(f"questions_with_supporting_target\t{summary['questions_with_supporting_target']}/{summary['questions_with_cross_edges']}\t{summary['questions_with_supporting_target_rate_among_cross_edge_questions']:.4f}")
    print(f"manual_review_template\t{review_path}")


if __name__ == "__main__":
    main()
