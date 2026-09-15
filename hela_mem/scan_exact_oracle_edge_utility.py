"""V0.5 exact paired oracle scan through HeLa-Mem's retrieval implementation.

For each question the scanner computes query keywords and the query embedding
once, then reuses them in two calls to ``HebbianMemoryGraph.retrieve`` over the
same unchanged graph.  The control uses original weights; the oracle arm only
multiplies directed Base-to-Non-Base edges using gold-aware evidence labels.
Both calls disable post-retrieval Hebbian reinforcement.

This is an oracle retrieval diagnostic.  It does not generate an answer, run
the LongMemEval judge, save a graph, or represent a deployable method.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .analyze_cross_edge_quality import build_records, classify_batch
from .runtime import atomic_write_json, model_for


MULTIPLIERS = {"supporting": 2.0, "redundant": 0.2, "irrelevant": 0.0, "uncertain": 1.0}
# Mirror eval_longmemeval.py: these rows never enter the real retrieval path.
CORRUPTED_INDICES = {74, 183, 278, 351, 380}


def load_quality_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["question_id"]): record for record in report.get("records", [])}


def compact_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "base_top_k_ids": trace.get("base_top_k_ids", []),
        "flipped_memory_ids": trace.get("flipped_memory_ids_after", []),
        "query_keywords": trace.get("query_keywords", []),
        "edge_weight_calibration_enabled": trace.get("edge_weight_calibration_enabled", False),
        "reinforcement_enabled": trace.get("reinforcement_enabled", True),
    }


def annotation_map(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row["candidate_id"]): row for row in record.get("candidate_annotations", [])}


def labels_for_current_base(
    current_record: Dict[str, Any],
    cached_record: Dict[str, Any] | None,
    evidence_model: str,
    batch_size: int,
    retries: int,
) -> tuple[Dict[str, Dict[str, Any]], str]:
    current_ids = {row["candidate_id"] for row in current_record["candidate_targets"]}
    if cached_record is not None and cached_record.get("base_top_k_ids") == current_record["base_top_k_ids"]:
        cached = annotation_map(cached_record)
        if current_ids.issubset(cached):
            return {candidate_id: cached[candidate_id] for candidate_id in current_ids}, "reused_quality_report"
    labels: Dict[str, Dict[str, Any]] = {}
    candidates = current_record["candidate_targets"]
    for start in range(0, len(candidates), batch_size):
        labels.update(classify_batch(current_record, candidates[start:start + batch_size], evidence_model, retries))
    return labels, "recomputed_for_current_base"


def multipliers_for_record(record: Dict[str, Any], labels: Dict[str, Dict[str, Any]]) -> Dict[tuple[str, str], float]:
    multipliers = {}
    for target in record["candidate_targets"]:
        candidate_id = target["candidate_id"]
        label = labels.get(candidate_id, {}).get("label", "uncertain")
        multiplier = MULTIPLIERS.get(label, 1.0)
        for base_id in target["connected_base_ids"]:
            multipliers[(str(base_id), str(candidate_id))] = multiplier
    return multipliers


def label_selected_ids(
    record: Dict[str, Any],
    selected_ids: List[str],
    labels: Dict[str, Dict[str, Any]],
    nodes: Dict[str, Dict[str, Any]],
    evidence_model: str,
    batch_size: int,
    retries: int,
) -> Dict[str, Dict[str, Any]]:
    """Label selected non-cross-edge memories for unbiased outcome accounting."""
    missing = [node_id for node_id in selected_ids if node_id not in labels and node_id in nodes]
    for start in range(0, len(missing), batch_size):
        candidates = [
            {"candidate_id": node_id, "candidate_text": nodes[node_id].get("content", "")[:700]}
            for node_id in missing[start:start + batch_size]
        ]
        labels.update(classify_batch(record, candidates, evidence_model, retries))
    return labels


def process_item(
    item: Dict[str, Any],
    graph_path: Path,
    saved_prediction: Dict[str, Any],
    cached_quality: Dict[str, Any] | None,
    evidence_model: str,
    top_k: int,
    batch_size: int,
    retries: int,
) -> Dict[str, Any]:
    from .hebbian_memory import HebbianMemoryGraph
    from .utils import get_embedding, get_timestamp, llm_extract_keywords

    question_id = str(item["question_id"])
    graph = HebbianMemoryGraph(file_path=str(graph_path))
    graph_payload = {"nodes": graph.nodes, "edges": {source: dict(neighbors) for source, neighbors in graph.edges.items()}}
    try:
        query_keywords = set(llm_extract_keywords(item["question"]))
        keyword_status = "ok"
    except Exception as error:
        query_keywords = set()
        keyword_status = f"fallback_empty:{type(error).__name__}"
    query_embedding = get_embedding(item["question"])
    paired_time = get_timestamp()

    baseline_results = graph.retrieve(
        item["question"], top_k=top_k,
        query_keywords_override=query_keywords,
        query_embedding_override=query_embedding,
        current_time_override=paired_time,
        update_graph=False,
    )
    baseline_trace = compact_trace(copy.deepcopy(graph.last_retrieval_trace or {}))
    current_prediction = {
        "question_id": question_id,
        "retrieved_episodic": [
            {"source": "base", "node_id": row["node"]["id"]}
            for row in baseline_results if row.get("source") == "base"
        ],
    }
    current_record = build_records(item, current_prediction, graph_payload, top_k)
    labels, label_source = labels_for_current_base(
        current_record, cached_quality, evidence_model, batch_size, retries,
    )
    edge_multipliers = multipliers_for_record(current_record, labels)

    oracle_results = graph.retrieve(
        item["question"], top_k=top_k,
        query_keywords_override=query_keywords,
        query_embedding_override=query_embedding,
        current_time_override=paired_time,
        edge_weight_multipliers=edge_multipliers,
        update_graph=False,
    )
    oracle_trace = compact_trace(copy.deepcopy(graph.last_retrieval_trace or {}))
    baseline_flipped = baseline_trace["flipped_memory_ids"]
    oracle_flipped = oracle_trace["flipped_memory_ids"]
    labels = label_selected_ids(
        current_record, list(dict.fromkeys(baseline_flipped + oracle_flipped)), labels,
        graph.nodes, evidence_model, batch_size, retries,
    )
    added = [node_id for node_id in oracle_flipped if node_id not in baseline_flipped]
    removed = [node_id for node_id in baseline_flipped if node_id not in oracle_flipped]
    supporting_added = [node_id for node_id in added if labels.get(node_id, {}).get("label") == "supporting"]
    irrelevant_removed = [node_id for node_id in removed if labels.get(node_id, {}).get("label") == "irrelevant"]
    baseline_useful = [node_id for node_id in baseline_flipped if labels.get(node_id, {}).get("label") == "supporting"]
    oracle_useful = [node_id for node_id in oracle_flipped if labels.get(node_id, {}).get("label") == "supporting"]
    saved_base_ids = [
        str(row["node_id"]) for row in saved_prediction.get("retrieved_episodic", [])
        if row.get("source") == "base" and row.get("node_id") is not None
    ][:top_k]
    return {
        "question_id": question_id, "question_type": item.get("question_type", ""),
        "keyword_status": keyword_status, "query_keywords": sorted(query_keywords),
        "paired_current_time": paired_time, "label_source": label_source,
        "saved_base_top_k_ids": saved_base_ids,
        "current_base_top_k_ids": baseline_trace["base_top_k_ids"],
        "saved_current_base_exact_match": saved_base_ids == baseline_trace["base_top_k_ids"],
        "baseline_flipped_ids": baseline_flipped, "oracle_flipped_ids": oracle_flipped,
        "selection_changed": baseline_flipped != oracle_flipped,
        "added_ids": added, "removed_ids": removed,
        "supporting_added_ids": supporting_added,
        "irrelevant_removed_ids": irrelevant_removed,
        "baseline_supporting_flipped_ids": baseline_useful,
        "oracle_supporting_flipped_ids": oracle_useful,
        "rescued_useful_question": not baseline_useful and bool(oracle_useful),
        "selected_labels": {
            node_id: labels.get(node_id, {"label": "uncertain", "rationale": "missing", "confidence": 0.0})
            for node_id in dict.fromkeys(baseline_flipped + oracle_flipped)
        },
        "calibrated_directed_edge_count": len(edge_multipliers),
        "oracle_edge_labels": {
            target["candidate_id"]: labels.get(
                target["candidate_id"],
                {"label": "uncertain", "rationale": "missing", "confidence": 0.0},
            )
            for target in current_record["candidate_targets"]
        },
        "calibrated_label_counts": dict(Counter(
            labels.get(target["candidate_id"], {}).get("label", "uncertain")
            for target in current_record["candidate_targets"]
            for _ in target["connected_base_ids"]
        )),
        "baseline_trace": baseline_trace, "oracle_trace": oracle_trace,
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in records if not row.get("skipped_corrupted", False)]
    changed = sum(row["selection_changed"] for row in valid)
    return {
        "total_questions": len(records),
        "valid_retrieval_questions": len(valid),
        "corrupted_questions_skipped": len(records) - len(valid),
        "flipped_selection_changed_questions": changed,
        "flipped_selection_changed_rate": changed / len(valid) if valid else 0.0,
        "supporting_candidates_added": sum(len(row["supporting_added_ids"]) for row in valid),
        "irrelevant_candidates_removed": sum(len(row["irrelevant_removed_ids"]) for row in valid),
        "rescued_useful_questions": sum(row["rescued_useful_question"] for row in valid),
        "saved_current_base_exact_match_questions": sum(row["saved_current_base_exact_match"] for row in valid),
        "quality_labels_reused_questions": sum(row["label_source"] == "reused_quality_report" for row in valid),
        "quality_labels_recomputed_questions": sum(row["label_source"] == "recomputed_for_current_base" for row in valid),
        "keyword_fallback_questions": sum(row["keyword_status"] != "ok" for row in valid),
    }


def fingerprint(args: argparse.Namespace) -> str:
    values = {
        "protocol_version": "exact_oracle_v05_2",
        "top_k": args.top_k, "max_flipped": args.max_flipped,
        "activation_alpha": args.activation_alpha,
        "spreading_threshold": args.spreading_threshold,
        "keyword_weight": args.keyword_weight,
        "supporting": 2.0, "redundant": 0.2, "irrelevant": 0.0,
        "evidence_model": args.evidence_model,
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact HeLa retrieval plus gold-derived oracle edge calibration")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence-model", default=os.environ.get("HEBBIAN_EVIDENCE_MODEL") or model_for("generation"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-flipped", type=int, default=3)
    parser.add_argument("--activation-alpha", type=float, default=0.1)
    parser.add_argument("--spreading-threshold", type=float, default=0.4)
    parser.add_argument("--keyword-weight", type=float, default=0.5)
    args = parser.parse_args()

    os.environ["HEBBIAN_MAX_FLIPPED"] = str(args.max_flipped)
    os.environ["HEBBIAN_ACTIVATION_ALPHA"] = str(args.activation_alpha)
    os.environ["HEBBIAN_SPREADING_THRESHOLD"] = str(args.spreading_threshold)
    os.environ["HEBBIAN_KEYWORD_WEIGHT"] = str(args.keyword_weight)
    os.environ["HEBBIAN_USE_INHIBITION"] = "false"
    output_dir = Path(args.output_dir)
    item_dir = output_dir / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    predictions = {
        str(row["question_id"]): row for row in (
            json.loads(line) for line in Path(args.base_predictions).read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }
    quality_cache = load_quality_cache(Path(args.quality_report))
    run_fingerprint = fingerprint(args)

    records: List[Dict[str, Any]] = []
    pending = []
    for item_index, item in enumerate(dataset):
        question_id = str(item["question_id"])
        item_path = item_dir / f"{question_id}.json"
        if item_path.exists():
            try:
                cached = json.loads(item_path.read_text(encoding="utf-8"))
                if cached.get("config_fingerprint") == run_fingerprint and cached.get("status") == "ok":
                    records.append(cached["result"])
                    continue
            except (OSError, ValueError, TypeError):
                pass
        if item_index in CORRUPTED_INDICES:
            skipped = {
                "question_id": question_id, "question_type": item.get("question_type", ""),
                "skipped_corrupted": True, "selection_changed": False,
                "supporting_added_ids": [], "irrelevant_removed_ids": [],
                "rescued_useful_question": False,
            }
            atomic_write_json(item_path, {"status": "ok", "config_fingerprint": run_fingerprint, "result": skipped})
            records.append(skipped)
            continue
        pending.append((item, item_path))
    print(f"Exact oracle scan: {len(records)} resumed, {len(pending)} pending", flush=True)

    def run_one(task: tuple[Dict[str, Any], Path]) -> Dict[str, Any]:
        item, item_path = task
        question_id = str(item["question_id"])
        result = process_item(
            item, Path(args.mem_dir) / f"{question_id}_hebbian.json",
            predictions[question_id], quality_cache.get(question_id),
            args.evidence_model, args.top_k, args.batch_size, args.retries,
        )
        atomic_write_json(item_path, {"status": "ok", "config_fingerprint": run_fingerprint, "result": result})
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one, task) for task in pending]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            records.append(future.result())
            print(f"Progress: {len(records)}/{len(dataset)}", flush=True)
    records.sort(key=lambda row: row["question_id"])
    report = {
        "oracle_only": True, "deployable_method": False, "exact_current_retrieval_code": True,
        "answer_generation_executed": False, "benchmark_judge_executed": False,
        "graph_mutation_executed": False, "paired_query_keywords_and_embedding": True,
        "edge_multipliers": MULTIPLIERS,
        "parameters": {"top_k": args.top_k, "max_flipped": args.max_flipped, "activation_alpha": args.activation_alpha, "spreading_threshold": args.spreading_threshold, "keyword_weight": args.keyword_weight},
        "inputs": {"data_path": args.data_path, "mem_dir": args.mem_dir, "base_predictions": args.base_predictions, "quality_report": args.quality_report},
        "evidence_model": args.evidence_model, "config_fingerprint": run_fingerprint,
        "summary": summarize(records), "records": records,
    }
    atomic_write_json(output_dir / "retrieval_scan.json", report)
    print("metric\tvalue")
    for key, value in report["summary"].items():
        print(f"{key}\t{value}")
    print(f"report\t{output_dir / 'retrieval_scan.json'}")


if __name__ == "__main__":
    main()
