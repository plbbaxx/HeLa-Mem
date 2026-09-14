"""Read-only oracle edge-utility counterfactual for LongMemEval graphs.

The original baseline did not persist full pre-retrieval activation traces.
Consequently this scanner uses a paired, semantic-only proxy: the saved Base
Top-K IDs are frozen, query similarities are recomputed with the existing
embedding model, and both arms use exactly the same activations and graph.
Only Base-to-Non-Base edge multipliers differ.  No LLM, answer generation,
benchmark judge, graph save, or Hebbian update is executed.

Gold-aware labels make this an oracle bottleneck test, not a deployable method.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .analyze_associative_candidate_expansion import base_top_k_ids, load_jsonl


LABEL_MULTIPLIERS = {"supporting": None, "redundant": None, "irrelevant": 0.0, "uncertain": 1.0}


def load_quality_labels(path: Path) -> Dict[str, Dict[str, str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    labels: Dict[str, Dict[str, str]] = {}
    for record in report.get("records", []):
        question_id = str(record["question_id"])
        labels[question_id] = {
            str(row["candidate_id"]): str(row["label"]).lower()
            for row in record.get("candidate_annotations", [])
        }
    return labels


def normalized_matrix(nodes: Dict[str, Dict[str, Any]], node_ids: List[str]) -> np.ndarray:
    matrix = np.asarray([nodes[node_id]["embedding"] for node_id in node_ids], dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def semantic_activations(node_matrix: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float64)
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    return (node_matrix @ query + 1.0) / 2.0


def edge_multiplier(
    source_id: str,
    target_id: str,
    base_set: set[str],
    target_labels: Dict[str, str],
    support_gain: float,
    redundant_alpha: float,
) -> tuple[float, str | None]:
    """Return the oracle multiplier only for directed Base-to-non-Base spread."""
    if source_id not in base_set or target_id in base_set:
        return 1.0, None
    label = target_labels.get(target_id)
    if label is None:
        return 1.0, None
    multipliers = dict(LABEL_MULTIPLIERS)
    multipliers["supporting"] = 1.0 + support_gain
    multipliers["redundant"] = redundant_alpha
    return float(multipliers.get(label, 1.0)), label


def spread_scores(
    activations: np.ndarray,
    node_ids: List[str],
    edges: Dict[str, Dict[str, Any]],
    activation_alpha: float,
    threshold: float,
    base_set: set[str] | None = None,
    target_labels: Dict[str, str] | None = None,
    support_gain: float = 1.0,
    redundant_alpha: float = 0.2,
) -> np.ndarray:
    scores = activations.copy()
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    calibrated = base_set is not None and target_labels is not None
    for index, source_score in enumerate(activations):
        if source_score <= threshold:
            continue
        source_id = node_ids[index]
        for target_id, raw_weight in edges.get(source_id, {}).items():
            target_index = id_to_index.get(str(target_id))
            if target_index is None:
                continue
            multiplier = 1.0
            if calibrated:
                multiplier, _ = edge_multiplier(
                    source_id, str(target_id), base_set, target_labels,
                    support_gain, redundant_alpha,
                )
            scores[target_index] += source_score * float(raw_weight) * multiplier * activation_alpha
    return scores


def select_flipped(scores: np.ndarray, node_ids: List[str], base_ids: List[str], top_k: int, max_flipped: int) -> List[str]:
    base_set = set(base_ids)
    # Match HebbianMemoryGraph.retrieve, including NumPy's tie ordering.
    ranking = np.argsort(scores)[::-1]
    return [node_ids[index] for index in ranking[:top_k] if node_ids[index] not in base_set][:max_flipped]


def analyze_item(
    item: Dict[str, Any],
    prediction: Dict[str, Any],
    graph: Dict[str, Any],
    query_embedding: np.ndarray,
    labels: Dict[str, str],
    top_k: int,
    max_flipped: int,
    activation_alpha: float,
    threshold: float,
    support_gain: float,
    redundant_alpha: float,
) -> Dict[str, Any]:
    nodes = graph.get("nodes", {})
    node_ids = list(nodes)
    base_ids = base_top_k_ids(prediction, top_k)
    missing_base = [node_id for node_id in base_ids if node_id not in nodes]
    if missing_base:
        raise ValueError(f"{item['question_id']}: Base IDs absent from graph: {missing_base}")
    activations = semantic_activations(normalized_matrix(nodes, node_ids), query_embedding)
    original_scores = spread_scores(activations, node_ids, graph.get("edges", {}), activation_alpha, threshold)
    oracle_scores = spread_scores(
        activations, node_ids, graph.get("edges", {}), activation_alpha, threshold,
        set(base_ids), labels, support_gain, redundant_alpha,
    )
    original = select_flipped(original_scores, node_ids, base_ids, top_k, max_flipped)
    oracle = select_flipped(oracle_scores, node_ids, base_ids, top_k, max_flipped)
    added = [node_id for node_id in oracle if node_id not in original]
    removed = [node_id for node_id in original if node_id not in oracle]
    added_labels = {node_id: labels.get(node_id, "unlabeled") for node_id in added}
    removed_labels = {node_id: labels.get(node_id, "unlabeled") for node_id in removed}
    base_set = set(base_ids)
    calibrated_edges = []
    for source_id, neighbors in graph.get("edges", {}).items():
        for target_id, weight in neighbors.items():
            multiplier, label = edge_multiplier(str(source_id), str(target_id), base_set, labels, support_gain, redundant_alpha)
            if label is not None:
                calibrated_edges.append({"source_id": str(source_id), "target_id": str(target_id), "weight": float(weight), "label": label, "multiplier": multiplier})
    return {
        "question_id": item["question_id"], "question_type": item.get("question_type", ""),
        "base_top_k_ids": base_ids, "original_flipped_ids": original, "oracle_flipped_ids": oracle,
        "selection_changed": original != oracle, "added_ids": added, "removed_ids": removed,
        "added_labels": added_labels, "removed_labels": removed_labels,
        "supporting_added_ids": [node_id for node_id in added if labels.get(node_id) == "supporting"],
        "non_supporting_added_ids": [node_id for node_id in added if labels.get(node_id) != "supporting"],
        "irrelevant_removed_ids": [node_id for node_id in removed if labels.get(node_id) == "irrelevant"],
        "zero_to_supporting": not original and any(labels.get(node_id) == "supporting" for node_id in oracle),
        "calibrated_directed_edges": calibrated_edges,
    }


def summarize(records: List[Dict[str, Any]], min_changed: int, min_supporting_share: float) -> Dict[str, Any]:
    changed = sum(record["selection_changed"] for record in records)
    supporting_added = sum(len(record["supporting_added_ids"]) for record in records)
    non_supporting_added = sum(len(record["non_supporting_added_ids"]) for record in records)
    all_added = supporting_added + non_supporting_added
    supporting_share = supporting_added / all_added if all_added else 0.0
    label_additions = Counter(label for record in records for label in record["added_labels"].values())
    label_removals = Counter(label for record in records for label in record["removed_labels"].values())
    return {
        "total_questions": len(records),
        "selection_changed_questions": changed,
        "selection_changed_rate": changed / len(records) if records else 0.0,
        "supporting_added_occurrences": supporting_added,
        "non_supporting_added_occurrences": non_supporting_added,
        "supporting_share_among_additions": supporting_share,
        "irrelevant_removed_occurrences": sum(len(record["irrelevant_removed_ids"]) for record in records),
        "zero_to_supporting_questions": sum(record["zero_to_supporting"] for record in records),
        "addition_labels": dict(label_additions), "removal_labels": dict(label_removals),
        "success_contract": {
            "minimum_selection_changed_questions": min_changed,
            "minimum_supporting_share_among_additions": min_supporting_share,
            "selection_change_pass": changed >= min_changed,
            "supporting_majority_pass": all_added > 0 and supporting_share > min_supporting_share,
            "overall_pass": changed >= min_changed and all_added > 0 and supporting_share > min_supporting_share,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic-only oracle edge-utility retrieval scan")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-model", default=os.environ.get("HEBBIAN_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--device", default="cpu", help="Embedding device; CPU avoids competing with a running vLLM server")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-flipped", type=int, default=3)
    parser.add_argument("--activation-alpha", type=float, default=0.1)
    parser.add_argument("--spreading-threshold", type=float, default=0.4)
    parser.add_argument("--support-gain", type=float, default=1.0)
    parser.add_argument("--redundant-alpha", type=float, default=0.2)
    parser.add_argument("--minimum-changed", type=int, default=10)
    parser.add_argument("--minimum-supporting-share", type=float, default=0.5)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    dataset = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    predictions = load_jsonl(Path(args.base_predictions))
    quality_labels = load_quality_labels(Path(args.quality_report))
    if {item["question_id"] for item in dataset} != set(predictions):
        raise SystemExit("dataset and baseline predictions must contain identical question IDs")
    model = SentenceTransformer(args.embedding_model, device=args.device)
    query_embeddings = model.encode(
        [item["question"] for item in dataset], convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    records = []
    for item, query_embedding in zip(dataset, query_embeddings):
        question_id = item["question_id"]
        graph_path = Path(args.mem_dir) / f"{question_id}_hebbian.json"
        if not graph_path.exists():
            raise SystemExit(f"missing graph: {graph_path}")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        records.append(analyze_item(
            item, predictions[question_id], graph, query_embedding,
            quality_labels.get(question_id, {}), args.top_k, args.max_flipped,
            args.activation_alpha, args.spreading_threshold,
            args.support_gain, args.redundant_alpha,
        ))
    report = {
        "oracle_only": True, "deployable_method": False, "read_only": True,
        "answer_generation_executed": False, "benchmark_judge_executed": False,
        "graph_mutation_executed": False, "protocol": "paired_semantic_only_fixed_base_top_k",
        "protocol_caveat": "The original run did not persist full activation traces. This paired proxy omits keyword and time-decay terms and is not a baseline reproduction.",
        "data_path": args.data_path, "mem_dir": args.mem_dir,
        "base_predictions": args.base_predictions, "quality_report": args.quality_report,
        "embedding_model": args.embedding_model, "embedding_device": args.device,
        "parameters": {"top_k": args.top_k, "max_flipped": args.max_flipped, "activation_alpha": args.activation_alpha, "spreading_threshold": args.spreading_threshold, "support_gain": args.support_gain, "redundant_alpha": args.redundant_alpha, "irrelevant_multiplier": 0.0, "uncertain_multiplier": 1.0},
        "summary": summarize(records, args.minimum_changed, args.minimum_supporting_share),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = report["summary"]
    for key in ("selection_changed_questions", "supporting_added_occurrences", "non_supporting_added_occurrences", "supporting_share_among_additions", "irrelevant_removed_occurrences", "zero_to_supporting_questions"):
        print(f"{key}\t{summary[key]}")
    print(f"success_contract\t{json.dumps(summary['success_contract'], ensure_ascii=False)}")
    print(f"report\t{output}")


if __name__ == "__main__":
    main()
