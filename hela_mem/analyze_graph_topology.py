"""Offline topology audit for the existing HeLa-Mem associative graphs.

No model, retrieval, or graph update is imported or executed.  Edge provenance
is not present in legacy graph artifacts, so this reports current topology only
and does not attribute individual edges to temporal initialization or later
co-retrieval reinforcement.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .analyze_associative_candidate_expansion import base_top_k_ids, load_jsonl


CATEGORIES = ("base_base", "base_nonbase", "nonbase_nonbase")


def undirected_edges(edges: Dict[str, Dict[str, Any]]) -> Iterable[Tuple[str, str, float]]:
    """Deduplicate bidirectional graph storage using the strongest stored weight."""
    pairs: Dict[Tuple[str, str], float] = {}
    for source, neighbors in edges.items():
        for target, raw_weight in neighbors.items():
            source_id, target_id = str(source), str(target)
            if source_id == target_id:
                continue
            pair = tuple(sorted((source_id, target_id)))
            pairs[pair] = max(pairs.get(pair, float("-inf")), float(raw_weight))
    for (left, right), weight in pairs.items():
        yield left, right, weight


def category(left: str, right: str, base_set: set[str]) -> str:
    left_base, right_base = left in base_set, right in base_set
    if left_base and right_base:
        return "base_base"
    if left_base or right_base:
        return "base_nonbase"
    return "nonbase_nonbase"


def empty_category_values() -> Dict[str, float]:
    return {name: 0.0 for name in CATEGORIES}


def analyze_item(item: Dict[str, Any], prediction: Dict[str, Any], mem_dir: Path, top_k: int) -> Dict[str, Any]:
    question_id = item["question_id"]
    base_ids = base_top_k_ids(prediction, top_k)
    graph_path = mem_dir / f"{question_id}_hebbian.json"
    if not graph_path.exists():
        return {
            "question_id": question_id,
            "question_type": item["question_type"],
            "base_top_k_ids": base_ids,
            "node_count": 0,
            "missing_graph": True,
            "edge_counts": empty_category_values(),
            "edge_weight_mass": empty_category_values(),
            "total_undirected_edges": 0,
            "total_edge_weight_mass": 0.0,
        }
    with graph_path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    base_set = set(base_ids)
    edge_counts = empty_category_values()
    edge_weight_mass = empty_category_values()
    for left, right, weight in undirected_edges(graph.get("edges", {})):
        edge_type = category(left, right, base_set)
        edge_counts[edge_type] += 1
        edge_weight_mass[edge_type] += weight
    total_edges = int(sum(edge_counts.values()))
    total_weight = sum(edge_weight_mass.values())
    return {
        "question_id": question_id,
        "question_type": item["question_type"],
        "base_top_k_ids": base_ids,
        "node_count": len(graph.get("nodes", {})),
        "missing_graph": False,
        "edge_counts": {key: int(value) for key, value in edge_counts.items()},
        "edge_weight_mass": edge_weight_mass,
        "total_undirected_edges": total_edges,
        "total_edge_weight_mass": total_weight,
        "has_base_nonbase_edge": edge_counts["base_nonbase"] > 0,
    }


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [record for record in records if not record["missing_graph"]]
    counts = Counter()
    weight_mass = Counter()
    for record in valid:
        counts.update(record["edge_counts"])
        weight_mass.update(record["edge_weight_mass"])
    total_edges = sum(counts.values())
    total_weight = sum(weight_mass.values())
    by_category = {}
    for edge_type in CATEGORIES:
        by_category[edge_type] = {
            "edge_count": int(counts[edge_type]),
            "edge_count_share": counts[edge_type] / total_edges if total_edges else 0.0,
            "edge_weight_mass": float(weight_mass[edge_type]),
            "edge_weight_share": weight_mass[edge_type] / total_weight if total_weight else 0.0,
        }
    total_per_graph = [record["total_undirected_edges"] for record in valid]
    return {
        "total_questions": len(records),
        "valid_graphs": len(valid),
        "missing_graphs": len(records) - len(valid),
        "questions_with_base_nonbase_edge": sum(record["has_base_nonbase_edge"] for record in valid),
        "questions_with_base_nonbase_edge_rate": sum(record["has_base_nonbase_edge"] for record in valid) / len(valid) if valid else 0.0,
        "total_undirected_edges": int(total_edges),
        "total_edge_weight_mass": float(total_weight),
        "edges_by_category": by_category,
        "mean_undirected_edges_per_graph": statistics.mean(total_per_graph) if total_per_graph else 0.0,
        "median_undirected_edges_per_graph": statistics.median(total_per_graph) if total_per_graph else 0.0,
        "max_undirected_edges_per_graph": max(total_per_graph) if total_per_graph else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Base/non-Base graph topology audit")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    with Path(args.data_path).open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    predictions = load_jsonl(Path(args.base_predictions))
    if set(item["question_id"] for item in dataset) != set(predictions):
        raise SystemExit("dataset and baseline predictions must contain identical question IDs")
    records = [analyze_item(item, predictions[item["question_id"]], Path(args.mem_dir), args.top_k) for item in dataset]
    report = {
        "offline_only": True,
        "edge_provenance_available": False,
        "edge_provenance_note": "Legacy graph JSON stores no temporal-versus-reinforcement edge provenance.",
        "data_path": args.data_path,
        "mem_dir": args.mem_dir,
        "base_predictions": args.base_predictions,
        "top_k": args.top_k,
        "summary": aggregate(records),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("category\tedge_count\tcount_share\tweight_mass\tweight_share")
    for edge_type in CATEGORIES:
        values = summary["edges_by_category"][edge_type]
        print(
            f"{edge_type}\t{values['edge_count']}\t{values['edge_count_share']:.4f}\t"
            f"{values['edge_weight_mass']:.4f}\t{values['edge_weight_share']:.4f}"
        )
    print(f"questions_with_base_nonbase_edge\t{summary['questions_with_base_nonbase_edge']}/{summary['valid_graphs']}\t{summary['questions_with_base_nonbase_edge_rate']:.4f}")


if __name__ == "__main__":
    main()
