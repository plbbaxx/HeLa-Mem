"""Offline coverage audit for seed-neighbor associative candidate expansion.

This module intentionally does not import the LLM client or retrieval classes.
It reuses Base Top-K IDs saved by the completed baseline and reads only encoded
Hebbian graph JSON files, so no answer generation, judging, or graph mutation
can occur.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    values = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                values[row["question_id"]] = row
    return values


def base_top_k_ids(prediction: Dict[str, Any], top_k: int) -> List[str]:
    """Recover the already-used Base evidence without rerunning retrieval."""
    episodic = prediction.get("retrieved_episodic", [])
    return [
        str(entry["node_id"])
        for entry in episodic
        if entry.get("source") == "base" and entry.get("node_id") is not None
    ][:top_k]


def ordered_neighbors(edges: Dict[str, Dict[str, Any]], seed_id: str) -> Iterable[tuple[str, float]]:
    neighbors = edges.get(seed_id, {})
    return sorted(
        ((str(node_id), float(weight)) for node_id, weight in neighbors.items()),
        key=lambda item: (-item[1], item[0]),
    )


def expand_top_neighbors(
    base_ids: List[str], edges: Dict[str, Dict[str, Any]], seed_k: int, neighbor_k: int
) -> tuple[List[str], List[str]]:
    seeds = base_ids[:seed_k]
    base_set = set(base_ids)
    candidates = []
    seen = set()
    for seed_id in seeds:
        for neighbor_id, _ in list(ordered_neighbors(edges, seed_id))[:neighbor_k]:
            if neighbor_id not in base_set and neighbor_id not in seen:
                seen.add(neighbor_id)
                candidates.append(neighbor_id)
    return seeds, candidates


def expand_threshold_neighbors(
    base_ids: List[str], edges: Dict[str, Dict[str, Any]], seed_k: int, threshold: float
) -> tuple[List[str], List[str]]:
    seeds = base_ids[:seed_k]
    base_set = set(base_ids)
    candidates = []
    seen = set()
    for seed_id in seeds:
        for neighbor_id, weight in ordered_neighbors(edges, seed_id):
            if weight < threshold:
                break
            if neighbor_id not in base_set and neighbor_id not in seen:
                seen.add(neighbor_id)
                candidates.append(neighbor_id)
    return seeds, candidates


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    sizes = [row["candidate_count"] for row in records]
    total = len(sizes)
    bins = {
        "zero": sum(size == 0 for size in sizes),
        "one": sum(size == 1 for size in sizes),
        "two_or_more": sum(size >= 2 for size in sizes),
    }
    return {
        "total_questions": total,
        "candidate_pool_zero": {"count": bins["zero"], "rate": bins["zero"] / total},
        "candidate_pool_one": {"count": bins["one"], "rate": bins["one"] / total},
        "candidate_pool_two_or_more": {"count": bins["two_or_more"], "rate": bins["two_or_more"] / total},
        "average_candidate_pool_size": sum(sizes) / total,
        "median_candidate_pool_size": statistics.median(sizes),
        "max_candidate_pool_size": max(sizes),
        "size_distribution": dict(sorted(Counter(sizes).items())),
    }


def analyze_config(
    dataset: List[Dict[str, Any]],
    predictions: Dict[str, Dict[str, Any]],
    mem_dir: Path,
    top_k: int,
    method: str,
    seed_k: int,
    value: float,
) -> Dict[str, Any]:
    records = []
    missing_graphs = []
    for item in dataset:
        question_id = item["question_id"]
        base_ids = base_top_k_ids(predictions.get(question_id, {}), top_k)
        graph_path = mem_dir / f"{question_id}_hebbian.json"
        if not graph_path.exists():
            missing_graphs.append(question_id)
            edges = {}
        else:
            with graph_path.open("r", encoding="utf-8") as handle:
                edges = json.load(handle).get("edges", {})
        if method == "top_neighbors":
            seeds, candidates = expand_top_neighbors(base_ids, edges, seed_k, int(value))
        else:
            seeds, candidates = expand_threshold_neighbors(base_ids, edges, seed_k, value)
        records.append({
            "question_id": question_id,
            "base_top_k_ids": base_ids,
            "seed_ids": seeds,
            "expanded_candidate_ids": candidates,
            "candidate_count": len(candidates),
        })
    return {
        "method": method,
        "seed_k": seed_k,
        "neighbor_k": int(value) if method == "top_neighbors" else None,
        "edge_threshold": value if method == "edge_threshold" else None,
        "offline_only": True,
        "missing_graphs": missing_graphs,
        "summary": summarize(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline associative candidate-pool coverage analysis")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--seed-ks", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--neighbor-ks", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--edge-thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    args = parser.parse_args()

    with Path(args.data_path).open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    predictions = load_jsonl(Path(args.base_predictions))
    if set(item["question_id"] for item in dataset) != set(predictions):
        raise SystemExit("dataset and baseline predictions must contain identical question IDs")

    groups = []
    for seed_k in args.seed_ks:
        for neighbor_k in args.neighbor_ks:
            groups.append(analyze_config(dataset, predictions, Path(args.mem_dir), args.top_k, "top_neighbors", seed_k, neighbor_k))
    for threshold in args.edge_thresholds:
        groups.append(analyze_config(dataset, predictions, Path(args.mem_dir), args.top_k, "edge_threshold", 5, threshold))

    report = {
        "offline_only": True,
        "data_path": args.data_path,
        "mem_dir": args.mem_dir,
        "base_predictions": args.base_predictions,
        "top_k": args.top_k,
        "groups": groups,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("method\tseed_k\tvalue\tzero\tone\ttwo_or_more\tmean\tmedian\tmax")
    for group in groups:
        summary = group["summary"]
        value = group["neighbor_k"] if group["method"] == "top_neighbors" else group["edge_threshold"]
        print(
            f"{group['method']}\t{group['seed_k']}\t{value}\t"
            f"{summary['candidate_pool_zero']['count']}\t{summary['candidate_pool_one']['count']}\t"
            f"{summary['candidate_pool_two_or_more']['count']}\t"
            f"{summary['average_candidate_pool_size']:.3f}\t{summary['median_candidate_pool_size']}\t"
            f"{summary['max_candidate_pool_size']}"
        )


if __name__ == "__main__":
    main()
