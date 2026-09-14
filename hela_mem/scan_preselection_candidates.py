"""Retrieval-only trigger audit for pre-selection competitive inhibition."""

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .hebbian_memory import HebbianMemoryGraph, apply_redundancy_aware_inhibition
from .runtime import atomic_write_json


def scan_item(item, args):
    question_id = item["question_id"]
    graph = HebbianMemoryGraph(str(Path(args.mem_dir) / f"{question_id}_hebbian.json"))
    graph.retrieve(item["question"], top_k=args.top_k)
    trace = graph.last_retrieval_trace
    candidate_ids = trace["candidate_memory_ids_before"]
    control_flipped = candidate_ids[:args.max_flipped]
    gamma_results = {}
    if candidate_ids:
        id_to_position = {node_id: pos for pos, node_id in enumerate(trace["node_ids"])}
        positions = [id_to_position[node_id] for node_id in candidate_ids]
        scores = np.asarray(trace["final_scores"])[positions]
        embeddings = np.asarray([graph.nodes[node_id]["embedding"] for node_id in candidate_ids])
        for gamma in args.gammas:
            new_scores, _ = apply_redundancy_aware_inhibition(scores, embeddings, gamma)
            order = np.argsort(new_scores)[::-1]
            selected = [candidate_ids[pos] for pos in order[:args.max_flipped]]
            gamma_results[str(gamma)] = {
                "selected": selected,
                "changed": selected != control_flipped,
            }
    return {
        "question_id": question_id,
        "question_type": item["question_type"],
        "candidate_count": len(candidate_ids),
        "candidate_memory_ids": candidate_ids,
        "control_flipped_memory_ids": control_flipped,
        "gamma_results": gamma_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-flipped", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    args = parser.parse_args()

    os.environ["HEBBIAN_USE_PRESELECTION_POOL"] = "true"
    os.environ["HEBBIAN_USE_REDUNDANCY_INHIBITION"] = "false"
    with open(args.data_path, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    size_distribution = Counter()
    replacement_counts = Counter()
    records = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        scanned = executor.map(lambda item: scan_item(item, args), dataset)
        for index, record in enumerate(scanned):
            size_distribution[record["candidate_count"]] += 1
            for gamma, result in record["gamma_results"].items():
                replacement_counts[gamma] += int(result["changed"])
            records.append(record)
            if (index + 1) % 50 == 0:
                print(f"Scanned {index + 1}/{len(dataset)}")

    total = len(records)
    report = {
        "total": total,
        "top_k": args.top_k,
        "max_flipped": args.max_flipped,
        "candidate_size_distribution": dict(sorted(size_distribution.items())),
        "questions_with_two_or_more_candidates": sum(row["candidate_count"] >= 2 for row in records),
        "questions_with_two_or_more_candidates_rate": sum(row["candidate_count"] >= 2 for row in records) / max(total, 1),
        "replacement_counts": dict(replacement_counts),
        "replacement_rates": {key: value / max(total, 1) for key, value in replacement_counts.items()},
        "records": records,
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: report[key] for key in report if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
