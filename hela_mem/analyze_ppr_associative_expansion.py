"""Read-only PPR candidate-expansion replay over frozen LongMemEval graphs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .analyze_associative_candidate_expansion import (
    base_top_k_ids,
    expand_top_neighbors,
    load_jsonl,
)
from .ppr_expansion import ppr_associative_candidates


LABELS = ("supporting", "redundant", "irrelevant", "uncertain")


def base_scores(prediction: dict[str, Any], base_ids: list[str]) -> dict[str, float]:
    by_id = {
        str(row["node_id"]): float(row.get("base_score") or 0.0)
        for row in prediction.get("retrieved_episodic", [])
        if row.get("node_id") is not None
    }
    return {node_id: max(0.0, by_id.get(node_id, 0.0)) for node_id in base_ids}


def load_quality_labels(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[str, dict[str, str]] = {}
    for record in report.get("records", []):
        qid = str(record["question_id"])
        labels[qid] = {
            str(row["candidate_id"]): str(row.get("label", "uncertain")).lower()
            for row in record.get("candidate_annotations", [])
            if str(row.get("label", "uncertain")).lower() in LABELS
        }
    return labels


def pool_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    sizes = [len(row[field]) for row in records]
    total = len(sizes)
    return {
        "questions": total,
        "questions_with_two_or_more_candidates": sum(size >= 2 for size in sizes),
        "questions_with_two_or_more_candidates_rate": sum(size >= 2 for size in sizes) / total if total else 0.0,
        "mean_candidate_count": statistics.mean(sizes) if sizes else 0.0,
        "median_candidate_count": statistics.median(sizes) if sizes else 0.0,
        "max_candidate_count": max(sizes, default=0),
        "total_nonbase_candidate_occurrences": sum(sizes),
        "candidate_size_distribution": dict(sorted(Counter(sizes).items())),
    }


def quality_summary(records: list[dict[str, Any]], quality: dict[str, dict[str, str]]) -> dict[str, Any]:
    selected = Counter()
    one_hop_selected = Counter()
    added = Counter()
    supporting_total = 0
    supporting_selected = 0
    labeled_selected = 0
    unlabeled_selected = 0
    for row in records:
        labels = quality.get(row["question_id"], {})
        supporting_total += sum(label == "supporting" for label in labels.values())
        ppr_ids = {candidate["node_id"] for candidate in row["ppr_candidates"]}
        one_hop_ids = set(row["one_hop_candidate_ids"])
        for node_id in one_hop_ids:
            label = labels.get(node_id)
            if label is not None:
                one_hop_selected[label] += 1
        supporting_selected += sum(labels.get(node_id) == "supporting" for node_id in ppr_ids)
        for node_id in ppr_ids:
            label = labels.get(node_id)
            if label is None:
                unlabeled_selected += 1
                continue
            labeled_selected += 1
            selected[label] += 1
            if node_id not in one_hop_ids:
                added[label] += 1
    supporting = selected["supporting"]
    irrelevant = selected["irrelevant"]
    one_hop_supporting = one_hop_selected["supporting"]
    one_hop_irrelevant = one_hop_selected["irrelevant"]
    total_ppr = labeled_selected + unlabeled_selected
    return {
        "quality_report_available": True,
        "label_scope_caveat": (
            "Existing labels cover previously observed Base-to-non-Base targets only; "
            "new multi-hop PPR candidates may be unlabeled."
        ),
        "selected_label_counts": dict(selected),
        "one_hop_selected_label_counts": dict(one_hop_selected),
        "new_vs_one_hop_label_counts": dict(added),
        "supporting_added": added["supporting"],
        "redundant_added": added["redundant"],
        "irrelevant_added": added["irrelevant"],
        "supporting_recall": supporting_selected / supporting_total if supporting_total else None,
        "one_hop_supporting_recall": one_hop_supporting / supporting_total if supporting_total else None,
        "supporting_labeled_target_total": supporting_total,
        "labeled_ppr_candidate_occurrences": labeled_selected,
        "unlabeled_ppr_candidate_occurrences": unlabeled_selected,
        "labeled_ppr_candidate_rate": labeled_selected / total_ppr if total_ppr else None,
        "irrelevant_to_supporting_ratio": irrelevant / supporting if supporting else None,
        "one_hop_irrelevant_to_supporting_ratio": (
            one_hop_irrelevant / one_hop_supporting if one_hop_supporting else None
        ),
    }


def analyze(
    dataset: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    mem_dir: Path,
    top_k: int,
    damping: float,
    ppr_top_n: int,
    one_hop_seed_k: int,
    one_hop_neighbor_k: int,
    quality: dict[str, dict[str, str]],
) -> dict[str, Any]:
    records = []
    missing_graphs = []
    invalid_personalization = []
    for item in dataset:
        qid = str(item["question_id"])
        prediction = predictions[qid]
        base_ids = base_top_k_ids(prediction, top_k)
        scores = base_scores(prediction, base_ids)
        graph_path = mem_dir / f"{qid}_hebbian.json"
        if not graph_path.exists():
            missing_graphs.append(qid)
            records.append({
                "question_id": qid,
                "question_type": item.get("question_type"),
                "missing_graph": True,
                "base_top_k_ids": base_ids,
                "base_seeds": [],
                "one_hop_candidate_ids": [],
                "ppr_candidates": [],
                "ppr_candidate_ids": [],
                "candidate_set_changed_vs_one_hop": False,
                "new_nonbase_ids_vs_one_hop": [],
                "ppr_iterations": 0,
                "ppr_converged": False,
                "ppr_residual": None,
            })
            continue
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", {})
        _, one_hop = expand_top_neighbors(base_ids, edges, one_hop_seed_k, one_hop_neighbor_k)
        if not base_ids or sum(scores.values()) <= 0:
            invalid_personalization.append(qid)
            ppr = {"scores": {}, "iterations": 0, "converged": False, "residual": None,
                   "base_seeds": [], "candidates": []}
        else:
            ppr = ppr_associative_candidates(
                node_ids=list(nodes), edges=edges, base_ids=base_ids, base_scores=scores,
                top_n=ppr_top_n, damping=damping, max_iter=50, tol=1e-6,
            )
        one_hop_set = set(one_hop)
        candidates = [
            {
                **candidate,
                "candidate_text": str(nodes.get(candidate["node_id"], {}).get("content", "")),
                "newly_added_to_base": True,
                "new_vs_one_hop": candidate["node_id"] not in one_hop_set,
                "utility_label_if_available": quality.get(qid, {}).get(candidate["node_id"]),
            }
            for candidate in ppr["candidates"]
        ]
        records.append({
            "question_id": qid,
            "question_type": item.get("question_type"),
            "missing_graph": False,
            "base_top_k_ids": base_ids,
            "base_seeds": ppr["base_seeds"],
            "one_hop_candidate_ids": one_hop,
            "ppr_candidates": candidates,
            "ppr_candidate_ids": [row["node_id"] for row in candidates],
            "candidate_set_changed_vs_one_hop": set(one_hop) != {row["node_id"] for row in candidates},
            "new_nonbase_ids_vs_one_hop": [row["node_id"] for row in candidates if row["new_vs_one_hop"]],
            "ppr_iterations": ppr["iterations"],
            "ppr_converged": ppr["converged"],
            "ppr_residual": ppr["residual"],
        })

    one_hop_summary = pool_summary(records, "one_hop_candidate_ids")
    ppr_summary = pool_summary(records, "ppr_candidate_ids")
    ppr_summary.update({
        "new_nonbase_occurrences_vs_one_hop": sum(len(row["new_nonbase_ids_vs_one_hop"]) for row in records),
        "questions_whose_candidate_set_changes": sum(row["candidate_set_changed_vs_one_hop"] for row in records),
        "questions_whose_candidate_set_changes_rate": (
            sum(row["candidate_set_changed_vs_one_hop"] for row in records) / len(records) if records else 0.0
        ),
    })
    report = {
        "protocol": "ppr_associative_expansion_diagnostic_v1",
        "offline_only": True,
        "qa_generation_calls": 0,
        "judge_calls": 0,
        "reencoding_performed": False,
        "pcst_enabled": False,
        "lateral_inhibition_modified": False,
        "formula": {
            "personalization": "p_i=max(S_base(i),0)/sum_j max(S_base(j),0), restricted to unchanged Base Top-K seeds",
            "transition": "P_ij=w_ij/sum_k w_ik for positive stored Hebbian edges",
            "iteration": "r_(t+1)=(1-d)p+d P^T r_t",
            "dangling_nodes": "redistribute dangling mass according to p",
        },
        "parameters": {
            "top_k": top_k, "ppr_damping": damping, "ppr_top_n": ppr_top_n,
            "ppr_max_iter": 50, "ppr_tol": 1e-6,
            "one_hop_seed_k": one_hop_seed_k, "one_hop_neighbor_k": one_hop_neighbor_k,
        },
        "missing_graphs": missing_graphs,
        "invalid_personalization_questions": invalid_personalization,
        "comparison": {"one_hop": one_hop_summary, "ppr": ppr_summary},
        "quality": quality_summary(records, quality) if quality else {"quality_report_available": False},
        "records": records,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline PPR associative candidate expansion replay")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--quality-report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--ppr-damping", type=float, default=0.5)
    parser.add_argument("--ppr-top-n", type=int, default=10)
    parser.add_argument("--one-hop-seed-k", type=int, default=8)
    parser.add_argument("--one-hop-neighbor-k", type=int, default=2)
    args = parser.parse_args()

    dataset = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    predictions = load_jsonl(Path(args.base_predictions))
    expected = {str(item["question_id"]) for item in dataset}
    if expected != set(predictions):
        raise SystemExit("dataset and baseline predictions must contain identical question IDs")
    quality_path = Path(args.quality_report) if args.quality_report else None
    report = analyze(
        dataset, predictions, Path(args.mem_dir), args.top_k, args.ppr_damping, args.ppr_top_n,
        args.one_hop_seed_k, args.one_hop_neighbor_k, load_quality_labels(quality_path),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("method\tquestions>=2\trate>=2\tmean\tmedian\tmax\ttotal_nonbase")
    for method in ("one_hop", "ppr"):
        row = report["comparison"][method]
        print(
            f"{method}\t{row['questions_with_two_or_more_candidates']}\t"
            f"{row['questions_with_two_or_more_candidates_rate']:.4f}\t"
            f"{row['mean_candidate_count']:.3f}\t{row['median_candidate_count']}\t"
            f"{row['max_candidate_count']}\t{row['total_nonbase_candidate_occurrences']}"
        )
    ppr = report["comparison"]["ppr"]
    print(f"questions_changed\t{ppr['questions_whose_candidate_set_changes']}")
    print(f"new_nonbase_vs_one_hop\t{ppr['new_nonbase_occurrences_vs_one_hop']}")
    if report["quality"]["quality_report_available"]:
        for key in ("supporting_added", "redundant_added", "irrelevant_added", "supporting_recall", "irrelevant_to_supporting_ratio"):
            print(f"{key}\t{report['quality'][key]}")
    print(f"report\t{output}")


if __name__ == "__main__":
    main()
