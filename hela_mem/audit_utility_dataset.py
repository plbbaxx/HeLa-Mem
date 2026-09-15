"""Read-only audit for a completed Base-conditioned utility dataset."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .runtime import atomic_write_json


EPSILON_FILES = {
    0.0: "pairwise_preferences_eps0.jsonl",
    0.02: "pairwise_preferences_eps002.jsonl",
    0.05: "pairwise_preferences_eps005.jsonl",
    0.10: "pairwise_preferences_eps010.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return {
        "count": len(values), "min": min(values), "max": max(values),
        "mean": statistics.mean(values), "std": statistics.pstdev(values),
        "p25": quartiles[0], "median": statistics.median(values), "p75": quartiles[2],
    }


def transition_sanity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["transition"]].append(float(row["utility_score"]))
    result = {}
    for transition, values in sorted(grouped.items()):
        expected = 1 if transition == "W2C" else -1 if transition == "C2W" else 0
        result[transition] = {
            **describe(values),
            "positive_count": sum(value > 0 for value in values),
            "negative_count": sum(value < 0 for value in values),
            "near_zero_count": sum(abs(value) <= 0.02 for value in values),
            "expected_sign": expected,
            "expected_sign_rate": (
                sum(value > 0 for value in values) / len(values) if expected == 1 else
                sum(value < 0 for value in values) / len(values) if expected == -1 else None
            ),
        }
    return result


def load_token_count(cache_path: Path) -> int | None:
    try:
        row = json.loads(cache_path.read_text(encoding="utf-8"))
        return int(row["token_count"]) if row.get("status") == "ok" else None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def extreme_rows(root: Path, rows: list[dict[str, Any]], base_records: dict[str, dict[str, Any]], top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = [float(row["utility_score"]) for row in rows]
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    selected = {id(row): row for row in sorted(rows, key=lambda row: abs(float(row["utility_score"])), reverse=True)[:top_n]}
    selected.update({id(row): row for row in rows if float(row["utility_score"]) < lower or float(row["utility_score"]) > upper})
    output = []
    for row in sorted(selected.values(), key=lambda value: abs(float(value["utility_score"])), reverse=True):
        qid, cid = row["question_id"], row["candidate_memory_id"]
        output.append({
            "question_id": qid, "candidate_memory_id": cid,
            "question": row["question"], "gold_answer": base_records.get(qid, {}).get("gold_answer"),
            "candidate_text": row["candidate_text"], "baseline_answer": row["baseline_answer"],
            "candidate_answer": row["candidate_answer"], "transition": row["transition"],
            "utility_score": row["utility_score"], "delta_answer": row["delta_answer"],
            "baseline_gold_mean_logprob": row["baseline_gold_mean_logprob"],
            "candidate_gold_mean_logprob": row["candidate_gold_mean_logprob"],
            "baseline_gold_token_count": load_token_count(root / "cache" / "baseline_logprob" / f"{qid}.json"),
            "candidate_gold_token_count": load_token_count(root / "cache" / "candidate_logprob" / qid / f"{cid}.json"),
            "iqr_outlier": float(row["utility_score"]) < lower or float(row["utility_score"]) > upper,
        })
    return output, {"q1": q1, "q3": q3, "iqr": iqr, "lower_fence": lower, "upper_fence": upper,
                    "iqr_outlier_count": sum(item["iqr_outlier"] for item in output), "exported_count": len(output)}


def split_and_pair_audit(root: Path, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    split_ids = {name: set(ids) for name, ids in manifest["question_ids"].items()}
    candidate_questions = {row["question_id"] for row in rows}
    errors = []
    names = list(split_ids)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                errors.append(f"split overlap {left}/{right}: {sorted(overlap)}")
    union = set().union(*split_ids.values())
    if union != candidate_questions:
        errors.append(f"split population mismatch: missing={sorted(candidate_questions-union)}, extra={sorted(union-candidate_questions)}")
    candidate_ids = {(row["question_id"], row["candidate_memory_id"]) for row in rows}
    report: dict[str, Any] = {"split_counts": {name: len(ids) for name, ids in split_ids.items()}, "epsilons": {}}
    for epsilon, filename in EPSILON_FILES.items():
        pairs = read_jsonl(root / filename)
        per_split = {}
        for name, ids in split_ids.items():
            selected = [pair for pair in pairs if pair["question_id"] in ids]
            per_split[name] = {
                "pair_count": len(selected), "question_count": len({pair["question_id"] for pair in selected}),
                "type_counts": dict(Counter(kind for pair in selected for kind in pair.get("pair_types", []))),
            }
        for pair in pairs:
            qid = pair["question_id"]
            if (qid, pair["preferred_candidate_id"]) not in candidate_ids or (qid, pair["rejected_candidate_id"]) not in candidate_ids:
                errors.append(f"unknown or cross-question candidate in {filename}: {pair}")
        report["epsilons"][str(epsilon)] = {
            "pair_count": len(pairs), "question_count": len({pair["question_id"] for pair in pairs}),
            "per_split": per_split,
        }
    report["integrity_errors"] = errors
    return report, errors


def audit(root: Path, output_dir: Path, top_n: int) -> dict[str, Any]:
    rows = read_jsonl(root / "candidate_utility.jsonl")
    bases = {row["question_id"]: row for row in read_jsonl(root / "base_contexts.jsonl")}
    invalid = []
    for row in rows:
        utility = row.get("utility_score")
        delta = row.get("delta_gold_mean_logprob")
        expected = None if row.get("baseline_gold_mean_logprob") is None or row.get("candidate_gold_mean_logprob") is None else row["candidate_gold_mean_logprob"] - row["baseline_gold_mean_logprob"]
        if not row.get("continuous_signal_available") or row.get("utility_source") != "delta_gold_mean_logprob":
            invalid.append({"question_id": row["question_id"], "candidate_memory_id": row["candidate_memory_id"], "error": "continuous utility unavailable"})
        elif not all(math.isfinite(float(value)) for value in (utility, delta, expected)):
            invalid.append({"question_id": row["question_id"], "candidate_memory_id": row["candidate_memory_id"], "error": "non-finite utility"})
        elif abs(float(delta) - float(expected)) > 1e-8 or abs(float(utility) - float(delta)) > 1e-8:
            invalid.append({"question_id": row["question_id"], "candidate_memory_id": row["candidate_memory_id"], "error": "utility arithmetic mismatch"})
    transitions = transition_sanity(rows)
    extremes, outlier_summary = extreme_rows(root, rows, bases, top_n)
    pair_report, pair_errors = split_and_pair_audit(root, rows)
    readiness_checks = {
        "all_rows_have_valid_continuous_utility": not invalid and bool(rows),
        "w2c_median_is_positive": transitions.get("W2C", {}).get("median", 0) > 0,
        "c2w_median_is_negative": transitions.get("C2W", {}).get("median", 0) < 0,
        "split_and_pair_integrity": not pair_errors,
        "dev_has_eps010_pairs": pair_report["epsilons"]["0.1"]["per_split"]["dev"]["question_count"] > 0,
        "test_has_eps010_pairs": pair_report["epsilons"]["0.1"]["per_split"]["test"]["question_count"] > 0,
    }
    report = {
        "protocol": "base_conditioned_utility_dataset_audit_v1",
        "read_only": True, "dataset_root": str(root), "candidate_count": len(rows),
        "candidate_question_count": len({row["question_id"] for row in rows}),
        "utility": describe([float(row["utility_score"]) for row in rows]),
        "transition_sanity": transitions, "outliers": outlier_summary,
        "split_pair_audit": pair_report, "invalid_row_count": len(invalid),
        "readiness_checks": readiness_checks, "ready_for_ranker_training": all(readiness_checks.values()),
        "note": "Readiness is a data-integrity/signal-coverage gate, not evidence that a trained ranker will improve final QA accuracy.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "utility_audit.json", report)
    write_jsonl(output_dir / "extreme_utility_samples.jsonl", extremes)
    write_jsonl(output_dir / "invalid_utility_rows.jsonl", invalid)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a completed continuous utility dataset without model calls")
    parser.add_argument("--dataset-dir", default="artifacts/utility_dataset_v1_1")
    parser.add_argument("--output-dir")
    parser.add_argument("--top-extremes", type=int, default=20)
    args = parser.parse_args()
    root = Path(args.dataset_dir)
    output = Path(args.output_dir) if args.output_dir else root / "audit"
    report = audit(root, output, args.top_extremes)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report\t{output / 'utility_audit.json'}")


if __name__ == "__main__":
    main()
