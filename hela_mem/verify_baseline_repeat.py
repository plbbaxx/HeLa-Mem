"""Fail fast unless two deterministic LongMemEval baselines are identical."""

import argparse
import json
from pathlib import Path

from .runtime import atomic_write_json


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return {row["question_id"]: row for line in handle if (row := json.loads(line))}


def episodic_ids(row):
    return [entry.get("node_id") for entry in row.get("retrieved_episodic", [])]


def base_ids(row):
    return (row.get("redundancy_inhibition_trace") or {}).get("base_top_k_ids", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    first, second = load(args.first), load(args.second)
    if first.keys() != second.keys():
        raise SystemExit("determinism failure: baseline question IDs do not match")
    differences = []
    for question_id in sorted(first):
        a, b = first[question_id], second[question_id]
        changed_fields = []
        comparisons = {
            "base_top_k": base_ids(a) == base_ids(b),
            "retrieved_episodic": episodic_ids(a) == episodic_ids(b),
            "prediction": a.get("prediction") == b.get("prediction"),
            "judge_result": a.get("judge_result") == b.get("judge_result"),
            "correct": a.get("correct") == b.get("correct"),
        }
        changed_fields.extend(key for key, identical in comparisons.items() if not identical)
        if changed_fields:
            differences.append({"question_id": question_id, "changed_fields": changed_fields})
    report = {
        "total": len(first),
        "identical": not differences,
        "different_questions": len(differences),
        "differences": differences,
    }
    atomic_write_json(Path(args.output), report)
    print(json.dumps({key: value for key, value in report.items() if key != "differences"}, indent=2))
    if differences:
        raise SystemExit("determinism failure: baseline repeats differ; stopping before V3")


if __name__ == "__main__":
    main()
