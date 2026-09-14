"""Validate deterministic baselines and compare V3 paired experiments."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return {row["question_id"]: row for line in handle if (row := json.loads(line))}


def ids(row, key="retrieved_episodic"):
    return [entry.get("node_id") for entry in row.get(key, [])]


def trace_ids(row, key):
    return (row.get("redundancy_inhibition_trace") or {}).get(key, [])


def compare(reference, variant):
    if reference.keys() != variant.keys():
        raise ValueError("question IDs do not match")
    rows = []
    per_type = defaultdict(lambda: {"total": 0, "reference_correct": 0, "variant_correct": 0, "w2c": 0, "c2w": 0, "retrieval_changed": 0, "base_changed": 0, "flipped_changed": 0, "prediction_changed": 0, "judge_changed": 0})
    for question_id in sorted(reference):
        before, after = reference[question_id], variant[question_id]
        before_correct, after_correct = int(before["correct"]), int(after["correct"])
        retrieval_changed = ids(before) != ids(after)
        base_changed = trace_ids(before, "base_top_k_ids") != trace_ids(after, "base_top_k_ids")
        flipped_changed = trace_ids(before, "flipped_memory_ids_after") != trace_ids(after, "flipped_memory_ids_after")
        prediction_changed = before.get("prediction") != after.get("prediction")
        judge_changed = before.get("judge_result") != after.get("judge_result")
        transition = "W2C" if before_correct == 0 and after_correct == 1 else "C2W" if before_correct == 1 and after_correct == 0 else "unchanged"
        bucket = per_type[before["question_type"]]
        bucket["total"] += 1
        bucket["reference_correct"] += before_correct
        bucket["variant_correct"] += after_correct
        bucket["w2c"] += int(transition == "W2C")
        bucket["c2w"] += int(transition == "C2W")
        bucket["retrieval_changed"] += int(retrieval_changed)
        bucket["base_changed"] += int(base_changed)
        bucket["flipped_changed"] += int(flipped_changed)
        bucket["prediction_changed"] += int(prediction_changed)
        bucket["judge_changed"] += int(judge_changed)
        rows.append({
            "question_id": question_id,
            "question_type": before["question_type"],
            "transition": transition,
            "retrieval_changed": retrieval_changed,
            "base_changed": base_changed,
            "flipped_changed": flipped_changed,
            "prediction_changed": prediction_changed,
            "judge_changed": judge_changed,
            "reference_prediction": before.get("prediction"),
            "variant_prediction": after.get("prediction"),
            "variant_trace": after.get("redundancy_inhibition_trace"),
        })
    for bucket in per_type.values():
        bucket["reference_accuracy"] = 100 * bucket["reference_correct"] / bucket["total"]
        bucket["variant_accuracy"] = 100 * bucket["variant_correct"] / bucket["total"]
        bucket["net"] = bucket["w2c"] - bucket["c2w"]
    total = len(rows)
    ref_correct = sum(int(row["correct"]) for row in reference.values())
    var_correct = sum(int(row["correct"]) for row in variant.values())
    return {
        "total": total,
        "reference_correct": ref_correct,
        "reference_accuracy": 100 * ref_correct / total,
        "variant_correct": var_correct,
        "variant_accuracy": 100 * var_correct / total,
        "w2c": sum(row["transition"] == "W2C" for row in rows),
        "c2w": sum(row["transition"] == "C2W" for row in rows),
        "net": sum(row["transition"] == "W2C" for row in rows) - sum(row["transition"] == "C2W" for row in rows),
        "retrieval_changed": sum(row["retrieval_changed"] for row in rows),
        "base_changed": sum(row["base_changed"] for row in rows),
        "flipped_changed": sum(row["flipped_changed"] for row in rows),
        "prediction_changed": sum(row["prediction_changed"] for row in rows),
        "judge_changed": sum(row["judge_changed"] for row in rows),
        "per_type": dict(sorted(per_type.items())),
        "transitions": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-a", required=True)
    parser.add_argument("--baseline-b", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline_a, baseline_b = load(args.baseline_a), load(args.baseline_b)
    deterministic = compare(baseline_a, baseline_b)
    report = {
        "deterministic_baseline_check": deterministic,
        "baseline_vs_preselection_control": compare(baseline_a, load(args.control)),
        "preselection_control_vs_v3": compare(load(args.control), load(args.v3)),
        "baseline_vs_v3": compare(baseline_a, load(args.v3)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        name: {key: value for key, value in section.items() if key not in ("per_type", "transitions")}
        for name, section in report.items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
