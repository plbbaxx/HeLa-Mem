"""Compare a baseline with one or more LongMemEval inhibition runs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as handle:
        return {row["question_id"]: row for line in handle if (row := json.loads(line))}


def retrieved_ids(row):
    return [entry.get("id") or entry.get("node_id") for entry in row.get("retrieved_episodic", [])]


def compare(baseline, variant):
    if baseline.keys() != variant.keys():
        raise ValueError("baseline and variant question IDs do not match")
    transitions = []
    per_type = defaultdict(lambda: {"total": 0, "baseline_correct": 0, "variant_correct": 0, "w2c": 0, "c2w": 0, "retrieval_changed": 0})
    for question_id in sorted(baseline):
        before, after = baseline[question_id], variant[question_id]
        before_correct, after_correct = int(before["correct"]), int(after["correct"])
        changed = retrieved_ids(before) != retrieved_ids(after)
        question_type = before["question_type"]
        bucket = per_type[question_type]
        bucket["total"] += 1
        bucket["baseline_correct"] += before_correct
        bucket["variant_correct"] += after_correct
        bucket["w2c"] += int(before_correct == 0 and after_correct == 1)
        bucket["c2w"] += int(before_correct == 1 and after_correct == 0)
        bucket["retrieval_changed"] += int(changed)
        transitions.append({
            "question_id": question_id,
            "question_type": question_type,
            "baseline_correct": before_correct,
            "variant_correct": after_correct,
            "transition": "W2C" if before_correct == 0 and after_correct == 1 else "C2W" if before_correct == 1 and after_correct == 0 else "unchanged",
            "retrieval_changed": changed,
            "baseline_prediction": before.get("prediction"),
            "variant_prediction": after.get("prediction"),
            "redundancy_inhibition_trace": after.get("redundancy_inhibition_trace"),
        })
    for bucket in per_type.values():
        bucket["baseline_accuracy"] = 100 * bucket["baseline_correct"] / bucket["total"]
        bucket["variant_accuracy"] = 100 * bucket["variant_correct"] / bucket["total"]
        bucket["net"] = bucket["w2c"] - bucket["c2w"]
    total = len(transitions)
    baseline_correct = sum(row["baseline_correct"] for row in transitions)
    variant_correct = sum(row["variant_correct"] for row in transitions)
    w2c = sum(row["transition"] == "W2C" for row in transitions)
    c2w = sum(row["transition"] == "C2W" for row in transitions)
    return {
        "total": total,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": 100 * baseline_correct / total,
        "variant_correct": variant_correct,
        "variant_accuracy": 100 * variant_correct / total,
        "w2c": w2c,
        "c2w": c2w,
        "net": w2c - c2w,
        "retrieval_changed": sum(row["retrieval_changed"] for row in transitions),
        "per_type": dict(sorted(per_type.items())),
        "transitions": transitions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", action="append", required=True, help="NAME=PREDICTIONS_JSONL")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = load_predictions(args.baseline)
    report = {"baseline": args.baseline, "variants": {}}
    for spec in args.variant:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise SystemExit(f"invalid --variant {spec!r}; expected NAME=PATH")
        report["variants"][name] = compare(baseline, load_predictions(path))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        name: {key: value for key, value in result.items() if key not in ("per_type", "transitions")}
        for name, result in report["variants"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
