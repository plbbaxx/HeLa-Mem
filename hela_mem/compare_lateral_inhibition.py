"""Compare paired LongMemEval baseline and lateral-inhibition predictions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as handle:
        return {row["question_id"]: row for line in handle if (row := json.loads(line))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--inhibition", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_predictions(args.baseline)
    inhibition = load_predictions(args.inhibition)
    if baseline.keys() != inhibition.keys():
        missing_baseline = sorted(inhibition.keys() - baseline.keys())
        missing_inhibition = sorted(baseline.keys() - inhibition.keys())
        raise SystemExit(
            f"question mismatch: missing_baseline={missing_baseline}, "
            f"missing_inhibition={missing_inhibition}"
        )

    transitions = []
    per_type = defaultdict(lambda: {"total": 0, "baseline_correct": 0, "inhibition_correct": 0, "w2c": 0, "c2w": 0})
    for question_id in sorted(baseline):
        before = baseline[question_id]
        after = inhibition[question_id]
        before_correct = int(before["correct"])
        after_correct = int(after["correct"])
        question_type = before["question_type"]
        bucket = per_type[question_type]
        bucket["total"] += 1
        bucket["baseline_correct"] += before_correct
        bucket["inhibition_correct"] += after_correct
        bucket["w2c"] += int(before_correct == 0 and after_correct == 1)
        bucket["c2w"] += int(before_correct == 1 and after_correct == 0)
        transitions.append({
            "question_id": question_id,
            "question_type": question_type,
            "baseline_correct": before_correct,
            "inhibition_correct": after_correct,
            "transition": "W2C" if before_correct == 0 and after_correct == 1 else "C2W" if before_correct == 1 and after_correct == 0 else "unchanged",
            "baseline_prediction": before.get("prediction"),
            "inhibition_prediction": after.get("prediction"),
            "episodic_inhibition_trace": after.get("episodic_inhibition_trace"),
            "semantic_inhibition_trace": after.get("semantic_inhibition_trace"),
        })

    total = len(transitions)
    baseline_correct = sum(row["baseline_correct"] for row in transitions)
    inhibition_correct = sum(row["inhibition_correct"] for row in transitions)
    w2c = sum(row["transition"] == "W2C" for row in transitions)
    c2w = sum(row["transition"] == "C2W" for row in transitions)
    for values in per_type.values():
        values["baseline_accuracy"] = 100 * values["baseline_correct"] / values["total"]
        values["inhibition_accuracy"] = 100 * values["inhibition_correct"] / values["total"]
        values["net"] = values["w2c"] - values["c2w"]
    report = {
        "total": total,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": 100 * baseline_correct / total,
        "inhibition_correct": inhibition_correct,
        "inhibition_accuracy": 100 * inhibition_correct / total,
        "w2c": w2c,
        "c2w": c2w,
        "net": w2c - c2w,
        "per_type": dict(sorted(per_type.items())),
        "transitions": transitions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total", "baseline_correct", "baseline_accuracy", "inhibition_correct", "inhibition_accuracy", "w2c", "c2w", "net")}, indent=2))


if __name__ == "__main__":
    main()
