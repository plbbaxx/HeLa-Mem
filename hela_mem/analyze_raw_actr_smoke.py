"""Sanity audit for matched Raw/Cue-IDF/ACT-R 10-question smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runtime import atomic_write_json


def load_results(directory: Path) -> dict[str, dict[str, Any]]:
    values = {}
    for path in directory.glob("result_*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        values[str(row["question_id"])] = row
    return values


def selected_ids(diagnostic: dict[str, Any]) -> list[str]:
    return [
        row["node_id"] for row in sorted(diagnostic["candidates"], key=lambda value: value["final_rank"])
        if row["selected_actr"]
    ]


def analyze(raw, cue_idf, actr, hybrid) -> dict[str, Any]:
    sets = [set(values) for values in (raw, cue_idf, actr, hybrid)]
    if not sets or any(values != sets[0] for values in sets[1:]):
        raise ValueError("smoke runs do not contain identical question IDs")
    records = []
    pool_mismatches = []
    graph_changes = []
    fan_scope_failures = []
    strength_failures = []
    ranking_changes = []
    for question_id in sorted(sets[0]):
        rows = {"raw": raw[question_id], "cue_idf": cue_idf[question_id], "actr": actr[question_id], "hybrid": hybrid[question_id]}
        diagnostics = {name: row["actr_diagnostic"] for name, row in rows.items()}
        pools = {name: tuple(value["coarse_candidate_ids"]) for name, value in diagnostics.items()}
        if len(set(pools.values())) != 1:
            pool_mismatches.append(question_id)
        if any(not row.get("memory_graph_unchanged", False) for row in rows.values() if not row.get("is_corrupted")):
            graph_changes.append(question_id)
        for name in ("cue_idf", "actr", "hybrid"):
            diagnostic = diagnostics[name]
            if diagnostic["fan_computed_over_memory_count"] != diagnostic["memory_bank_size"]:
                fan_scope_failures.append(f"{question_id}:{name}")
            cues = diagnostic["cues"]
            for left in cues:
                for right in cues:
                    if left["fan"] < right["fan"] and not left["fan_strength"] > right["fan_strength"]:
                        strength_failures.append(f"{question_id}:{name}")
        raw_top = selected_ids(diagnostics["raw"])
        actr_top = selected_ids(diagnostics["actr"])
        changed = raw_top != actr_top
        if changed:
            ranking_changes.append(question_id)
        records.append({
            "question_id": question_id,
            "question": diagnostics["actr"]["question"],
            "cues": diagnostics["actr"]["cues"],
            "raw_top_k": raw_top,
            "actr_top_k": actr_top,
            "ranking_changed": changed,
            "ranking_delta": [
                {
                    "node_id": candidate["node_id"],
                    "base_rank": candidate["base_rank"],
                    "actr_rank": candidate["actr_rank"],
                    "rank_delta": candidate["rank_delta"],
                }
                for candidate in diagnostics["actr"]["candidates"]
                if candidate["rank_delta"] != 0
            ],
        })
    return {
        "question_count": len(records),
        "candidate_pool_ids_identical": not pool_mismatches,
        "candidate_pool_mismatch_questions": pool_mismatches,
        "memory_graphs_unchanged": not graph_changes,
        "memory_graph_changed_questions": graph_changes,
        "fan_uses_complete_memory_bank": not fan_scope_failures,
        "fan_scope_failures": fan_scope_failures,
        "low_fan_has_stronger_activation": not strength_failures,
        "fan_strength_failures": sorted(set(strength_failures)),
        "ranking_changed_questions": ranking_changes,
        "first_ranking_changed_case": next((row for row in records if row["ranking_changed"]), None),
        "cue_review_records": records[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--cue-idf-dir", required=True)
    parser.add_argument("--actr-dir", required=True)
    parser.add_argument("--hybrid-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze(*[
        load_results(Path(value))
        for value in (args.raw_dir, args.cue_idf_dir, args.actr_dir, args.hybrid_dir)
    ])
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"cue_review_records"}}, ensure_ascii=False, indent=2))
    print(f"report\t{args.output}")


if __name__ == "__main__":
    main()
