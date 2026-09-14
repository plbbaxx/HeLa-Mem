"""V4 Pairwise Redundancy Utility Scorer for HeLa-Mem.

Stage 1 reuses V2 residual-gap artifacts.  V4 determines redundancy by a
rank-ordered, early-stopping candidate/Base pair comparison.  Only candidates
that match no Base memory receive residual-gap matching.  Oracle labels are
never included in prompts and are evaluation-only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .runtime import atomic_write_json, model_for
from .scan_exact_oracle_edge_utility import CORRUPTED_INDICES
from .structured_residual_utility_v2 import (
    GAP_PROMPT_TEMPLATE,
    build_contexts,
    cache_key,
    chat,
    gap_metrics,
    strip_code_fence,
)
from .evidence_constrained_utility_v3 import load_v2_gap_cache, irrelevant_to_supporting_rate
from .utility_scorer_v1 import CLASSES, classification_metrics, load_v05, replay_item, replay_metrics


PAIR_PROMPT_TEMPLATE = """QUESTION:
{question}

BASE MEMORY:
{base_memory}

CANDIDATE MEMORY:
{candidate}

Determine whether the Base Memory and Candidate Memory express the same answer-relevant fact for the Question.

Return SAME_FACT only when they convey essentially the same useful information needed for answering the Question, even if phrased differently.

Return DIFFERENT_FACT when:
- they mention the same entity/topic but provide different information;
- one contains additional distinct evidence;
- they are only loosely related;
- they are irrelevant to each other.

Do not use external knowledge.
Do not use the gold answer.

Return only:
SAME_FACT
or
DIFFERENT_FACT"""

GAP_MATCH_PROMPT_TEMPLATE = """You are evaluating whether a candidate memory fills an information gap.

QUESTION:
{question}

INFORMATION ALREADY COVERED BY BASE MEMORIES:
{base_coverage}

ANSWERABLE FROM BASE:
{answerable_from_base}

MISSING INFORMATION SLOTS:
{missing_slots}

CANDIDATE MEMORY:
{candidate}

The Candidate has already been compared against every Base Memory and was not found to repeat the same answer-relevant fact.

Return FILLS_GAP only when the Candidate explicitly supplies new evidence that fills at least one listed missing slot needed to answer the Question.
Return NO_MATCH otherwise.

Rules:
- Judge evidence stated in the Candidate, not what might plausibly be true.
- Do not use external knowledge or a gold answer.
- If answerable_from_base is true, FILLS_GAP is not allowed.
- Shared topics or entities alone are not enough.

Return only FILLS_GAP or NO_MATCH."""


def build_pair_prompt(question: str, base_memory: str, candidate: str) -> str:
    return PAIR_PROMPT_TEMPLATE.format(question=question, base_memory=(base_memory or "").strip(), candidate=(candidate or "").strip())


def build_gap_match_prompt(question: str, residual: Dict[str, Any], candidate: str) -> str:
    return GAP_MATCH_PROMPT_TEMPLATE.format(
        question=question,
        base_coverage=json.dumps(residual["base_coverage"], ensure_ascii=False),
        answerable_from_base=str(residual["answerable_from_base"]).lower(),
        missing_slots=json.dumps(residual["missing_slots"], ensure_ascii=False),
        candidate=(candidate or "").strip(),
    )


def _parse_label(text: str, allowed: Tuple[str, ...], stage: str) -> str:
    value = strip_code_fence(text).upper()
    first = value.splitlines()[0].strip() if value else ""
    if first not in allowed:
        raise ValueError(f"invalid {stage} output: {value[:160]!r}")
    return first


def parse_pair_response(text: str) -> str:
    return _parse_label(text, ("SAME_FACT", "DIFFERENT_FACT"), "pairwise")


def parse_gap_match_response(text: str) -> str:
    return _parse_label(text, ("FILLS_GAP", "NO_MATCH"), "gap matching")


def _load_cached(path: Path, model: str, prompt_sha: str) -> Dict[str, Any] | None:
    try:
        row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError, TypeError):
        return None
    if row and row.get("status") == "ok" and row.get("model") == model and row.get("prompt_sha256") == prompt_sha:
        return row
    return None


def run_candidate_pipeline(
    contexts: Dict[str, Dict[str, Any]], gaps: Dict[str, Dict[str, Any]], output_dir: Path,
    model: str, workers: int, retries: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    pair_dir, gap_dir = output_dir / "pairwise_items", output_dir / "gap_match_items"
    pair_dir.mkdir(parents=True, exist_ok=True)
    gap_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(context, candidate) for context in contexts.values() for candidate in context["candidates"]]
    stats: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    print(f"V4 candidates: {len(tasks)}", flush=True)

    def run_one(task):
        context, candidate = task
        calls = cache_hits = comparisons = 0
        matched_base = None
        pair_trace = []
        for rank, base in enumerate(context["base_memories"], start=1):
            prompt = build_pair_prompt(context["question"], base["content"], candidate["content"])
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            path = pair_dir / f"{cache_key(context['question_id'], candidate['candidate_id'], base['memory_id'])}.json"
            cached = _load_cached(path, model, prompt_sha)
            if cached:
                decision, raw = cached["pairwise_label"], cached.get("raw_output", "")
                cache_hits += 1
            else:
                decision, raw = chat(prompt, model, 16, parse_pair_response, retries)
                calls += 1
                cached = {
                    "status": "ok", "stage": "pairwise_redundancy", "model": model,
                    "question_id": context["question_id"], "candidate_id": candidate["candidate_id"],
                    "base_memory_id": base["memory_id"], "base_rank": rank, "prompt_sha256": prompt_sha,
                    "pairwise_label": decision, "raw_output": raw, "gold_answer_in_prompt": False,
                }
                atomic_write_json(path, cached)
            comparisons += 1
            pair_trace.append({"base_memory_id": base["memory_id"], "base_rank": rank, "label": decision})
            if decision == "SAME_FACT":
                matched_base = base
                break
        if matched_base is not None:
            prediction, stage2_label, gap_calls, gap_cache_hits, gap_raw = "REDUNDANT", "SAME_FACT", 0, 0, None
        else:
            residual = gaps[context["question_id"]]["residual"]
            prompt = build_gap_match_prompt(context["question"], residual, candidate["content"])
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            path = gap_dir / f"{cache_key(context['question_id'], candidate['candidate_id'])}.json"
            cached = _load_cached(path, model, prompt_sha)
            if cached:
                stage2_label, gap_raw, gap_calls, gap_cache_hits = cached["gap_label"], cached.get("raw_output", ""), 0, 1
            else:
                stage2_label, gap_raw = chat(prompt, model, 16, parse_gap_match_response, retries)
                gap_calls, gap_cache_hits = 1, 0
                atomic_write_json(path, {
                    "status": "ok", "stage": "residual_gap_matching", "model": model,
                    "question_id": context["question_id"], "candidate_id": candidate["candidate_id"],
                    "prompt_sha256": prompt_sha, "gap_label": stage2_label, "raw_output": gap_raw,
                    "gold_answer_in_prompt": False,
                })
            prediction = "SUPPORTING" if stage2_label == "FILLS_GAP" else "IRRELEVANT"
        return {
            "status": "ok", "stage": "pairwise_then_gap", "model": model,
            "question_id": context["question_id"], "question_type": context["question_type"],
            "candidate_id": candidate["candidate_id"], "candidate_text": candidate["content"],
            "oracle_label": candidate["oracle_label"], "predicted_label": prediction,
            "stage2_label": stage2_label, "matched_base_memory_id": matched_base["memory_id"] if matched_base else None,
            "matched_base_memory_text": matched_base["content"] if matched_base else None,
            "pair_trace": pair_trace, "pairwise_calls": calls, "pairwise_cache_hits": cache_hits,
            "pairwise_comparisons": comparisons, "early_stopped": matched_base is not None and comparisons < len(context["base_memories"]),
            "gap_calls": gap_calls, "gap_cache_hits": gap_cache_hits, "gap_raw_output": gap_raw,
            "gold_answer_in_prompt": False,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            for key in ("pairwise_calls", "pairwise_cache_hits", "pairwise_comparisons", "gap_calls", "gap_cache_hits"):
                stats[key] += row[key]
            stats["early_stop_count"] += int(row["early_stopped"])
            print(f"V4 complete: {len(rows)}/{len(tasks)}", flush=True)
    rows.sort(key=lambda row: (row["question_id"], row["candidate_id"]))
    stats["total_candidates"] = len(rows)
    stats["avg_pairwise_calls_per_candidate"] = stats["pairwise_calls"] / len(rows) if rows else 0.0
    stats["avg_pairwise_comparisons_per_candidate"] = stats["pairwise_comparisons"] / len(rows) if rows else 0.0
    return rows, dict(stats)


def redundancy_diagnostics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    redundant = [row for row in rows if row["predicted_label"] == "REDUNDANT"]
    requested = {
        "oracle_redundant_to_predicted_supporting": sum(row["oracle_label"] == "REDUNDANT" and row["predicted_label"] == "SUPPORTING" for row in rows),
        "oracle_redundant_to_predicted_irrelevant": sum(row["oracle_label"] == "REDUNDANT" and row["predicted_label"] == "IRRELEVANT" for row in rows),
        "oracle_supporting_to_predicted_redundant": sum(row["oracle_label"] == "SUPPORTING" and row["predicted_label"] == "REDUNDANT" for row in rows),
        "oracle_irrelevant_to_predicted_redundant": sum(row["oracle_label"] == "IRRELEVANT" and row["predicted_label"] == "REDUNDANT" for row in rows),
    }
    records = [{
        "question_id": row["question_id"], "candidate_id": row["candidate_id"],
        "matched_base_memory_id": row["matched_base_memory_id"], "candidate_text": row["candidate_text"],
        "base_memory_text": row["matched_base_memory_text"], "oracle_label": row["oracle_label"],
    } for row in redundant]
    return {"summary": {"predicted_redundant_count": len(records), **requested}, "records": records}


def replay_eligibility(classification: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "minimum_redundant_recall": 0.20, "minimum_supporting_precision": 0.30,
        "maximum_irrelevant_to_supporting_rate": 0.35,
        "redundant_recall_pass": classification["per_class"]["REDUNDANT"]["recall"] >= 0.20,
        "supporting_precision_pass": classification["per_class"]["SUPPORTING"]["precision"] >= 0.30,
        "irrelevant_to_supporting_rate": irrelevant_to_supporting_rate(classification),
        "irrelevant_to_supporting_pass": irrelevant_to_supporting_rate(classification) <= 0.35,
    }
    result["run_retrieval_replay"] = all(value for key, value in result.items() if key.endswith("_pass"))
    return result


def comparison(report: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, float]:
    old = report["classification_metrics"]
    return {
        "supporting_precision_delta": current["per_class"]["SUPPORTING"]["precision"] - old["per_class"]["SUPPORTING"]["precision"],
        "supporting_recall_delta": current["per_class"]["SUPPORTING"]["recall"] - old["per_class"]["SUPPORTING"]["recall"],
        "redundant_precision_delta": current["per_class"]["REDUNDANT"]["precision"] - old["per_class"]["REDUNDANT"]["precision"],
        "redundant_recall_delta": current["per_class"]["REDUNDANT"]["recall"] - old["per_class"]["REDUNDANT"]["recall"],
        "macro_f1_delta": current["macro_f1"] - old["macro_f1"],
        "irrelevant_to_supporting_rate_delta": irrelevant_to_supporting_rate(current) - irrelevant_to_supporting_rate(old),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 Pairwise Redundancy Utility Scorer")
    parser.add_argument("--data-path", required=True); parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--oracle-v05", required=True); parser.add_argument("--v2-report", required=True)
    parser.add_argument("--v3-report", required=True); parser.add_argument("--v2-gap-dir", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--model", default=os.environ.get("HEBBIAN_UTILITY_MODEL") or model_for("generation"))
    parser.add_argument("--workers", type=int, default=8); parser.add_argument("--retries", type=int, default=3); parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_BASE_URL"):
        raise SystemExit("OPENAI_BASE_URL is required")
    dataset = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    v05_report, v05_records = load_v05(Path(args.oracle_v05))
    v2_report = json.loads(Path(args.v2_report).read_text(encoding="utf-8"))
    v3_report = json.loads(Path(args.v3_report).read_text(encoding="utf-8"))
    parameters = v05_report.get("parameters", {})
    if int(parameters.get("top_k", args.top_k)) != args.top_k:
        raise SystemExit("--top-k must match V0.5")
    for env_name, parameter, default in (("HEBBIAN_MAX_FLIPPED", "max_flipped", 3), ("HEBBIAN_ACTIVATION_ALPHA", "activation_alpha", 0.1), ("HEBBIAN_SPREADING_THRESHOLD", "spreading_threshold", 0.4), ("HEBBIAN_KEYWORD_WEIGHT", "keyword_weight", 0.5)):
        os.environ[env_name] = str(parameters.get(parameter, default))
    os.environ.update({"HEBBIAN_USE_KEYWORD_MATCH": "true", "HEBBIAN_USE_TIME_DECAY": "true", "HEBBIAN_USE_INHIBITION": "false"})
    contexts = build_contexts(dataset, v05_records, Path(args.mem_dir))
    gaps = load_v2_gap_cache(contexts, Path(args.v2_gap_dir), args.model)
    output_dir = Path(args.output_dir)
    rows, costs = run_candidate_pipeline(contexts, gaps, output_dir, args.model, args.workers, args.retries)
    classification = classification_metrics(rows)
    eligibility = replay_eligibility(classification)
    replay, replay_rows = None, []
    if eligibility["run_retrieval_replay"]:
        by_question: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in rows:
            by_question.setdefault(row["question_id"], {})[row["candidate_id"]] = row
        for index, item in enumerate(dataset):
            if index not in CORRUPTED_INDICES:
                qid = str(item["question_id"])
                replay_rows.append(replay_item(item, v05_records[qid], by_question.get(qid, {}), Path(args.mem_dir) / f"{qid}_hebbian.json", args.top_k))
        replay = replay_metrics(replay_rows)
    diagnostics = redundancy_diagnostics(rows)
    report = {
        "method": "V4 Pairwise Redundancy Utility Scorer", "gold_answer_used_by_stage1": False, "gold_answer_used_by_stage2": False,
        "oracle_labels_used_only_for_evaluation": True, "answer_generation_executed": False, "benchmark_judge_executed": False, "graph_mutation_executed": False,
        "model": args.model, "temperature": 0.0, "stage1_reused_from_v2": True, "gap_prompt_template": GAP_PROMPT_TEMPLATE,
        "pairwise_prompt_template": PAIR_PROMPT_TEMPLATE, "gap_match_prompt_template": GAP_MATCH_PROMPT_TEMPLATE,
        "retrieval_parameters": parameters, "classification_metrics": classification, "pairwise_cost": costs,
        "comparison_with_v2": comparison(v2_report, classification), "comparison_with_v3": comparison(v3_report, classification),
        "retrieval_replay_eligibility": eligibility, "retrieval_replay_executed": eligibility["run_retrieval_replay"], "retrieval_replay_metrics": replay,
        "redundancy_diagnostics": diagnostics, "match_outputs": rows, "retrieval_records": replay_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "pairwise_redundancy_utility_v4_report.json", report)
    atomic_write_json(output_dir / "match_predictions.json", rows)
    atomic_write_json(output_dir / "redundancy_diagnostics.json", diagnostics)
    if replay is not None:
        atomic_write_json(output_dir / "retrieval_replay.json", {"metrics": replay, "records": replay_rows})
    print("classification_metric\tvalue")
    for key in ("overall_accuracy", "macro_f1", "false_supporting_count", "supporting_to_redundant", "supporting_to_irrelevant", "irrelevant_to_supporting"):
        print(f"{key}\t{classification[key]}")
    for label in CLASSES:
        for metric in ("precision", "recall", "f1"):
            print(f"{label}_{metric}\t{classification['per_class'][label][metric]}")
    print("confusion_matrix_rows_oracle_columns_prediction")
    print("oracle\\predicted\t" + "\t".join(CLASSES))
    for label in CLASSES:
        print(label + "\t" + "\t".join(str(classification["confusion_matrix"]["counts"][label][pred]) for pred in CLASSES))
    print("pairwise_cost\t" + json.dumps(costs, ensure_ascii=False))
    print("redundancy_diagnostic\t" + json.dumps(diagnostics["summary"], ensure_ascii=False))
    print("retrieval_replay_eligibility\t" + json.dumps(eligibility, ensure_ascii=False))
    if replay is not None:
        for key, value in replay.items(): print(f"retrieval_{key}\t{value}")
    print(f"report\t{output_dir / 'pairwise_redundancy_utility_v4_report.json'}")


if __name__ == "__main__":
    main()
