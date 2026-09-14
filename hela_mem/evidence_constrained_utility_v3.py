"""V3 Evidence-Constrained Utility Scorer for HeLa-Mem.

V3 reuses V2's cached Stage-1 residual gaps verbatim.  It changes only the
candidate decision: SUPPORTING needs an explicit gap fill; REDUNDANT needs an
explicit Base-memory identifier and duplicated fact.  Oracle labels remain
evaluation-only and no answer generation, judging, graph mutation, or encoding
is performed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .runtime import atomic_write_json, model_for, strip_reasoning
from .scan_exact_oracle_edge_utility import CORRUPTED_INDICES
from .structured_residual_utility_v2 import (
    GAP_PROMPT_TEMPLATE,
    build_contexts,
    build_gap_prompt,
    cache_key,
    chat,
    gap_metrics,
    strip_code_fence,
)
from .utility_scorer_v1 import (
    CLASSES,
    classification_metrics,
    load_v05,
    replay_item,
    replay_metrics,
)


V2_SUPPORTING_PRECISION = 0.38461538461538464
MAX_SUPPORTING_PRECISION_DROP = 0.02
MIN_REDUNDANT_RECALL = 0.20
MAX_IRRELEVANT_TO_SUPPORTING_RATE = 0.35

MATCH_PROMPT_TEMPLATE = """You are evaluating the incremental utility of a candidate memory.

QUESTION:
{question}

CURRENT BASE MEMORIES:
{base_memories_with_ids}

MISSING INFORMATION:
{missing_gaps}

CANDIDATE MEMORY:
{candidate}

Follow these steps strictly.

STEP 1: GAP FILLING
Does the Candidate provide new information that is NOT already contained in the Base Memories and directly fills at least one missing information gap needed to answer the Question?

If YES, return:
{{
  "label": "SUPPORTING",
  "filled_gap": "...",
  "evidence": "..."
}}

STEP 2: REDUNDANCY CHECK
If the Candidate does NOT fill a missing gap, check whether it repeats or paraphrases an answer-relevant fact that is already present in the Base Memories.

A Candidate may be labeled REDUNDANT only if you can identify:
1. the specific Base Memory ID containing the same useful fact;
2. the fact that is duplicated.

If YES, return:
{{
  "label": "REDUNDANT",
  "base_memory_id": "...",
  "duplicated_fact": "..."
}}

STEP 3: OTHERWISE
If the Candidate neither fills a missing gap nor duplicates an answer-relevant fact from the Base Memories, return:
{{
  "label": "IRRELEVANT"
}}

Important rules:
- Do NOT use the gold answer.
- Do NOT use external knowledge.
- Semantic or topical similarity alone is not enough for SUPPORTING.
- Semantic or topical similarity alone is not enough for REDUNDANT.
- SUPPORTING requires new missing information.
- REDUNDANT requires explicit evidence from an existing Base Memory.
- If no missing gap exists, the Candidate cannot be SUPPORTING.
- If no matching Base Memory can be identified, the Candidate cannot be REDUNDANT.

Return valid JSON only."""


def format_base_memories_with_ids(memories: List[Dict[str, str]]) -> str:
    return "\n\n".join(
        f"[Base Memory ID: {memory['memory_id']}]\n{(memory.get('content') or '').strip()}"
        for memory in memories
    )


def build_match_prompt(question: str, base_memories: List[Dict[str, str]], residual: Dict[str, Any], candidate: str) -> str:
    """Build Stage 2 input without an oracle/gold-answer argument."""
    return MATCH_PROMPT_TEMPLATE.format(
        question=question,
        base_memories_with_ids=format_base_memories_with_ids(base_memories),
        missing_gaps=json.dumps(residual["missing_slots"], ensure_ascii=False),
        candidate=(candidate or "").strip(),
    )


def parse_stage2_response(text: str) -> Dict[str, str]:
    value = json.loads(strip_code_fence(text))
    if not isinstance(value, dict) or not isinstance(value.get("label"), str):
        raise ValueError("Stage 2 output must be a JSON object with a string label")
    label = value["label"].strip().upper()
    expected_fields = {
        "SUPPORTING": {"label", "filled_gap", "evidence"},
        "REDUNDANT": {"label", "base_memory_id", "duplicated_fact"},
        "IRRELEVANT": {"label"},
    }
    if label not in expected_fields or set(value) != expected_fields[label]:
        raise ValueError(f"invalid Stage 2 JSON fields for {label!r}")
    parsed = {"label": label}
    for field in expected_fields[label] - {"label"}:
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{label} requires a nonempty {field}")
        parsed[field] = value[field].strip()
    return parsed


def load_v2_gap_cache(contexts: Dict[str, Dict[str, Any]], gap_dir: Path, model: str) -> Dict[str, Dict[str, Any]]:
    """Reuse, rather than regenerate, the V2 Stage-1 cache with full validation."""
    gaps: Dict[str, Dict[str, Any]] = {}
    missing = []
    for question_id, context in contexts.items():
        if not context["candidates"]:
            continue
        prompt = build_gap_prompt(context["question"], context["base_memories"])
        expected_sha = hashlib.sha256(prompt.encode()).hexdigest()
        path = gap_dir / f"{cache_key(question_id)}.json"
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            missing.append(question_id)
            continue
        if not (
            row.get("status") == "ok" and row.get("model") == model
            and row.get("question_id") == question_id and row.get("prompt_sha256") == expected_sha
            and isinstance(row.get("residual"), dict)
        ):
            missing.append(question_id)
            continue
        gaps[question_id] = row
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            f"V2 Stage-1 cache is missing or incompatible for {len(missing)} questions ({preview}). "
            "Run V2 first; V3 must not regenerate Stage 1."
        )
    print(f"Stage 1 reused from V2 cache: {len(gaps)} questions", flush=True)
    return gaps


def _base_memory_map(context: Dict[str, Any]) -> Dict[str, str]:
    return {str(memory["memory_id"]): memory.get("content", "") for memory in context["base_memories"]}


def run_match_stage(
    contexts: Dict[str, Dict[str, Any]], gaps: Dict[str, Dict[str, Any]], output_dir: Path,
    model: str, workers: int, retries: int,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: List[Dict[str, Any]] = []
    pending: List[Tuple[Dict[str, Any], Dict[str, Any], str, str, Path]] = []
    for question_id, context in contexts.items():
        if not context["candidates"]:
            continue
        residual = gaps[question_id]["residual"]
        for candidate in context["candidates"]:
            prompt = build_match_prompt(context["question"], context["base_memories"], residual, candidate["content"])
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            path = output_dir / f"{cache_key(question_id, candidate['candidate_id'])}.json"
            try:
                row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
                if row and row.get("status") == "ok" and row.get("model") == model and row.get("prompt_sha256") == prompt_sha:
                    completed.append(row)
                    continue
            except (OSError, ValueError, TypeError):
                pass
            pending.append((context, candidate, prompt, prompt_sha, path))
    total = len(completed) + len(pending)
    print(f"V3 Stage 2 Evidence-Constrained Matching: {len(completed)} resumed, {len(pending)} pending", flush=True)

    def run_one(task):
        context, candidate, prompt, prompt_sha, path = task
        decision, raw = chat(prompt, model, 300, parse_stage2_response, retries)
        base_memories = _base_memory_map(context)
        base_id = decision.get("base_memory_id")
        row = {
            "status": "ok", "stage": "evidence_constrained_candidate_matching", "model": model,
            "question_id": context["question_id"], "question_type": context["question_type"],
            "candidate_id": candidate["candidate_id"], "candidate": candidate["content"],
            "prompt_sha256": prompt_sha, "stage2_label": decision["label"],
            "predicted_label": decision["label"], "oracle_label": candidate["oracle_label"],
            "decision": decision, "base_memory_id_exists": base_id in base_memories if base_id else None,
            "referenced_base_memory": base_memories.get(base_id) if base_id else None,
            "raw_output": raw, "gold_answer_in_prompt": False,
        }
        atomic_write_json(path, row)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run_one, task) for task in pending]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
            print(f"V3 Stage 2 complete: {len(completed)}/{total}", flush=True)
    completed.sort(key=lambda row: (row["question_id"], row["candidate_id"]))
    return completed


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if len(token) > 2}


def redundant_diagnostics(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = []
    for row in rows:
        if row["predicted_label"] != "REDUNDANT":
            continue
        decision = row["decision"]
        fact_tokens = _tokens(decision["duplicated_fact"])
        base_tokens = _tokens(row.get("referenced_base_memory") or "")
        overlap = len(fact_tokens & base_tokens) / len(fact_tokens) if fact_tokens else 0.0
        records.append({
            "question_id": row["question_id"], "candidate_id": row["candidate_id"],
            "candidate": row["candidate"], "predicted_base_memory_id": decision["base_memory_id"],
            "referenced_base_memory": row.get("referenced_base_memory"),
            "duplicated_fact": decision["duplicated_fact"], "oracle_label": row["oracle_label"],
            "base_memory_id_exists": row["base_memory_id_exists"],
            "duplicated_fact_lexical_grounding_proxy": overlap >= 0.60,
            "duplicated_fact_token_overlap": overlap,
            "false_redundant": row["oracle_label"] != "REDUNDANT",
        })
    false_by_oracle = Counter(record["oracle_label"] for record in records if record["false_redundant"])
    return records, {
        "predicted_redundant_count": len(records),
        "base_memory_id_exists_count": sum(record["base_memory_id_exists"] is True for record in records),
        "duplicated_fact_lexically_grounded_count": sum(record["duplicated_fact_lexical_grounding_proxy"] for record in records),
        "false_redundant_count": sum(record["false_redundant"] for record in records),
        "false_redundant_by_oracle_label": dict(sorted(false_by_oracle.items())),
        "grounding_note": "Lexical grounding is an automatic proxy; candidate, cited Base memory, and fact are retained for manual audit.",
    }


def irrelevant_to_supporting_rate(classification: Dict[str, Any]) -> float:
    support = classification["per_class"]["IRRELEVANT"]["support"]
    return classification["irrelevant_to_supporting"] / support if support else 0.0


def replay_eligibility(classification: Dict[str, Any], v2_report: Dict[str, Any]) -> Dict[str, Any]:
    v2_precision = v2_report["classification_metrics"]["per_class"]["SUPPORTING"]["precision"]
    supporting_precision = classification["per_class"]["SUPPORTING"]["precision"]
    redundant_recall = classification["per_class"]["REDUNDANT"]["recall"]
    irrelevant_rate = irrelevant_to_supporting_rate(classification)
    result = {
        "minimum_redundant_recall": MIN_REDUNDANT_RECALL,
        "v2_supporting_precision": v2_precision,
        "maximum_supporting_precision_drop": MAX_SUPPORTING_PRECISION_DROP,
        "maximum_irrelevant_to_supporting_rate": MAX_IRRELEVANT_TO_SUPPORTING_RATE,
        "redundant_recall_pass": redundant_recall >= MIN_REDUNDANT_RECALL,
        "supporting_precision_preserved_pass": supporting_precision >= v2_precision - MAX_SUPPORTING_PRECISION_DROP,
        "irrelevant_to_supporting_rate": irrelevant_rate,
        "irrelevant_to_supporting_pass": irrelevant_rate <= MAX_IRRELEVANT_TO_SUPPORTING_RATE,
    }
    result["run_retrieval_replay"] = all(value for key, value in result.items() if key.endswith("_pass"))
    return result


def compare_with_v2(v2_report: Dict[str, Any], classification: Dict[str, Any]) -> Dict[str, Any]:
    old = v2_report["classification_metrics"]
    return {
        "supporting_precision_delta": classification["per_class"]["SUPPORTING"]["precision"] - old["per_class"]["SUPPORTING"]["precision"],
        "supporting_recall_delta": classification["per_class"]["SUPPORTING"]["recall"] - old["per_class"]["SUPPORTING"]["recall"],
        "redundant_precision_delta": classification["per_class"]["REDUNDANT"]["precision"] - old["per_class"]["REDUNDANT"]["precision"],
        "redundant_recall_delta": classification["per_class"]["REDUNDANT"]["recall"] - old["per_class"]["REDUNDANT"]["recall"],
        "macro_f1_delta": classification["macro_f1"] - old["macro_f1"],
        "irrelevant_to_supporting_rate_v2": irrelevant_to_supporting_rate(old),
        "irrelevant_to_supporting_rate_v3": irrelevant_to_supporting_rate(classification),
        "irrelevant_to_supporting_rate_delta": irrelevant_to_supporting_rate(classification) - irrelevant_to_supporting_rate(old),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 Evidence-Constrained Utility Scorer")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--oracle-v05", required=True)
    parser.add_argument("--v2-report", required=True)
    parser.add_argument("--v2-gap-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=os.environ.get("HEBBIAN_UTILITY_MODEL") or model_for("generation"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()
    if not os.environ.get("OPENAI_BASE_URL"):
        raise SystemExit("OPENAI_BASE_URL is required")

    dataset = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    v05_report, v05_records = load_v05(Path(args.oracle_v05))
    v2_report = json.loads(Path(args.v2_report).read_text(encoding="utf-8"))
    parameters = v05_report.get("parameters", {})
    if int(parameters.get("top_k", args.top_k)) != args.top_k:
        raise SystemExit("--top-k must match V0.5")
    for env_name, parameter, default in (
        ("HEBBIAN_MAX_FLIPPED", "max_flipped", 3),
        ("HEBBIAN_ACTIVATION_ALPHA", "activation_alpha", 0.1),
        ("HEBBIAN_SPREADING_THRESHOLD", "spreading_threshold", 0.4),
        ("HEBBIAN_KEYWORD_WEIGHT", "keyword_weight", 0.5),
    ):
        os.environ[env_name] = str(parameters.get(parameter, default))
    os.environ.update({"HEBBIAN_USE_KEYWORD_MATCH": "true", "HEBBIAN_USE_TIME_DECAY": "true", "HEBBIAN_USE_INHIBITION": "false"})

    output_dir = Path(args.output_dir)
    contexts = build_contexts(dataset, v05_records, Path(args.mem_dir))
    gaps = load_v2_gap_cache(contexts, Path(args.v2_gap_dir), args.model)
    matches = run_match_stage(contexts, gaps, output_dir / "match_items", args.model, args.workers, args.retries)
    classification = classification_metrics(matches)
    diagnostics, diagnostic_summary = redundant_diagnostics(matches)
    eligibility = replay_eligibility(classification, v2_report)
    replay, replay_rows = None, []
    if eligibility["run_retrieval_replay"]:
        predictions_by_question: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in matches:
            predictions_by_question.setdefault(row["question_id"], {})[row["candidate_id"]] = row
        for index, item in enumerate(dataset):
            if index in CORRUPTED_INDICES:
                continue
            question_id = str(item["question_id"])
            replay_rows.append(replay_item(item, v05_records[question_id], predictions_by_question.get(question_id, {}), Path(args.mem_dir) / f"{question_id}_hebbian.json", args.top_k))
        replay = replay_metrics(replay_rows)
    comparison = compare_with_v2(v2_report, classification)
    report = {
        "method": "V3 Evidence-Constrained Utility Scorer", "gold_answer_used_by_stage1": False,
        "gold_answer_used_by_stage2": False, "oracle_labels_used_only_for_evaluation": True,
        "answer_generation_executed": False, "benchmark_judge_executed": False,
        "graph_mutation_executed": False, "model": args.model, "temperature": 0.0,
        "stage1_reused_from_v2": True, "gap_prompt_template": GAP_PROMPT_TEMPLATE,
        "match_prompt_template": MATCH_PROMPT_TEMPLATE, "retrieval_parameters": parameters,
        "inputs": {"data_path": args.data_path, "mem_dir": args.mem_dir, "oracle_v05": args.oracle_v05, "v2_report": args.v2_report, "v2_gap_dir": args.v2_gap_dir},
        "gap_metrics": gap_metrics(gaps), "classification_metrics": classification,
        "comparison_with_v2": comparison, "retrieval_replay_eligibility": eligibility,
        "retrieval_replay_executed": eligibility["run_retrieval_replay"], "retrieval_replay_metrics": replay,
        "redundant_diagnostic_summary": diagnostic_summary, "match_outputs": matches,
        "redundant_diagnostics": diagnostics, "retrieval_records": replay_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "evidence_constrained_utility_v3_report.json", report)
    atomic_write_json(output_dir / "match_predictions.json", matches)
    atomic_write_json(output_dir / "redundant_diagnostics.json", {"summary": diagnostic_summary, "records": diagnostics})
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
    matrix = classification["confusion_matrix"]["counts"]
    for label in CLASSES:
        print(label + "\t" + "\t".join(str(matrix[label][pred]) for pred in CLASSES))
    print("redundant_diagnostic\t" + json.dumps(diagnostic_summary, ensure_ascii=False))
    print("retrieval_replay_eligibility\t" + json.dumps(eligibility, ensure_ascii=False))
    if replay is not None:
        for key, value in replay.items():
            print(f"retrieval_{key}\t{value}")
    print("comparison_with_v2\t" + json.dumps(comparison, ensure_ascii=False))
    print(f"report\t{output_dir / 'evidence_constrained_utility_v3_report.json'}")


if __name__ == "__main__":
    main()
