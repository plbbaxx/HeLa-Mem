"""V2 Structured Residual Utility Estimation for HeLa-Mem.

Stage 1 extracts a question-conditioned evidence state from the complete Base
Top-15: covered information, answerability, and missing information slots.
Stage 2 classifies each candidate by whether it fills a missing slot, duplicates
covered evidence, or does not match.  Neither stage receives a gold answer.

Gold-derived V0.5 labels are joined only after Stage 2 for evaluation.  The
retrieval replay is read-only and reuses V0.5's exact retrieval state.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .runtime import atomic_write_json, chat_extra_body, model_for, strip_reasoning
from .scan_exact_oracle_edge_utility import CORRUPTED_INDICES
from .utility_scorer_v1 import (
    CLASSES,
    classification_metrics,
    load_v05,
    replay_item,
    replay_metrics,
)


STAGE2_LABELS = ("FILLS_GAP", "DUPLICATE", "NO_MATCH")
STAGE2_TO_CLASS = {
    "FILLS_GAP": "SUPPORTING",
    "DUPLICATE": "REDUNDANT",
    "NO_MATCH": "IRRELEVANT",
}

GAP_PROMPT_TEMPLATE = """You are identifying the residual information needed to answer a question after reading the current memory evidence.

QUESTION:
{question}

CURRENT BASE MEMORIES:
{base_memories}

Analyze only the Question and Base Memories above. You do not have access to a gold answer.

Return a JSON object with exactly these fields:
{{
  "base_coverage": ["atomic fact already established by the Base Memories"],
  "answerable_from_base": true or false,
  "missing_slots": [
    {{"slot_id": "G1", "description": "specific type of information still required"}}
  ]
}}

Rules:
- Include only facts relevant to answering the Question in base_coverage.
- A missing slot describes what information is needed, never a guessed value.
- Do not invent people, places, dates, events, relations, or answer values.
- If the Base Memories already contain enough evidence, set answerable_from_base to true and missing_slots to [].
- If evidence is missing, set answerable_from_base to false and list only question-required gaps.
- For multi-hop questions, include unresolved bridge facts.
- For temporal, list, count, update, and disambiguation questions, represent the unresolved constraint explicitly.
- Return JSON only."""

MATCH_PROMPT_TEMPLATE = """You are evaluating the incremental evidence supplied by one candidate memory.

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

Classify the Candidate as exactly one of:

FILLS_GAP:
The Candidate explicitly supplies new evidence that fills at least one listed missing slot needed to answer the Question.

DUPLICATE:
The Candidate is relevant, but its useful information is already present in the Base coverage and it fills no missing slot.

NO_MATCH:
The Candidate neither fills a missing slot nor duplicates answer-relevant Base coverage. Shared topics or entities alone are not enough.

Rules:
- Judge evidence stated in the Candidate, not what might plausibly be true.
- Do not use or assume access to a gold answer.
- If answerable_from_base is true, FILLS_GAP is not allowed.
- Return only FILLS_GAP, DUPLICATE, or NO_MATCH."""


def format_base_memories(memories: List[Dict[str, str]]) -> str:
    return "\n".join(
        f"[{index}] {(memory.get('content') or '').strip()}"
        for index, memory in enumerate(memories, start=1)
    )


def build_gap_prompt(question: str, base_memories: List[Dict[str, str]]) -> str:
    """No answer/gold parameter exists, preventing gold leakage by interface."""
    return GAP_PROMPT_TEMPLATE.format(
        question=question,
        base_memories=format_base_memories(base_memories),
    )


def build_match_prompt(question: str, residual: Dict[str, Any], candidate: str) -> str:
    """Construct Stage 2 input from Stage 1 state; no raw gold label is accepted."""
    return MATCH_PROMPT_TEMPLATE.format(
        question=question,
        base_coverage=json.dumps(residual["base_coverage"], ensure_ascii=False),
        answerable_from_base=str(residual["answerable_from_base"]).lower(),
        missing_slots=json.dumps(residual["missing_slots"], ensure_ascii=False),
        candidate=(candidate or "").strip(),
    )


def strip_code_fence(text: str) -> str:
    value = strip_reasoning(text).strip()
    return re.sub(r"^```(?:json|text)?\s*|\s*```$", "", value, flags=re.IGNORECASE).strip()


def parse_gap_response(text: str) -> Dict[str, Any]:
    value = json.loads(strip_code_fence(text))
    if set(value) != {"base_coverage", "answerable_from_base", "missing_slots"}:
        raise ValueError("gap output must contain exactly base_coverage, answerable_from_base, missing_slots")
    if not isinstance(value["base_coverage"], list) or not all(isinstance(item, str) for item in value["base_coverage"]):
        raise ValueError("base_coverage must be a list of strings")
    if not isinstance(value["answerable_from_base"], bool):
        raise ValueError("answerable_from_base must be boolean")
    if not isinstance(value["missing_slots"], list):
        raise ValueError("missing_slots must be a list")
    normalized_slots = []
    for index, slot in enumerate(value["missing_slots"], start=1):
        if not isinstance(slot, dict) or not isinstance(slot.get("description"), str):
            raise ValueError("every missing slot requires a string description")
        description = slot["description"].strip()
        if not description:
            raise ValueError("missing slot descriptions cannot be empty")
        normalized_slots.append({
            "slot_id": str(slot.get("slot_id") or f"G{index}"),
            "description": description,
        })
    if value["answerable_from_base"] and normalized_slots:
        raise ValueError("answerable_from_base=true requires an empty missing_slots list")
    if not value["answerable_from_base"] and not normalized_slots:
        raise ValueError("answerable_from_base=false requires at least one missing slot")
    return {
        "base_coverage": [item.strip() for item in value["base_coverage"] if item.strip()],
        "answerable_from_base": value["answerable_from_base"],
        "missing_slots": normalized_slots,
    }


def parse_stage2_response(text: str) -> str:
    value = strip_code_fence(text).upper()
    if value not in STAGE2_LABELS:
        raise ValueError(f"invalid Stage 2 output: {value[:160]!r}")
    return value


def chat(prompt: str, model: str, max_tokens: int, parser, retries: int) -> tuple[Any, str]:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("V2 requires openai>=1.0") from error
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Perform structured residual evidence analysis exactly as instructed."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                **chat_extra_body(),
            )
            raw = response.choices[0].message.content if response.choices else ""
            return parser(raw or ""), strip_reasoning(raw or "")
        except Exception as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"V2 model call failed after {retries} attempts: {last_error}")


def build_contexts(
    dataset: List[Dict[str, Any]],
    v05_records: Dict[str, Dict[str, Any]],
    mem_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    contexts = {}
    for index, item in enumerate(dataset):
        if index in CORRUPTED_INDICES:
            continue
        question_id = str(item["question_id"])
        v05 = v05_records[question_id]
        graph = json.loads((mem_dir / f"{question_id}_hebbian.json").read_text(encoding="utf-8"))
        nodes = graph.get("nodes", {})
        base_ids = [str(value) for value in v05.get("current_base_top_k_ids", [])]
        candidates = []
        for candidate_id, oracle in v05.get("oracle_edge_labels", {}).items():
            candidate_id = str(candidate_id)
            candidates.append({
                "candidate_id": candidate_id,
                "content": nodes.get(candidate_id, {}).get("content", ""),
                "oracle_label": str(oracle.get("label", "uncertain")).upper(),
            })
        contexts[question_id] = {
            "question_id": question_id,
            "question_type": item.get("question_type", ""),
            "question": item["question"],
            "base_memories": [
                {"memory_id": node_id, "content": nodes.get(node_id, {}).get("content", "")}
                for node_id in base_ids
            ],
            "candidates": candidates,
        }
    return contexts


def cache_key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]


def run_gap_stage(
    contexts: Dict[str, Dict[str, Any]],
    output_dir: Path,
    model: str,
    workers: int,
    retries: int,
) -> Dict[str, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: Dict[str, Dict[str, Any]] = {}
    pending = []
    # Stage 1 is required only where a candidate utility decision will follow.
    for question_id, context in contexts.items():
        if not context["candidates"]:
            continue
        prompt = build_gap_prompt(context["question"], context["base_memories"])
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        path = output_dir / f"{cache_key(question_id)}.json"
        if path.exists():
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("status") == "ok" and row.get("model") == model and row.get("prompt_sha256") == prompt_sha:
                    completed[question_id] = row
                    continue
            except (OSError, ValueError, TypeError):
                pass
        pending.append((question_id, prompt, prompt_sha, path))
    total = len(completed) + len(pending)
    print(f"Stage 1 Gap Extraction: {len(completed)} resumed, {len(pending)} pending", flush=True)

    def run_one(task):
        question_id, prompt, prompt_sha, path = task
        residual, raw = chat(prompt, model, 1200, parse_gap_response, retries)
        row = {
            "status": "ok", "stage": "gap_extraction", "model": model,
            "question_id": question_id, "prompt_sha256": prompt_sha,
            "residual": residual, "raw_output": raw, "gold_answer_in_prompt": False,
        }
        atomic_write_json(path, row)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run_one, task) for task in pending]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed[row["question_id"]] = row
            print(f"Stage 1 complete: {len(completed)}/{total}", flush=True)
    return completed


def run_match_stage(
    contexts: Dict[str, Dict[str, Any]],
    gaps: Dict[str, Dict[str, Any]],
    output_dir: Path,
    model: str,
    workers: int,
    retries: int,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    pending = []
    for question_id, context in contexts.items():
        if not context["candidates"]:
            continue
        residual = gaps[question_id]["residual"]
        for candidate in context["candidates"]:
            prompt = build_match_prompt(context["question"], residual, candidate["content"])
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            path = output_dir / f"{cache_key(question_id, candidate['candidate_id'])}.json"
            if path.exists():
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if row.get("status") == "ok" and row.get("model") == model and row.get("prompt_sha256") == prompt_sha:
                        completed.append(row)
                        continue
                except (OSError, ValueError, TypeError):
                    pass
            pending.append((context, candidate, prompt, prompt_sha, path))
    total = len(completed) + len(pending)
    print(f"Stage 2 Gap Matching: {len(completed)} resumed, {len(pending)} pending", flush=True)

    def run_one(task):
        context, candidate, prompt, prompt_sha, path = task
        stage2_label, raw = chat(prompt, model, 16, parse_stage2_response, retries)
        row = {
            "status": "ok", "stage": "candidate_gap_matching", "model": model,
            "question_id": context["question_id"], "question_type": context["question_type"],
            "candidate_id": candidate["candidate_id"], "prompt_sha256": prompt_sha,
            "stage2_label": stage2_label,
            "predicted_label": STAGE2_TO_CLASS[stage2_label],
            # Joined after inference; never included in the prompt.
            "oracle_label": candidate["oracle_label"],
            "raw_output": raw, "gold_answer_in_prompt": False,
        }
        atomic_write_json(path, row)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(run_one, task) for task in pending]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
            print(f"Stage 2 complete: {len(completed)}/{total}", flush=True)
    completed.sort(key=lambda row: (row["question_id"], row["candidate_id"]))
    return completed


def gap_metrics(gaps: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    residuals = [row["residual"] for row in gaps.values()]
    sizes = [len(row["missing_slots"]) for row in residuals]
    return {
        "questions_analyzed": len(residuals),
        "answerable_from_base_questions": sum(row["answerable_from_base"] for row in residuals),
        "answerable_from_base_rate": sum(row["answerable_from_base"] for row in residuals) / len(residuals) if residuals else 0.0,
        "empty_gap_questions": sum(not row["missing_slots"] for row in residuals),
        "empty_gap_rate": sum(not row["missing_slots"] for row in residuals) / len(residuals) if residuals else 0.0,
        "mean_missing_slots": statistics.mean(sizes) if sizes else 0.0,
        "median_missing_slots": statistics.median(sizes) if sizes else 0.0,
        "max_missing_slots": max(sizes) if sizes else 0,
    }


def evaluation_gate(classification: Dict[str, Any], replay: Dict[str, Any]) -> Dict[str, Any]:
    supporting = classification["per_class"]["SUPPORTING"]
    redundant = classification["per_class"]["REDUNDANT"]
    irrelevant_support_rate = (
        classification["irrelevant_to_supporting"] / classification["per_class"]["IRRELEVANT"]["support"]
        if classification["per_class"]["IRRELEVANT"]["support"] else 0.0
    )
    gate = {
        "minimum_supporting_precision": 0.30,
        "minimum_supporting_recall": 0.40,
        "minimum_redundant_recall": 0.20,
        "maximum_irrelevant_to_supporting_rate": 0.35,
        "minimum_oracle_selection_precision_on_changed_questions": 0.35,
        "requires_useful_rescue": True,
        "supporting_precision_pass": supporting["precision"] >= 0.30,
        "supporting_recall_pass": supporting["recall"] >= 0.40,
        "redundant_recall_pass": redundant["recall"] >= 0.20,
        "irrelevant_to_supporting_rate": irrelevant_support_rate,
        "irrelevant_to_supporting_pass": irrelevant_support_rate <= 0.35,
        "oracle_changed_selection_precision_pass": replay["oracle_selection_precision_on_changed_questions"] >= 0.35,
        "useful_rescue_pass": replay["useful_questions_rescued"] > 0,
    }
    gate["recommended_for_final_answer_evaluation"] = all(
        gate[key] for key in (
            "supporting_precision_pass", "supporting_recall_pass", "redundant_recall_pass",
            "irrelevant_to_supporting_pass", "oracle_changed_selection_precision_pass",
            "useful_rescue_pass",
        )
    )
    return gate


def compare_with_v1(v1_report: Dict[str, Any], classification: Dict[str, Any], replay: Dict[str, Any]) -> Dict[str, Any]:
    v1_classification = v1_report["classification_metrics"]
    v1_replay = v1_report["retrieval_replay_metrics"]
    if "oracle_selection_precision_on_changed_questions" not in v1_replay and v1_report.get("retrieval_records"):
        v1_replay = replay_metrics(v1_report["retrieval_records"])
    v1_irrelevant_support_rate = (
        v1_classification["irrelevant_to_supporting"] / v1_classification["per_class"]["IRRELEVANT"]["support"]
        if v1_classification["per_class"]["IRRELEVANT"]["support"] else 0.0
    )
    v2_irrelevant_support_rate = (
        classification["irrelevant_to_supporting"] / classification["per_class"]["IRRELEVANT"]["support"]
        if classification["per_class"]["IRRELEVANT"]["support"] else 0.0
    )
    return {
        "supporting_precision_delta": classification["per_class"]["SUPPORTING"]["precision"] - v1_classification["per_class"]["SUPPORTING"]["precision"],
        "supporting_recall_delta": classification["per_class"]["SUPPORTING"]["recall"] - v1_classification["per_class"]["SUPPORTING"]["recall"],
        "redundant_recall_delta": classification["per_class"]["REDUNDANT"]["recall"] - v1_classification["per_class"]["REDUNDANT"]["recall"],
        "macro_f1_delta": classification["macro_f1"] - v1_classification["macro_f1"],
        "irrelevant_to_supporting_rate_v1": v1_irrelevant_support_rate,
        "irrelevant_to_supporting_rate_v2": v2_irrelevant_support_rate,
        "irrelevant_to_supporting_rate_delta": v2_irrelevant_support_rate - v1_irrelevant_support_rate,
        "oracle_changed_selection_precision_delta": replay["oracle_selection_precision_on_changed_questions"] - v1_replay.get("oracle_selection_precision_on_changed_questions", v1_replay["oracle_selection_precision"]),
        "useful_questions_rescued_delta": replay["useful_questions_rescued"] - v1_replay["useful_questions_rescued"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 Structured Residual Utility Estimation")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--oracle-v05", required=True)
    parser.add_argument("--v1-report")
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
    parameters = v05_report.get("parameters", {})
    if int(parameters.get("top_k", args.top_k)) != args.top_k:
        raise SystemExit("--top-k must match V0.5")
    os.environ["HEBBIAN_MAX_FLIPPED"] = str(parameters.get("max_flipped", 3))
    os.environ["HEBBIAN_ACTIVATION_ALPHA"] = str(parameters.get("activation_alpha", 0.1))
    os.environ["HEBBIAN_SPREADING_THRESHOLD"] = str(parameters.get("spreading_threshold", 0.4))
    os.environ["HEBBIAN_KEYWORD_WEIGHT"] = str(parameters.get("keyword_weight", 0.5))
    os.environ["HEBBIAN_USE_KEYWORD_MATCH"] = "true"
    os.environ["HEBBIAN_USE_TIME_DECAY"] = "true"
    os.environ["HEBBIAN_USE_INHIBITION"] = "false"

    output_dir = Path(args.output_dir)
    contexts = build_contexts(dataset, v05_records, Path(args.mem_dir))
    gaps = run_gap_stage(contexts, output_dir / "gap_items", args.model, args.workers, args.retries)
    matches = run_match_stage(contexts, gaps, output_dir / "match_items", args.model, args.workers, args.retries)
    classification = classification_metrics(matches)
    predictions_by_question: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in matches:
        predictions_by_question.setdefault(row["question_id"], {})[row["candidate_id"]] = row

    replay_rows = []
    for index, item in enumerate(dataset):
        if index in CORRUPTED_INDICES:
            continue
        question_id = str(item["question_id"])
        replay_rows.append(replay_item(
            item, v05_records[question_id], predictions_by_question.get(question_id, {}),
            Path(args.mem_dir) / f"{question_id}_hebbian.json", args.top_k,
        ))
        if len(replay_rows) % 25 == 0:
            print(f"Retrieval replay: {len(replay_rows)}/{len(dataset) - len(CORRUPTED_INDICES)}", flush=True)
    replay = replay_metrics(replay_rows)
    gaps_summary = gap_metrics(gaps)
    gate = evaluation_gate(classification, replay)
    v1_comparison = None
    if args.v1_report and Path(args.v1_report).exists():
        v1_comparison = compare_with_v1(
            json.loads(Path(args.v1_report).read_text(encoding="utf-8")),
            classification, replay,
        )
    report = {
        "method": "V2 Structured Residual Utility Estimation",
        "gold_answer_used_by_stage1": False, "gold_answer_used_by_stage2": False,
        "oracle_labels_used_only_for_evaluation": True,
        "answer_generation_executed": False, "benchmark_judge_executed": False,
        "graph_mutation_executed": False, "model": args.model, "temperature": 0.0,
        "gap_prompt_template": GAP_PROMPT_TEMPLATE,
        "match_prompt_template": MATCH_PROMPT_TEMPLATE,
        "retrieval_parameters": parameters,
        "inputs": {"data_path": args.data_path, "mem_dir": args.mem_dir, "oracle_v05": args.oracle_v05},
        "gap_metrics": gaps_summary, "classification_metrics": classification,
        "retrieval_replay_metrics": replay, "evaluation_gate": gate,
        "comparison_with_v1": v1_comparison,
        "gap_outputs": list(gaps.values()), "match_outputs": matches,
        "retrieval_records": replay_rows,
    }
    atomic_write_json(output_dir / "structured_residual_utility_v2_report.json", report)
    atomic_write_json(output_dir / "gap_outputs.json", list(gaps.values()))
    atomic_write_json(output_dir / "match_predictions.json", matches)
    atomic_write_json(output_dir / "retrieval_replay.json", {"metrics": replay, "records": replay_rows})

    print("gap_metric\tvalue")
    for key, value in gaps_summary.items():
        print(f"{key}\t{value}")
    print("classification_metric\tvalue")
    for key in ("overall_accuracy", "macro_f1", "false_supporting_count", "supporting_to_redundant", "supporting_to_irrelevant", "irrelevant_to_supporting"):
        print(f"{key}\t{classification[key]}")
    for label in CLASSES:
        row = classification["per_class"][label]
        for metric in ("precision", "recall", "f1"):
            print(f"{label}_{metric}\t{row[metric]}")
    print("confusion_matrix_rows_oracle_columns_prediction")
    print("oracle\\predicted\t" + "\t".join(CLASSES))
    matrix = classification["confusion_matrix"]["counts"]
    for label in CLASSES:
        print(label + "\t" + "\t".join(str(matrix[label][pred]) for pred in CLASSES))
    print("stage2_prediction_counts\t" + json.dumps(Counter(row["stage2_label"] for row in matches), ensure_ascii=False))
    print("retrieval_metric\tvalue")
    for key, value in replay.items():
        print(f"{key}\t{value}")
    print("evaluation_gate\t" + json.dumps(gate, ensure_ascii=False))
    if v1_comparison is not None:
        print("comparison_with_v1\t" + json.dumps(v1_comparison, ensure_ascii=False))
    print(f"report\t{output_dir / 'structured_residual_utility_v2_report.json'}")


if __name__ == "__main__":
    main()
