"""Deployable, gold-free utility scorer and retrieval replay for HeLa-Mem.

The scorer sees only a question, all current Base Top-15 memories, and one
Non-Base candidate.  Gold-derived V0.5 labels are joined only after inference
for evaluation and never enter the model prompt.

The replay reuses V0.5's query keywords, paired timestamp, Base context, and
unmodified graph.  It performs retrieval only: no answer generation, benchmark
judge, graph save, or Hebbian reinforcement.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .analyze_cross_edge_quality import build_records
from .runtime import atomic_write_json, chat_extra_body, model_for, strip_reasoning
from .scan_exact_oracle_edge_utility import CORRUPTED_INDICES, MULTIPLIERS, compact_trace


CLASSES = ("SUPPORTING", "REDUNDANT", "IRRELEVANT")
PROMPT_TEMPLATE = """You are evaluating whether a candidate memory provides useful additional evidence for answering a question.

QUESTION:
{question}

CURRENT BASE MEMORIES:
{base_memories}

CANDIDATE MEMORY:
{candidate}

First determine what information is already covered by the Base Memories and what information is still missing to answer the Question.

Then classify the Candidate as exactly one of:

SUPPORTING:
Provides new information not already covered by Base Memories and helps fill a missing piece needed to answer the question.

REDUNDANT:
Relevant to the question, but the useful information is already covered by Base Memories.

IRRELEVANT:
Does not provide information needed to answer the question, even if topically related.

Important:
- Judge incremental value relative to the FULL Base Memories.
- Do not use or assume access to a gold answer.
- Similar wording or shared entities alone does not imply SUPPORTING.
- Multi-hop bridge facts, missing temporal information, missing list or count items, updates, and disambiguating evidence can be SUPPORTING.

Return only:
SUPPORTING
or
REDUNDANT
or
IRRELEVANT"""


def build_scorer_prompt(question: str, base_memories: List[Dict[str, str]], candidate: str) -> str:
    """Construct the scorer input.  No answer/gold parameter exists by design."""
    base_text = "\n".join(
        f"[{index}] {memory.get('content', '').strip()}"
        for index, memory in enumerate(base_memories, start=1)
    )
    return PROMPT_TEMPLATE.format(
        question=question,
        base_memories=base_text,
        candidate=(candidate or "").strip(),
    )


def parse_label(text: str) -> str:
    value = strip_reasoning(text).strip().upper()
    value = re.sub(r"^```(?:TEXT)?\s*|\s*```$", "", value).strip()
    if value not in CLASSES:
        raise ValueError(f"invalid scorer output: {value[:160]!r}")
    return value


def classify_candidate(prompt: str, model: str, retries: int) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("utility scorer requires openai>=1.0") from error
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
                    {"role": "system", "content": "Classify incremental memory utility. Return one allowed label only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=16,
                **chat_extra_body(),
            )
            raw = response.choices[0].message.content if response.choices else ""
            return {"predicted_label": parse_label(raw or ""), "raw_output": strip_reasoning(raw or "")}
        except Exception as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"utility scorer failed after {retries} attempts: {last_error}")


def candidate_key(question_id: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{question_id}\0{candidate_id}".encode()).hexdigest()[:24]


def load_v05(path: Path) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("oracle_only") or not report.get("exact_current_retrieval_code"):
        raise ValueError("--oracle-v05 must be an Exact Oracle V0.5 report")
    return report, {str(row["question_id"]): row for row in report["records"]}


def scorer_tasks(
    dataset: List[Dict[str, Any]],
    v05_records: Dict[str, Dict[str, Any]],
    mem_dir: Path,
) -> List[Dict[str, Any]]:
    tasks = []
    for item in dataset:
        question_id = str(item["question_id"])
        v05 = v05_records.get(question_id, {})
        if v05.get("skipped_corrupted"):
            continue
        graph = json.loads((mem_dir / f"{question_id}_hebbian.json").read_text(encoding="utf-8"))
        nodes = graph.get("nodes", {})
        base_ids = [str(value) for value in v05.get("current_base_top_k_ids", [])]
        base_memories = [
            {"memory_id": node_id, "content": nodes.get(node_id, {}).get("content", "")}
            for node_id in base_ids
        ]
        oracle_labels = v05.get("oracle_edge_labels", {})
        for candidate_id, annotation in oracle_labels.items():
            candidate_id = str(candidate_id)
            tasks.append({
                "question_id": question_id,
                "question_type": item.get("question_type", ""),
                "candidate_id": candidate_id,
                "prompt": build_scorer_prompt(item["question"], base_memories, nodes.get(candidate_id, {}).get("content", "")),
                # Kept outside prompt and joined only after inference.
                "oracle_label": str(annotation.get("label", "uncertain")).upper(),
            })
            tasks[-1]["prompt_sha256"] = hashlib.sha256(tasks[-1]["prompt"].encode()).hexdigest()
    return tasks


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = [row for row in rows if row["oracle_label"] in CLASSES]
    excluded = len(rows) - len(evaluated)
    matrix = {true: {pred: 0 for pred in CLASSES} for true in CLASSES}
    for row in evaluated:
        matrix[row["oracle_label"]][row["predicted_label"]] += 1
    per_class = {}
    for label in CLASSES:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in CLASSES if other != label)
        fn = sum(matrix[label][other] for other in CLASSES if other != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        per_class[label] = {
            "support": sum(matrix[label].values()), "precision": precision,
            "recall": recall, "f1": safe_div(2 * precision * recall, precision + recall),
        }
    correct = sum(matrix[label][label] for label in CLASSES)
    false_supporting = sum(
        matrix[true]["SUPPORTING"] for true in CLASSES if true != "SUPPORTING"
    )
    return {
        "total_candidates": len(rows), "evaluated_candidates": len(evaluated),
        "uncertain_oracle_candidates_excluded": excluded,
        "overall_accuracy": safe_div(correct, len(evaluated)),
        "macro_f1": sum(per_class[label]["f1"] for label in CLASSES) / len(CLASSES),
        "per_class": per_class,
        "confusion_matrix": {"row_is_oracle": True, "column_is_prediction": True, "labels": list(CLASSES), "counts": matrix},
        "false_supporting_count": false_supporting,
        "supporting_to_redundant": matrix["SUPPORTING"]["REDUNDANT"],
        "supporting_to_irrelevant": matrix["SUPPORTING"]["IRRELEVANT"],
        "irrelevant_to_supporting": matrix["IRRELEVANT"]["SUPPORTING"],
        "predicted_label_counts": dict(Counter(row["predicted_label"] for row in rows)),
        "oracle_label_counts": dict(Counter(row["oracle_label"] for row in rows)),
    }


def predicted_multipliers(record: Dict[str, Any], predictions: Dict[str, str]) -> Dict[tuple[str, str], float]:
    values = {}
    for target in record["candidate_targets"]:
        candidate_id = str(target["candidate_id"])
        label = predictions[candidate_id]
        for base_id in target["connected_base_ids"]:
            values[(str(base_id), candidate_id)] = MULTIPLIERS[label.lower()]
    return values


def replay_item(
    item: Dict[str, Any],
    v05: Dict[str, Any],
    prediction_rows: Dict[str, Dict[str, Any]],
    graph_path: Path,
    top_k: int,
) -> Dict[str, Any]:
    from .hebbian_memory import HebbianMemoryGraph
    from .utils import get_embedding

    graph = HebbianMemoryGraph(file_path=str(graph_path))
    graph_payload = {"nodes": graph.nodes, "edges": {source: dict(neighbors) for source, neighbors in graph.edges.items()}}
    current_base_ids = [str(value) for value in v05["current_base_top_k_ids"]]
    synthetic_prediction = {"retrieved_episodic": [{"source": "base", "node_id": node_id} for node_id in current_base_ids]}
    candidate_record = build_records(item, synthetic_prediction, graph_payload, top_k)
    predicted = {
        target["candidate_id"]: prediction_rows[target["candidate_id"]]["predicted_label"]
        for target in candidate_record["candidate_targets"]
    }
    multipliers = predicted_multipliers(candidate_record, predicted)
    graph.retrieve(
        item["question"], top_k=top_k,
        query_keywords_override=set(v05.get("query_keywords", [])),
        query_embedding_override=get_embedding(item["question"]),
        current_time_override=v05["paired_current_time"],
        edge_weight_multipliers=multipliers,
        update_graph=False,
    )
    automatic_trace = compact_trace(graph.last_retrieval_trace or {})
    if automatic_trace["base_top_k_ids"] != current_base_ids:
        raise RuntimeError(f"{item['question_id']}: replay Base Top-K does not match V0.5")
    baseline = [str(value) for value in v05.get("baseline_flipped_ids", [])]
    oracle = [str(value) for value in v05.get("oracle_flipped_ids", [])]
    automatic = [str(value) for value in automatic_trace["flipped_memory_ids"]]
    gold_labels = {
        str(candidate_id): str(annotation.get("label", "uncertain")).upper()
        for candidate_id, annotation in v05.get("oracle_edge_labels", {}).items()
    }
    gold_labels.update({
        str(candidate_id): str(annotation.get("label", "uncertain")).upper()
        for candidate_id, annotation in v05.get("selected_labels", {}).items()
    })
    added = [node_id for node_id in automatic if node_id not in baseline]
    removed = [node_id for node_id in baseline if node_id not in automatic]
    baseline_useful = [node_id for node_id in baseline if gold_labels.get(node_id) == "SUPPORTING"]
    automatic_useful = [node_id for node_id in automatic if gold_labels.get(node_id) == "SUPPORTING"]
    return {
        "question_id": str(item["question_id"]), "question_type": item.get("question_type", ""),
        "baseline_flipped_ids": baseline, "oracle_flipped_ids": oracle,
        "automatic_flipped_ids": automatic,
        "selection_changed_from_baseline": automatic != baseline,
        "supporting_added_ids": [node_id for node_id in added if gold_labels.get(node_id) == "SUPPORTING"],
        "irrelevant_removed_ids": [node_id for node_id in removed if gold_labels.get(node_id) == "IRRELEVANT"],
        "irrelevant_added_ids": [node_id for node_id in added if gold_labels.get(node_id) == "IRRELEVANT"],
        "useful_question_rescued": not baseline_useful and bool(automatic_useful),
        "oracle_changed_from_baseline": oracle != baseline,
        "oracle_exact_match": automatic == oracle,
        "oracle_intersection": len(set(automatic) & set(oracle)),
        "oracle_union": len(set(automatic) | set(oracle)),
        "automatic_count": len(automatic), "oracle_count": len(oracle),
    }


def replay_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    oracle_total = sum(row["oracle_count"] for row in rows)
    automatic_total = sum(row["automatic_count"] for row in rows)
    intersection = sum(row["oracle_intersection"] for row in rows)
    union = sum(row["oracle_union"] for row in rows)
    oracle_changed = [row for row in rows if row["oracle_changed_from_baseline"]]
    return {
        "valid_retrieval_questions": len(rows),
        "selection_changed_questions": sum(row["selection_changed_from_baseline"] for row in rows),
        "supporting_added": sum(len(row["supporting_added_ids"]) for row in rows),
        "irrelevant_removed": sum(len(row["irrelevant_removed_ids"]) for row in rows),
        "irrelevant_added": sum(len(row["irrelevant_added_ids"]) for row in rows),
        "useful_questions_rescued": sum(row["useful_question_rescued"] for row in rows),
        "oracle_selection_exact_match_questions": sum(row["oracle_exact_match"] for row in rows),
        "oracle_selection_exact_match_rate": safe_div(sum(row["oracle_exact_match"] for row in rows), len(rows)),
        "oracle_changed_questions": len(oracle_changed),
        "oracle_exact_match_on_changed_questions": sum(row["oracle_exact_match"] for row in oracle_changed),
        "oracle_exact_match_rate_on_changed_questions": safe_div(sum(row["oracle_exact_match"] for row in oracle_changed), len(oracle_changed)),
        "oracle_selection_overlap_count": intersection,
        "oracle_selection_recall": safe_div(intersection, oracle_total),
        "oracle_selection_precision": safe_div(intersection, automatic_total),
        "oracle_selection_jaccard": safe_div(intersection, union),
        "oracle_selection_precision_on_changed_questions": safe_div(
            sum(row["oracle_intersection"] for row in oracle_changed),
            sum(row["automatic_count"] for row in oracle_changed),
        ),
        "oracle_selection_recall_on_changed_questions": safe_div(
            sum(row["oracle_intersection"] for row in oracle_changed),
            sum(row["oracle_count"] for row in oracle_changed),
        ),
    }


def print_metrics(classification: Dict[str, Any], replay: Dict[str, Any]) -> None:
    print("classification_metric\tvalue")
    for key in ("overall_accuracy", "macro_f1", "false_supporting_count", "supporting_to_redundant", "supporting_to_irrelevant", "irrelevant_to_supporting"):
        print(f"{key}\t{classification[key]}")
    for label in CLASSES:
        row = classification["per_class"][label]
        print(f"{label}_precision\t{row['precision']}")
        print(f"{label}_recall\t{row['recall']}")
        print(f"{label}_f1\t{row['f1']}")
    print("confusion_matrix_rows_oracle_columns_prediction")
    print("oracle\\predicted\t" + "\t".join(CLASSES))
    matrix = classification["confusion_matrix"]["counts"]
    for label in CLASSES:
        print(label + "\t" + "\t".join(str(matrix[label][pred]) for pred in CLASSES))
    print("retrieval_metric\tvalue")
    for key, value in replay.items():
        print(f"{key}\t{value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold-free Qwen utility scorer plus exact retrieval replay")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--oracle-v05", required=True)
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
    v05_parameters = v05_report.get("parameters", {})
    if int(v05_parameters.get("top_k", args.top_k)) != args.top_k:
        raise SystemExit("--top-k must match the V0.5 report")
    os.environ["HEBBIAN_MAX_FLIPPED"] = str(v05_parameters.get("max_flipped", 3))
    os.environ["HEBBIAN_ACTIVATION_ALPHA"] = str(v05_parameters.get("activation_alpha", 0.1))
    os.environ["HEBBIAN_SPREADING_THRESHOLD"] = str(v05_parameters.get("spreading_threshold", 0.4))
    os.environ["HEBBIAN_KEYWORD_WEIGHT"] = str(v05_parameters.get("keyword_weight", 0.5))
    os.environ["HEBBIAN_USE_KEYWORD_MATCH"] = "true"
    os.environ["HEBBIAN_USE_TIME_DECAY"] = "true"
    os.environ["HEBBIAN_USE_INHIBITION"] = "false"
    output_dir = Path(args.output_dir)
    scorer_dir = output_dir / "scorer_items"
    scorer_dir.mkdir(parents=True, exist_ok=True)
    tasks = scorer_tasks(dataset, v05_records, Path(args.mem_dir))
    completed: List[Dict[str, Any]] = []
    pending = []
    for task in tasks:
        path = scorer_dir / f"{candidate_key(task['question_id'], task['candidate_id'])}.json"
        if path.exists():
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                if (
                    row.get("model") == args.model
                    and row.get("prompt_sha256") == task["prompt_sha256"]
                    and row.get("status") == "ok"
                ):
                    completed.append(row)
                    continue
            except (OSError, ValueError, TypeError):
                pass
        pending.append((task, path))
    print(f"Utility scoring: {len(completed)} resumed, {len(pending)} pending", flush=True)

    def run_task(value: tuple[Dict[str, Any], Path]) -> Dict[str, Any]:
        task, path = value
        # Inference happens before oracle_label is copied into the saved row.
        prediction = classify_candidate(task["prompt"], args.model, args.retries)
        row = {
            "status": "ok", "model": args.model,
            "question_id": task["question_id"], "question_type": task["question_type"],
            "candidate_id": task["candidate_id"],
            "prompt_sha256": task["prompt_sha256"],
            "predicted_label": prediction["predicted_label"],
            "oracle_label": task["oracle_label"],
            "raw_output": prediction["raw_output"],
            "gold_answer_in_prompt": False,
        }
        atomic_write_json(path, row)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_task, value) for value in pending]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
            print(f"Scored candidates: {len(completed)}/{len(tasks)}", flush=True)
    completed.sort(key=lambda row: (row["question_id"], row["candidate_id"]))
    classification = classification_metrics(completed)
    predictions_by_question: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in completed:
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
    supporting = classification["per_class"]["SUPPORTING"]
    gate = {
        "minimum_supporting_precision": 0.50,
        "minimum_supporting_recall": 0.40,
        "minimum_oracle_selection_recall": 0.50,
        "requires_at_least_one_useful_rescue": True,
        "supporting_precision_pass": supporting["precision"] >= 0.50,
        "supporting_recall_pass": supporting["recall"] >= 0.40,
        "oracle_selection_recall_pass": replay["oracle_selection_recall"] >= 0.50,
        "useful_rescue_pass": replay["useful_questions_rescued"] > 0,
    }
    gate["recommended_for_final_answer_evaluation"] = all(
        gate[key] for key in (
            "supporting_precision_pass", "supporting_recall_pass",
            "oracle_selection_recall_pass", "useful_rescue_pass",
        )
    )
    report = {
        "deployable_scorer": True, "gold_answer_used_by_scorer": False,
        "answer_generation_executed": False, "benchmark_judge_executed": False,
        "graph_mutation_executed": False, "model": args.model, "temperature": 0.0,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_context_policy": "All Base Top-15 memories and the candidate are included without text truncation.",
        "retrieval_parameters": v05_parameters,
        "inputs": {"data_path": args.data_path, "mem_dir": args.mem_dir, "oracle_v05": args.oracle_v05},
        "classification_metrics": classification, "retrieval_replay_metrics": replay,
        "evaluation_gate": gate, "scorer_outputs": completed, "retrieval_records": replay_rows,
    }
    atomic_write_json(output_dir / "utility_scorer_v1_report.json", report)
    atomic_write_json(output_dir / "utility_scorer_predictions.json", completed)
    atomic_write_json(output_dir / "retrieval_replay.json", {"metrics": replay, "records": replay_rows})
    print_metrics(classification, replay)
    print(f"recommended_for_final_answer_evaluation\t{gate['recommended_for_final_answer_evaluation']}")
    print(f"report\t{output_dir / 'utility_scorer_v1_report.json'}")


if __name__ == "__main__":
    main()
