"""V5 Question-Conditioned Fact Alignment Utility Scorer.

The only experimental variable is the utility scorer representation.  V5
reuses V2's residual-gap cache and the V0.5 candidate/oracle evaluation set,
but represents Base and candidate evidence as question-conditioned claims
before a single semantic alignment decision.  No retrieval replay is run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .runtime import atomic_write_json, model_for
from .structured_residual_utility_v2 import GAP_PROMPT_TEMPLATE, build_contexts, cache_key, chat, strip_code_fence
from .evidence_constrained_utility_v3 import load_v2_gap_cache, irrelevant_to_supporting_rate
from .utility_scorer_v1 import CLASSES, classification_metrics, load_v05


BASE_CLAIM_PROMPT = """You are extracting answer-relevant facts from retrieved memories.

QUESTION:
{question}

BASE MEMORIES:
{base_memories_with_ids}

For every Base Memory, determine whether it contains a concrete fact that helps answer the Question.  Include every answer-relevant Base Memory in claims; you may omit memories that are not answer-relevant.

If it does, extract:
- slot: the specific answer component or sub-question addressed by this fact
- value: the concrete factual value provided for that slot

If it does not contain information useful for answering the Question, set answer_relevant to false and set slot and value to null.

Important rules:
- Extract facts relative to the Question, not general topics.
- Do not use external knowledge or the gold answer.
- Do not merge different facts merely because they mention the same entity or topic.
- Keep slot descriptions specific enough to distinguish different answer components.
- For list/count/aggregation questions, different items should remain distinguishable.

Return valid JSON only:
{{
  "claims": [
    {{"memory_id": "...", "answer_relevant": true, "slot": "...", "value": "..."}}
  ]
}}
"""

CANDIDATE_CLAIM_PROMPT = """You are extracting the answer-relevant fact from a candidate memory.

QUESTION:
{question}

CANDIDATE MEMORY:
{candidate}

Determine whether the Candidate contains a concrete fact useful for answering the Question.

If yes, extract:
- slot: the specific answer component or sub-question addressed by the Candidate
- value: the concrete factual value it provides

If the Candidate does not contain an answer-relevant fact, set answer_relevant to false and set slot and value to null.

Important rules:
- Do not classify based on topical similarity.
- Mentioning the same entity as the Question is not enough.
- Extract only concrete facts that contribute to answering the Question.
- Do not use external knowledge or the gold answer.

Return valid JSON only:
{{"answer_relevant": true, "slot": "...", "value": "..."}}
"""

ALIGNMENT_PROMPT = """You are evaluating the incremental utility of a Candidate Claim relative to the current Base Claims.

QUESTION:
{question}

MISSING INFORMATION:
{missing_gaps}

BASE CLAIMS:
{base_claims}

CANDIDATE CLAIM:
{candidate_claim}

Classify the Candidate into exactly one of:

REDUNDANT
The Candidate expresses the same answer-relevant fact as an existing Base Claim.
For REDUNDANT, both must hold:
1. They address the same answer slot or sub-question.
2. Their factual values are equivalent or paraphrases of the same fact.

SUPPORTING
The Candidate provides a new answer-relevant fact not already covered by the Base Claims and helps fill at least one missing information gap.

IRRELEVANT
The Candidate neither duplicates an answer-relevant fact already in Base nor provides new information that fills a missing gap.

Important rules:
- Same topic is NOT enough for REDUNDANT.
- Same entity is NOT enough for REDUNDANT.
- Same slot with a different factual value is NOT automatically REDUNDANT.
- For list, count, aggregation, multi-hop, temporal, or update questions, a new distinct item or value may be SUPPORTING.
- If Candidate Claim has answer_relevant=false, normally classify it as IRRELEVANT.
- Do not use the gold answer or external knowledge.

Return valid JSON only:
{{"label": "SUPPORTING | REDUNDANT | IRRELEVANT", "matched_base_memory_id": "... or null", "matched_slot": "... or null", "reason": "brief structured reason"}}
"""


def format_base_memories(memories: List[Dict[str, str]]) -> str:
    return "\n\n".join(f"[Base Memory ID: {memory['memory_id']}]\n{(memory.get('content') or '').strip()}" for memory in memories)


def build_base_claim_prompt(question: str, base_memories: List[Dict[str, str]]) -> str:
    return BASE_CLAIM_PROMPT.format(question=question, base_memories_with_ids=format_base_memories(base_memories))


def build_candidate_claim_prompt(question: str, candidate: str) -> str:
    return CANDIDATE_CLAIM_PROMPT.format(question=question, candidate=(candidate or "").strip())


def build_alignment_prompt(question: str, residual: Dict[str, Any], base_claims: List[Dict[str, Any]], candidate_claim: Dict[str, Any]) -> str:
    return ALIGNMENT_PROMPT.format(question=question, missing_gaps=json.dumps(residual["missing_slots"], ensure_ascii=False), base_claims=json.dumps(base_claims, ensure_ascii=False), candidate_claim=json.dumps(candidate_claim, ensure_ascii=False))


def _json(text: str) -> Any:
    return json.loads(strip_code_fence(text))


def _claim(value: Dict[str, Any], needs_memory_id: bool) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("claim must be an object")
    if needs_memory_id and (not isinstance(value.get("memory_id"), str) or not value["memory_id"].strip()):
        raise ValueError("claim requires memory_id")
    # Do not let optional explanations, omitted nulls, or incomplete claims
    # terminate a long run.  A claim is usable only when the model explicitly
    # marks it relevant *and* provides both factual fields; otherwise safely
    # downgrade it to irrelevant rather than inventing a slot or value.
    relevant = value.get("answer_relevant") is True
    slot, factual_value = value.get("slot"), value.get("value")
    usable = relevant and isinstance(slot, str) and slot.strip() and isinstance(factual_value, str) and factual_value.strip()
    result = {"answer_relevant": bool(usable), "slot": slot.strip() if usable else None, "value": factual_value.strip() if usable else None}
    if needs_memory_id: result["memory_id"] = value["memory_id"].strip()
    return result


def parse_base_claims(text: str, expected_ids: Iterable[str]) -> List[Dict[str, Any]]:
    value = _json(text)
    if not isinstance(value, dict) or set(value) != {"claims"} or not isinstance(value["claims"], list):
        raise ValueError("base output must be {claims: [...]}")
    claims = []
    for row in value["claims"]:
        try:
            claims.append(_claim(row, True))
        except ValueError:
            # An entry without a verifiable memory ID cannot be grounded and is
            # ignored; the real Base memory will be restored as irrelevant.
            continue
    ordered_ids, expected = [str(x) for x in expected_ids], {str(x) for x in expected_ids}
    # A small model occasionally repeats an ID or emits an invented ID despite
    # the prompt.  Such a claim cannot be safely grounded in the Base context,
    # so retain only the first claim with a real, unique ID.  Omitted real IDs
    # remain explicit irrelevant claims below.  This recovery never maps an
    # invented claim onto a different memory.
    valid_claims, seen = [], set()
    for claim in claims:
        memory_id = claim["memory_id"]
        if memory_id in expected and memory_id not in seen:
            valid_claims.append(claim)
            seen.add(memory_id)
    # Models often return only relevant claims.  Restore omitted Base entries as
    # explicit irrelevant claims so alignment still receives the complete Top-K
    # evidence state without fabricating slot/value content.
    by_id = {row["memory_id"]: row for row in valid_claims}
    return [by_id.get(memory_id, {"memory_id": memory_id, "answer_relevant": False, "slot": None, "value": None}) for memory_id in ordered_ids]


def parse_candidate_claim(text: str) -> Dict[str, Any]:
    try:
        return _claim(_json(text), False)
    except ValueError:
        return {"answer_relevant": False, "slot": None, "value": None}


def parse_alignment(text: str, valid_base_ids: Iterable[str]) -> Dict[str, Any]:
    value = _json(text)
    if not isinstance(value, dict):
        raise ValueError("alignment output must be an object")
    label = str(value.get("label", "")).strip().upper()
    reason = value.get("reason")
    reason = reason.strip() if isinstance(reason, str) and reason.strip() else "No structured reason returned by model."
    base_id, slot = value.get("matched_base_memory_id"), value.get("matched_slot")
    # A positive REDUNDANT decision has an asymmetric safety requirement: it
    # must be grounded in a real Base claim.  An incomplete or invented match
    # is therefore an abstention (IRRELEVANT), not a guessed redundancy.
    if label == "REDUNDANT":
        if not isinstance(base_id, str) or base_id not in {str(x) for x in valid_base_ids} or not isinstance(slot, str) or not slot.strip():
            return {"label": "IRRELEVANT", "matched_base_memory_id": None, "matched_slot": None, "reason": "Unverifiable REDUNDANT alignment downgraded to IRRELEVANT."}
        return {"label": label, "matched_base_memory_id": base_id, "matched_slot": slot.strip(), "reason": reason}
    if label in {"SUPPORTING", "IRRELEVANT"}:
        return {"label": label, "matched_base_memory_id": None, "matched_slot": None, "reason": reason}
    # Never manufacture a positive utility label from malformed output.
    return {"label": "IRRELEVANT", "matched_base_memory_id": None, "matched_slot": None, "reason": "Unusable alignment label downgraded to IRRELEVANT."}


def _load(path: Path, model: str, prompt_sha: str) -> Dict[str, Any] | None:
    try: row = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError, TypeError): return None
    return row if row and row.get("status") == "ok" and row.get("model") == model and row.get("prompt_sha256") == prompt_sha else None


def parse_error_observer(stats: Counter):
    """Count malformed/schema outputs that were retried by the shared chat helper."""
    lock = threading.Lock()
    def observe(error: Exception) -> None:
        if not isinstance(error, (ValueError, json.JSONDecodeError)):
            return
        with lock:
            stats["parse_failures"] += 1
            if isinstance(error, json.JSONDecodeError):
                stats["malformed_json"] += 1
    return observe


def run_base_claims(contexts: Dict[str, Dict[str, Any]], out: Path, model: str, workers: int, retries: int) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    out.mkdir(parents=True, exist_ok=True); results, pending, stats = {}, [], Counter()
    for qid, context in contexts.items():
        if not context["candidates"]: continue
        prompt = build_base_claim_prompt(context["question"], context["base_memories"]); sha = hashlib.sha256(prompt.encode()).hexdigest(); path = out / f"{cache_key(qid)}.json"; row = _load(path, model, sha)
        if row: results[qid] = row["claims"]; stats["base_claim_cache_hits"] += 1
        else: pending.append((qid, context, prompt, sha, path))
    observer = parse_error_observer(stats)
    def one(task):
        qid, context, prompt, sha, path = task
        claims, raw = chat(prompt, model, 2400, lambda value: parse_base_claims(value, [m["memory_id"] for m in context["base_memories"]]), retries, observer)
        atomic_write_json(path, {"status":"ok", "stage":"base_claim_extraction", "model":model, "question_id":qid, "prompt_sha256":sha, "claims":claims, "raw_output":raw, "gold_answer_in_prompt":False})
        return qid, claims
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for future in concurrent.futures.as_completed([ex.submit(one, task) for task in pending]):
            qid, claims = future.result(); results[qid] = claims; stats["base_claim_extraction_calls"] += 1; print(f"Base claims: {len(results)}/{len(results)+len(pending)-stats['base_claim_extraction_calls']}", flush=True)
    return results, dict(stats)


def run_candidate_claims(contexts: Dict[str, Dict[str, Any]], out: Path, model: str, workers: int, retries: int) -> Tuple[Dict[Tuple[str,str], Dict[str, Any]], Dict[str, int]]:
    out.mkdir(parents=True, exist_ok=True); results, pending, stats = {}, [], Counter()
    for qid, context in contexts.items():
        for candidate in context["candidates"]:
            prompt = build_candidate_claim_prompt(context["question"], candidate["content"]); sha = hashlib.sha256(prompt.encode()).hexdigest(); path = out / f"{cache_key(qid, candidate['candidate_id'])}.json"; row = _load(path, model, sha)
            key = (qid, candidate["candidate_id"])
            if row: results[key] = row["claim"]; stats["candidate_claim_cache_hits"] += 1
            else: pending.append((qid, candidate, prompt, sha, path))
    observer = parse_error_observer(stats)
    def one(task):
        qid, candidate, prompt, sha, path = task
        claim, raw = chat(prompt, model, 400, parse_candidate_claim, retries, observer)
        atomic_write_json(path, {"status":"ok", "stage":"candidate_claim_extraction", "model":model, "question_id":qid, "candidate_id":candidate["candidate_id"], "prompt_sha256":sha, "claim":claim, "raw_output":raw, "gold_answer_in_prompt":False})
        return (qid, candidate["candidate_id"]), claim
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for future in concurrent.futures.as_completed([ex.submit(one, task) for task in pending]):
            key, claim = future.result(); results[key] = claim; stats["candidate_claim_extraction_calls"] += 1; print(f"Candidate claims: {len(results)}/{len(results)+len(pending)-stats['candidate_claim_extraction_calls']}", flush=True)
    return results, dict(stats)


def run_alignment(contexts: Dict[str, Dict[str, Any]], gaps: Dict[str, Dict[str, Any]], base_claims: Dict[str, List[Dict[str, Any]]], candidate_claims: Dict[Tuple[str,str], Dict[str, Any]], out: Path, model: str, workers: int, retries: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    out.mkdir(parents=True, exist_ok=True); rows, pending, stats = [], [], Counter()
    for qid, context in contexts.items():
        for candidate in context["candidates"]:
            claim = candidate_claims[(qid, candidate["candidate_id"])]; prompt = build_alignment_prompt(context["question"], gaps[qid]["residual"], base_claims[qid], claim); sha = hashlib.sha256(prompt.encode()).hexdigest(); path = out / f"{cache_key(qid, candidate['candidate_id'])}.json"; row = _load(path, model, sha)
            if row: rows.append(row); stats["alignment_cache_hits"] += 1
            else: pending.append((context, candidate, claim, base_claims[qid], prompt, sha, path))
    observer = parse_error_observer(stats)
    def one(task):
        context, candidate, candidate_claim, claims, prompt, sha, path = task
        decision, raw = chat(prompt, model, 500, lambda value: parse_alignment(value, [claim["memory_id"] for claim in claims]), retries, observer)
        by_id = {claim["memory_id"]: claim for claim in claims}; matched = by_id.get(decision["matched_base_memory_id"])
        row = {"status":"ok", "stage":"claim_alignment", "model":model, "question_id":context["question_id"], "question_type":context["question_type"], "candidate_id":candidate["candidate_id"], "candidate_text":candidate["content"], "candidate_answer_relevant":candidate_claim["answer_relevant"], "candidate_slot":candidate_claim["slot"], "candidate_value":candidate_claim["value"], "predicted_label":decision["label"], "oracle_label":candidate["oracle_label"], "matched_base_memory_id":decision["matched_base_memory_id"], "matched_base_slot":matched["slot"] if matched else None, "matched_base_value":matched["value"] if matched else None, "reason":decision["reason"], "prompt_sha256":sha, "raw_output":raw, "gold_answer_in_prompt":False}
        atomic_write_json(path, row); return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for future in concurrent.futures.as_completed([ex.submit(one, task) for task in pending]):
            rows.append(future.result()); stats["alignment_calls"] += 1; print(f"Alignment: {len(rows)}/{len(rows)+len(pending)-stats['alignment_calls']}", flush=True)
    rows.sort(key=lambda row:(row["question_id"], row["candidate_id"])); return rows, dict(stats)


def _tokens(text: str | None) -> set[str]: return {word.lower() for word in re.findall(r"[A-Za-z0-9]+", text or "") if len(word) > 2}

def error_type(row: Dict[str, Any]) -> str:
    oracle, pred = row["oracle_label"], row["predicted_label"]
    if oracle == pred: return "correct"
    if not row["candidate_answer_relevant"]: return "claim extraction error"
    if pred == "REDUNDANT":
        if (row["candidate_slot"] or "").lower() != (row["matched_base_slot"] or "").lower(): return "same topic but different fact"
        overlap = len(_tokens(row["candidate_value"]) & _tokens(row["matched_base_value"])) / max(1, len(_tokens(row["candidate_value"])))
        if overlap < .4: return "same slot but different value"
        if row["question_type"] in {"temporal-reasoning", "knowledge-update"}: return "temporal/update fact mismatch"
        return "alignment error"
    if oracle == "REDUNDANT": return "alignment error"
    return "claim extraction error" if not row["candidate_slot"] else "alignment error"

def error_exports(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    mapping = {"oracle_supporting_to_predicted_redundant":("SUPPORTING","REDUNDANT"), "oracle_irrelevant_to_predicted_redundant":("IRRELEVANT","REDUNDANT"), "oracle_redundant_to_predicted_irrelevant":("REDUNDANT","IRRELEVANT"), "oracle_redundant_to_predicted_supporting":("REDUNDANT","SUPPORTING")}
    return {name:[{**row, "suspected_error_type":error_type(row)} for row in rows if row["oracle_label"] == pair[0] and row["predicted_label"] == pair[1]] for name,pair in mapping.items()}

def comparison(report: Dict[str, Any], current: Dict[str, Any]) -> Dict[str,float]:
    old=report["classification_metrics"]
    return {f"{label.lower()}_{metric}_delta":current["per_class"][label][metric]-old["per_class"][label][metric] for label in CLASSES for metric in ("precision","recall","f1")} | {"macro_f1_delta":current["macro_f1"]-old["macro_f1"], "irrelevant_to_supporting_rate_delta":irrelevant_to_supporting_rate(current)-irrelevant_to_supporting_rate(old)}

def main() -> None:
    p=argparse.ArgumentParser(description="V5 Question-Conditioned Fact Alignment Utility Scorer")
    for flag in ("data-path","mem-dir","oracle-v05","v2-report","v3-report","v4-report","v2-gap-dir","output-dir"): p.add_argument(f"--{flag}",required=True)
    p.add_argument("--model",default=os.environ.get("HEBBIAN_UTILITY_MODEL") or model_for("generation")); p.add_argument("--workers",type=int,default=8); p.add_argument("--retries",type=int,default=3); args=p.parse_args()
    if not os.environ.get("OPENAI_BASE_URL"): raise SystemExit("OPENAI_BASE_URL is required")
    dataset=json.loads(Path(args.data_path).read_text(encoding="utf-8")); _,v05=load_v05(Path(args.oracle_v05)); contexts=build_contexts(dataset,v05,Path(args.mem_dir)); gaps=load_v2_gap_cache(contexts,Path(args.v2_gap_dir),args.model); out=Path(args.output_dir)
    base,base_cost=run_base_claims(contexts,out/"base_claim_items",args.model,args.workers,args.retries)
    candidate,candidate_cost=run_candidate_claims(contexts,out/"candidate_claim_items",args.model,args.workers,args.retries)
    rows,alignment_cost=run_alignment(contexts,gaps,base,candidate,out/"alignment_items",args.model,args.workers,args.retries)
    metrics=classification_metrics(rows); errors=error_exports(rows); costs={**base_cost,**candidate_cost,**alignment_cost}; total=len(rows); costs["cache_hits"]=sum(value for key,value in costs.items() if key.endswith("cache_hits")); costs["malformed_json"]=costs.get("malformed_json",0); costs["parse_failures"]=costs.get("parse_failures",0); costs["avg_llm_calls_per_candidate"]=(costs.get("base_claim_extraction_calls",0)+costs.get("candidate_claim_extraction_calls",0)+costs.get("alignment_calls",0))/total if total else 0.0
    prior={name:json.loads(Path(path).read_text(encoding="utf-8")) for name,path in (("v2",args.v2_report),("v3",args.v3_report),("v4",args.v4_report))}
    report={"method":"V5 Question-Conditioned Fact Alignment Utility Scorer","gold_answer_used_by_stage1":False,"gold_answer_used_by_stage2":False,"oracle_labels_used_only_for_evaluation":True,"answer_generation_executed":False,"benchmark_judge_executed":False,"graph_mutation_executed":False,"retrieval_replay_executed":False,"model":args.model,"temperature":0.0,"stage1_reused_from_v2":True,"gap_prompt_template":GAP_PROMPT_TEMPLATE,"base_claim_prompt_template":BASE_CLAIM_PROMPT,"candidate_claim_prompt_template":CANDIDATE_CLAIM_PROMPT,"alignment_prompt_template":ALIGNMENT_PROMPT,"classification_metrics":metrics,"comparison_with_v2":comparison(prior["v2"],metrics),"comparison_with_v3":comparison(prior["v3"],metrics),"comparison_with_v4":comparison(prior["v4"],metrics),"call_cost":costs,"error_cases":errors,"match_outputs":rows}
    out.mkdir(parents=True,exist_ok=True); atomic_write_json(out/"question_conditioned_fact_alignment_v5_report.json",report); atomic_write_json(out/"match_predictions.json",rows); atomic_write_json(out/"error_cases.json",errors)
    print("classification_metric\tvalue");
    for key in ("overall_accuracy","macro_f1","false_supporting_count","supporting_to_redundant","supporting_to_irrelevant","irrelevant_to_supporting"): print(f"{key}\t{metrics[key]}")
    for label in CLASSES:
        for metric in ("precision","recall","f1"): print(f"{label}_{metric}\t{metrics['per_class'][label][metric]}")
    print("confusion_matrix_rows_oracle_columns_prediction"); print("oracle\\predicted\t"+"\t".join(CLASSES))
    for label in CLASSES: print(label+"\t"+"\t".join(str(metrics["confusion_matrix"]["counts"][label][pred]) for pred in CLASSES))
    print("call_cost\t"+json.dumps(costs,ensure_ascii=False)); print("error_case_counts\t"+json.dumps({key:len(value) for key,value in errors.items()},ensure_ascii=False)); print(f"report\t{out/'question_conditioned_fact_alignment_v5_report.json'}")

if __name__=="__main__": main()
