"""
LongMemEval Evaluation for Hebbian Memory

Loads encoded Hebbian memory graphs and evaluates on LongMemEval-S questions.
Uses GPT-4o-mini as judge with type-specific prompts (same as LightMem).

Usage:
    python -m hela_mem.eval_longmemeval \
        --data_path /path/to/longmemeval_s.json \
        --mem_dir results/longmemeval_mem_XXXX \
        [--num_items 500] [--start_item 0] [--top_k 20] \
        [--use_consolidation] [--workers 10]
"""

import json
import os
import argparse
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .hebbian_memory import HebbianMemoryGraph
from .hebbian_retriever import HebbianRetriever
from .hebbian_knowledge_memory import HebbianKnowledgeMemory
from .utils import (
    gpt_generate_answer_with_rotation,
)
from .runtime import append_jsonl, atomic_write_json, config_fingerprint, llm_scope, model_for, read_valid_json, sha256_file, usage_snapshot


# ========== GPT Judge (from LongMemEval / LightMem) ==========

# Corrupted sample indices from LightMem paper (treated as incorrect)
CORRUPTED_INDICES = {74, 183, 278, 351, 380}


def get_anscheck_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool = False
) -> str:
    """
    Build GPT judge prompt for LongMemEval evaluation.
    Each question type has a tailored evaluation template.
    Directly adapted from LongMemEval official evaluation.
    """
    if not abstention:
        if task in ("single-session-user", "single-session-assistant", "multi-session"):
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. "
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. "
                "In addition, do not penalize off-by-one errors for the number of days. "
                "If the question asks for the number of days/weeks/months, etc., and the model makes "
                "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response "
                "is still correct. "
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, "
                "the response should be considered as correct as long as the updated answer is the "
                "required answer."
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        elif task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response "
                "from a model. Please answer yes if the response satisfies the desired response. "
                "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
                "The response is correct as long as it recalls and utilizes the user's personal "
                "information correctly."
                "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        else:
            # Fallback for unknown types
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no."
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
    else:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is "
            "given but the asked information is not."
            "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )

    return template.format(question, answer, response)


def parse_judge_response(response: Optional[str]) -> bool:
    """Parse GPT judge yes/no response into boolean."""
    if response is None:
        return False
    normalized = str(response).strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = first_line.replace(".", "").replace("!", "").replace(":", "").replace(";", "").split()
    if not tokens:
        return False
    head = tokens[0]
    if head in ("yes", "y"):
        return True
    if head in ("no", "n"):
        return False
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return False


def compact_retrieval(results: list) -> list:
    """Keep evidence useful for analysis without duplicating embedding vectors."""
    compact = []
    for result in results:
        node = result.get("node", {})
        compact.append({
            "node_id": node.get("id"), "content": node.get("content", ""),
            "timestamp": node.get("timestamp", ""), "score": result.get("score"),
            "base_score": result.get("base_score"), "source": result.get("source"),
            "flipped_by_spreading": result.get("flipped_by_spreading", False),
        })
    return compact


# ========== Consolidation (same as NarrativeQA) ==========

CONSOLIDATION_THRESHOLD = int(os.environ.get("HEBBIAN_CONSOLIDATION_THRESHOLD", "3"))
MAX_CLUSTERS = int(os.environ.get("HEBBIAN_MAX_CLUSTERS", "20"))
MAX_CLUSTER_SIZE = int(os.environ.get("HEBBIAN_MAX_CLUSTER_SIZE", "8"))


def consolidate_memory(
    memory_graph: HebbianMemoryGraph,
    knowledge_memory: HebbianKnowledgeMemory,
) -> int:
    """
    Hebbian-driven consolidation: Episodic -> Semantic Memory.
    1. Build similarity edges to enrich graph structure
    2. Find hub nodes (high-degree)
    3. Collect clusters, use LLM to extract key facts
    4. Store facts in HebbianKnowledgeMemory

    Returns number of facts extracted.
    """
    import numpy as np

    nodes = memory_graph.nodes
    edges = memory_graph.edges

    # Step 0: Build similarity edges
    SIMILARITY_K = 5
    node_ids = list(nodes.keys())
    if len(node_ids) < 3:
        return 0

    embeddings = {}
    for nid in node_ids:
        emb = nodes[nid].get("embedding")
        if emb is not None:
            embeddings[nid] = np.array(emb)

    if len(embeddings) >= 3:
        emb_ids = list(embeddings.keys())
        emb_matrix = np.array([embeddings[nid] for nid in emb_ids])
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = emb_matrix / norms
        sim_matrix = normed @ normed.T

        edges_added = 0
        for i, nid in enumerate(emb_ids):
            sims = sim_matrix[i].copy()
            sims[i] = -1
            top_indices = np.argsort(sims)[-SIMILARITY_K:]
            for j in top_indices:
                if sims[j] > 0.5:
                    other_id = emb_ids[j]
                    if other_id not in edges.get(nid, {}):
                        memory_graph.add_edge(nid, other_id, weight=sims[j] * 0.3, bidirectional=True)
                        edges_added += 1
        print(f"    [Consolidation] Built {edges_added} similarity edges")

    # Step 1: Find hub nodes
    hub_candidates = []
    for node_id in nodes:
        degree = len(edges.get(node_id, {}))
        total_weight = sum(edges.get(node_id, {}).values())
        if degree >= CONSOLIDATION_THRESHOLD:
            hub_candidates.append((node_id, degree, total_weight))

    if not hub_candidates:
        return 0

    hub_candidates.sort(key=lambda x: x[1] * x[2], reverse=True)
    hub_candidates = hub_candidates[:MAX_CLUSTERS]

    processed = set()
    total_facts = 0

    for hub_id, degree, total_weight in hub_candidates:
        if hub_id in processed:
            continue

        neighbor_edges = edges.get(hub_id, {})
        sorted_neighbors = sorted(neighbor_edges.items(), key=lambda x: x[1], reverse=True)

        cluster_ids = [hub_id]
        for nid, w in sorted_neighbors[: MAX_CLUSTER_SIZE - 1]:
            if nid not in processed and nid in nodes:
                cluster_ids.append(nid)

        processed.update(cluster_ids)

        cluster_texts = []
        for cid in cluster_ids:
            content = nodes[cid].get("content", "")
            if content:
                cluster_texts.append(content[:1500])

        if len(cluster_texts) < 2:
            continue

        # Step 2: LLM extracts key facts
        combined = "\n\n---\n\n".join(cluster_texts)
        prompt = (
            "You are a knowledge extraction engine.\n"
            "Below are several related conversation passages.\n"
            "Extract the key facts as a list. Each fact should be a single, "
            "self-contained sentence covering people, events, preferences, "
            "dates, or important details.\n"
            "Output one fact per line, prefixed with '- '.\n"
            "Extract 3-8 facts. Output ONLY the fact list.\n\n"
            f"Passages:\n{combined}\n\nFacts:"
        )
        messages = [
            {"role": "system", "content": "Extract key facts from conversation passages."},
            {"role": "user", "content": prompt},
        ]

        try:
            with llm_scope("consolidation_calls"):
                result = gpt_generate_answer_with_rotation(prompt, messages, role="extraction")
            if not result:
                continue
            for line in result.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    line = line[2:].strip()
                if len(line) > 10 and line.lower() != "none":
                    knowledge_memory.add_knowledge(line)
                    total_facts += 1
        except Exception as e:
            print(f"    [Consolidation] Error: {e}")
            continue

    memory_graph.save()
    knowledge_memory.save()
    print(f"    [Consolidation] Extracted {total_facts} facts -> Semantic Memory")
    return total_facts


# ========== Answer Generation ==========

def build_longmemeval_prompt(
    context_text: str,
    knowledge_text: str,
    profile_text: str,
    assistant_knowledge_text: str,
    question: str,
    question_date: str,
) -> Tuple[str, str]:
    """
    Build system prompt and user prompt for LongMemEval QA.
    Adapted from HebbianRetriever.answer() with conciseness and format
    instructions aligned with the original internal experiment prompt settings.

    Returns:
        (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are a helpful assistant with access to the user's conversation history. "
        "Your task is to answer questions about the user or past conversations "
        "in an extremely concise manner.\n"
        "When the question is: \"What did the charity race raise awareness for?\", "
        "you should not answer in the form of: \"The charity race raised awareness "
        "for mental health.\" Instead, it should be: \"mental health\", as this is "
        "more concise."
    )
    if assistant_knowledge_text:
        system_prompt += f"\n{assistant_knowledge_text}"

    user_prompt = (
        f"<CONTEXT>\n"
        f"Current date: {question_date}\n\n"
        f"Relevant memories from conversation history:\n"
        f"{context_text}\n\n"
    )

    if knowledge_text:
        user_prompt += f"<KNOWLEDGE BASE>\n{knowledge_text}\n\n"

    if profile_text and profile_text != "None":
        user_prompt += f"<CHARACTER TRAITS>\nCharacteristics of the user:\n{profile_text}\n\n"

    user_prompt += (
        f"Question: {question}\n"
        f"Please only provide the content of the answer, without including 'answer:'\n"
        f"For questions that require answering a date or time, strictly follow the "
        f"format \"15 July 2023\" and provide a specific date whenever possible. "
        f"For example, if you need to answer \"last year,\" give the specific year "
        f"rather than just saying \"last year.\" Only provide one year, date, or time, "
        f"without any extra responses.\n"
        f"If the question is about the duration, answer in the form of several years, "
        f"months, or days.\n"
        f"Generate answers primarily composed of concrete entities."
    )

    return system_prompt, user_prompt


def answer_question(
    retriever: HebbianRetriever,
    question: str,
    question_date: str,
    top_k: int = 20,
    knowledge_memory: Optional[HebbianKnowledgeMemory] = None,
    semantic_top_k: int = 5,
    item_id: str = "",
) -> Tuple[str, list, list]:
    """
    Answer a single LongMemEval question using Hebbian retrieval.

    Returns:
        (answer_text, retrieved_results)
    """
    # 1. Episodic retrieval
    results = retriever.graph.retrieve(question, top_k=top_k)

    # Build episodic context
    context_blocks = []
    for res in results[:top_k]:
        node = res["node"]
        score = res["score"]
        source_label = "Direct Match" if res.get("base_score", 0) > 0.6 else "Associative Memory"
        block = (
            f"[{source_label} | Relevancy: {score:.2f}]\n"
            f"Time: {node.get('timestamp', 'unknown')}\n"
            f"Content: {node['content']}"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    # 2. Semantic retrieval from Knowledge Memory
    knowledge_text = ""
    kb_results = []
    if knowledge_memory:
        try:
            kb_results = knowledge_memory.search_knowledge(question, top_k=semantic_top_k)
            if kb_results:
                kb_lines = [f"- {kn['knowledge']}" for kn in kb_results]
                knowledge_text = "\n".join(kb_lines)
        except Exception as e:
            print(f"    [{item_id}] Semantic retrieval error: {e}")

    # 3. Get user profile
    profile_text = "None"
    if knowledge_memory and hasattr(knowledge_memory, "get_raw_user_profile"):
        profile_text = knowledge_memory.get_raw_user_profile(item_id) or "None"

    # 4. Get assistant knowledge
    assistant_knowledge_text = ""
    if knowledge_memory and hasattr(knowledge_memory, "get_assistant_knowledge"):
        try:
            ak_list = knowledge_memory.get_assistant_knowledge()
            if ak_list:
                assistant_knowledge_text = "Here are some of your character traits and knowledge:\n"
                for ak_item in ak_list:
                    k_text = ak_item["knowledge"].strip()
                    if k_text:
                        assistant_knowledge_text += f"- {k_text}\n"
        except Exception as e:
            print(f"    [{item_id}] Assistant knowledge error: {e}")

    # 5. Build prompt and generate answer
    system_prompt, user_prompt = build_longmemeval_prompt(
        context_text, knowledge_text, profile_text, assistant_knowledge_text,
        question, question_date,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with llm_scope("answer_generation_calls"):
        response = gpt_generate_answer_with_rotation(user_prompt, messages, role="generation")
    return response, results, kb_results


# ========== Single Item Evaluation ==========

def evaluate_single_item(
    item: Dict[str, Any],
    item_idx: int,
    mem_dir: str,
    top_k: int = 20,
    semantic_top_k: int = 5,
    use_consolidation: bool = False,
    results_dir: Optional[str] = None,
    fingerprint: str = "",
) -> Dict[str, Any]:
    """
    Evaluate a single LongMemEval item.

    Steps:
        1. Load encoded Hebbian memory graph
        2. Optionally run consolidation
        3. Retrieve + generate answer
        4. Judge with GPT-4o-mini
        5. Save per-item result

    Returns:
        Dict with question_id, question_type, correct, generated_answer, etc.
    """
    item_id = item["question_id"]
    question_type = item["question_type"]
    question = item["question"]
    answer = item["answer"]
    question_date = item.get("question_date", "")
    is_abstention = "abs" in item_id

    t_start = time.time()

    # Check for corrupted samples
    if item_idx in CORRUPTED_INDICES:
        print(f"  [{item_idx}] {item_id} - CORRUPTED (skipped, marked incorrect)")
        result = {
            "question_id": item_id,
            "question_type": question_type,
            "question": question,
            "correct": 0,
            "prediction": "[CORRUPTED]",
            "generated_answer": "[CORRUPTED]",
            "gold_answer": str(answer),
            "ground_truth": str(answer),
            "judge_result": "corrupted_index",
            "retrieved_episodic": [],
            "retrieved_semantic": [],
            "model": model_for("generation"),
            "judge_model": model_for("judge"),
            "status": "ok",
            "config_fingerprint": fingerprint,
            "is_corrupted": True,
            "eval_time": 0.0,
        }
        if results_dir:
            atomic_write_json(os.path.join(results_dir, f"result_{item_id}.json"), result)
        return result

    # Load memory graph
    mem_path = os.path.join(mem_dir, f"{item_id}_hebbian.json")
    if not os.path.exists(mem_path):
        raise RuntimeError(f"encoding_failure: memory file not found: {mem_path}")

    memory_graph = HebbianMemoryGraph(file_path=mem_path)
    if not memory_graph.nodes:
        raise RuntimeError("encoding_failure: empty memory graph")

    # Load knowledge memory
    kb_path = os.path.join(mem_dir, f"{item_id}_long_term.json")
    knowledge_memory = HebbianKnowledgeMemory(file_path=kb_path)

    # Optional consolidation
    if use_consolidation:
        consolidate_memory(memory_graph, knowledge_memory)

    # Create retriever
    retriever = HebbianRetriever(memory_graph, profile_memory=knowledge_memory)

    # Generate answer
    try:
        retrieval_started = time.perf_counter()
        with llm_scope("other_calls", item_id):
            generated_answer, retrieved, semantic_retrieved = answer_question(
                retriever, question, question_date, top_k=top_k,
                knowledge_memory=knowledge_memory, semantic_top_k=semantic_top_k,
                item_id=item_id,
            )
        generation_seconds = time.perf_counter() - retrieval_started
    except Exception as e:
        raise RuntimeError(f"generation_failure: {e}") from e

    # Judge with GPT
    try:
        judge_prompt = get_anscheck_prompt(
            question_type, question, str(answer), generated_answer,
            abstention=is_abstention,
        )
        judge_messages = [{"role": "user", "content": judge_prompt}]
        judge_started = time.perf_counter()
        with llm_scope("judge_calls", item_id):
            judge_response = gpt_generate_answer_with_rotation(judge_prompt, judge_messages, role="judge")
        correct = 1 if parse_judge_response(judge_response) else 0
        judge_seconds = time.perf_counter() - judge_started
    except Exception as e:
        raise RuntimeError(f"judge_failure: {e}") from e

    eval_time = time.time() - t_start

    result = {
        "question_id": item_id,
        "question_type": question_type,
        "question": question,
        "gold_answer": str(answer),
        "ground_truth": str(answer),
        "prediction": generated_answer,
        "generated_answer": generated_answer,
        "judge_result": judge_response,
        "correct": correct,
        "is_abstention": is_abstention,
        "num_nodes": len(memory_graph.nodes),
        "num_retrieved": len(retrieved),
        "retrieved_episodic": compact_retrieval(retrieved),
        "retrieved_semantic": [{key: value for key, value in entry.items() if key != "knowledge_embedding"} for entry in semantic_retrieved],
        "redundancy_inhibition_trace": memory_graph.last_retrieval_trace,
        "model": model_for("generation"),
        "judge_model": model_for("judge"),
        "status": "ok",
        "config_fingerprint": fingerprint,
        "generation_seconds": generation_seconds,
        "judge_seconds": judge_seconds,
        "llm_usage": usage_snapshot(item_id),
        "eval_time": eval_time,
    }

    # Persist retrieval-induced Hebbian updates before marking evaluation complete.
    memory_graph.save()

    # Save per-item result last; this is the resume completion marker.
    if results_dir:
        result_path = os.path.join(results_dir, f"result_{item_id}.json")
        atomic_write_json(result_path, result)

    status = "CORRECT" if correct else "WRONG"
    print(f"  [{item_idx}] {item_id} ({question_type}) -> {status} ({eval_time:.1f}s)")

    return result


# ========== Main Evaluation ==========

def eval_longmemeval(
    data_path: str,
    mem_dir: str,
    num_items: Optional[int] = None,
    start_item: int = 0,
    top_k: int = 20,
    semantic_top_k: int = 5,
    use_consolidation: bool = False,
    workers: int = 20,
    results_dir: Optional[str] = None,
    resume: bool = True,
) -> None:
    """
    Run LongMemEval-S evaluation on encoded Hebbian memories.

    Args:
        data_path: Path to longmemeval_s.json
        mem_dir: Directory with encoded memory files
        num_items: Number of items to evaluate (default: all)
        start_item: Start index
        top_k: Top-K episodic retrieval
        semantic_top_k: Top-K semantic retrieval
        use_consolidation: Whether to run consolidation before eval
        workers: Number of parallel workers
    """
    print("=" * 70)
    print("LongMemEval Hebbian Evaluation")
    print("=" * 70)
    print(f"Data: {data_path}")
    print(f"Memory dir: {mem_dir}")
    print(f"top_k={top_k}, semantic_top_k={semantic_top_k}")
    print(f"Consolidation: {'ON' if use_consolidation else 'OFF'}")
    print(f"Workers: {workers}")
    print(f"Hebbian params: max_flipped={os.environ.get('HEBBIAN_MAX_FLIPPED', '5')}, "
          f"lr={os.environ.get('HEBBIAN_LEARNING_RATE', '0.02')}, "
          f"alpha={os.environ.get('HEBBIAN_ACTIVATION_ALPHA', '0.1')}, "
          f"preselection_pool={os.environ.get('HEBBIAN_USE_PRESELECTION_POOL', 'false')}, "
          f"redundancy_inhibition={os.environ.get('HEBBIAN_USE_REDUNDANCY_INHIBITION', 'false')}, "
          f"gamma={os.environ.get('HEBBIAN_INHIBITION_GAMMA', '0.2')}")
    print("=" * 70)

    # Load dataset
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} items")

    # Select items
    items = dataset[start_item:]
    if num_items is not None:
        items = items[:num_items]
    print(f"Evaluating items [{start_item}:{start_item + len(items)}]")

    # Create results directory
    results_dir = results_dir or os.path.join(mem_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    # Evaluate items in parallel
    all_results = []
    pending = []
    fingerprint = config_fingerprint({
        "dataset_sha256": sha256_file(data_path),
        "generation_model": model_for("generation"),
        "judge_model": model_for("judge"),
        "top_k": top_k,
        "semantic_top_k": semantic_top_k,
        "use_consolidation": use_consolidation,
        "use_preselection_pool": os.environ.get("HEBBIAN_USE_PRESELECTION_POOL", "false").lower() == "true",
        "use_redundancy_inhibition": os.environ.get("HEBBIAN_USE_REDUNDANCY_INHIBITION", "false").lower() == "true",
        "inhibition_gamma": os.environ.get("HEBBIAN_INHIBITION_GAMMA", "0.2"),
    })
    for i, item in enumerate(items):
        result_path = os.path.join(results_dir, f"result_{item['question_id']}.json")
        cached = read_valid_json(result_path, ("question_id", "status", "prediction", "judge_result")) if resume else None
        if cached and cached["status"] == "ok" and cached.get("config_fingerprint") == fingerprint:
            all_results.append(cached)
        else:
            pending.append((i, item))
    print(f"Evaluation resume: {len(all_results)} complete, {len(pending)} pending")
    errors_path = Path(results_dir).parent / "errors.jsonl"
    errors_path.touch(exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                evaluate_single_item,
                item,
                start_item + i,
                mem_dir,
                top_k,
                semantic_top_k,
                use_consolidation,
                results_dir,
                fingerprint,
            ): start_item + i
            for i, item in pending
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                all_results.append(result)
                if len(all_results) % 50 == 0:
                    correct_so_far = sum(r["correct"] for r in all_results)
                    print(f"\n  Progress: {len(all_results)}/{len(items)} | "
                          f"Accuracy: {correct_so_far}/{len(all_results)} "
                          f"({correct_so_far / len(all_results) * 100:.1f}%)\n")
            except Exception as e:
                print(f"  [ERROR] Item {idx}: {e}")
                failed_item = items[idx - start_item] if start_item <= idx < start_item + len(items) else {}
                message = str(e)
                stage = message.split(":", 1)[0] if "_failure:" in message else "evaluation_failure"
                append_jsonl(errors_path, {"timestamp": datetime.now().isoformat(), "stage": stage, "question_id": failed_item.get("question_id", "unknown"), "error_type": type(e).__name__, "message": message})

    # Sort results by question_id for consistency
    all_results.sort(key=lambda x: x["question_id"])

    # Compute metrics
    total = len(all_results)
    total_correct = sum(r["correct"] for r in all_results)
    overall_accuracy = total_correct / total * 100 if total > 0 else 0.0

    # Per-type metrics
    type_metrics = {}
    for r in all_results:
        qt = r["question_type"]
        if qt not in type_metrics:
            type_metrics[qt] = {"correct": 0, "total": 0}
        type_metrics[qt]["total"] += 1
        type_metrics[qt]["correct"] += r["correct"]

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"Total items: {total}")
    print(f"Overall accuracy: {total_correct}/{total} ({overall_accuracy:.2f}%)")
    print(f"\nPer-type accuracy:")
    for qt in sorted(type_metrics.keys()):
        m = type_metrics[qt]
        acc = m["correct"] / m["total"] * 100 if m["total"] > 0 else 0.0
        print(f"  {qt:30s}: {m['correct']:3d}/{m['total']:3d} ({acc:.2f}%)")
    print(f"{'=' * 70}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "mem_dir": mem_dir,
        "total_items": total,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "per_type": {
            qt: {
                "correct": m["correct"],
                "total": m["total"],
                "accuracy": m["correct"] / m["total"] * 100 if m["total"] > 0 else 0.0,
            }
            for qt, m in type_metrics.items()
        },
        "params": {
            "top_k": top_k,
            "semantic_top_k": semantic_top_k,
            "use_consolidation": use_consolidation,
            "max_flipped": os.environ.get("HEBBIAN_MAX_FLIPPED", "5"),
            "learning_rate": os.environ.get("HEBBIAN_LEARNING_RATE", "0.02"),
            "activation_alpha": os.environ.get("HEBBIAN_ACTIVATION_ALPHA", "0.1"),
            "spreading_threshold": os.environ.get("HEBBIAN_SPREADING_THRESHOLD", "0.4"),
            "decay_rate": os.environ.get("HEBBIAN_DECAY_RATE", "0.995"),
            "keyword_weight": os.environ.get("HEBBIAN_KEYWORD_WEIGHT", "0.5"),
            "tau": os.environ.get("HEBBIAN_TAU", "1e7"),
            "use_preselection_pool": os.environ.get("HEBBIAN_USE_PRESELECTION_POOL", "false").lower() == "true",
            "use_redundancy_inhibition": os.environ.get("HEBBIAN_USE_REDUNDANCY_INHIBITION", "false").lower() == "true",
            "inhibition_gamma": os.environ.get("HEBBIAN_INHIBITION_GAMMA", "0.2"),
            "generation_temperature": os.environ.get("HEBBIAN_GENERATION_TEMPERATURE", os.environ.get("HEBBIAN_TEMPERATURE", "0.7")),
            "extraction_temperature": os.environ.get("HEBBIAN_EXTRACTION_TEMPERATURE", os.environ.get("HEBBIAN_TEMPERATURE", "0.7")),
            "judge_temperature": os.environ.get("HEBBIAN_JUDGE_TEMPERATURE", os.environ.get("HEBBIAN_TEMPERATURE", "0.7")),
            "generation_model": model_for("generation"),
            "judge_model": model_for("judge"),
        },
        "results": all_results,
    }

    summary_path = os.path.join(results_dir, "eval_summary.json")
    atomic_write_json(summary_path, summary)
    predictions_path = Path(results_dir).parent / "predictions.jsonl"
    predictions_path.unlink(missing_ok=True)
    for result in all_results:
        append_jsonl(predictions_path, result)
    atomic_write_json(Path(results_dir).parent / "metrics.json", {key: summary[key] for key in ("total_items", "total_correct", "overall_accuracy", "per_type")})
    run_dir = Path(results_dir).parent
    eval_usage = usage_snapshot()
    atomic_write_json(run_dir / "eval_llm_usage.json", eval_usage)
    encode_usage = read_valid_json(run_dir / "encode_llm_usage.json") or {}
    atomic_write_json(run_dir / "llm_usage.json", {"encode": encode_usage, "eval": eval_usage})
    old_timing = read_valid_json(run_dir / "timing.json") or {}
    eval_seconds = sum(float(result.get("eval_time", 0.0)) for result in all_results)
    atomic_write_json(run_dir / "timing.json", old_timing | {"eval_time": eval_seconds, "total_time": float(old_timing.get("encode_time", 0.0)) + eval_seconds, "avg_eval_time_per_item": eval_seconds / max(len(all_results), 1)})
    print(f"Summary saved: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LongMemEval-S with Hebbian Memory"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to longmemeval_s.json",
    )
    parser.add_argument(
        "--mem_dir", type=str, required=True,
        help="Memory directory (output of encode_longmemeval)",
    )
    parser.add_argument("--num_items", type=int, default=None, help="Number of items")
    parser.add_argument("--start_item", type=int, default=0, help="Start index")
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="Top-K episodic retrieval (default: from HEBBIAN_TOP_K env or 20)",
    )
    parser.add_argument(
        "--semantic_top_k", type=int, default=5,
        help="Top-K semantic retrieval (default: 5)",
    )
    parser.add_argument(
        "--use_consolidation", action="store_true",
        help="Run consolidation before evaluation",
    )
    parser.add_argument("--workers", "--concurrency", dest="workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--use-redundancy-inhibition", action="store_true")
    parser.add_argument("--use-preselection-pool", action="store_true")
    parser.add_argument("--inhibition-gamma", type=float, default=0.2)

    args = parser.parse_args()

    top_k = args.top_k or int(os.environ.get("HEBBIAN_TOP_K", "20"))
    os.environ["HEBBIAN_USE_REDUNDANCY_INHIBITION"] = str(args.use_redundancy_inhibition).lower()
    os.environ["HEBBIAN_USE_PRESELECTION_POOL"] = str(args.use_preselection_pool).lower()
    os.environ["HEBBIAN_INHIBITION_GAMMA"] = str(args.inhibition_gamma)

    eval_longmemeval(
        data_path=args.data_path,
        mem_dir=args.mem_dir,
        num_items=args.num_items,
        start_item=args.start_item,
        top_k=top_k,
        semantic_top_k=args.semantic_top_k,
        use_consolidation=args.use_consolidation,
        workers=args.workers,
        results_dir=args.results_dir,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
