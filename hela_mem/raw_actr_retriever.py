"""Read-only raw episodic retrieval with ACT-R-inspired cue competition."""

from __future__ import annotations

import json
import math
import re
from statistics import median
from typing import Any, Callable, Mapping

import numpy as np

from .runtime import chat_extra_body, llm_scope, model_for, record_llm_request, record_llm_usage, strip_reasoning
from .utils import _create_client, get_embedding, normalize_vector


CUE_PROMPT = """You extract relation-aware retrieval cues from a question.

QUESTION:
{query}

Return valid JSON with this schema:
{{
  "cues": [
    {{"type": "relation|event|temporal_relation|time|attribute", "text": "a complete relational phrase"}}
  ]
}}

Rules:
- Each cue must be a complete phrase expressing an event, relation, attribute, or temporal constraint useful for locating answer-bearing memories.
- Do not output isolated entities, isolated locations, isolated dates, or single keywords.
- Do not score cue importance, rewrite the question, infer the answer, or use external knowledge.
- Return at most 8 cues. Return JSON only.
"""


def _parse_cues(raw: str, query: str) -> list[dict[str, str]]:
    text = strip_reasoning(raw)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    payload = json.loads(text)
    rows = payload.get("cues")
    if not isinstance(rows, list):
        raise ValueError("cue response has no cues list")
    cues = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        cue_type = str(row.get("type", "relation")).strip() or "relation"
        cue_text = str(row.get("text", "")).strip()
        tokens = re.findall(r"[A-Za-z0-9]+", cue_text)
        isolated_ascii_value = bool(tokens) and len(tokens) == 1 and re.fullmatch(r"[A-Za-z0-9 .'-]+", cue_text)
        if cue_text and not isolated_ascii_value:
            cues.append({"type": cue_type, "text": cue_text})
    if not cues:
        raise ValueError("cue response contains no usable cues")
    return cues


def extract_relational_cues(query: str) -> list[dict[str, str]]:
    """Extract relation-level cues with Qwen; fall back to the whole question."""
    prompt = CUE_PROMPT.format(query=query)
    messages = [
        {"role": "system", "content": "You are a deterministic structured retrieval-cue parser."},
        {"role": "user", "content": prompt},
    ]
    try:
        client = _create_client()
        with llm_scope("other_calls"):
            record_llm_request()
            response = client.chat.completions.create(
                model=model_for("generation"),
                messages=messages,
                temperature=0,
                max_tokens=500,
                **chat_extra_body(),
            )
            record_llm_usage(response)
        if not response or not response.choices:
            raise ValueError("empty cue response")
        return _parse_cues(response.choices[0].message.content or "", query)
    except Exception as error:
        print(f"ACT-R cue extraction fallback to whole query: {type(error).__name__}: {error}")
        return [{"type": "query_fallback", "text": query}]


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


class RawACTRRetriever:
    """Retrieve from ``memory_graph.nodes`` without reading or mutating edges."""

    def __init__(
        self,
        memory_graph: Any,
        cue_extractor: Callable[[str], list[dict[str, str]]] = extract_relational_cues,
        embedding_fn: Callable[[str], Any] = get_embedding,
    ) -> None:
        self._nodes: Mapping[str, Mapping[str, Any]] = memory_graph.nodes
        self._cue_extractor = cue_extractor
        self._embedding_fn = embedding_fn
        self.last_diagnostic: dict[str, Any] | None = None

    def retrieve(
        self,
        query: str,
        mode: str = "raw",
        candidate_k: int = 30,
        top_k: int = 15,
        fan_threshold: float = 0.5,
        score_mode: str = "actr",
        alpha: float = 0.5,
        question_id: str = "",
    ) -> list[dict[str, Any]]:
        if mode not in {"raw", "cue_idf", "actr"}:
            raise ValueError("mode must be raw, cue_idf, or actr")
        if score_mode not in {"actr", "hybrid"}:
            raise ValueError("score_mode must be actr or hybrid")
        if candidate_k <= 0 or top_k <= 0:
            raise ValueError("candidate_k and top_k must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("actr alpha must satisfy 0 <= alpha <= 1")
        if not -1.0 <= fan_threshold <= 1.0:
            raise ValueError("fan threshold must be a cosine value in [-1, 1]")

        node_ids = [str(node_id) for node_id in self._nodes]
        if not node_ids:
            self.last_diagnostic = self._empty_diagnostic(
                question_id, query, mode, candidate_k, top_k, fan_threshold, score_mode, alpha
            )
            return []
        memory_matrix = np.stack([
            normalize_vector(self._nodes[node_id]["embedding"]) for node_id in node_ids
        ])
        query_vec = normalize_vector(self._embedding_fn(query))
        raw_scores = memory_matrix @ query_vec
        raw_order = sorted(range(len(node_ids)), key=lambda index: (-float(raw_scores[index]), node_ids[index]))
        candidate_indices = raw_order[: min(candidate_k, len(raw_order))]
        candidate_ids = [node_ids[index] for index in candidate_indices]

        # Raw is deliberately a clean whole-query cosine baseline and makes no
        # cue-extraction LLM call.
        cues = [] if mode == "raw" else self._cue_extractor(query)
        cue_rows = []
        candidate_associations = np.zeros((len(cues), len(candidate_indices)), dtype=float)
        actr_activation = np.zeros(len(candidate_indices), dtype=float)
        idf_score = np.zeros(len(candidate_indices), dtype=float)
        bank_size = len(node_ids)
        source_activation = 1.0 / len(cues) if cues else 0.0
        s_max = math.log(bank_size + 1)
        for cue_index, cue in enumerate(cues):
            cue_vec = normalize_vector(self._embedding_fn(cue["text"]))
            all_associations = memory_matrix @ cue_vec
            fan = max(1, int(np.count_nonzero(all_associations >= fan_threshold)))
            fan_strength = s_max - math.log(fan)
            candidate_values = all_associations[candidate_indices]
            gates = (candidate_values >= fan_threshold).astype(float)
            candidate_associations[cue_index] = candidate_values
            # Semantic approximation of ACT-R associative-strength fan
            # dilution under a finite, uniformly divided source activation.
            actr_activation += source_activation * gates * fan_strength
            idf_score += candidate_values * fan_strength
            cue_rows.append({
                "type": cue.get("type", "relation"),
                "text": cue["text"],
                "fan": fan,
                "source_activation": source_activation,
                "fan_strength": fan_strength,
            })

        candidate_raw = raw_scores[candidate_indices].astype(float)
        if mode == "raw":
            final_scores = candidate_raw.copy()
        elif mode == "cue_idf":
            final_scores = idf_score.copy()
        elif score_mode == "actr":
            final_scores = actr_activation.copy()
        else:
            final_scores = alpha * _minmax(candidate_raw) + (1.0 - alpha) * _minmax(actr_activation)

        final_order = sorted(
            range(len(candidate_ids)),
            key=lambda index: (-float(final_scores[index]), candidate_indices[index], candidate_ids[index]),
        )
        selected_positions = final_order[: min(top_k, len(final_order))]
        raw_selected = set(range(min(top_k, len(candidate_ids))))
        final_selected = set(selected_positions)
        actr_order = sorted(
            range(len(candidate_ids)),
            key=lambda index: (-float(actr_activation[index]), candidate_indices[index], candidate_ids[index]),
        )
        actr_rank = {position: rank for rank, position in enumerate(actr_order, start=1)}
        final_rank = {position: rank for rank, position in enumerate(final_order, start=1)}
        candidates = []
        for position, node_id in enumerate(candidate_ids):
            node = self._nodes[node_id]
            candidates.append({
                "node_id": node_id,
                "content": str(node.get("content", "")),
                "base_score": float(candidate_raw[position]),
                "actr_activation": float(actr_activation[position]),
                "cue_idf_score": float(idf_score[position]),
                "final_score": float(final_scores[position]),
                "base_rank": position + 1,
                "actr_rank": actr_rank[position],
                "final_rank": final_rank[position],
                "rank_delta": (position + 1) - final_rank[position],
                "selected_raw": position in raw_selected,
                "selected_actr": position in final_selected,
                "cue_matches": [
                    {
                        "cue": cue_rows[cue_index]["text"],
                        "similarity": float(candidate_associations[cue_index, position]),
                        "gate": int(candidate_associations[cue_index, position] >= fan_threshold),
                    }
                    for cue_index in range(len(cue_rows))
                ],
            })
        self.last_diagnostic = {
            "question_id": question_id,
            "retrieval_mode": mode,
            "question": query,
            "candidate_k": candidate_k,
            "final_k": top_k,
            "fan_threshold": fan_threshold,
            "score_mode": score_mode,
            "alpha": alpha,
            "memory_bank_size": bank_size,
            "coarse_candidate_ids": candidate_ids,
            "fan_computed_over_memory_count": bank_size,
            "cues": cue_rows,
            "candidates": candidates,
        }
        results = []
        for position in selected_positions:
            node_id = candidate_ids[position]
            results.append({
                "node": self._nodes[node_id],
                "score": float(final_scores[position]),
                "base_score": float(candidate_raw[position]),
                "actr_activation": float(actr_activation[position]),
                "source": mode,
                "flipped_by_spreading": False,
            })
        return results

    @staticmethod
    def _empty_diagnostic(question_id, query, mode, candidate_k, top_k, threshold, score_mode, alpha):
        return {
            "question_id": question_id, "retrieval_mode": mode, "question": query,
            "candidate_k": candidate_k, "final_k": top_k, "fan_threshold": threshold,
            "score_mode": score_mode, "alpha": alpha, "memory_bank_size": 0,
            "coarse_candidate_ids": [], "fan_computed_over_memory_count": 0,
            "cues": [], "candidates": [],
        }


def summarize_fans(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    cues = [cue for diagnostic in diagnostics for cue in diagnostic.get("cues", [])]
    fans = [int(cue["fan"]) for cue in cues]
    question_count = len(diagnostics)
    histogram = {str(value): fans.count(value) for value in sorted(set(fans))}
    high = 0
    low = 0
    for diagnostic in diagnostics:
        size = max(1, int(diagnostic.get("memory_bank_size", 0)))
        for cue in diagnostic.get("cues", []):
            ratio = cue["fan"] / size
            high += ratio >= 0.5
            low += ratio <= 0.1
    return {
        "question_count": question_count,
        "cue_count": len(cues),
        "average_cue_count": len(cues) / question_count if question_count else 0.0,
        "average_fan": sum(fans) / len(fans) if fans else 0.0,
        "median_fan": median(fans) if fans else 0.0,
        "min_fan": min(fans, default=0),
        "max_fan": max(fans, default=0),
        "high_fan_definition": "fan / memory_bank_size >= 0.5",
        "low_fan_definition": "fan / memory_bank_size <= 0.1",
        "high_fan_cue_rate": high / len(cues) if cues else 0.0,
        "low_fan_cue_rate": low / len(cues) if cues else 0.0,
        "fan_histogram": histogram,
    }
