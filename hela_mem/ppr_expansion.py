"""Minimal Personalized PageRank expansion over an encoded Hebbian graph."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np


def personalized_pagerank(
    node_ids: Sequence[str],
    edges: Mapping[str, Mapping[str, Any]],
    personalization: Mapping[str, float],
    damping: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Compute PPR with positive Hebbian weights and personalized dangling mass."""
    if not 0.0 <= damping < 1.0:
        raise ValueError("ppr damping must satisfy 0 <= damping < 1")
    if max_iter <= 0 or tol <= 0:
        raise ValueError("ppr max_iter and tol must be positive")
    ids = [str(node_id) for node_id in node_ids]
    if not ids:
        return {"scores": {}, "iterations": 0, "converged": True, "residual": 0.0}
    index = {node_id: position for position, node_id in enumerate(ids)}
    p = np.asarray([max(0.0, float(personalization.get(node_id, 0.0))) for node_id in ids], dtype=float)
    total = float(p.sum())
    if total <= 0:
        raise ValueError("PPR personalization has no positive mass")
    p /= total

    transitions: list[list[tuple[int, float]]] = []
    dangling = []
    for source_id in ids:
        neighbors = [
            (index[str(target_id)], float(weight))
            for target_id, weight in edges.get(source_id, {}).items()
            if str(target_id) in index and str(target_id) != source_id and float(weight) > 0
        ]
        row_total = sum(weight for _, weight in neighbors)
        if row_total > 0:
            transitions.append([(target, weight / row_total) for target, weight in neighbors])
            dangling.append(False)
        else:
            transitions.append([])
            dangling.append(True)

    rank = p.copy()
    residual = float("inf")
    for iteration in range(1, max_iter + 1):
        propagated = np.zeros_like(rank)
        dangling_mass = 0.0
        for source, row in enumerate(transitions):
            if dangling[source]:
                dangling_mass += float(rank[source])
                continue
            for target, probability in row:
                propagated[target] += rank[source] * probability
        # Standard personalized dangling handling keeps P stochastic.
        propagated += dangling_mass * p
        updated = (1.0 - damping) * p + damping * propagated
        residual = float(np.abs(updated - rank).sum())
        rank = updated
        if residual <= tol:
            return {
                "scores": {node_id: float(rank[position]) for position, node_id in enumerate(ids)},
                "iterations": iteration,
                "converged": True,
                "residual": residual,
            }
    return {
        "scores": {node_id: float(rank[position]) for position, node_id in enumerate(ids)},
        "iterations": max_iter,
        "converged": False,
        "residual": residual,
    }


def strongest_shortest_paths(
    node_ids: Sequence[str],
    edges: Mapping[str, Mapping[str, Any]],
    seed_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Record hop distance and strongest predecessor within a shortest seed path."""
    valid = {str(node_id) for node_id in node_ids}
    seeds = [str(node_id) for node_id in seed_ids if str(node_id) in valid]
    incoming: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in valid}
    for source_id, neighbors in edges.items():
        source = str(source_id)
        if source not in valid:
            continue
        for target_id, raw_weight in neighbors.items():
            target = str(target_id)
            weight = float(raw_weight)
            if target in valid and target != source and weight > 0:
                incoming[target].append((source, weight))
    distance = {seed_id: 0 for seed_id in seeds}
    queue = deque(seeds)
    while queue:
        source = queue.popleft()
        for target, raw_weight in edges.get(source, {}).items():
            target_id = str(target)
            if target_id not in valid or target_id == source or float(raw_weight) <= 0:
                continue
            if target_id not in distance:
                distance[target_id] = distance[source] + 1
                queue.append(target_id)

    predecessor: dict[str, str | None] = {seed_id: None for seed_id in seeds}
    for target_id, hop in sorted(distance.items(), key=lambda item: (item[1], item[0])):
        if hop == 0:
            continue
        choices = [
            (weight, source_id)
            for source_id, weight in incoming[target_id]
            if distance.get(source_id) == hop - 1
        ]
        predecessor[target_id] = max(choices, key=lambda item: (item[0], item[1]))[1] if choices else None

    result = {}
    for node_id, hop in distance.items():
        path = [node_id]
        cursor = node_id
        seen = {cursor}
        while predecessor.get(cursor) is not None:
            cursor = str(predecessor[cursor])
            if cursor in seen:
                break
            seen.add(cursor)
            path.append(cursor)
        path.reverse()
        result[node_id] = {
            "hop_distance": hop,
            "strongest_predecessor": predecessor.get(node_id),
            "hebbian_path": path,
        }
    return result


def ppr_associative_candidates(
    node_ids: Sequence[str],
    edges: Mapping[str, Mapping[str, Any]],
    base_ids: Sequence[str],
    base_scores: Mapping[str, float],
    top_n: int,
    damping: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Generate non-Base candidates; PPR scores are not final retrieval scores."""
    if top_n < 0:
        raise ValueError("ppr top_n must be non-negative")
    base = [str(node_id) for node_id in base_ids]
    base_set = set(base)
    personalization = {node_id: max(0.0, float(base_scores.get(node_id, 0.0))) for node_id in base}
    ppr = personalized_pagerank(node_ids, edges, personalization, damping, max_iter, tol)
    paths = strongest_shortest_paths(node_ids, edges, base)
    ranked = sorted(
        (
            (node_id, score) for node_id, score in ppr["scores"].items()
            if node_id not in base_set and score > 0.0 and node_id in paths
        ),
        key=lambda item: (-item[1], item[0]),
    )[:top_n]
    candidates = []
    for node_id, score in ranked:
        candidates.append({"node_id": node_id, "ppr_score": score, **paths[node_id]})
    positive = sum(personalization.values())
    seeds = [
        {
            "node_id": node_id,
            "base_score": float(base_scores.get(node_id, 0.0)),
            "personalization": personalization[node_id] / positive if positive > 0 else 0.0,
        }
        for node_id in base
    ]
    return {**ppr, "base_seeds": seeds, "candidates": candidates}
