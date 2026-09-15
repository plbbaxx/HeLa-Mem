"""Qwen3-Reranker-0.6B Base-conditioned utility ranking experiment.

This is deliberately separate from the historical MemReranker-4B trainer.
It consumes the frozen V1.1 utility dataset and eps=0.05 preferences verbatim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .train_memreranker_utility import (
    OFFICIAL_PREFIX,
    OFFICIAL_SUFFIX,
    UTILITY_INSTRUCTION,
    atomic_json,
    load_dataset,
    ndcg,
    percentile,
    resolve_yes_no_token_ids,
    spearman,
    split_pairs,
)


BASE_BLOCK = re.compile(r"(?=^\[(?:Direct Match|Associative Memory) \| Relevancy:)", re.MULTILINE)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_max_length(raw_lengths: Sequence[int]) -> tuple[int, str]:
    """Apply the frozen length policy without assuming 16K."""
    if not raw_lengths:
        raise ValueError("cannot choose max_length from an empty dataset")
    share_4096 = sum(value <= 4096 for value in raw_lengths) / len(raw_lengths)
    p95 = percentile(raw_lengths, .95)
    # "Most" is made conservative: at least 90%, rather than a bare majority.
    if share_4096 >= .90:
        return 4096, "at_least_90_percent_at_or_below_4096"
    if p95 <= 8192:
        return 8192, "p95_at_or_below_8192"
    return 16384, "p95_above_8192_use_16384_cap"


def split_base_blocks(context: str) -> list[str]:
    blocks = [piece.strip() for piece in BASE_BLOCK.split(context.strip()) if piece.strip()]
    return blocks or ([context.strip()] if context.strip() else [])


@dataclass(frozen=True)
class EncodedUtilityExample:
    input_ids: list[int]
    raw_token_count: int
    actual_token_count: int
    truncated_token_count: int
    truncation_ratio: float
    base_blocks_total: int
    base_blocks_kept: int
    question_preserved: bool
    candidate_preserved: bool


class BaseTailTruncatingEncoder:
    """Keep Question and Candidate intact; remove Base memories from the tail."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = tokenizer.encode(OFFICIAL_PREFIX, add_special_tokens=False)
        self.suffix = tokenizer.encode(OFFICIAL_SUFFIX, add_special_tokens=False)

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode(self, row: dict[str, Any], base_conditioned: bool) -> EncodedUtilityExample:
        question = str(row["question"]).strip()
        candidate = str(row["candidate_text"]).strip()
        if base_conditioned:
            head = (
                f"<Instruct>: {UTILITY_INSTRUCTION}\n<Query>: Question:\n{question}"
                "\n\nCurrent Base Memories:\n"
            )
            blocks = split_base_blocks(str(row.get("actual_baseline_context") or ""))
        else:
            head = f"<Instruct>: {UTILITY_INSTRUCTION}\n<Query>: {question}"
            blocks = []
        document = f"\n<Document>: {candidate}"
        head_tokens, document_tokens = self._tokens(head), self._tokens(document)
        block_tokens = [self._tokens(("" if index == 0 else "\n\n") + block) for index, block in enumerate(blocks)]
        fixed = self.prefix + head_tokens
        tail = document_tokens + self.suffix
        raw_token_count = len(fixed) + sum(map(len, block_tokens)) + len(tail)
        if len(fixed) + len(tail) > self.max_length:
            raise RuntimeError(
                "Question plus Candidate exceeds max_length; refusing to violate the preservation invariant "
                f"(required={len(fixed)+len(tail)}, max_length={self.max_length})"
            )
        remaining = self.max_length - len(fixed) - len(tail)
        kept: list[list[int]] = []
        for tokens in block_tokens:
            if len(tokens) > remaining:
                break
            kept.append(tokens)
            remaining -= len(tokens)
        input_ids = fixed + [token for block in kept for token in block] + tail
        truncated = raw_token_count - len(input_ids)
        return EncodedUtilityExample(
            input_ids=input_ids,
            raw_token_count=raw_token_count,
            actual_token_count=len(input_ids),
            truncated_token_count=truncated,
            truncation_ratio=truncated / raw_token_count if raw_token_count else 0.0,
            base_blocks_total=len(blocks),
            base_blocks_kept=len(kept),
            question_preserved=True,
            candidate_preserved=True,
        )


class FullQuestionBalancedSampler:
    """Draw len(pairs) samples, choosing question uniformly before pair."""

    def __init__(self, pairs: Sequence[dict[str, Any]], seed: int, epoch: int) -> None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, pair in enumerate(pairs):
            grouped[str(pair["question_id"])].append(index)
        if not grouped:
            raise ValueError("no training pair is available")
        self.grouped = dict(grouped)
        self.num_samples = len(pairs)
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        questions = list(self.grouped)
        emitted = 0
        while emitted < self.num_samples:
            rng.shuffle(questions)
            for question_id in questions:
                if emitted >= self.num_samples:
                    return
                yield rng.choice(self.grouped[question_id])
                emitted += 1


class PairDataset:
    def __init__(self, pairs: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], encoder: BaseTailTruncatingEncoder) -> None:
        self.pairs = list(pairs)
        self.rows = {(str(row["question_id"]), str(row["candidate_memory_id"])): row for row in rows}
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        qid = str(pair["question_id"])
        preferred = self.rows[(qid, str(pair["preferred_candidate_id"]))]
        rejected = self.rows[(qid, str(pair["rejected_candidate_id"]))]
        if preferred["question"] != rejected["question"]:
            raise RuntimeError(f"pair {qid} does not share one frozen Question")
        if preferred.get("actual_baseline_context") != rejected.get("actual_baseline_context"):
            raise RuntimeError(f"pair {qid} does not share one frozen Base Top-15 context")
        return {
            "preferred": self.encoder.encode(preferred, True).input_ids,
            "rejected": self.encoder.encode(rejected, True).input_ids,
            "question_id": qid,
        }


class DynamicPairCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch
        sequences = [item["preferred"] for item in batch] + [item["rejected"] for item in batch]
        output = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        if "attention_mask" not in output:
            output["attention_mask"] = (output["input_ids"] != self.tokenizer.pad_token_id).long()
        if not bool(output["attention_mask"][:, -1].all()):
            raise RuntimeError("last tensor position is not the final valid scoring token")
        output["pair_batch_size"] = torch.tensor(len(batch))
        output["nonpad_tokens"] = output["attention_mask"].sum()
        output["question_ids"] = [str(item["question_id"]) for item in batch]
        return output


def length_statistics(rows: Sequence[dict[str, Any]], tokenizer: Any) -> tuple[dict[str, Any], int]:
    raw_encoder = BaseTailTruncatingEncoder(tokenizer, 1_000_000_000)
    raw = [raw_encoder.encode(row, True).raw_token_count for row in rows]
    selected, reason = choose_max_length(raw)
    minimum_required = []
    for row in rows:
        without_base = dict(row)
        without_base["actual_baseline_context"] = ""
        minimum_required.append(raw_encoder.encode(without_base, True).raw_token_count)
    largest_minimum = max(minimum_required, default=0)
    if largest_minimum > selected:
        safe_caps = [cap for cap in (4096, 8192, 16384) if cap >= largest_minimum]
        if not safe_caps:
            raise RuntimeError(
                f"Question plus Candidate needs {largest_minimum} tokens, exceeding the 16384 safety cap"
            )
        selected = safe_caps[0]
        reason += f";raised_to_preserve_question_and_candidate_minimum_{largest_minimum}"
    encoder = BaseTailTruncatingEncoder(tokenizer, selected)
    encoded = [encoder.encode(row, True) for row in rows]
    details = []
    for row, item in zip(rows, encoded):
        details.append({
            "question_id": str(row["question_id"]),
            "candidate_memory_id": str(row["candidate_memory_id"]),
            **asdict(item),
        })
        details[-1].pop("input_ids")
    report = {
        "protocol": "qwen3_reranker_0_6b_token_length_v1",
        "candidate_count": len(raw),
        "p50": percentile(raw, .50),
        "p90": percentile(raw, .90),
        "p95": percentile(raw, .95),
        "p99": percentile(raw, .99),
        "max": max(raw, default=0),
        "share_over_4096": sum(value > 4096 for value in raw) / len(raw),
        "share_over_8192": sum(value > 8192 for value in raw) / len(raw),
        "share_over_16384": sum(value > 16384 for value in raw) / len(raw),
        "selected_max_length": selected,
        "selection_reason": reason,
        "maximum_question_candidate_required_tokens": largest_minimum,
        "truncated_sample_count": sum(item.truncated_token_count > 0 for item in encoded),
        "mean_truncation_ratio": statistics.mean(item.truncation_ratio for item in encoded),
        "max_truncation_ratio": max((item.truncation_ratio for item in encoded), default=0.0),
        "invariants": {"question_always_preserved": True, "candidate_always_preserved": True, "base_tail_only": True},
        "samples": details,
    }
    return report, selected


def inspect_local_model(model_path: Path, tokenizer: Any, yes_id: int, no_id: int) -> dict[str, Any]:
    inspected: dict[str, Any] = {}
    for name in ("config.json", "tokenizer_config.json"):
        path = model_path / name
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            inspected[name] = {
                key: value.get(key) for key in (
                    "model_type", "architectures", "max_position_embeddings", "torch_dtype", "chat_template"
                ) if key in value
            }
    for name in ("chat_template.jinja", "README.md"):
        path = model_path / name
        inspected[name] = {"exists": path.exists(), "path": str(path)}
    return {
        "model_path": str(model_path),
        "local_checkpoint_bytes": sum(
            path.stat().st_size for path in model_path.rglob("*") if path.is_file()
        ),
        "assets": inspected,
        "yes_token": "yes",
        "yes_token_id": yes_id,
        "no_token": "no",
        "no_token_id": no_id,
        "token_ids_resolved_by_tokenizer": True,
        "final_valid_token_rule": "left padding; score logits at tensor position -1 after official suffix",
        "score": "yes_logit_minus_no_logit",
        "official_prefix": OFFICIAL_PREFIX,
        "official_suffix": OFFICIAL_SUFFIX,
        "instruction": UTILITY_INSTRUCTION,
        "tokenizer_class": type(tokenizer).__name__,
    }


def score_batch(model: Any, batch: dict[str, Any], yes_id: int, no_id: int) -> Any:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, -1, :]
    return logits[:, yes_id].float() - logits[:, no_id].float()


def reciprocal_rank(scores: Sequence[float], utilities: Sequence[float]) -> float | None:
    if not scores:
        return None
    best = max(utilities)
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    return 1.0 / (next(rank for rank, index in enumerate(order, 1) if utilities[index] == best))


def evaluate(
    model: Any,
    tokenizer: Any,
    encoder: BaseTailTruncatingEncoder,
    rows: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    question_ids: set[str],
    yes_id: int,
    no_id: int,
    base_conditioned: bool,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    selected_rows = [row for row in rows if str(row["question_id"]) in question_ids]
    score_map: dict[tuple[str, str], float] = {}
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(selected_rows), batch_size):
            chunk = selected_rows[start:start + batch_size]
            sequences = [encoder.encode(row, base_conditioned).input_ids for row in chunk]
            batch = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = score_batch(model, batch, yes_id, no_id).cpu().tolist()
            for row, score in zip(chunk, scores):
                score_map[(str(row["question_id"]), str(row["candidate_memory_id"]))] = float(score)
    selected_pairs = split_pairs(pairs, question_ids)
    margins = []
    for pair in selected_pairs:
        qid = str(pair["question_id"])
        margins.append(
            score_map[(qid, str(pair["preferred_candidate_id"]))]
            - score_map[(qid, str(pair["rejected_candidate_id"]))]
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[str(row["question_id"])].append(row)
    correlations, ndcgs, mrrs = [], [], []
    for qid, group in grouped.items():
        scores = [score_map[(qid, str(row["candidate_memory_id"]))] for row in group]
        utilities = [float(row["utility_score"]) for row in group]
        correlation, ranking, rr = spearman(scores, utilities), ndcg(scores, utilities), reciprocal_rank(scores, utilities)
        if correlation is not None:
            correlations.append(correlation)
        if ranking is not None:
            ndcgs.append(ranking)
        if rr is not None:
            mrrs.append(rr)
    return {
        "pair_count": len(margins),
        "pair_question_count": len({str(pair["question_id"]) for pair in selected_pairs}),
        "candidate_count": len(selected_rows),
        "pairwise_accuracy": sum(value > 0 for value in margins) / len(margins) if margins else None,
        "tie_count": sum(value == 0 for value in margins),
        "mean_pair_margin": statistics.mean(margins) if margins else None,
        "mean_question_spearman": statistics.mean(correlations) if correlations else None,
        "mean_question_ndcg": statistics.mean(ndcgs) if ndcgs else None,
        "mrr_of_max_utility_candidate": statistics.mean(mrrs) if mrrs else None,
        "scored_question_count": len(grouped),
    }


def load_tokenizer(model_path: str) -> tuple[Any, int, int]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_id, no_id = resolve_yes_no_token_ids(tokenizer)
    return tokenizer, yes_id, no_id


def load_base_model(model_path: str, attention: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if attention != "auto":
        kwargs["attn_implementation"] = attention
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs).cuda()


def attach_lora(model: Any, args: argparse.Namespace) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model.config.use_cache = False
    return get_peft_model(model, config)


def profile_training_options(
    model: Any,
    tokenizer: Any,
    dataset: PairDataset,
    yes_id: int,
    no_id: int,
    requested_batch_size: int,
) -> dict[str, Any]:
    """Pick the largest safe batch and prefer checkpointing OFF."""
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, Subset

    longest = sorted(range(len(dataset)), key=lambda index: max(
        len(dataset[index]["preferred"]), len(dataset[index]["rejected"])
    ), reverse=True)
    candidates = [requested_batch_size] if requested_batch_size > 0 else [4, 2, 1]
    attempts = []
    collator = DynamicPairCollator(tokenizer)
    for checkpointing in (False, True):
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        else:
            model.gradient_checkpointing_disable()
        for batch_size in candidates:
            indexes = longest[:batch_size]
            if len(indexes) < batch_size:
                continue
            loader = DataLoader(Subset(dataset, indexes), batch_size=batch_size, collate_fn=collator)
            try:
                batch = next(iter(loader))
                pair_batch = int(batch.pop("pair_batch_size"))
                token_count = int(batch.pop("nonpad_tokens"))
                batch.pop("question_ids")
                batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    scores = score_batch(model, batch, yes_id, no_id)
                    loss = -functional.logsigmoid(scores[:pair_batch] - scores[pair_batch:]).mean()
                loss.backward()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated()
                model.zero_grad(set_to_none=True)
                result = {
                    "status": "fit",
                    "batch_size_pairs": batch_size,
                    "sequence_count": batch_size * 2,
                    "gradient_checkpointing": checkpointing,
                    "seconds_per_step": elapsed,
                    "nonpad_tokens": token_count,
                    "tokens_per_second": token_count / elapsed,
                    "peak_gpu_memory_bytes": peak,
                    "peak_gpu_memory_gib": peak / (1024 ** 3),
                }
                attempts.append(result)
                return {"selected": result, "attempts": attempts}
            except torch.cuda.OutOfMemoryError as error:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                attempts.append({
                    "status": "oom", "batch_size_pairs": batch_size,
                    "gradient_checkpointing": checkpointing, "error": str(error)[:500],
                })
    raise RuntimeError(f"profiling found no safe training batch: {attempts}")


def run_dev_baselines(
    model: Any, tokenizer: Any, encoder: BaseTailTruncatingEncoder, rows: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]], split_ids: dict[str, set[str]], yes_id: int, no_id: int, batch_size: int,
) -> dict[str, Any]:
    return {
        "protocol": "qwen3_reranker_0_6b_utility_baselines_v1",
        "score": "yes_logit_minus_no_logit",
        "test_sealed": True,
        "methods": {
            "original_qwen3_reranker_0_6b": {
                "dev": evaluate(model, tokenizer, encoder, rows, pairs, split_ids["dev"], yes_id, no_id, False, batch_size)
            },
            "base_conditioned_zero_shot_qwen3_reranker_0_6b": {
                "dev": evaluate(model, tokenizer, encoder, rows, pairs, split_ids["dev"], yes_id, no_id, True, batch_size)
            },
        },
    }


def train(
    model: Any, tokenizer: Any, encoder: BaseTailTruncatingEncoder, rows: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]], split_ids: dict[str, set[str]], yes_id: int, no_id: int,
    output: Path, args: argparse.Namespace,
) -> tuple[dict[str, Any], Any]:
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    train_pairs = split_pairs(pairs, split_ids["train"])
    dev_pairs = split_pairs(pairs, split_ids["dev"])
    question_count = len({str(pair["question_id"]) for pair in train_pairs})
    if question_count != 41:
        raise RuntimeError(f"frozen eps005 train population must have 41 pair-bearing questions, got {question_count}")
    model = attach_lora(model, args)
    dataset = PairDataset(train_pairs, rows, encoder)
    profile = profile_training_options(model, tokenizer, dataset, yes_id, no_id, args.train_batch_size)
    selected = profile["selected"]
    batch_size = int(selected["batch_size_pairs"])
    checkpointing = bool(selected["gradient_checkpointing"])
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()
    collator = DynamicPairCollator(tokenizer)
    batches_per_epoch = math.ceil(len(train_pairs) / batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_updates * args.warmup_ratio)),
        num_training_steps=total_updates,
    )
    train_log = output / "train_log.jsonl"
    if train_log.exists():
        raise RuntimeError(f"training output already exists; use a new output directory: {train_log}")
    dev_history, runtime_steps = [], []
    best_accuracy, best_epoch = -1.0, None
    best = output / "best_checkpoint"
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        sampler = FullQuestionBalancedSampler(train_pairs, args.seed, epoch)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collator, num_workers=0)
        losses = []
        model.train()
        torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(loader, 1):
            pair_batch = int(batch.pop("pair_batch_size"))
            nonpad_tokens = int(batch.pop("nonpad_tokens"))
            sampled_question_ids = batch.pop("question_ids")
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = score_batch(model, batch, yes_id, no_id)
                loss = -functional.logsigmoid(scores[:pair_batch] - scores[pair_batch:]).mean()
                scaled = loss / args.gradient_accumulation_steps
            scaled.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            losses.append(float(loss.detach().cpu()))
            record = {
                "epoch": epoch, "step": step, "steps_in_epoch": len(loader),
                "sampled_pair_count": pair_batch, "loss": losses[-1],
                "sampled_question_ids": sampled_question_ids,
                "learning_rate": scheduler.get_last_lr()[0], "seconds": elapsed,
                "nonpad_tokens": nonpad_tokens, "tokens_per_second": nonpad_tokens / elapsed,
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
            }
            runtime_steps.append(record)
            append_jsonl(train_log, record)
            if step == 1 or step == len(loader) or step % 20 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} {step}/{len(loader)} "
                    f"loss={record['loss']:.6f} sec={elapsed:.2f} tok/s={record['tokens_per_second']:.1f}",
                    flush=True,
                )
        dev = evaluate(model, tokenizer, encoder, rows, dev_pairs, split_ids["dev"], yes_id, no_id, True, args.eval_batch_size)
        epoch_row = {"epoch": epoch, "train_loss": statistics.mean(losses), **dev}
        dev_history.append(epoch_row)
        atomic_json(output / "dev_metrics_by_epoch.json", dev_history)
        accuracy = float(dev["pairwise_accuracy"] or 0.0)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            model.save_pretrained(best)
            tokenizer.save_pretrained(best)
            atomic_json(best / "selection.json", {
                "selected_epoch": epoch,
                "primary_metric": "dev_pairwise_accuracy",
                "dev_pairwise_accuracy": accuracy,
                "dev_spearman": dev["mean_question_spearman"],
                "dev_ndcg": dev["mean_question_ndcg"],
            })
    runtime = {
        "profiling": profile,
        "training": {
            "step_count": len(runtime_steps),
            "mean_seconds_per_step": statistics.mean(row["seconds"] for row in runtime_steps),
            "median_seconds_per_step": statistics.median(row["seconds"] for row in runtime_steps),
            "mean_tokens_per_second": statistics.mean(row["tokens_per_second"] for row in runtime_steps),
            "peak_gpu_memory_gib": max(row["peak_gpu_memory_gib"] for row in runtime_steps),
        },
    }
    atomic_json(output / "runtime_stats.json", runtime)
    summary = {
        "train_pair_count": len(train_pairs),
        "train_pair_question_count": question_count,
        "samples_per_epoch": len(train_pairs),
        "sampling": "uniform_question_then_random_pair",
        "batch_size_pairs": batch_size,
        "gradient_checkpointing": checkpointing,
        "best_epoch": best_epoch,
        "best_dev_pairwise_accuracy": best_accuracy,
        "best_checkpoint": str(best),
    }
    return summary, model


def frozen_test(
    base_model: Any, tokenizer: Any, encoder: BaseTailTruncatingEncoder, rows: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]], split_ids: dict[str, set[str]], yes_id: int, no_id: int,
    best: Path, batch_size: int,
) -> dict[str, Any]:
    from peft import PeftModel
    original = evaluate(base_model, tokenizer, encoder, rows, pairs, split_ids["test"], yes_id, no_id, False, batch_size)
    conditioned = evaluate(base_model, tokenizer, encoder, rows, pairs, split_ids["test"], yes_id, no_id, True, batch_size)
    tuned = PeftModel.from_pretrained(base_model, best).eval()
    trained = evaluate(tuned, tokenizer, encoder, rows, pairs, split_ids["test"], yes_id, no_id, True, batch_size)
    return {
        "protocol": "qwen3_reranker_0_6b_frozen_test_v1",
        "checkpoint_selection": json.loads((best / "selection.json").read_text(encoding="utf-8")),
        "methods": {
            "original_qwen3_reranker_0_6b": original,
            "base_conditioned_zero_shot_qwen3_reranker_0_6b": conditioned,
            "base_conditioned_lora_utility_ranker": trained,
        },
    }


def cost_comparison(output: Path, model: Any, historical: Path) -> dict[str, Any]:
    current_runtime_path = output / "runtime_stats.json"
    current_baseline_path = output / "baseline_metrics.json"
    best_selection = output / "best_checkpoint" / "selection.json"
    current = {
        "model": "Qwen3-Reranker-0.6B",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "runtime": json.loads(current_runtime_path.read_text(encoding="utf-8")) if current_runtime_path.exists() else None,
        "baseline": json.loads(current_baseline_path.read_text(encoding="utf-8")) if current_baseline_path.exists() else None,
        "trained_dev_best": json.loads(best_selection.read_text(encoding="utf-8")) if best_selection.exists() else None,
    }
    historical_baseline = historical / "baseline_results.json"
    historical_runtime = historical / "runtime_stats.json"
    previous = {
        "model": "MemReranker-4B",
        "parameter_count": None,
        "local_checkpoint_bytes": sum(
            path.stat().st_size for path in Path("/mnt/disk2/caoxue/models/MemReranker-4B").rglob("*") if path.is_file()
        ) if Path("/mnt/disk2/caoxue/models/MemReranker-4B").exists() else None,
        "runtime": json.loads(historical_runtime.read_text(encoding="utf-8")) if historical_runtime.exists() else None,
        "baseline": json.loads(historical_baseline.read_text(encoding="utf-8")) if historical_baseline.exists() else None,
        "note": "Missing fields remain null; historical artifacts are read-only and are never recomputed or overwritten.",
    }
    return {"purpose": "engineering_cost_only_not_paper_main_baseline", "qwen3_reranker_0_6b": current, "historical_memreranker_4b": previous}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-Reranker-0.6B with frozen Base-conditioned utility preferences")
    parser.add_argument("--dataset-dir", default="artifacts/utility_dataset_v1_1")
    parser.add_argument("--model-path", default="/mnt/disk2/caoxue/models/Qwen3-Reranker-0.6B")
    parser.add_argument("--output-dir", default="artifacts/qwen3_reranker_0_6b_utility_lora_v1")
    parser.add_argument("--historical-4b-dir", default="artifacts/memreranker_utility_lora_v1")
    parser.add_argument("--stage", choices=("audit", "baselines", "train", "test", "all"), default="all")
    parser.add_argument("--max-length", choices=("auto", "4096", "8192", "16384"), default="auto")
    parser.add_argument("--train-batch-size", type=int, default=0, help="0 profiles 4, 2, 1 pairs and selects the largest safe batch")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=.1)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    parser.add_argument("--attention", choices=("auto", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--force-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    dataset_root, output = Path(args.dataset_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, pairs, split_ids = load_dataset(dataset_root)
    tokenizer, yes_id, no_id = load_tokenizer(args.model_path)
    auto_stats, auto_length = length_statistics(rows, tokenizer)
    selected_length = auto_length if args.max_length == "auto" else int(args.max_length)
    if selected_length != auto_length:
        auto_stats["selected_max_length"] = selected_length
        auto_stats["selection_reason"] = f"explicit_cli_override_from_auto_{auto_length}"
        explicit_encoder = BaseTailTruncatingEncoder(tokenizer, selected_length)
        explicit = [explicit_encoder.encode(row, True) for row in rows]
        auto_stats["truncated_sample_count"] = sum(item.truncated_token_count > 0 for item in explicit)
        auto_stats["mean_truncation_ratio"] = statistics.mean(item.truncation_ratio for item in explicit)
        auto_stats["max_truncation_ratio"] = max(item.truncation_ratio for item in explicit)
        for detail, item in zip(auto_stats["samples"], explicit):
            detail.update({key: value for key, value in asdict(item).items() if key != "input_ids"})
    atomic_json(output / "token_length_stats.json", auto_stats)
    scoring = inspect_local_model(Path(args.model_path), tokenizer, yes_id, no_id)
    train_pairs = split_pairs(pairs, split_ids["train"])
    config = {
        "protocol": "qwen3_reranker_0_6b_base_conditioned_utility_v1",
        "arguments": vars(args),
        "frozen_data": {
            "dataset_dir": str(dataset_root), "pair_file": "pairwise_preferences_eps005.jsonl",
            "epsilon": .05, "train_pair_count": len(train_pairs),
            "train_pair_question_count": len({str(pair['question_id']) for pair in train_pairs}),
        },
        "input": "Question + Frozen Base Top-15 + Candidate",
        "loss": "-F.logsigmoid(preferred_score-rejected_score).mean()",
        "scoring": scoring,
        "selected_max_length": selected_length,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    }
    atomic_json(output / "config.json", config)
    print(json.dumps({"scoring": scoring, "token_lengths": {key: value for key, value in auto_stats.items() if key != "samples"}}, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "audit":
        return

    encoder = BaseTailTruncatingEncoder(tokenizer, selected_length)
    model = load_base_model(args.model_path, args.attention)
    import torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config["loaded_base_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    atomic_json(output / "config.json", config)
    if args.stage in ("baselines", "all"):
        baseline_path = output / "baseline_metrics.json"
        if baseline_path.exists():
            print(f"Preserving existing baseline metrics: {baseline_path}", flush=True)
        else:
            baseline = run_dev_baselines(model, tokenizer, encoder, rows, pairs, split_ids, yes_id, no_id, args.eval_batch_size)
            atomic_json(baseline_path, baseline)
            print(json.dumps(baseline, ensure_ascii=False, indent=2), flush=True)
        if args.stage == "baselines":
            return

    trained_model = None
    if args.stage in ("train", "all"):
        summary, trained_model = train(model, tokenizer, encoder, rows, pairs, split_ids, yes_id, no_id, output, args)
        atomic_json(output / "training_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if args.stage == "train":
            comparison = cost_comparison(output, trained_model, Path(args.historical_4b_dir))
            atomic_json(output / "cost_comparison.json", comparison)
            return

    if args.stage in ("test", "all"):
        test_path = output / "test_metrics.json"
        if test_path.exists() and not args.force_test:
            print(f"Frozen Test already exists; refusing to rerun without --force-test: {test_path}", flush=True)
            return
        if trained_model is not None:
            del trained_model
        del model
        import torch
        torch.cuda.empty_cache()
        base = load_base_model(args.model_path, args.attention)
        best = output / "best_checkpoint"
        if not best.exists():
            raise RuntimeError(f"Dev-selected checkpoint is missing: {best}")
        result = frozen_test(base, tokenizer, encoder, rows, pairs, split_ids, yes_id, no_id, best, args.eval_batch_size)
        atomic_json(test_path, result)
        comparison = cost_comparison(output, base, Path(args.historical_4b_dir))
        atomic_json(output / "cost_comparison.json", comparison)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
