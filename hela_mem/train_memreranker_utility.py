"""Question-balanced pairwise LoRA training for base-conditioned memory utility.

The module deliberately consumes the frozen V1.1 utility dataset.  It never
rebuilds Base Top-K memories or utility labels and never runs answer generation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


OFFICIAL_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
OFFICIAL_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
UTILITY_INSTRUCTION = (
    "Given a question and any supplied current Base memories, determine whether the Candidate "
    "Memory provides useful additional evidence for answering the question."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def describe_lengths(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50": percentile(values, .50),
        "p90": percentile(values, .90),
        "p95": percentile(values, .95),
        "max": max(values, default=0),
    }


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    a, b = rankdata(left), rankdata(right)
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return numerator / denominator if denominator else None


def ndcg(scores: Sequence[float], utilities: Sequence[float]) -> float | None:
    if len(scores) < 2:
        return None
    floor = min(utilities)
    gains = [value - floor for value in utilities]
    def dcg(order: Sequence[int]) -> float:
        return sum(gains[index] / math.log2(rank + 2) for rank, index in enumerate(order))
    predicted = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    ideal = sorted(range(len(scores)), key=lambda index: utilities[index], reverse=True)
    denominator = dcg(ideal)
    return dcg(predicted) / denominator if denominator > 0 else None


def format_query(row: dict[str, Any], base_conditioned: bool) -> str:
    question = str(row["question"]).strip()
    if not base_conditioned:
        return question
    context = str(row.get("actual_baseline_context") or "").strip()
    return f"Question:\n{question}\n\nCurrent Base Memories:\n{context}"


def format_instruction(query: str, candidate: str) -> str:
    return f"<Instruct>: {UTILITY_INSTRUCTION}\n<Query>: {query}\n<Document>: {candidate.strip()}"


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    raw_length: int
    truncated: bool


class PromptEncoder:
    """Official MemReranker template with deterministic tail truncation.

    The official prefix and suffix are always retained.  When necessary, the
    formatted body is truncated from its tail, matching the model-card recipe.
    Because Base memories precede Document in the official body, we additionally
    expose truncation counts in the mandatory length audit.
    """

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = tokenizer.encode(OFFICIAL_PREFIX, add_special_tokens=False)
        self.suffix = tokenizer.encode(OFFICIAL_SUFFIX, add_special_tokens=False)
        if len(self.prefix) + len(self.suffix) >= max_length:
            raise ValueError("max_length is too small for the official MemReranker template")

    def encode(self, row: dict[str, Any], base_conditioned: bool) -> EncodedExample:
        body = self.tokenizer.encode(
            format_instruction(format_query(row, base_conditioned), str(row["candidate_text"])),
            add_special_tokens=False,
        )
        raw_length = len(self.prefix) + len(body) + len(self.suffix)
        budget = self.max_length - len(self.prefix) - len(self.suffix)
        body = body[:budget]
        return EncodedExample(self.prefix + body + self.suffix, raw_length, raw_length > self.max_length)


def resolve_yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Resolve official classification tokens without hard-coded IDs."""
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    invalid = {None, getattr(tokenizer, "unk_token_id", None)}
    if yes_id in invalid or no_id in invalid or yes_id == no_id:
        raise RuntimeError(f"invalid MemReranker yes/no token ids: yes={yes_id}, no={no_id}")
    for token, token_id in (("yes", yes_id), ("no", no_id)):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] != token_id:
            raise RuntimeError(f"{token!r} is not a single official scoring token: {encoded}, id={token_id}")
    return int(yes_id), int(no_id)


def load_dataset(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[str]]]:
    rows = read_jsonl(root / "candidate_utility.jsonl")
    pairs = read_jsonl(root / "pairwise_preferences_eps005.jsonl")
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    split_ids = {name: set(map(str, ids)) for name, ids in manifest["question_ids"].items()}
    by_key = {(str(row["question_id"]), str(row["candidate_memory_id"])): row for row in rows}
    for pair in pairs:
        qid = str(pair["question_id"])
        if (qid, str(pair["preferred_candidate_id"])) not in by_key or (qid, str(pair["rejected_candidate_id"])) not in by_key:
            raise RuntimeError(f"pair references an unknown candidate: {pair}")
    return rows, pairs, split_ids


class QuestionBalancedSampler:
    """Sample exactly ``len(pairs)`` items while cycling uniformly over questions."""

    def __init__(self, pairs: Sequence[dict[str, Any]], seed: int, epoch: int = 0) -> None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, pair in enumerate(pairs):
            grouped[str(pair["question_id"])].append(index)
        if not grouped:
            raise ValueError("question-balanced sampler received no pairs")
        self.grouped = dict(grouped)
        self.seed = seed
        self.epoch = epoch
        self.num_samples = len(pairs)

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
    def __init__(self, pairs: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], encoder: PromptEncoder) -> None:
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
        encoded = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = (encoded["input_ids"] != self.tokenizer.pad_token_id).long()
        encoded["pair_batch_size"] = torch.tensor(len(batch))
        return encoded


def split_pairs(pairs: Sequence[dict[str, Any]], question_ids: set[str]) -> list[dict[str, Any]]:
    return [pair for pair in pairs if str(pair["question_id"]) in question_ids]


def length_audit(rows: Sequence[dict[str, Any]], encoder: PromptEncoder, max_length: int) -> dict[str, Any]:
    report: dict[str, Any] = {"max_length": max_length, "dynamic_padding": True}
    for name, conditioned in (("original", False), ("base_conditioned", True)):
        examples = [encoder.encode(row, conditioned) for row in rows]
        report[name] = {
            **describe_lengths([item.raw_length for item in examples]),
            "truncated_count": sum(item.truncated for item in examples),
            "truncated_rate": sum(item.truncated for item in examples) / len(examples) if examples else 0.0,
        }
    return report


def model_score(model: Any, batch: dict[str, Any], yes_id: int, no_id: int) -> Any:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, -1, :]
    return logits[:, yes_id].float() - logits[:, no_id].float()


def evaluate(
    model: Any,
    tokenizer: Any,
    encoder: PromptEncoder,
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
            token_rows = [encoder.encode(row, base_conditioned).input_ids for row in chunk]
            batch = tokenizer.pad({"input_ids": token_rows}, padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = model_score(model, batch, yes_id, no_id).cpu().tolist()
            for row, score in zip(chunk, scores):
                score_map[(str(row["question_id"]), str(row["candidate_memory_id"]))] = float(score)
            if start == 0 or start + batch_size >= len(selected_rows) or (start // batch_size + 1) % 25 == 0:
                print(f"Scoring candidates: {min(start + batch_size, len(selected_rows))}/{len(selected_rows)}", flush=True)
    selected_pairs = split_pairs(pairs, question_ids)
    margins = []
    for pair in selected_pairs:
        qid = str(pair["question_id"])
        margins.append(
            score_map[(qid, str(pair["preferred_candidate_id"]))]
            - score_map[(qid, str(pair["rejected_candidate_id"]))]
        )
    correlations, ndcgs = [], []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[str(row["question_id"])].append(row)
    for qid, group in grouped.items():
        scores = [score_map[(qid, str(row["candidate_memory_id"]))] for row in group]
        utilities = [float(row["utility_score"]) for row in group]
        correlation, ranking = spearman(scores, utilities), ndcg(scores, utilities)
        if correlation is not None:
            correlations.append(correlation)
        if ranking is not None:
            ndcgs.append(ranking)
    return {
        "pair_count": len(margins),
        "pair_question_count": len({str(pair["question_id"]) for pair in selected_pairs}),
        "candidate_count": len(selected_rows),
        "pairwise_accuracy": sum(margin > 0 for margin in margins) / len(margins) if margins else None,
        "tie_count": sum(margin == 0 for margin in margins),
        "mean_pair_margin": statistics.mean(margins) if margins else None,
        "mean_question_spearman": statistics.mean(correlations) if correlations else None,
        "mean_question_ndcg": statistics.mean(ndcgs) if ndcgs else None,
        "scored_question_count": len(grouped),
    }


def load_runtime(model_path: str, max_length: int, attn_implementation: str) -> tuple[Any, Any, PromptEncoder, int, int]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("training requires transformers>=4.51.0 and torch") from error
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_id, no_id = resolve_yes_no_token_ids(tokenizer)
    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).cuda()
    return model, tokenizer, PromptEncoder(tokenizer, max_length), yes_id, no_id


def run_baselines(
    model: Any, tokenizer: Any, encoder: PromptEncoder, rows: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]],
    split_ids: dict[str, set[str]], yes_id: int, no_id: int, batch_size: int,
) -> dict[str, Any]:
    result = {"protocol": "memreranker_utility_baselines_v1", "score": "yes_logit_minus_no_logit", "methods": {}}
    for name, conditioned in (("original_memreranker", False), ("base_conditioned_zero_shot", True)):
        print(f"Evaluating baseline: {name}", flush=True)
        # Test remains sealed until the Dev-selected LoRA checkpoint is final.
        result["methods"][name] = {
            "dev": evaluate(model, tokenizer, encoder, rows, pairs, split_ids["dev"], yes_id, no_id, conditioned, batch_size)
        }
    return result


def train_lora(
    model: Any, tokenizer: Any, encoder: PromptEncoder, rows: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]],
    split_ids: dict[str, set[str]], yes_id: int, no_id: int, output_dir: Path, args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import DataLoader
        from transformers import get_cosine_schedule_with_warmup
    except ImportError as error:
        raise RuntimeError("LoRA training requires peft and transformers>=4.51.0") from error
    train_pairs = split_pairs(pairs, split_ids["train"])
    dev_pairs = split_pairs(pairs, split_ids["dev"])
    train_questions = {str(pair["question_id"]) for pair in train_pairs}
    if len(train_questions) != args.expected_train_questions:
        raise RuntimeError(
            f"eps005 train pair population changed: expected {args.expected_train_questions} questions, got {len(train_questions)}"
        )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, config)
    dataset = PairDataset(train_pairs, rows, encoder)
    collator = DynamicPairCollator(tokenizer)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    batches_per_epoch = math.ceil(len(train_pairs) / args.train_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(total_updates * args.warmup_ratio)),
        num_training_steps=total_updates,
    )
    history = []
    best_accuracy, best_epoch = -1.0, None
    best_dir = output_dir / "best_adapter"
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        sampler = QuestionBalancedSampler(train_pairs, args.seed, epoch)
        loader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=sampler, collate_fn=collator, num_workers=0)
        model.train()
        losses = []
        for step, batch in enumerate(loader, 1):
            pair_batch_size = int(batch.pop("pair_batch_size"))
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = model_score(model, batch, yes_id, no_id)
                preferred, rejected = scores[:pair_batch_size], scores[pair_batch_size:]
                loss = -functional.logsigmoid(preferred - rejected).mean()
                scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            losses.append(float(loss.detach().cpu()))
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step == len(loader) or step % 25 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} train {step}/{len(loader)} loss={losses[-1]:.6f}",
                    flush=True,
                )
        print(f"Epoch {epoch}/{args.epochs} Dev evaluation", flush=True)
        dev = evaluate(model, tokenizer, encoder, rows, dev_pairs, split_ids["dev"], yes_id, no_id, True, args.eval_batch_size)
        record = {"epoch": epoch, "train_loss": statistics.mean(losses), "dev": dev}
        history.append(record)
        atomic_json(output_dir / "training_history.json", history)
        accuracy = float(dev["pairwise_accuracy"] or 0.0)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            atomic_json(best_dir / "selection.json", {"selected_epoch": epoch, "dev_pairwise_accuracy": accuracy})
    return {
        "protocol": "base_conditioned_pairwise_lora_v1",
        "train_pair_count": len(train_pairs),
        "train_pair_question_count": len(train_questions),
        "dev_pair_count": len(dev_pairs),
        "question_balanced_samples_per_epoch": len(train_pairs),
        "best_epoch": best_epoch,
        "best_dev_pairwise_accuracy": best_accuracy,
        "history": history,
        "best_adapter": str(best_dir),
    }


def load_adapter_for_test(base_model: Any, adapter_path: Path) -> Any:
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, adapter_path).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Base-conditioned MemReranker utility LoRA")
    parser.add_argument("--dataset-dir", default="artifacts/utility_dataset_v1_1")
    parser.add_argument("--model-path", default="/mnt/disk2/caoxue/models/MemReranker-4B")
    parser.add_argument("--output-dir", default="artifacts/memreranker_utility_lora_v1")
    parser.add_argument("--stage", choices=("audit", "baselines", "train", "test", "all"), default="all")
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    parser.add_argument("--expected-train-questions", type=int, default=41)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--attn-implementation", choices=("auto", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--force-test", action="store_true", help="Explicitly overwrite the frozen LoRA test result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    root, output = Path(args.dataset_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, pairs, split_ids = load_dataset(root)

    # Tokenizer-only setup makes the audit cheap and executable without loading 4B weights.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    yes_id, no_id = resolve_yes_no_token_ids(tokenizer)
    encoder = PromptEncoder(tokenizer, args.max_length)
    audit = length_audit(rows, encoder, args.max_length)
    audit.update({
        "model_path": args.model_path,
        "official_template": {"prefix": OFFICIAL_PREFIX, "suffix": OFFICIAL_SUFFIX, "instruction": UTILITY_INSTRUCTION},
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "eps005_pair_count": len(pairs),
        "eps005_pair_question_count": len({str(pair["question_id"]) for pair in pairs}),
        "eps005_train_pair_count": len(split_pairs(pairs, split_ids["train"])),
        "eps005_train_pair_question_count": len({str(pair["question_id"]) for pair in split_pairs(pairs, split_ids["train"])}),
    })
    atomic_json(output / "input_length_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "audit":
        return

    model, tokenizer, encoder, yes_id, no_id = load_runtime(args.model_path, args.max_length, args.attn_implementation)
    manifest = vars(args).copy()
    manifest.update({"yes_token_id": yes_id, "no_token_id": no_id, "score": "yes_logit_minus_no_logit", "loss": "-logsigmoid(s_positive-s_negative)"})
    atomic_json(output / "run_config.json", manifest)

    if args.stage in ("baselines", "all"):
        baseline_path = output / "baseline_results.json"
        if baseline_path.exists():
            print(f"Baseline results already exist; preserving {baseline_path}", flush=True)
        else:
            baseline = run_baselines(model, tokenizer, encoder, rows, pairs, split_ids, yes_id, no_id, args.eval_batch_size)
            atomic_json(baseline_path, baseline)
            print(json.dumps(baseline, ensure_ascii=False, indent=2), flush=True)
        if args.stage == "baselines":
            return

    if args.stage in ("train", "all"):
        training = train_lora(model, tokenizer, encoder, rows, pairs, split_ids, yes_id, no_id, output, args)
        atomic_json(output / "training_summary.json", training)
        print(json.dumps(training, ensure_ascii=False, indent=2), flush=True)
        if args.stage == "train":
            return
        del model
        import torch
        torch.cuda.empty_cache()

    if args.stage in ("test", "all"):
        final_path = output / "final_test_results.json"
        if final_path.exists() and not args.force_test:
            print(f"Frozen test result already exists; refusing to rerun without --force-test: {final_path}", flush=True)
            return
        best_dir = output / "best_adapter"
        if not best_dir.exists():
            raise RuntimeError(f"selected Dev checkpoint does not exist: {best_dir}")
        if args.stage == "all":
            del model
            import torch
            torch.cuda.empty_cache()
            model, tokenizer, encoder, yes_id, no_id = load_runtime(args.model_path, args.max_length, args.attn_implementation)
        print("Final frozen Test: original MemReranker", flush=True)
        original = evaluate(model, tokenizer, encoder, rows, pairs, split_ids["test"], yes_id, no_id, False, args.eval_batch_size)
        print("Final frozen Test: Base-conditioned zero-shot", flush=True)
        conditioned = evaluate(model, tokenizer, encoder, rows, pairs, split_ids["test"], yes_id, no_id, True, args.eval_batch_size)
        tuned = load_adapter_for_test(model, best_dir)
        print("Final frozen Test: Base-conditioned + LoRA", flush=True)
        result = {
            "protocol": "frozen_final_test_v1",
            "checkpoint_selection": json.loads((best_dir / "selection.json").read_text(encoding="utf-8")),
            "methods": {
                "original_memreranker": original,
                "base_conditioned_zero_shot": conditioned,
                "base_conditioned_lora": evaluate(
                    tuned, tokenizer, encoder, rows, pairs, split_ids["test"], yes_id, no_id, True, args.eval_batch_size
                ),
            },
        }
        atomic_json(final_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
