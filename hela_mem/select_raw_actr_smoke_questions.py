"""Select a fixed smoke subset whose raw memory banks exceed final Top-K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval_longmemeval import CORRUPTED_INDICES
from .runtime import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--mem-dir", required=True)
    parser.add_argument("--num-items", type=int, default=10)
    parser.add_argument("--minimum-size", type=int, default=16)
    parser.add_argument("--preferred-size", type=int, default=30)
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    dataset = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    eligible = []
    for index, item in enumerate(dataset):
        if index in CORRUPTED_INDICES:
            continue
        graph_path = Path(args.mem_dir) / f"{item['question_id']}_hebbian.json"
        if not graph_path.exists():
            continue
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        size = len(graph.get("nodes", {}))
        if size >= args.minimum_size:
            eligible.append((index, item, size))
    eligible.sort(key=lambda row: (row[2] < args.preferred_size, row[0]))
    selected = eligible[: args.num_items]
    if len(selected) < args.num_items:
        raise SystemExit(
            f"only {len(selected)} questions have memory_bank_size >= {args.minimum_size}; "
            f"need {args.num_items}"
        )
    atomic_write_json(args.output_data, [item for _, item, _ in selected])
    manifest = {
        "source_data_path": args.data_path,
        "mem_dir": args.mem_dir,
        "num_items": len(selected),
        "minimum_size": args.minimum_size,
        "preferred_size": args.preferred_size,
        "preferred_count": sum(size >= args.preferred_size for _, _, size in selected),
        "records": [
            {
                "source_index": index,
                "question_id": item["question_id"],
                "question_type": item.get("question_type"),
                "memory_bank_size": size,
            }
            for index, item, size in selected
        ],
    }
    atomic_write_json(args.output_manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
