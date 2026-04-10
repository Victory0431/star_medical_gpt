#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build trainable subsets from curated SFT quality buckets")
    parser.add_argument("--input-file", required=True, help="scored_all jsonl produced by light_quality_score.py")
    parser.add_argument(
        "--output-root",
        default="/home/qjh/llm_learning/my_medical_gpt/data/sft/curation/subsets",
        help="Output directory for curated train subsets",
    )
    parser.add_argument("--output-name", required=True, help="Subset output name without suffix")
    parser.add_argument("--target-size", type=int, required=True)
    parser.add_argument(
        "--strategy",
        choices=["high_bucket_only", "high_then_borderline", "source_stratified_high"],
        default="high_bucket_only",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_high_bucket_only(rows: List[Dict[str, Any]], target_size: int) -> List[Dict[str, Any]]:
    high_rows = [row for row in rows if row["quality_bucket"] == "high_confidence_keep"]
    high_rows.sort(key=lambda item: (-item["quality_score"], item.get("curation_hash", "")))
    return high_rows[:target_size]


def pick_high_then_borderline(rows: List[Dict[str, Any]], target_size: int) -> List[Dict[str, Any]]:
    high_rows = [row for row in rows if row["quality_bucket"] == "high_confidence_keep"]
    borderline_rows = [row for row in rows if row["quality_bucket"] == "borderline_review"]
    high_rows.sort(key=lambda item: (-item["quality_score"], item.get("curation_hash", "")))
    borderline_rows.sort(key=lambda item: (-item["quality_score"], item.get("curation_hash", "")))
    picked = high_rows[:target_size]
    if len(picked) < target_size:
        picked.extend(borderline_rows[: target_size - len(picked)])
    return picked


def pick_source_stratified_high(rows: List[Dict[str, Any]], target_size: int, seed: int) -> List[Dict[str, Any]]:
    high_rows = [row for row in rows if row["quality_bucket"] == "high_confidence_keep"]
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in high_rows:
        grouped[row.get("curation_source", "unknown")].append(row)

    for source_rows in grouped.values():
        source_rows.sort(key=lambda item: (-item["quality_score"], item.get("curation_hash", "")))

    total_high = len(high_rows)
    quotas: Dict[str, int] = {}
    allocated = 0
    sources = sorted(grouped)
    for source in sources:
        ratio = len(grouped[source]) / total_high if total_high else 0.0
        quota = int(target_size * ratio)
        quotas[source] = min(quota, len(grouped[source]))
        allocated += quotas[source]

    rng = random.Random(seed)
    while allocated < target_size:
        updated = False
        for source in sources:
            if allocated >= target_size:
                break
            if quotas[source] < len(grouped[source]):
                quotas[source] += 1
                allocated += 1
                updated = True
        if not updated:
            break

    picked: List[Dict[str, Any]] = []
    for source in sources:
        picked.extend(grouped[source][: quotas[source]])
    rng.shuffle(picked)
    picked.sort(key=lambda item: (-item["quality_score"], item.get("curation_hash", "")))
    return picked[:target_size]


def to_train_record(scored_row: Dict[str, Any], subset_name: str) -> Dict[str, Any]:
    return {
        "curation_source": scored_row.get("curation_source"),
        "curation_hash": scored_row.get("curation_hash"),
        "quality_score": scored_row.get("quality_score"),
        "quality_bucket": scored_row.get("quality_bucket"),
        "subset_name": subset_name,
        "conversations": scored_row["conversations"],
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input_file).resolve()
    output_root = Path(args.output_root).resolve()
    output_path = output_root / f"{args.output_name}.jsonl"
    report_path = output_root / f"{args.output_name}.report.json"

    rows = list(read_jsonl(input_path))
    if args.strategy == "high_bucket_only":
        picked = pick_high_bucket_only(rows, args.target_size)
    elif args.strategy == "source_stratified_high":
        picked = pick_source_stratified_high(rows, args.target_size, args.seed)
    else:
        picked = pick_high_then_borderline(rows, args.target_size)

    train_rows = [to_train_record(row, args.output_name) for row in picked]
    write_jsonl(output_path, train_rows)

    source_counts = Counter(row.get("curation_source", "unknown") for row in picked)
    bucket_counts = Counter(row.get("quality_bucket", "unknown") for row in picked)
    avg_score = sum(float(row.get("quality_score", 0.0)) for row in picked) / len(picked) if picked else 0.0
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "output_name": args.output_name,
        "strategy": args.strategy,
        "target_size": args.target_size,
        "written_rows": len(train_rows),
        "avg_quality_score": round(avg_score, 4),
        "source_counts": dict(source_counts),
        "bucket_counts": dict(bucket_counts),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
