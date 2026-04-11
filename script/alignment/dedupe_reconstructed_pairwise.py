#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deduplicate reconstructed pairwise jsonl files by sample_id")
    parser.add_argument("--input-file", required=True, help="Raw reconstructed jsonl")
    parser.add_argument("--audit-file", default=None, help="Optional matching audit jsonl")
    parser.add_argument("--key-field", default="sample_id")
    parser.add_argument("--keep", choices=["first", "last"], default="last")
    parser.add_argument("--report-file", default=None, help="Optional report output path")
    return parser


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_rows(rows: List[Dict[str, Any]], key_field: str, keep: str) -> Tuple[List[Dict[str, Any]], int]:
    indexed: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    duplicate_count = 0
    for index, row in enumerate(rows):
        key = str(row.get(key_field, ""))
        if not key:
            key = f"__missing__{index}"
        if key in indexed:
            duplicate_count += 1
            if keep == "last":
                indexed[key] = (index, row)
        else:
            indexed[key] = (index, row)
    kept = [row for _, row in sorted(indexed.values(), key=lambda item: item[0])]
    return kept, duplicate_count


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input_file).resolve()
    audit_path = Path(args.audit_file).resolve() if args.audit_file else None
    report_path = Path(args.report_file).resolve() if args.report_file else None

    raw_rows = read_jsonl(input_path)
    deduped_raw, raw_duplicates = dedupe_rows(raw_rows, args.key_field, args.keep)
    write_jsonl(input_path, deduped_raw)

    audit_duplicates = None
    if audit_path is not None and audit_path.exists():
        audit_rows = read_jsonl(audit_path)
        deduped_audit, audit_duplicates = dedupe_rows(audit_rows, args.key_field, args.keep)
        write_jsonl(audit_path, deduped_audit)

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path),
        "audit_file": str(audit_path) if audit_path else None,
        "key_field": args.key_field,
        "keep": args.keep,
        "raw_before": len(raw_rows),
        "raw_after": len(deduped_raw),
        "raw_duplicates_removed": raw_duplicates,
        "audit_duplicates_removed": audit_duplicates,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
