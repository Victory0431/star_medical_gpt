#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare SFT datasets into a unified ShareGPT-style `conversations` format.

Supported input formats:
1. sharegpt:
   {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
2. instruction:
   {"instruction": "...", "input": "...", "output": "..."}

The script writes normalized JSONL files that can be consumed directly by the
MedicalGPT SFT pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


ALLOWED_ROLES = {"system", "human", "gpt"}
ROLE_MAP = {
    "system": "system",
    "human": "human",
    "user": "human",
    "assistant": "gpt",
    "gpt": "gpt",
    "bot": "gpt",
}


def default_workers() -> int:
    cpu_count = os.cpu_count() or 8
    return max(4, min(32, cpu_count // 4 or 1))


def chunk_size_for(total: int, workers: int) -> int:
    if total <= 0:
        return max(512, min(8192, workers * 256))
    return max(128, min(4096, math.ceil(total / max(1, workers * 8))))


def clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    return value.strip()


def detect_format(record: Dict) -> str:
    if isinstance(record.get("conversations"), list):
        return "sharegpt"
    if "instruction" in record and "output" in record:
        return "instruction"
    raise ValueError("Unsupported record format")


def normalize_sharegpt(record: Dict) -> List[Dict[str, str]]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Missing or empty conversations")

    normalized: List[Dict[str, str]] = []
    for item in conversations:
        if not isinstance(item, dict):
            raise ValueError("Conversation item must be a dict")
        raw_role = clean_text(item.get("from", "")).lower()
        role = ROLE_MAP.get(raw_role)
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Unsupported role: {raw_role}")
        value = clean_text(item.get("value", ""))
        if not value:
            raise ValueError("Empty conversation value")
        normalized.append({"from": role, "value": value})

    if normalized and normalized[0]["from"] == "system":
        lead = normalized[0]
        normalized = [lead] + normalized[1:]

    if normalized and normalized[0]["from"] != "system" and normalized[0]["from"] != "human":
        raise ValueError("The first non-system role must be human")

    checked: List[Dict[str, str]] = []
    expected_role = "human"
    start_index = 0
    if normalized and normalized[0]["from"] == "system":
        checked.append(normalized[0])
        start_index = 1

    for item in normalized[start_index:]:
        if item["from"] != expected_role:
            raise ValueError("Conversation roles must alternate between human and gpt")
        checked.append(item)
        expected_role = "gpt" if expected_role == "human" else "human"

    conversation_turns = checked[1:] if checked and checked[0]["from"] == "system" else checked
    if len(conversation_turns) < 2 or len(conversation_turns) % 2 != 0:
        raise ValueError("Conversation must contain complete human/gpt turns")

    return checked


def normalize_instruction(record: Dict) -> List[Dict[str, str]]:
    instruction = clean_text(record.get("instruction", ""))
    input_text = clean_text(record.get("input", ""))
    output = clean_text(record.get("output", ""))

    if not instruction and not input_text:
        raise ValueError("Instruction and input cannot both be empty")
    if not output:
        raise ValueError("Output cannot be empty")

    human_value = instruction
    if input_text:
        human_value = f"{instruction}\n\n{input_text}" if instruction else input_text

    return [
        {"from": "human", "value": human_value},
        {"from": "gpt", "value": output},
    ]


def normalize_record(record: Dict, input_format: str) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        fmt = detect_format(record) if input_format == "auto" else input_format
        if fmt == "sharegpt":
            normalized = normalize_sharegpt(record)
        elif fmt == "instruction":
            normalized = normalize_instruction(record)
        else:
            raise ValueError(f"Unsupported input format: {fmt}")

        return {"conversations": normalized}, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def detect_file_layout(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        while True:
            char = f.read(1)
            if not char:
                return "jsonl"
            if not char.isspace():
                return "json_array" if char == "[" else "jsonl"


def iter_rows(path: Path) -> Iterator[Tuple[int, str]]:
    layout = detect_file_layout(path)
    if layout == "json_array":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"JSON array file must contain a list: {path}")
        for index, record in enumerate(payload, start=1):
            yield index, json.dumps(record, ensure_ascii=False)
        return

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_number, line


def iter_chunks(rows: Iterator[Tuple[int, str]], chunk_size: int) -> Iterator[List[Tuple[int, str]]]:
    buffer: List[Tuple[int, str]] = []
    for row in rows:
        buffer.append(row)
        if len(buffer) >= chunk_size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def process_chunk(chunk: List[Tuple[int, str]], input_format: str) -> Tuple[List[Dict], List[Dict]]:
    valid: List[Dict] = []
    invalid: List[Dict] = []
    for line_number, line in chunk:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid.append({"line": line_number, "error": f"JSON decode error: {exc}"})
            continue

        normalized, error = normalize_record(record, input_format=input_format)
        if normalized is None:
            invalid.append({"line": line_number, "error": error})
            continue
        valid.append(normalized)
    return valid, invalid


def write_jsonl(records: Iterable[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def build_output_name(input_path: Path, output_name: Optional[str] = None) -> str:
    if output_name:
        return output_name if output_name.endswith(".jsonl") else f"{output_name}.processed.jsonl"
    suffix = ".processed.jsonl"
    if input_path.suffix == ".jsonl":
        return input_path.name.replace(".jsonl", suffix)
    if input_path.suffix == ".json":
        return input_path.name.replace(".json", suffix)
    return f"{input_path.name}{suffix}"


def prepare_file(
    input_path: Path,
    output_dir: Path,
    report_dir: Path,
    input_format: str,
    workers: int,
    deduplicate: bool,
    output_name: Optional[str] = None,
) -> Dict:
    chunk_size = chunk_size_for(total=0, workers=workers)
    output_path = output_dir / build_output_name(input_path, output_name=output_name)
    total_rows = 0
    valid_rows = 0
    invalid_rows = 0
    duplicate_count = 0
    invalid_examples: List[Dict] = []
    seen = set()

    def flush_result(valid_chunk: List[Dict], invalid_chunk: List[Dict], writer) -> None:
        nonlocal valid_rows, invalid_rows, duplicate_count

        for record in valid_chunk:
            if deduplicate:
                key = json.dumps(record, ensure_ascii=False, sort_keys=True)
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
            json.dump(record, writer, ensure_ascii=False)
            writer.write("\n")
            valid_rows += 1

        invalid_rows += len(invalid_chunk)
        for item in invalid_chunk:
            if len(invalid_examples) >= 20:
                break
            invalid_examples.append(item)

    with output_path.open("w", encoding="utf-8") as writer, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for chunk in iter_chunks(iter_rows(input_path), chunk_size=chunk_size):
            total_rows += len(chunk)
            futures.append(executor.submit(process_chunk, chunk, input_format))
            if len(futures) >= max(1, workers * 4):
                future = futures.pop(0)
                valid_chunk, invalid_chunk = future.result()
                flush_result(valid_chunk, invalid_chunk, writer)

        for future in futures:
            valid_chunk, invalid_chunk = future.result()
            flush_result(valid_chunk, invalid_chunk, writer)

    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_format": input_format,
        "workers": workers,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "duplicate_rows": duplicate_count,
        "invalid_examples": invalid_examples,
    }
    report_path = report_dir / f"{output_path.stem}.report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT datasets for MedicalGPT/Qwen training")
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="One or more input JSON/JSONL files",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split name for output organization",
    )
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "sharegpt", "instruction"],
        help="Input dataset format",
    )
    parser.add_argument(
        "--output-root",
        default="/home/qjh/llm_learning/my_medical_gpt/data/sft/processed",
        help="Root directory for processed dataset outputs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Worker threads used for preprocessing",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable exact-record deduplication",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional custom output filename stem for single-file processing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_name and len(args.input_files) != 1:
        raise ValueError("--output-name only supports single-file processing")

    output_root = Path(args.output_root)
    output_dir = output_root / args.split
    report_dir = output_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict] = []
    for input_file in args.input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        report = prepare_file(
            input_path=input_path,
            output_dir=output_dir,
            report_dir=report_dir,
            input_format=args.input_format,
            workers=max(1, args.workers),
            deduplicate=not args.no_deduplicate,
            output_name=args.output_name,
        )
        reports.append(report)
        print(
            f"[done] {input_path.name}: total={report['total_rows']} "
            f"valid={report['valid_rows']} invalid={report['invalid_rows']} "
            f"duplicates={report['duplicate_rows']} -> {report['output_file']}"
        )

    summary = {
        "files": reports,
        "total_input_rows": sum(item["total_rows"] for item in reports),
        "total_valid_rows": sum(item["valid_rows"] for item in reports),
        "total_invalid_rows": sum(item["invalid_rows"] for item in reports),
        "total_duplicate_rows": sum(item["duplicate_rows"] for item in reports),
    }
    summary_path = output_root / "reports" / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[summary] written to {summary_path}")


if __name__ == "__main__":
    main()
