#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pairwise preference data for DPO/ORPO/RM training")
    parser.add_argument("--input-files", nargs="+", required=True, help="Raw json/jsonl files")
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument(
        "--output-root",
        default="/home/qjh/llm_learning/my_medical_gpt/data/alignment/processed/dpo",
        help="Processed DPO data root",
    )
    parser.add_argument("--output-name", default=None, help="Output file prefix")
    parser.add_argument("--system-prompt", default="", help="Optional system prompt injected into every sample")
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser


def read_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        yield payload
        return
    raise ValueError(f"Unsupported JSON structure in {path}")


def normalize_history(history: Any) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if not history:
        return messages

    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict):
                role = str(item.get("role", "")).strip().lower()
                content = str(item.get("content", "")).strip()
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                user_text = str(item[0]).strip()
                assistant_text = str(item[1]).strip()
                if user_text:
                    messages.append({"role": "user", "content": user_text})
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})
    return messages


def build_prompt_messages(record: Dict[str, Any], system_prompt: str) -> List[Dict[str, str]]:
    prompt: List[Dict[str, str]] = []
    injected_system = system_prompt.strip()
    raw_system = str(record.get("system", "")).strip()
    if injected_system:
        prompt.append({"role": "system", "content": injected_system})
    elif raw_system:
        prompt.append({"role": "system", "content": raw_system})

    prompt.extend(normalize_history(record.get("history")))

    question = str(record.get("question", "")).strip()
    if not question:
        raise ValueError("Missing question field")
    prompt.append({"role": "user", "content": question})
    return prompt


def transform_record(record: Dict[str, Any], system_prompt: str, source_name: str) -> Optional[Dict[str, Any]]:
    chosen = str(record.get("response_chosen", "")).strip()
    rejected = str(record.get("response_rejected", "")).strip()
    if not chosen or not rejected:
        return None

    prompt = build_prompt_messages(record, system_prompt)
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "source": source_name,
    }


def infer_output_name(input_files: List[Path]) -> str:
    if len(input_files) == 1:
        stem = input_files[0].stem
    else:
        stem = input_files[0].stem + "_multi"
    return stem.replace(".processed", "")


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_arg_parser().parse_args()
    input_files = [Path(item).resolve() for item in args.input_files]
    output_name = args.output_name or infer_output_name(input_files)
    output_root = Path(args.output_root).resolve()
    output_path = output_root / args.split / f"{output_name}.processed.jsonl"
    report_path = output_root / "reports" / f"{output_name}.{args.split}.report.json"

    records: List[Dict[str, Any]] = []
    invalid_count = 0
    for input_path in input_files:
        source_name = input_path.stem
        for raw_record in read_json_records(input_path):
            try:
                transformed = transform_record(raw_record, args.system_prompt, source_name)
            except Exception:
                transformed = None
            if transformed is None:
                invalid_count += 1
                continue
            records.append(transformed)
            if args.max_samples > 0 and len(records) >= args.max_samples:
                break
        if args.max_samples > 0 and len(records) >= args.max_samples:
            break

    write_jsonl(records, output_path)
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split,
        "input_files": [str(path) for path in input_files],
        "output_path": str(output_path),
        "output_name": output_name,
        "record_count": len(records),
        "invalid_count": invalid_count,
        "system_prompt_enabled": bool(args.system_prompt.strip()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
