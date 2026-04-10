#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


GENERIC_DEFERRAL_PATTERNS = [
    "建议到医院检查",
    "建议去医院检查",
    "建议及时就医",
    "建议尽快就医",
    "请咨询医生",
    "遵医嘱",
    "去正规医院",
]

LOW_SIGNAL_PATTERNS = [
    "是的",
    "不是",
    "可以",
    "不可以",
    "有可能",
    "不好说",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based coarse filtering for processed SFT datasets")
    parser.add_argument("--input-files", nargs="+", required=True, help="Processed SFT jsonl files")
    parser.add_argument(
        "--output-root",
        default="/home/qjh/llm_learning/my_medical_gpt/data/sft/curation",
        help="Output root for curation artifacts",
    )
    parser.add_argument("--output-name", default="medical_sft_27w_rule_filtered", help="Merged filtered dataset name")
    parser.add_argument("--min-user-chars", type=int, default=4)
    parser.add_argument("--min-assistant-chars", type=int, default=24)
    parser.add_argument("--max-user-chars", type=int, default=1500)
    parser.add_argument("--max-assistant-chars", type=int, default=4000)
    parser.add_argument("--max-total-turns", type=int, default=12)
    parser.add_argument("--max-line-repeat", type=float, default=0.45, help="Max repeated line ratio")
    parser.add_argument("--max-char-repeat", type=float, default=0.35, help="Max repeated char ratio")
    parser.add_argument("--min-assistant-punct", type=int, default=1, help="Require at least N punctuation marks in answer")
    parser.add_argument("--keep-rejected", action="store_true", help="Also write rejected rows for audit")
    return parser


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload["_line_no"] = line_no
            yield payload


def normalize_text(text: Any) -> str:
    text = str(text or "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_conversations(conversations: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    if not isinstance(conversations, list):
        return normalized
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = normalize_text(item.get("from", "")).lower()
        value = normalize_text(item.get("value", ""))
        if role and value:
            normalized.append({"from": role, "value": value})
    return normalized


def text_hash(conversations: List[Dict[str, str]]) -> str:
    digest = hashlib.sha1()
    for item in conversations:
        digest.update(item["from"].encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(item["value"].encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def repeated_char_ratio(text: str) -> float:
    cleaned = re.sub(r"\s+", "", text)
    if not cleaned:
        return 1.0
    counter = Counter(cleaned)
    return max(counter.values()) / len(cleaned)


def repeated_line_ratio(text: str) -> float:
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(raw_lines) <= 1:
        return 0.0
    counter = Counter(raw_lines)
    return max(counter.values()) / len(raw_lines)


def punctuation_count(text: str) -> int:
    return len(re.findall(r"[，。！？；：,.!?;:]", text))


def low_signal_answer(text: str) -> bool:
    normalized = normalize_text(text)
    if normalized in LOW_SIGNAL_PATTERNS:
        return True
    return len(normalized) <= 8 and any(token == normalized for token in LOW_SIGNAL_PATTERNS)


def generic_deferral(text: str) -> bool:
    if len(text) >= 80:
        return False
    return any(pattern in text for pattern in GENERIC_DEFERRAL_PATTERNS)


def validate_conversations(
    conversations: List[Dict[str, str]],
    args: argparse.Namespace,
) -> List[str]:
    reasons: List[str] = []
    if not conversations:
        return ["empty_conversations"]
    if len(conversations) < 2:
        return ["too_few_turns"]
    if len(conversations) > args.max_total_turns:
        reasons.append("too_many_turns")
    if len(conversations) % 2 != 0:
        reasons.append("odd_turn_count")

    for index, item in enumerate(conversations):
        expected_role = "human" if index % 2 == 0 else "gpt"
        if item["from"] != expected_role:
            reasons.append("role_order_invalid")
            break

    user_text = conversations[-2]["value"] if len(conversations) >= 2 else ""
    assistant_text = conversations[-1]["value"] if len(conversations) >= 1 else ""

    if len(user_text) < args.min_user_chars:
        reasons.append("user_too_short")
    if len(user_text) > args.max_user_chars:
        reasons.append("user_too_long")
    if len(assistant_text) < args.min_assistant_chars:
        reasons.append("assistant_too_short")
    if len(assistant_text) > args.max_assistant_chars:
        reasons.append("assistant_too_long")
    if punctuation_count(assistant_text) < args.min_assistant_punct:
        reasons.append("assistant_low_structure")
    if repeated_char_ratio(assistant_text) > args.max_char_repeat:
        reasons.append("assistant_char_repeat")
    if repeated_line_ratio(assistant_text) > args.max_line_repeat:
        reasons.append("assistant_line_repeat")
    if low_signal_answer(assistant_text):
        reasons.append("assistant_low_signal")
    if generic_deferral(assistant_text):
        reasons.append("assistant_generic_deferral")

    return sorted(set(reasons))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_arg_parser().parse_args()

    output_root = Path(args.output_root).resolve()
    filtered_output = output_root / "rule_filtered" / f"{args.output_name}.jsonl"
    rejected_output = output_root / "rule_filtered" / f"{args.output_name}.rejected.jsonl"
    report_output = output_root / "reports" / f"{args.output_name}.summary.json"

    kept_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": [str(Path(path).resolve()) for path in args.input_files],
        "output_file": str(filtered_output),
        "rejected_output_file": str(rejected_output) if args.keep_rejected else None,
        "thresholds": {
            "min_user_chars": args.min_user_chars,
            "min_assistant_chars": args.min_assistant_chars,
            "max_user_chars": args.max_user_chars,
            "max_assistant_chars": args.max_assistant_chars,
            "max_total_turns": args.max_total_turns,
            "max_line_repeat": args.max_line_repeat,
            "max_char_repeat": args.max_char_repeat,
            "min_assistant_punct": args.min_assistant_punct,
        },
    }

    source_seen_hashes: set[str] = set()
    source_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "kept": 0, "duplicates": 0, "rejected": 0, "reason_counts": Counter()}
    )

    for input_file in args.input_files:
        input_path = Path(input_file).resolve()
        source_name = input_path.stem.replace(".processed", "")
        for record in read_jsonl(input_path):
            source_stats[source_name]["total"] += 1
            conversations = normalize_conversations(record.get("conversations"))
            record_hash = text_hash(conversations)
            reasons = validate_conversations(conversations, args)
            if record_hash in source_seen_hashes:
                reasons.append("duplicate_conversation")
                source_stats[source_name]["duplicates"] += 1

            if reasons:
                source_stats[source_name]["rejected"] += 1
                source_stats[source_name]["reason_counts"].update(reasons)
                if args.keep_rejected:
                    rejected_rows.append(
                        {
                            "source": source_name,
                            "line_no": record["_line_no"],
                            "reasons": reasons,
                            "conversations": conversations,
                        }
                    )
                continue

            source_seen_hashes.add(record_hash)
            source_stats[source_name]["kept"] += 1
            kept_rows.append(
                {
                    "curation_source": source_name,
                    "curation_hash": record_hash,
                    "conversations": conversations,
                }
            )

    write_jsonl(filtered_output, kept_rows)
    if args.keep_rejected:
        write_jsonl(rejected_output, rejected_rows)

    serializable_source_stats: Dict[str, Dict[str, Any]] = {}
    global_reason_counts: Counter[str] = Counter()
    for source_name, payload in source_stats.items():
        global_reason_counts.update(payload["reason_counts"])
        serializable_source_stats[source_name] = {
            "total": payload["total"],
            "kept": payload["kept"],
            "rejected": payload["rejected"],
            "duplicates": payload["duplicates"],
            "keep_ratio": round(payload["kept"] / payload["total"], 4) if payload["total"] else 0.0,
            "reason_counts": dict(payload["reason_counts"].most_common()),
        }

    stats["total_rows"] = sum(item["total"] for item in source_stats.values())
    stats["kept_rows"] = len(kept_rows)
    stats["rejected_rows"] = stats["total_rows"] - len(kept_rows)
    stats["keep_ratio"] = round(len(kept_rows) / stats["total_rows"], 4) if stats["total_rows"] else 0.0
    stats["reason_counts"] = dict(global_reason_counts.most_common())
    stats["source_stats"] = serializable_source_stats

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
