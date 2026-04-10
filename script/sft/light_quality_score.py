#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


STRUCTURE_MARKERS = ["1.", "2.", "3.", "一、", "二、", "三、", "首先", "其次", "另外", "最后"]
REFERRAL_PATTERNS = ["建议就医", "及时就医", "尽快就医", "到医院", "去医院", "急诊", "门诊", "检查", "复查"]
HEDGING_PATTERNS = ["可能", "一般", "通常", "建议", "需要结合", "不能单凭", "还需", "要结合"]
ABSOLUTE_PATTERNS = ["一定是", "肯定是", "绝对是", "必须就是", "包治", "根治", "完全治愈", "没问题"]
LOW_SIGNAL_PATTERNS = [
    "建议咨询医生",
    "建议去医院检查",
    "请到医院检查",
    "遵医嘱",
    "具体请咨询医生",
]
HIGH_RISK_PATTERNS = [
    "胸痛",
    "呼吸困难",
    "昏迷",
    "抽搐",
    "大出血",
    "剧烈腹痛",
    "意识不清",
    "急性",
    "心肌梗塞",
    "心梗",
    "主动脉夹层",
    "中风",
    "卒中",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight heuristic quality scorer for SFT data")
    parser.add_argument("--input-file", required=True)
    parser.add_argument(
        "--output-root",
        default="/home/qjh/llm_learning/my_medical_gpt/data/sft/curation",
        help="Root directory for curation outputs",
    )
    parser.add_argument("--output-name", default="medical_sft_huatuo_27w_rule_filtered")
    parser.add_argument("--high-threshold", type=float, default=0.72)
    parser.add_argument("--low-threshold", type=float, default=0.45)
    parser.add_argument("--sample-per-bucket", type=int, default=25)
    return parser


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\u3000", " ").split())


def count_hits(text: str, patterns: List[str]) -> int:
    return sum(1 for pattern in patterns if pattern in text)


def repeated_char_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 1.0
    counter = Counter(compact)
    return max(counter.values()) / len(compact)


def repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0.0
    counter = Counter(lines)
    return max(counter.values()) / len(lines)


def punctuation_count(text: str) -> int:
    return len(re.findall(r"[，。！？；：,.!?;:]", text))


def ideal_band_score(value: int, low: int, high: int, hard_low: int, hard_high: int) -> float:
    if value < hard_low or value > hard_high:
        return 0.0
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, (value - hard_low) / max(1, (low - hard_low)))
    return max(0.0, (hard_high - value) / max(1, (hard_high - high)))


def build_features(record: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    conversations = record["conversations"]
    user_text = normalize_text(conversations[-2]["value"]) if len(conversations) >= 2 else ""
    assistant_text = normalize_text(conversations[-1]["value"]) if len(conversations) >= 1 else ""

    assistant_chars = len(assistant_text)
    user_chars = len(user_text)
    punct = punctuation_count(assistant_text)
    paragraphs = len([part for part in re.split(r"[。！？\n]", assistant_text) if part.strip()])
    structure_hits = count_hits(assistant_text, STRUCTURE_MARKERS)
    referral_hits = count_hits(assistant_text, REFERRAL_PATTERNS)
    hedging_hits = count_hits(assistant_text, HEDGING_PATTERNS)
    absolute_hits = count_hits(assistant_text, ABSOLUTE_PATTERNS)
    low_signal_hits = count_hits(assistant_text, LOW_SIGNAL_PATTERNS)
    high_risk_question = count_hits(user_text, HIGH_RISK_PATTERNS) > 0
    char_repeat = repeated_char_ratio(assistant_text)
    line_repeat = repeated_line_ratio(assistant_text)

    features = {
        "length_score": ideal_band_score(assistant_chars, low=80, high=900, hard_low=24, hard_high=2200),
        "question_length_score": ideal_band_score(user_chars, low=12, high=280, hard_low=4, hard_high=1200),
        "structure_score": min(1.0, 0.35 * (punct >= 3) + 0.35 * (paragraphs >= 2) + 0.30 * min(structure_hits, 2)),
        "referral_score": min(1.0, 0.6 if referral_hits >= 1 else 0.0 + 0.4 if (high_risk_question and referral_hits >= 1) else 0.0),
        "hedging_score": min(1.0, hedging_hits / 2.0),
        "low_signal_penalty": min(1.0, 0.55 if low_signal_hits >= 1 else 0.0 + 0.45 if assistant_chars < 48 else 0.0),
        "absolute_penalty": min(1.0, absolute_hits / 2.0),
        "repetition_penalty": min(1.0, max(char_repeat - 0.18, 0.0) * 2.2 + max(line_repeat - 0.2, 0.0) * 2.0),
        "high_risk_no_referral_penalty": 1.0 if high_risk_question and referral_hits == 0 else 0.0,
    }
    extra = {
        "user_text": user_text,
        "assistant_text": assistant_text,
        "assistant_chars": assistant_chars,
        "user_chars": user_chars,
        "punctuation_count": punct,
        "paragraphs": paragraphs,
        "structure_hits": structure_hits,
        "referral_hits": referral_hits,
        "hedging_hits": hedging_hits,
        "absolute_hits": absolute_hits,
        "low_signal_hits": low_signal_hits,
        "high_risk_question": high_risk_question,
        "char_repeat": round(char_repeat, 4),
        "line_repeat": round(line_repeat, 4),
    }
    return features, extra


def compute_total_score(features: Dict[str, float]) -> float:
    score = (
        0.24 * features["length_score"]
        + 0.08 * features["question_length_score"]
        + 0.20 * features["structure_score"]
        + 0.18 * features["referral_score"]
        + 0.12 * features["hedging_score"]
        - 0.10 * features["low_signal_penalty"]
        - 0.10 * features["absolute_penalty"]
        - 0.08 * features["repetition_penalty"]
        - 0.10 * features["high_risk_no_referral_penalty"]
    )
    return max(0.0, min(1.0, score))


def assign_bucket(score: float, high_threshold: float, low_threshold: float) -> str:
    if score >= high_threshold:
        return "high_confidence_keep"
    if score < low_threshold:
        return "high_risk_drop"
    return "borderline_review"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input_file).resolve()
    output_root = Path(args.output_root).resolve()
    quality_root = output_root / "quality"
    quality_root.mkdir(parents=True, exist_ok=True)

    bucket_paths = {
        "high_confidence_keep": quality_root / f"{args.output_name}.high_confidence_keep.jsonl",
        "borderline_review": quality_root / f"{args.output_name}.borderline_review.jsonl",
        "high_risk_drop": quality_root / f"{args.output_name}.high_risk_drop.jsonl",
        "scored_all": quality_root / f"{args.output_name}.scored_all.jsonl",
    }
    report_path = output_root / "reports" / f"{args.output_name}.light_quality.summary.json"
    samples_path = output_root / "reports" / f"{args.output_name}.light_quality.samples.json"

    bucket_counts = Counter()
    source_bucket_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    source_score_sums = Counter()
    score_hist = [0] * 10
    top_samples: Dict[str, List[Dict[str, Any]]] = {
        "high_confidence_keep": [],
        "borderline_review": [],
        "high_risk_drop": [],
    }
    feature_sums = Counter()
    total_rows = 0

    writers = {name: path.open("w", encoding="utf-8") for name, path in bucket_paths.items()}
    try:
        for record in read_jsonl(input_path):
            total_rows += 1
            features, extra = build_features(record)
            score = compute_total_score(features)
            bucket = assign_bucket(score, args.high_threshold, args.low_threshold)
            source = record.get("curation_source", "unknown")

            scored_row = {
                "curation_source": source,
                "curation_hash": record.get("curation_hash"),
                "quality_score": round(score, 6),
                "quality_bucket": bucket,
                "features": {key: round(value, 6) for key, value in features.items()},
                "meta": extra,
                "conversations": record["conversations"],
            }

            line = json.dumps(scored_row, ensure_ascii=False) + "\n"
            writers["scored_all"].write(line)
            writers[bucket].write(line)

            bucket_counts[bucket] += 1
            source_bucket_counts[source][bucket] += 1
            source_score_sums[source] += score
            score_hist[min(9, int(score * 10))] += 1
            for key, value in features.items():
                feature_sums[key] += value

            if len(top_samples[bucket]) < args.sample_per_bucket:
                top_samples[bucket].append(
                    {
                        "source": source,
                        "quality_score": round(score, 4),
                        "user_text": extra["user_text"][:220],
                        "assistant_text": extra["assistant_text"][:320],
                    }
                )

    finally:
        for handle in writers.values():
            handle.close()

    source_stats: Dict[str, Dict[str, Any]] = {}
    for source, counts in source_bucket_counts.items():
        source_total = sum(counts.values())
        source_stats[source] = {
            "total": source_total,
            "avg_quality_score": round(source_score_sums[source] / source_total, 4) if source_total else 0.0,
            "bucket_counts": dict(counts),
            "bucket_ratios": {key: round(value / source_total, 4) for key, value in counts.items()},
        }

    feature_means = {key: round(value / total_rows, 4) for key, value in feature_sums.items()}
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path),
        "output_files": {key: str(value) for key, value in bucket_paths.items()},
        "thresholds": {
            "high_threshold": args.high_threshold,
            "low_threshold": args.low_threshold,
        },
        "total_rows": total_rows,
        "bucket_counts": dict(bucket_counts),
        "bucket_ratios": {key: round(value / total_rows, 4) for key, value in bucket_counts.items()},
        "source_stats": source_stats,
        "feature_means": feature_means,
        "score_histogram_bins_0_1": score_hist,
        "interpretation": {
            "high_confidence_keep": "可直接作为后续精选 SFT 候选池",
            "borderline_review": "建议后续结合分布抽样或更强 judge 二次筛选",
            "high_risk_drop": "当前规则下更像低信息/高风险/结构较差样本，默认不优先进入训练集",
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples_path.write_text(json.dumps(top_samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
