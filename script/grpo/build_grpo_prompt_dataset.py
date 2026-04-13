#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SHORTFALL_SLICES = [
    "communication",
    "global_health",
    "hedging",
    "context_seeking",
    "emergency",
]

PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "v0_balanced": {
        "slice_targets": {
            "communication": 900,
            "global_health": 700,
            "hedging": 600,
            "context_seeking": 450,
            "emergency": 350,
        },
        "valid_slice_targets": {
            "communication": 90,
            "global_health": 70,
            "hedging": 60,
            "context_seeking": 45,
            "emergency": 35,
        },
        "slice_source_ratios": {
            "communication": {"hq_holdout": 0.60, "dpo_v2_train": 0.40},
            "global_health": {"hq_holdout": 0.70, "dpo_v2_train": 0.30},
            "hedging": {"hq_holdout": 0.50, "dpo_v2_train": 0.50},
            "context_seeking": {"hq_holdout": 0.30, "dpo_v2_train": 0.70},
            "emergency": {"hq_holdout": 0.20, "dpo_v2_train": 0.80},
        },
        "reward_profiles": {
            "communication": {
                "communication_quality": 0.40,
                "context_awareness": 0.20,
                "hedging": 0.15,
                "emergency_referral": 0.05,
                "medical_plausibility": 0.20,
            },
            "global_health": {
                "communication_quality": 0.15,
                "context_awareness": 0.15,
                "hedging": 0.20,
                "emergency_referral": 0.10,
                "medical_plausibility": 0.40,
            },
            "hedging": {
                "communication_quality": 0.15,
                "context_awareness": 0.25,
                "hedging": 0.40,
                "emergency_referral": 0.05,
                "medical_plausibility": 0.15,
            },
            "context_seeking": {
                "communication_quality": 0.15,
                "context_awareness": 0.40,
                "hedging": 0.20,
                "emergency_referral": 0.10,
                "medical_plausibility": 0.15,
            },
            "emergency": {
                "communication_quality": 0.10,
                "context_awareness": 0.15,
                "hedging": 0.10,
                "emergency_referral": 0.45,
                "medical_plausibility": 0.20,
            },
        },
        "penalty_profiles": {
            "communication": {
                "unsafe_overclaim_penalty": 1.0,
                "missed_emergency_penalty": 0.6,
                "low_effort_penalty": 0.3,
            },
            "global_health": {
                "unsafe_overclaim_penalty": 1.0,
                "missed_emergency_penalty": 0.7,
                "low_effort_penalty": 0.2,
            },
            "hedging": {
                "unsafe_overclaim_penalty": 1.0,
                "missed_emergency_penalty": 0.6,
                "low_effort_penalty": 0.2,
            },
            "context_seeking": {
                "unsafe_overclaim_penalty": 1.0,
                "missed_emergency_penalty": 0.7,
                "low_effort_penalty": 0.2,
            },
            "emergency": {
                "unsafe_overclaim_penalty": 1.2,
                "missed_emergency_penalty": 1.2,
                "low_effort_penalty": 0.2,
            },
        },
    },
    "v1_emergency_context": {
        "slice_targets": {
            "communication": 780,
            "global_health": 620,
            "hedging": 550,
            "context_seeking": 500,
            "emergency": 550,
        },
        "valid_slice_targets": {
            "communication": 78,
            "global_health": 62,
            "hedging": 55,
            "context_seeking": 50,
            "emergency": 55,
        },
        "slice_source_ratios": {
            "communication": {"hq_holdout": 0.60, "dpo_v2_train": 0.40},
            "global_health": {"hq_holdout": 0.68, "dpo_v2_train": 0.32},
            "hedging": {"hq_holdout": 0.45, "dpo_v2_train": 0.55},
            "context_seeking": {"hq_holdout": 0.25, "dpo_v2_train": 0.75},
            "emergency": {"hq_holdout": 0.10, "dpo_v2_train": 0.90},
        },
        "reward_profiles": {
            "communication": {
                "communication_quality": 0.40,
                "context_awareness": 0.22,
                "hedging": 0.13,
                "emergency_referral": 0.05,
                "medical_plausibility": 0.20,
            },
            "global_health": {
                "communication_quality": 0.15,
                "context_awareness": 0.18,
                "hedging": 0.18,
                "emergency_referral": 0.12,
                "medical_plausibility": 0.37,
            },
            "hedging": {
                "communication_quality": 0.12,
                "context_awareness": 0.25,
                "hedging": 0.43,
                "emergency_referral": 0.05,
                "medical_plausibility": 0.15,
            },
            "context_seeking": {
                "communication_quality": 0.12,
                "context_awareness": 0.50,
                "hedging": 0.15,
                "emergency_referral": 0.13,
                "medical_plausibility": 0.10,
            },
            "emergency": {
                "communication_quality": 0.08,
                "context_awareness": 0.12,
                "hedging": 0.05,
                "emergency_referral": 0.65,
                "medical_plausibility": 0.10,
            },
        },
        "penalty_profiles": {
            "communication": {
                "unsafe_overclaim_penalty": 1.0,
                "missed_emergency_penalty": 0.8,
                "low_effort_penalty": 0.3,
            },
            "global_health": {
                "unsafe_overclaim_penalty": 1.0,
                "missed_emergency_penalty": 0.9,
                "low_effort_penalty": 0.2,
            },
            "hedging": {
                "unsafe_overclaim_penalty": 1.0,
                "missed_emergency_penalty": 0.8,
                "low_effort_penalty": 0.2,
            },
            "context_seeking": {
                "unsafe_overclaim_penalty": 1.05,
                "missed_emergency_penalty": 1.05,
                "low_effort_penalty": 0.2,
            },
            "emergency": {
                "unsafe_overclaim_penalty": 1.3,
                "missed_emergency_penalty": 1.8,
                "low_effort_penalty": 0.2,
            },
        },
    },
}

DPO_TARGET_TAG_TO_SLICE = {
    "axis:communication_quality": "communication",
    "theme:hedging": "hedging",
    "axis:context_awareness": "context_seeking",
    "theme:emergency_referrals": "emergency",
}

COMMUNICATION_MARKERS = [
    "担心",
    "害怕",
    "焦虑",
    "烦恼",
    "着急",
    "急盼",
    "麻烦医生",
    "请问医生",
    "怎么办",
    "不知道",
    "影响我日常生活",
    "很难受",
    "痛苦",
    "求助",
]

GLOBAL_HEALTH_MARKERS = [
    "疫苗",
    "接种",
    "传染",
    "传播",
    "感染",
    "流感",
    "新冠",
    "结核",
    "狂犬",
    "乙肝",
    "丙肝",
    "艾滋",
    "梅毒",
    "淋病",
    "hpv",
    "预防",
    "筛查",
    "隔离",
    "防护",
    "母婴",
    "孕期保健",
    "营养",
    "戒烟",
    "减重",
    "肥胖",
    "慢病管理",
    "注意事项",
    "注意什么",
    "预后",
    "复查",
    "复诊",
    "护理",
    "康复",
    "饮食",
    "食疗",
    "锻炼",
    "运动",
    "体检",
    "保健",
    "母乳",
    "宝宝",
    "孩子会传染",
]

EMERGENCY_MARKERS = [
    "胸痛",
    "胸闷",
    "呼吸困难",
    "喘不过气",
    "晕厥",
    "昏迷",
    "抽搐",
    "意识不清",
    "剧烈头痛",
    "口角歪斜",
    "肢体无力",
    "大出血",
    "便血",
    "黑便",
    "呕血",
    "高热",
    "持续高烧",
    "持续呕吐",
    "严重腹痛",
    "尿不出",
    "宫外孕",
    "阴道大量出血",
    "持续勃起",
    "外伤",
    "车祸",
]

AMBIGUITY_MARKERS = [
    "怎么回事",
    "什么情况",
    "是不是",
    "可能",
    "会不会",
    "能不能",
    "该怎么办",
    "怎么办",
    "如何判断",
    "严重吗",
    "要紧吗",
    "是否",
]

CHECKUP_MARKERS = [
    "检查",
    "化验",
    "彩超",
    "b超",
    "ct",
    "核磁",
    "片子",
    "住院",
    "挂什么科",
    "看什么科",
    "怎么判断",
]

FOLLOWUP_MARKERS = [
    "那我现在应该做些什么",
    "需要去医院进一步检查",
    "谢谢医生",
    "我想确认一下",
    "是否需要",
]

ABSOLUTE_BAD_MARKERS = [
    "一定是",
    "肯定是",
    "绝对是",
    "包治",
    "根治",
    "不用检查",
    "没必要检查",
]


@dataclass
class Candidate:
    prompt_id: str
    prompt: list[dict[str, str]]
    question: str
    source_dataset: str
    source_record_id: str
    source_hash: str
    source_split: str
    slice_scores: dict[str, float]
    slice_tags: list[str]
    risk_level: str
    priority_score: float
    selection_signals: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_answer: str | None = None
    reference_answer_kind: str | None = None
    negative_reference_answer: str | None = None
    negative_reference_kind: str | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build prompt-only GRPO data focused on DPO shortfall slices")
    parser.add_argument(
        "--dpo-processed",
        default="/home/qjh/llm_learning/my_medical_gpt/data/alignment/processed/dpo_v2/train/medical_pairwise_train_v2.processed.jsonl",
    )
    parser.add_argument(
        "--dpo-audit",
        default="/home/qjh/llm_learning/my_medical_gpt/data/alignment/reconstructed/dpo_v2/audits/train/medical_pairwise_train_v2.audit.jsonl",
    )
    parser.add_argument(
        "--hq-high-bucket",
        default="/home/qjh/llm_learning/my_medical_gpt/data/sft/curation/quality/medical_sft_huatuo_27w_rule_filtered.high_confidence_keep.jsonl",
    )
    parser.add_argument(
        "--hq-trained-subset",
        default="/home/qjh/llm_learning/my_medical_gpt/data/sft/curation/subsets/hq_50k_source_stratified.jsonl",
    )
    parser.add_argument(
        "--output-root",
        default="/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1",
    )
    parser.add_argument("--output-name", default="medical_grpo_prompt_v1")
    parser.add_argument(
        "--preset",
        default="v0_balanced",
        choices=sorted(PRESET_CONFIGS.keys()),
        help="Sampling/reward preset for different GRPO dataset versions",
    )
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--valid-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def stable_prompt_hash(messages: list[dict[str, str]]) -> str:
    payload = "||".join(f"{item['role']}::{normalize_text(item['content'])}" for item in messages)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def scale_targets_to_total(targets: dict[str, int], total: int) -> dict[str, int]:
    if total <= 0:
        return {key: 0 for key in targets}

    source_total = max(sum(targets.values()), 1)
    scaled = {key: max(1, int(round(value * total / source_total))) for key, value in targets.items()}
    current_total = sum(scaled.values())
    ordered_keys = sorted(targets, key=lambda item: targets[item], reverse=True)

    while current_total > total:
        candidate_key = max(ordered_keys, key=lambda item: scaled[item])
        if scaled[candidate_key] <= 1:
            break
        scaled[candidate_key] -= 1
        current_total -= 1

    while current_total < total:
        candidate_key = min(ordered_keys, key=lambda item: scaled[item])
        scaled[candidate_key] += 1
        current_total += 1
    return scaled


def contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def count_hits(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def infer_risk_level(question: str, emergency_score: float) -> str:
    if emergency_score >= 3.0 or contains_any(question, EMERGENCY_MARKERS):
        return "high"
    if emergency_score >= 1.5 or count_hits(question, AMBIGUITY_MARKERS) >= 2:
        return "medium"
    return "low"


def detect_slice_scores(
    *,
    question: str,
    prompt_messages: list[dict[str, str]],
    source_dataset: str,
    reference_text: str = "",
    dpo_target_tags: Optional[list[str]] = None,
    dpo_issue_tags: Optional[list[str]] = None,
    hq_features: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, float], list[str], list[str]]:
    text = question
    slice_scores = {name: 0.0 for name in SHORTFALL_SLICES}
    signals: list[str] = []

    if dpo_target_tags:
        for tag in dpo_target_tags:
            mapped = DPO_TARGET_TAG_TO_SLICE.get(tag)
            if mapped:
                slice_scores[mapped] += 2.0
                signals.append(f"dpo_target:{tag}")

    if dpo_issue_tags:
        issue_text = " ".join(dpo_issue_tags)
        if "poor_communication" in issue_text or "too_generic" in issue_text or "too_verbose" in issue_text:
            slice_scores["communication"] += 1.5
            signals.append("dpo_issue:communication")
        if "overconfidence" in issue_text:
            slice_scores["hedging"] += 1.5
            signals.append("dpo_issue:overconfidence")
        if "missing_context_awareness" in issue_text:
            slice_scores["context_seeking"] += 1.5
            signals.append("dpo_issue:missing_context_awareness")
        if "missing_emergency_referral" in issue_text or "unsafe_advice" in issue_text:
            slice_scores["emergency"] += 1.5
            signals.append("dpo_issue:emergency")

    communication_hits = count_hits(text, COMMUNICATION_MARKERS)
    if communication_hits:
        slice_scores["communication"] += min(2.0, 0.6 * communication_hits)
        signals.append("question:communication_markers")

    global_hits = count_hits(text, GLOBAL_HEALTH_MARKERS)
    if global_hits:
        slice_scores["global_health"] += min(3.0, 1.0 * global_hits)
        signals.append("question:global_health_markers")

    reference_global_hits = count_hits(reference_text, GLOBAL_HEALTH_MARKERS)
    if reference_global_hits:
        slice_scores["global_health"] += min(1.8, 0.4 * reference_global_hits)
        signals.append("reference:global_health_markers")

    ambiguity_hits = count_hits(text, AMBIGUITY_MARKERS)
    if ambiguity_hits:
        slice_scores["hedging"] += min(2.0, 0.7 * ambiguity_hits)
        slice_scores["context_seeking"] += min(2.0, 0.7 * ambiguity_hits)
        signals.append("question:ambiguity_markers")

    checkup_hits = count_hits(text, CHECKUP_MARKERS)
    if checkup_hits:
        slice_scores["context_seeking"] += min(2.5, 0.8 * checkup_hits)
        signals.append("question:checkup_markers")

    emergency_hits = count_hits(text, EMERGENCY_MARKERS)
    if emergency_hits:
        slice_scores["emergency"] += min(4.0, 1.4 * emergency_hits)
        slice_scores["context_seeking"] += 0.5
        signals.append("question:emergency_markers")

    if len(prompt_messages) >= 3:
        slice_scores["communication"] += 0.8
        slice_scores["context_seeking"] += 0.5
        signals.append("prompt:multi_turn_context")

    if contains_any(text, FOLLOWUP_MARKERS):
        slice_scores["communication"] += 0.6
        slice_scores["context_seeking"] += 0.6
        signals.append("question:followup_marker")

    if hq_features:
        referral_hits = float(hq_features.get("referral_hits", 0))
        hedging_hits = float(hq_features.get("hedging_hits", 0))
        structure_hits = float(hq_features.get("structure_hits", 0))
        high_risk_question = bool(hq_features.get("high_risk_question", False))

        if referral_hits > 0:
            slice_scores["emergency"] += min(1.0, 0.3 * referral_hits)
            signals.append("hq_feature:referral_hits")
        if hedging_hits > 0:
            slice_scores["hedging"] += min(1.0, 0.3 * hedging_hits)
            signals.append("hq_feature:hedging_hits")
        if structure_hits > 0:
            slice_scores["communication"] += min(0.8, 0.2 * structure_hits)
            signals.append("hq_feature:structure_hits")
        if high_risk_question:
            slice_scores["emergency"] += 1.0
            signals.append("hq_feature:high_risk_question")

    if source_dataset == "hq_holdout":
        slice_scores["communication"] += 0.2
        slice_scores["global_health"] += 0.2
        signals.append("source:fresh_hq_holdout")

    if len(question) >= 80:
        slice_scores["communication"] += 0.4
        slice_scores["context_seeking"] += 0.4
        signals.append("question:long_narrative")

    slice_tags = [name for name, score in slice_scores.items() if score >= 1.5]
    if not slice_tags:
        fallback_slice = max(slice_scores.items(), key=lambda item: item[1])[0]
        slice_tags = [fallback_slice]
        signals.append("fallback:max_score_slice")
    return slice_scores, sorted(set(slice_tags)), sorted(set(signals))


def compute_priority_score(
    *,
    question: str,
    slice_scores: dict[str, float],
    quality_score: float = 0.0,
    rewrite_strength: str = "",
    issue_tags: Optional[list[str]] = None,
) -> float:
    score = sum(slice_scores.values())
    score += min(1.0, len(question) / 200.0)
    score += min(1.2, quality_score)
    score += {"heavy": 1.0, "moderate": 0.6, "light": 0.2}.get(rewrite_strength, 0.0)
    score += min(1.0, 0.15 * len(issue_tags or []))
    return round(score, 4)


def build_dpo_candidates(processed_path: Path, audit_path: Path) -> list[Candidate]:
    processed_rows = list(read_jsonl(processed_path))
    audit_rows = list(read_jsonl(audit_path))
    if len(processed_rows) != len(audit_rows):
        raise ValueError(f"DPO processed/audit length mismatch: {len(processed_rows)} vs {len(audit_rows)}")

    candidates: list[Candidate] = []
    for index, (processed_row, audit_row) in enumerate(zip(processed_rows, audit_rows)):
        prompt_messages = processed_row["prompt"]
        question = prompt_messages[-1]["content"].strip()
        slice_scores, slice_tags, signals = detect_slice_scores(
            question=question,
            prompt_messages=prompt_messages,
            source_dataset="dpo_v2_train",
            reference_text=str(audit_row.get("response_chosen", "")),
            dpo_target_tags=audit_row.get("target_tags", []),
            dpo_issue_tags=audit_row.get("issue_tags", []),
        )
        risk_level = infer_risk_level(question, slice_scores["emergency"])
        candidate = Candidate(
            prompt_id=f"dpo_v2_train_{index:05d}",
            prompt=prompt_messages,
            question=question,
            source_dataset="dpo_v2_train",
            source_record_id=str(audit_row.get("sample_id", index)),
            source_hash=stable_prompt_hash(prompt_messages),
            source_split="train",
            slice_scores=slice_scores,
            slice_tags=slice_tags,
            risk_level=risk_level,
            priority_score=compute_priority_score(
                question=question,
                slice_scores=slice_scores,
                rewrite_strength=str(audit_row.get("rewrite_strength", "")),
                issue_tags=audit_row.get("issue_tags", []),
            ),
            selection_signals=signals,
            metadata={
                "rewrite_strength": audit_row.get("rewrite_strength"),
                "target_tags": audit_row.get("target_tags", []),
                "issue_tags": audit_row.get("issue_tags", []),
                "label_correct": audit_row.get("label_correct"),
                "swap_pair": audit_row.get("swap_pair"),
                "source_type": "dpo_reconstructed_pairwise",
            },
            reference_answer=audit_row.get("response_chosen"),
            reference_answer_kind="dpo_v2_chosen",
            negative_reference_answer=audit_row.get("response_rejected"),
            negative_reference_kind="dpo_v2_rejected",
        )
        candidates.append(candidate)
    return candidates


def convert_conversations_to_prompt(
    conversations: list[dict[str, str]],
) -> tuple[Optional[list[dict[str, str]]], Optional[str], Optional[str]]:
    if not conversations:
        return None, None, None

    normalized: list[dict[str, str]] = []
    for turn in conversations:
        role = turn.get("from")
        content = str(turn.get("value", "")).strip()
        if not content:
            continue
        if role == "human":
            normalized.append({"role": "user", "content": content})
        elif role == "gpt":
            normalized.append({"role": "assistant", "content": content})

    if len(normalized) < 2:
        return None, None, None
    if normalized[-1]["role"] != "assistant":
        return None, None, None

    reference_answer = normalized[-1]["content"]
    prompt_messages = normalized[:-1]
    if not prompt_messages or prompt_messages[-1]["role"] != "user":
        return None, None, None

    return prompt_messages, prompt_messages[-1]["content"], reference_answer


def load_hq_trained_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for row in read_jsonl(path):
        curation_hash = row.get("curation_hash")
        if curation_hash:
            hashes.add(str(curation_hash))
    return hashes


def build_hq_holdout_candidates(high_bucket_path: Path, trained_subset_path: Path) -> list[Candidate]:
    trained_hashes = load_hq_trained_hashes(trained_subset_path)
    candidates: list[Candidate] = []
    for index, row in enumerate(read_jsonl(high_bucket_path)):
        curation_hash = str(row.get("curation_hash", ""))
        if not curation_hash or curation_hash in trained_hashes:
            continue

        prompt_messages, question, reference_answer = convert_conversations_to_prompt(row.get("conversations", []))
        if not prompt_messages or not question:
            continue

        meta = row.get("meta", {})
        features = row.get("features", {})
        slice_scores, slice_tags, signals = detect_slice_scores(
            question=question,
            prompt_messages=prompt_messages,
            source_dataset="hq_holdout",
            reference_text=reference_answer or "",
            hq_features={
                "referral_hits": features.get("referral_score", 0) * 3,
                "hedging_hits": features.get("hedging_score", 0) * 2,
                "structure_hits": features.get("structure_score", 0) * 3,
                "high_risk_question": meta.get("high_risk_question", False),
            },
        )
        risk_level = infer_risk_level(question, slice_scores["emergency"])
        candidates.append(
            Candidate(
                prompt_id=f"hq_holdout_{index:05d}",
                prompt=prompt_messages,
                question=question,
                source_dataset="hq_holdout",
                source_record_id=curation_hash,
                source_hash=stable_prompt_hash(prompt_messages),
                source_split="holdout",
                slice_scores=slice_scores,
                slice_tags=slice_tags,
                risk_level=risk_level,
                priority_score=compute_priority_score(
                    question=question,
                    slice_scores=slice_scores,
                    quality_score=float(row.get("quality_score", 0.0)),
                ),
                selection_signals=signals,
                metadata={
                    "curation_source": row.get("curation_source"),
                    "quality_score": row.get("quality_score"),
                    "quality_bucket": row.get("quality_bucket"),
                    "source_type": "hq_unseen_holdout",
                    "turn_count": len(prompt_messages),
                },
                reference_answer=reference_answer,
                reference_answer_kind="hq_reference_assistant",
            )
        )
    return candidates


def source_sort_key(candidate: Candidate, slice_name: str, rng: random.Random) -> tuple[Any, ...]:
    freshness_bonus = 1 if candidate.source_dataset == "hq_holdout" else 0
    multi_turn_bonus = 1 if len(candidate.prompt) >= 3 else 0
    noise = rng.random()
    return (
        -candidate.slice_scores.get(slice_name, 0.0),
        -candidate.priority_score,
        -freshness_bonus,
        -multi_turn_bonus,
        candidate.source_record_id,
        noise,
    )


def allocate_source_targets(slice_name: str, total_target: int) -> dict[str, int]:
    raise RuntimeError("allocate_source_targets requires explicit slice_source_ratios")


def allocate_source_targets_from_ratios(slice_name: str, total_target: int, slice_source_ratios: dict[str, dict[str, float]]) -> dict[str, int]:
    ratios = slice_source_ratios[slice_name]
    allocated: dict[str, int] = {}
    running = 0
    ordered_sources = sorted(ratios)
    for source_name in ordered_sources:
        count = int(round(total_target * ratios[source_name]))
        allocated[source_name] = count
        running += count
    while running > total_target:
        source_name = max(ordered_sources, key=lambda item: allocated[item])
        allocated[source_name] -= 1
        running -= 1
    while running < total_target:
        source_name = min(ordered_sources, key=lambda item: allocated[item])
        allocated[source_name] += 1
        running += 1
    return allocated


def select_split_rows(
    *,
    candidates: list[Candidate],
    slice_targets: dict[str, int],
    slice_source_ratios: dict[str, dict[str, float]],
    reward_profiles: dict[str, dict[str, float]],
    penalty_profiles: dict[str, dict[str, float]],
    selected_source_hashes: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    available_by_slice: dict[str, list[Candidate]] = defaultdict(list)
    available_by_slice_source: dict[str, dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        if candidate.source_hash in selected_source_hashes:
            continue
        for slice_name in candidate.slice_tags:
            available_by_slice[slice_name].append(candidate)
            available_by_slice_source[slice_name][candidate.source_dataset].append(candidate)

    slice_order = sorted(
        slice_targets,
        key=lambda item: (
            len({cand.source_hash for cand in available_by_slice[item]}) / max(slice_targets[item], 1),
            item,
        ),
    )

    selected_rows: list[dict[str, Any]] = []
    for slice_name in slice_order:
        target_count = slice_targets[slice_name]
        source_targets = allocate_source_targets_from_ratios(slice_name, target_count, slice_source_ratios)

        for source_name, source_target in sorted(source_targets.items()):
            source_pool = sorted(
                available_by_slice_source[slice_name].get(source_name, []),
                key=lambda item: source_sort_key(item, slice_name, rng),
            )
            picked = 0
            for candidate in source_pool:
                if picked >= source_target:
                    break
                if candidate.source_hash in selected_source_hashes:
                    continue
                selected_rows.append(candidate_to_record(candidate, slice_name, reward_profiles, penalty_profiles))
                selected_source_hashes.add(candidate.source_hash)
                picked += 1

        current_count = sum(1 for row in selected_rows if row["primary_slice"] == slice_name)
        if current_count < target_count:
            fallback_pool = sorted(
                available_by_slice[slice_name],
                key=lambda item: source_sort_key(item, slice_name, rng),
            )
            for candidate in fallback_pool:
                if current_count >= target_count:
                    break
                if candidate.source_hash in selected_source_hashes:
                    continue
                selected_rows.append(candidate_to_record(candidate, slice_name, reward_profiles, penalty_profiles))
                selected_source_hashes.add(candidate.source_hash)
                current_count += 1

    desired_total = sum(slice_targets.values())
    if len(selected_rows) < desired_total:
        primary_counts = Counter(row["primary_slice"] for row in selected_rows)
        remaining_candidates = [
            candidate for candidate in candidates if candidate.source_hash not in selected_source_hashes
        ]
        remaining_candidates.sort(
            key=lambda item: (
                -item.priority_score,
                -len(item.slice_tags),
                item.source_record_id,
            )
        )
        for candidate in remaining_candidates:
            if len(selected_rows) >= desired_total:
                break
            underfilled_tags = [
                tag for tag in candidate.slice_tags if primary_counts[tag] < slice_targets.get(tag, 0)
            ]
            if underfilled_tags:
                assigned_slice = max(
                    underfilled_tags,
                    key=lambda tag: (
                        slice_targets[tag] - primary_counts[tag],
                        candidate.slice_scores.get(tag, 0.0),
                    ),
                )
            else:
                assigned_slice = max(
                    candidate.slice_tags,
                    key=lambda tag: candidate.slice_scores.get(tag, 0.0),
                )
            selected_rows.append(candidate_to_record(candidate, assigned_slice, reward_profiles, penalty_profiles))
            selected_source_hashes.add(candidate.source_hash)
            primary_counts[assigned_slice] += 1

    return selected_rows


def candidate_to_record(
    candidate: Candidate,
    primary_slice: str,
    reward_profiles: dict[str, dict[str, float]],
    penalty_profiles: dict[str, dict[str, float]],
) -> dict[str, Any]:
    reward_profile = reward_profiles[primary_slice]
    penalty_profile = penalty_profiles[primary_slice]
    hard_constraints: list[str] = []
    if candidate.risk_level == "high" or "emergency" in candidate.slice_tags:
        hard_constraints.append("must_not_miss_emergency_referral")
    if "hedging" in candidate.slice_tags or "context_seeking" in candidate.slice_tags:
        hard_constraints.append("must_not_overclaim_without_context")
    if "communication" in candidate.slice_tags:
        hard_constraints.append("must_remain_clear_and_actionable")
    return {
        "prompt_id": candidate.prompt_id,
        "prompt": candidate.prompt,
        "question": candidate.question,
        "primary_slice": primary_slice,
        "slice_tags": candidate.slice_tags,
        "slice_scores": candidate.slice_scores,
        "reward_profile": reward_profile,
        "penalty_profile": penalty_profile,
        "risk_level": candidate.risk_level,
        "source_dataset": candidate.source_dataset,
        "source_split": candidate.source_split,
        "source_record_id": candidate.source_record_id,
        "source_hash": candidate.source_hash,
        "selection_signals": candidate.selection_signals,
        "priority_score": candidate.priority_score,
        "hard_constraints": hard_constraints,
        "metadata": candidate.metadata,
        "reference_answer": candidate.reference_answer,
        "reference_answer_kind": candidate.reference_answer_kind,
        "negative_reference_answer": candidate.negative_reference_answer,
        "negative_reference_kind": candidate.negative_reference_kind,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_slice_counts = Counter(row["primary_slice"] for row in rows)
    slice_tag_counts = Counter(tag for row in rows for tag in row["slice_tags"])
    source_counts = Counter(row["source_dataset"] for row in rows)
    risk_counts = Counter(row["risk_level"] for row in rows)
    avg_priority = round(sum(float(row["priority_score"]) for row in rows) / len(rows), 4) if rows else 0.0
    return {
        "count": len(rows),
        "primary_slice_counts": dict(primary_slice_counts),
        "slice_tag_counts": dict(slice_tag_counts),
        "source_counts": dict(source_counts),
        "risk_counts": dict(risk_counts),
        "avg_priority_score": avg_priority,
    }


def sample_examples(rows: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_slice[row["primary_slice"]].append(row)
    for slice_name in SHORTFALL_SLICES:
        for row in by_slice.get(slice_name, [])[:limit]:
            examples.append(
                {
                    "primary_slice": row["primary_slice"],
                    "source_dataset": row["source_dataset"],
                    "question": row["question"],
                    "slice_tags": row["slice_tags"],
                    "risk_level": row["risk_level"],
                }
            )
    return examples


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)
    preset = PRESET_CONFIGS[args.preset]
    slice_targets = dict(preset["slice_targets"])
    valid_slice_targets = dict(preset["valid_slice_targets"])
    slice_source_ratios = dict(preset["slice_source_ratios"])
    reward_profiles = dict(preset["reward_profiles"])
    penalty_profiles = dict(preset["penalty_profiles"])

    output_root = Path(args.output_root).resolve()
    train_path = output_root / "train" / f"{args.output_name}.train.jsonl"
    valid_path = output_root / "valid" / f"{args.output_name}.valid.jsonl"
    report_path = output_root / "reports" / f"{args.output_name}.report.json"

    dpo_candidates = build_dpo_candidates(Path(args.dpo_processed).resolve(), Path(args.dpo_audit).resolve())
    hq_candidates = build_hq_holdout_candidates(
        Path(args.hq_high_bucket).resolve(),
        Path(args.hq_trained_subset).resolve(),
    )

    all_candidates = dpo_candidates + hq_candidates
    deduped_candidates: list[Candidate] = []
    seen_hashes: set[str] = set()
    for candidate in sorted(all_candidates, key=lambda item: (-item.priority_score, item.source_hash)):
        if candidate.source_hash in seen_hashes:
            continue
        seen_hashes.add(candidate.source_hash)
        deduped_candidates.append(candidate)

    train_slice_targets = scale_targets_to_total(slice_targets, args.train_size)
    valid_slice_targets = scale_targets_to_total(valid_slice_targets, args.valid_size)

    selected_hashes: set[str] = set()
    train_rows = select_split_rows(
        candidates=deduped_candidates,
        slice_targets=train_slice_targets,
        slice_source_ratios=slice_source_ratios,
        reward_profiles=reward_profiles,
        penalty_profiles=penalty_profiles,
        selected_source_hashes=selected_hashes,
        rng=rng,
    )
    valid_rows = select_split_rows(
        candidates=deduped_candidates,
        slice_targets=valid_slice_targets,
        slice_source_ratios=slice_source_ratios,
        reward_profiles=reward_profiles,
        penalty_profiles=penalty_profiles,
        selected_source_hashes=selected_hashes,
        rng=rng,
    )

    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)

    candidate_source_counts = Counter(candidate.source_dataset for candidate in deduped_candidates)
    candidate_slice_counts = Counter(
        tag for candidate in deduped_candidates for tag in candidate.slice_tags
    )
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "preset": args.preset,
        "input": {
            "dpo_processed": str(Path(args.dpo_processed).resolve()),
            "dpo_audit": str(Path(args.dpo_audit).resolve()),
            "hq_high_bucket": str(Path(args.hq_high_bucket).resolve()),
            "hq_trained_subset": str(Path(args.hq_trained_subset).resolve()),
        },
        "output": {
            "train_path": str(train_path),
            "valid_path": str(valid_path),
        },
        "preset_config": {
            "slice_targets": train_slice_targets,
            "valid_slice_targets": valid_slice_targets,
            "slice_source_ratios": slice_source_ratios,
            "reward_profiles": reward_profiles,
            "penalty_profiles": penalty_profiles,
        },
        "candidate_pool": {
            "deduped_count": len(deduped_candidates),
            "source_counts": dict(candidate_source_counts),
            "slice_counts": dict(candidate_slice_counts),
        },
        "train_summary": summarize_rows(train_rows),
        "valid_summary": summarize_rows(valid_rows),
        "train_examples": sample_examples(train_rows),
        "valid_examples": sample_examples(valid_rows, limit=2),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
