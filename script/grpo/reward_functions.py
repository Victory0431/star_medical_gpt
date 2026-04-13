#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence


EMPATHY_MARKERS = [
    "别太担心",
    "先别太紧张",
    "先别慌",
    "可以理解",
    "理解你的担心",
    "能理解你的焦虑",
    "确实会让人担心",
]

ACTIONABLE_MARKERS = [
    "建议",
    "可以先",
    "尽快",
    "及时",
    "复查",
    "检查",
    "监测",
    "观察",
    "就诊",
    "挂",
    "评估",
    "完善",
]

CHECKUP_MARKERS = [
    "化验",
    "抽血",
    "复查",
    "检查",
    "彩超",
    "b超",
    "ct",
    "核磁",
    "阴道镜",
    "tct",
    "hpv",
    "听力",
    "视力",
]

FOLLOWUP_MARKERS = [
    "请问",
    "有没有",
    "是否有",
    "还需要了解",
    "建议补充",
    "方便说一下",
    "需要补充",
    "如果方便",
]

HEDGING_MARKERS = [
    "可能",
    "不能仅凭",
    "还不能完全确定",
    "无法直接判断",
    "需要结合",
    "还要结合",
    "一般来说",
    "通常",
    "不一定",
    "建议进一步",
    "如果",
    "若",
]

EMERGENCY_PROMPT_MARKERS = [
    "胸痛",
    "胸闷",
    "呼吸困难",
    "喘不过气",
    "晕厥",
    "昏迷",
    "意识不清",
    "抽搐",
    "大出血",
    "阴道大量出血",
    "呕血",
    "黑便",
    "持续高烧",
    "持续呕吐",
    "严重腹痛",
    "宫外孕",
]

EMERGENCY_ACTION_MARKERS = [
    "立即就医",
    "尽快就医",
    "及时就医",
    "立即去急诊",
    "尽快去急诊",
    "建议急诊",
    "拨打120",
    "马上到医院",
    "建议住院",
    "急诊",
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

LOW_INFO_MARKERS = [
    "建议咨询医生",
    "建议去医院检查一下",
    "注意休息",
    "多喝水",
    "保持好心情",
    "遵医嘱",
    "祝你健康",
    "这个不好说",
]

STRUCTURE_MARKERS = [
    "首先",
    "其次",
    "另外",
    "同时",
    "第一",
    "第二",
    "1.",
    "2.",
]

USER_DISTRESS_MARKERS = [
    "担心",
    "害怕",
    "焦虑",
    "着急",
    "快急疯了",
    "不知道怎么办",
    "很难受",
    "痛苦",
]

NORMALIZE_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    if not text:
        return False
    return any(marker in text for marker in markers)


def _normalize_text(text: str) -> str:
    return NORMALIZE_RE.sub("", text.lower())


def _split_sentences(text: str) -> List[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _messages_to_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        parts: List[str] = []
        for item in messages:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)
    return str(messages or "")


def _last_user_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        for item in reversed(prompt):
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content", "")
                if isinstance(content, str):
                    return content
    return _messages_to_text(prompt)


def _completion_text(completion: Any) -> str:
    return _messages_to_text(completion)


def _char_ngrams(text: str, n: int = 2) -> Counter[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return Counter()
    if len(normalized) < n:
        return Counter(normalized)
    return Counter(normalized[i : i + n] for i in range(len(normalized) - n + 1))


def _overlap_f1(pred: str, ref: str) -> float:
    pred_counts = _char_ngrams(pred)
    ref_counts = _char_ngrams(ref)
    if not pred_counts or not ref_counts:
        return 0.0
    overlap = sum(min(count, ref_counts[token]) for token, count in pred_counts.items())
    pred_total = sum(pred_counts.values())
    ref_total = sum(ref_counts.values())
    if pred_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _sentence_repetition_penalty(text: str) -> float:
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return 0.0
    counts = Counter(sentences)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / max(1, len(sentences))


def _length_score(text: str, good_min: int = 40, good_max: int = 420) -> float:
    length = len(_normalize_text(text))
    if length == 0:
        return -1.0
    if length < 16:
        return -0.8
    if length < good_min:
        return -0.25
    if length <= good_max:
        return 0.25
    if length <= 700:
        return 0.1
    return -0.2


def _weight(profile: Dict[str, float] | None, key: str, default: float = 1.0) -> float:
    if not isinstance(profile, dict):
        return default
    value = profile.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _penalty_weight(profile: Dict[str, float] | None, key: str, default: float = 1.0) -> float:
    return _weight(profile, key, default)


def _clamp(value: float, min_value: float = -1.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _iter_rows(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    kwargs: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    size = len(prompts)
    for idx in range(size):
        row = {
            "prompt_text": _last_user_text(prompts[idx]),
            "completion_text": _completion_text(completions[idx]),
        }
        for key, value in kwargs.items():
            if isinstance(value, list) and idx < len(value):
                row[key] = value[idx]
            else:
                row[key] = value
        yield row


def communication_quality_reward(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    scores: List[float] = []
    for row in _iter_rows(prompts, completions, kwargs):
        completion = row["completion_text"]
        prompt_text = row["prompt_text"]
        reward_profile = row.get("reward_profile")

        raw = _length_score(completion)
        if _contains_any(completion, EMPATHY_MARKERS):
            raw += 0.35
        if _contains_any(completion, ACTIONABLE_MARKERS):
            raw += 0.25
        if _contains_any(completion, STRUCTURE_MARKERS) or "\n" in completion:
            raw += 0.10
        if _contains_any(prompt_text, USER_DISTRESS_MARKERS) and not _contains_any(completion, EMPATHY_MARKERS):
            raw -= 0.20
        if _contains_any(completion, LOW_INFO_MARKERS):
            raw -= 0.25
        raw -= 0.35 * _sentence_repetition_penalty(completion)

        scores.append(round(_clamp(raw, -1.2, 1.2) * _weight(reward_profile, "communication_quality", 1.0), 6))
    return scores


def context_awareness_reward(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    scores: List[float] = []
    for row in _iter_rows(prompts, completions, kwargs):
        completion = row["completion_text"]
        prompt_text = row["prompt_text"]
        slice_tags = row.get("slice_tags") or []
        hard_constraints = row.get("hard_constraints") or []
        reward_profile = row.get("reward_profile")

        needs_context = (
            "context_seeking" in slice_tags
            or "must_not_overclaim_without_context" in hard_constraints
            or _contains_any(prompt_text, ["怎么回事", "什么情况", "是不是", "会不会", "严重吗", "是否"])
        )

        raw = 0.0
        if needs_context:
            if _contains_any(completion, FOLLOWUP_MARKERS):
                raw += 0.45
            if _contains_any(completion, CHECKUP_MARKERS):
                raw += 0.30
            if _contains_any(completion, HEDGING_MARKERS):
                raw += 0.15
            if _contains_any(completion, ABSOLUTE_BAD_MARKERS):
                raw -= 0.85
        else:
            if _contains_any(completion, CHECKUP_MARKERS):
                raw += 0.10

        scores.append(round(_clamp(raw, -1.0, 1.0) * _weight(reward_profile, "context_awareness", 1.0), 6))
    return scores


def hedging_reward(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    scores: List[float] = []
    for row in _iter_rows(prompts, completions, kwargs):
        completion = row["completion_text"]
        prompt_text = row["prompt_text"]
        slice_tags = row.get("slice_tags") or []
        reward_profile = row.get("reward_profile")

        need_hedging = "hedging" in slice_tags or _contains_any(prompt_text, ["是不是", "可能", "会不会", "能不能", "如何判断"])

        raw = 0.0
        if need_hedging and _contains_any(completion, HEDGING_MARKERS):
            raw += 0.60
        if need_hedging and _contains_any(completion, CHECKUP_MARKERS):
            raw += 0.10
        if _contains_any(completion, ABSOLUTE_BAD_MARKERS):
            raw -= 0.80

        scores.append(round(_clamp(raw, -1.0, 1.0) * _weight(reward_profile, "hedging", 1.0), 6))
    return scores


def emergency_referral_reward(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    scores: List[float] = []
    for row in _iter_rows(prompts, completions, kwargs):
        completion = row["completion_text"]
        prompt_text = row["prompt_text"]
        slice_tags = row.get("slice_tags") or []
        hard_constraints = row.get("hard_constraints") or []
        risk_level = row.get("risk_level", "")
        reward_profile = row.get("reward_profile")

        emergency_expected = (
            "emergency" in slice_tags
            or "must_not_miss_emergency_referral" in hard_constraints
            or risk_level == "high"
            or _contains_any(prompt_text, EMERGENCY_PROMPT_MARKERS)
        )

        raw = 0.0
        if emergency_expected:
            if _contains_any(completion, EMERGENCY_ACTION_MARKERS):
                raw += 0.85
            elif _contains_any(completion, ["尽快", "及时", "就医", "住院"]):
                raw += 0.45
            else:
                raw -= 0.75
        elif _contains_any(completion, EMERGENCY_ACTION_MARKERS):
            raw -= 0.10

        scores.append(round(_clamp(raw, -1.0, 1.0) * _weight(reward_profile, "emergency_referral", 1.0), 6))
    return scores


def reference_alignment_reward(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    scores: List[float] = []
    for row in _iter_rows(prompts, completions, kwargs):
        completion = row["completion_text"]
        reference_answer = row.get("reference_answer") or ""
        negative_reference_answer = row.get("negative_reference_answer") or ""
        reward_profile = row.get("reward_profile")

        pos_overlap = _overlap_f1(completion, reference_answer)
        neg_overlap = _overlap_f1(completion, negative_reference_answer) if negative_reference_answer else 0.0

        raw = pos_overlap - 0.45 * neg_overlap
        if reference_answer and _contains_any(reference_answer, CHECKUP_MARKERS) and _contains_any(completion, CHECKUP_MARKERS):
            raw += 0.08
        if reference_answer and _contains_any(reference_answer, EMERGENCY_ACTION_MARKERS) and _contains_any(
            completion, EMERGENCY_ACTION_MARKERS
        ):
            raw += 0.08

        scores.append(round(_clamp(raw, -1.0, 1.0) * _weight(reward_profile, "medical_plausibility", 1.0), 6))
    return scores


def safety_penalty_reward(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    scores: List[float] = []
    for row in _iter_rows(prompts, completions, kwargs):
        completion = row["completion_text"]
        prompt_text = row["prompt_text"]
        slice_tags = row.get("slice_tags") or []
        hard_constraints = row.get("hard_constraints") or []
        risk_level = row.get("risk_level", "")
        penalty_profile = row.get("penalty_profile")

        raw = 0.0
        normalized = _normalize_text(completion)
        if len(normalized) < 18:
            raw -= 0.55 * _penalty_weight(penalty_profile, "low_effort_penalty", 1.0)
        if _contains_any(completion, LOW_INFO_MARKERS):
            raw -= 0.30 * _penalty_weight(penalty_profile, "low_effort_penalty", 1.0)
        if _contains_any(completion, ABSOLUTE_BAD_MARKERS):
            raw -= 0.60 * _penalty_weight(penalty_profile, "unsafe_overclaim_penalty", 1.0)

        emergency_expected = (
            "emergency" in slice_tags
            or "must_not_miss_emergency_referral" in hard_constraints
            or risk_level == "high"
            or _contains_any(prompt_text, EMERGENCY_PROMPT_MARKERS)
        )
        if emergency_expected and not _contains_any(completion, EMERGENCY_ACTION_MARKERS + ["尽快", "及时", "就医"]):
            raw -= 0.55 * _penalty_weight(penalty_profile, "missed_emergency_penalty", 1.0)

        raw -= 0.35 * _sentence_repetition_penalty(completion)
        scores.append(round(_clamp(raw, -1.5, 0.0), 6))
    return scores


DEFAULT_REWARD_FUNCS = [
    communication_quality_reward,
    context_awareness_reward,
    hedging_reward,
    emergency_referral_reward,
    reference_alignment_reward,
    safety_penalty_reward,
]


DEFAULT_REWARD_DESCRIPTIONS = {
    "communication_quality_reward": "鼓励更清晰、安抚性更强、可执行性更好的医疗沟通。",
    "context_awareness_reward": "鼓励在信息不充分时先补上下文、补检查，而不是直接过度断言。",
    "hedging_reward": "鼓励在不确定医疗场景里用合适的保守表达。",
    "emergency_referral_reward": "鼓励高风险样本里保留及时就医/急诊建议。",
    "reference_alignment_reward": "用正参考答案与负参考答案做轻量 lexical margin，约束医学方向别跑偏。",
    "safety_penalty_reward": "对过度断言、低信息回答、漏急诊建议和明显重复做惩罚。",
}


__all__ = [
    "DEFAULT_REWARD_DESCRIPTIONS",
    "DEFAULT_REWARD_FUNCS",
    "communication_quality_reward",
    "context_awareness_reward",
    "hedging_reward",
    "emergency_referral_reward",
    "reference_alignment_reward",
    "safety_penalty_reward",
]
