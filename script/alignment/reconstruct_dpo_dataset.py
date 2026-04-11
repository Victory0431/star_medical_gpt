#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests


HEALTHBENCH_REWRITE_SYSTEM_PROMPT = """
你是一名“医疗偏好数据重构器”，负责把旧的 medical pairwise 数据重构成更适合 DPO v2 的高质量偏好样本。

你的唯一目标不是让 chosen 更长，而是让 chosen 更符合以下 HealthBench 风格优化方向：
1. medical accuracy：避免明显错误、过度确定、错误治疗建议。
2. communication quality：表达清晰、自然、友好，不堆砌无关内容。
3. context awareness：信息不足时不过度下结论，必要时先提示需要补充信息或检查。
4. emergency referrals：一旦存在危险信号、持续恶化或需要面诊/急诊的可能，应明确建议及时就医。
5. hedging：在不确定时保持审慎，不把猜测说成事实。

重构原则：
- 必须始终输出一个更优的 final_chosen 和一个更差的 final_rejected。
- final_chosen 应该是“更适合医疗助手上线”的回答。
- final_rejected 应该仍然像一个 plausible 的回答，但明显缺少某些关键质量，不要写成胡言乱语。
- 尽量保留原始问句语境，使用中文回答。
- 不要提到 HealthBench、打分、评测、chosen/rejected、训练数据等元信息。
- 不要无端引入非常具体但缺乏依据的医学结论。
- 如果问题信息不足，final_chosen 可以先给安全范围内的初步建议，并补充“建议就医/检查/补充关键信息”。
- 如果原始 chosen/rejected 标签方向错了，可以交换。
- 如果两边都不好，请重写两边，但仍需保留一个明显更优、一个明显更差。
- final_chosen 和 final_rejected 都应避免过长。多数样本控制在 1 到 5 句；只有确实需要时再更详细。

输出必须是合法 JSON，不要输出任何额外文本。
""".strip()


HEALTHBENCH_REWRITE_USER_TEMPLATE = """
请重构下面这条医疗 pairwise 数据。

【问题】
{question}

【原始 chosen】
{chosen}

【原始 rejected】
{rejected}

请按以下要求输出 JSON：
{{
  "label_correct": true,
  "swap_pair": false,
  "rewrite_strength": "light|moderate|heavy",
  "issue_tags": ["可多选，示例：factual_risk", "overconfidence", "poor_communication", "missing_context_awareness", "missing_emergency_referral", "too_generic", "too_verbose", "irrelevant_content", "unsafe_advice", "label_direction_wrong"],
  "target_tags": ["可多选，示例：axis:accuracy", "axis:communication_quality", "axis:context_awareness", "theme:emergency_referrals", "theme:hedging"],
  "rationale": "一句到两句中文说明，解释为什么这样改。",
  "final_chosen": "重构后的更优回答",
  "final_rejected": "重构后的更差回答"
}}

要求：
- final_chosen 必须优于 final_rejected。
- 如果原标签方向不对，请把 swap_pair 设为 true，并确保 final_chosen 是真正更优的一边。
- final_rejected 可以比较差，但不要完全离题。
- final_chosen 优先优化：准确、安全、沟通清晰、信息不足时保持审慎、必要时建议检查或就医。
""".strip()


BATCH_REWRITE_USER_TEMPLATE = """
请批量重构下面这些医疗 pairwise 数据。返回一个 JSON 对象，格式必须为：
{
  "items": [
    {
      "sample_id": "...",
      "label_correct": true,
      "swap_pair": false,
      "rewrite_strength": "light|moderate|heavy",
      "issue_tags": ["..."],
      "target_tags": ["..."],
      "rationale": "...",
      "final_chosen": "...",
      "final_rejected": "..."
    }
  ]
}

要求：
- items 数量必须和输入样本数量完全一致。
- 每个 sample_id 都必须原样返回。
- final_chosen 必须优于 final_rejected，并且更符合以下方向：准确、安全、清晰、上下文感知、必要时建议检查/就医、不过度确定。
- final_rejected 仍需保持像一个 plausible 的较差回答，不要写成胡言乱语。
- 只返回 JSON，不要输出任何解释性文本。

样本列表：
{samples_json}
""".strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct medical pairwise DPO data with an OpenAI-compatible judge model.")
    parser.add_argument("--input-file", required=True, help="Raw medical pairwise jsonl file")
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument("--output-name", required=True, help="Output file prefix")
    parser.add_argument(
        "--output-root",
        default="/home/qjh/llm_learning/my_medical_gpt/data/alignment/reconstructed/dpo_v2",
        help="Root directory for reconstructed datasets",
    )
    parser.add_argument("--model", default="gpt-5.2", help="OpenAI-compatible model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL; falls back to OPENAI_BASE_URL")
    parser.add_argument("--api-key", default=None, help="API key; falls back to OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--min-request-interval", type=float, default=0.8, help="Minimum interval in seconds between request starts")
    parser.add_argument("--retry-base-sleep", type=float, default=3.0, help="Base backoff seconds for retryable errors")
    parser.add_argument("--retry-max-sleep", type=float, default=45.0, help="Max backoff seconds for retryable errors")
    parser.add_argument("--progress-log-interval", type=int, default=20, help="Log progress every N completed samples")
    parser.add_argument("--strict-serial", action="store_true", help="Force single-thread, one-sample-at-a-time reconstruction")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser


def normalize_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base_url: {base_url}")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/chat/completions"


def setup_logger(log_paths: list[Path]):
    import logging

    logger = logging.getLogger("my_medical_gpt.reconstruct_dpo_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    for log_path in log_paths:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(row["sample_id"]): row for row in rows if "sample_id" in row}


def build_sample_id(split: str, index: int, question: str) -> str:
    digest = hashlib.md5(question.encode("utf-8")).hexdigest()[:10]
    return f"{split}_{index:05d}_{digest}"


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Failed to locate JSON object in response: {text[:200]}")
    payload = cleaned[start : end + 1]
    return json.loads(payload)


class OpenAICompatibleJsonClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout_seconds: int,
        max_retries: int,
        temperature: float,
        min_request_interval: float,
        retry_base_sleep: float,
        retry_max_sleep: float,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = normalize_chat_completions_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature
        self.min_request_interval = max(0.0, min_request_interval)
        self.retry_base_sleep = max(0.1, retry_base_sleep)
        self.retry_max_sleep = max(self.retry_base_sleep, retry_max_sleep)
        self.session = requests.Session()
        self._request_gate = threading.Lock()
        self._last_request_started_at = 0.0

    def _wait_for_turn(self) -> None:
        if self.min_request_interval <= 0:
            return
        with self._request_gate:
            now = time.monotonic()
            wait_seconds = self.min_request_interval - (now - self._last_request_started_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_request_started_at = time.monotonic()

    def _compute_backoff(self, attempt: int, response: Optional[requests.Response] = None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), self.retry_max_sleep)
                except ValueError:
                    pass
        base = min(self.retry_base_sleep * (2 ** max(0, attempt - 1)), self.retry_max_sleep)
        jitter = random.uniform(0.0, min(1.5, base * 0.2))
        return min(base + jitter, self.retry_max_sleep)

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        for attempt in range(1, self.max_retries + 1):
            response: Optional[requests.Response] = None
            try:
                self._wait_for_turn()
                response = self.session.post(
                    self.base_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                response_payload = response.json()
                content = response_payload["choices"][0]["message"]["content"]
                parsed = extract_json_object(content)
                parsed["_raw_response"] = content
                parsed["_actual_model"] = response_payload.get("model")
                return parsed
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI-compatible request failed after {self.max_retries} attempts: {exc}") from exc
                status_code = getattr(response, "status_code", None)
                if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
                    time.sleep(self._compute_backoff(attempt, response))
                    continue
                if isinstance(exc, requests.RequestException):
                    time.sleep(self._compute_backoff(attempt))
                    continue
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError("Unreachable")


def sanitize_issue_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        tag = str(item).strip()
        if tag:
            cleaned.append(tag)
    return cleaned


def sanitize_rewrite_result(result: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    final_chosen = str(result.get("final_chosen", "")).strip()
    final_rejected = str(result.get("final_rejected", "")).strip()
    if not final_chosen:
        final_chosen = str(record["response_chosen"]).strip()
    if not final_rejected:
        final_rejected = str(record["response_rejected"]).strip()
    if final_chosen == final_rejected:
        final_rejected = str(record["response_rejected"]).strip()
    return {
        "label_correct": bool(result.get("label_correct", True)),
        "swap_pair": bool(result.get("swap_pair", False)),
        "rewrite_strength": str(result.get("rewrite_strength", "moderate")).strip() or "moderate",
        "issue_tags": sanitize_issue_tags(result.get("issue_tags")),
        "target_tags": sanitize_issue_tags(result.get("target_tags")),
        "rationale": str(result.get("rationale", "")).strip(),
        "final_chosen": final_chosen,
        "final_rejected": final_rejected,
        "actual_model": result.get("_actual_model"),
        "raw_response": result.get("_raw_response", ""),
    }


def sanitize_batch_result_item(result: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return sanitize_rewrite_result(result, record)


def reconstruct_one(
    client: OpenAICompatibleJsonClient,
    split: str,
    index: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    question = str(record["question"]).strip()
    chosen = str(record["response_chosen"]).strip()
    rejected = str(record["response_rejected"]).strip()
    sample_id = build_sample_id(split, index, question)
    user_prompt = HEALTHBENCH_REWRITE_USER_TEMPLATE.format(
        question=question,
        chosen=chosen,
        rejected=rejected,
    )
    response = client.chat_json(HEALTHBENCH_REWRITE_SYSTEM_PROMPT, user_prompt)
    normalized = sanitize_rewrite_result(response, record)

    raw_row = {
        "sample_id": sample_id,
        "question": question,
        "response_chosen": normalized["final_chosen"],
        "response_rejected": normalized["final_rejected"],
        "source": "medical_pairwise_reconstructed_v2",
        "original_response_chosen": chosen,
        "original_response_rejected": rejected,
        "label_correct": normalized["label_correct"],
        "swap_pair": normalized["swap_pair"],
        "rewrite_strength": normalized["rewrite_strength"],
        "issue_tags": normalized["issue_tags"],
        "target_tags": normalized["target_tags"],
        "rationale": normalized["rationale"],
        "reconstruction_model": normalized["actual_model"] or client.model,
    }
    audit_row = dict(raw_row)
    audit_row.update(
        {
            "raw_response": normalized["raw_response"],
            "split": split,
            "index": index,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return {
        "sample_id": sample_id,
        "raw_row": raw_row,
        "audit_row": audit_row,
    }


def reconstruct_batch(
    client: OpenAICompatibleJsonClient,
    split: str,
    items: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    payload_items = []
    record_by_sample_id: dict[str, dict[str, Any]] = {}
    index_by_sample_id: dict[str, int] = {}
    for index, record in items:
        sample_id = build_sample_id(split, index, str(record.get("question", "")))
        payload_items.append(
            {
                "sample_id": sample_id,
                "question": str(record["question"]).strip(),
                "chosen": str(record["response_chosen"]).strip(),
                "rejected": str(record["response_rejected"]).strip(),
            }
        )
        record_by_sample_id[sample_id] = record
        index_by_sample_id[sample_id] = index

    user_prompt = BATCH_REWRITE_USER_TEMPLATE.format(
        samples_json=json.dumps(payload_items, ensure_ascii=False, indent=2)
    )
    response = client.chat_json(HEALTHBENCH_REWRITE_SYSTEM_PROMPT, user_prompt)
    items_payload = response.get("items")
    if not isinstance(items_payload, list):
        raise ValueError(f"Batch response missing items list: {response}")

    normalized_results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items_payload:
        sample_id = str(item.get("sample_id", "")).strip()
        if not sample_id or sample_id not in record_by_sample_id:
            raise ValueError(f"Unexpected sample_id in batch response: {sample_id}")
        seen.add(sample_id)
        record = record_by_sample_id[sample_id]
        normalized = sanitize_batch_result_item(
            {
                **item,
                "_actual_model": response.get("_actual_model"),
                "_raw_response": response.get("_raw_response", ""),
            },
            record,
        )
        raw_row = {
            "sample_id": sample_id,
            "question": str(record["question"]).strip(),
            "response_chosen": normalized["final_chosen"],
            "response_rejected": normalized["final_rejected"],
            "source": "medical_pairwise_reconstructed_v2",
            "original_response_chosen": str(record["response_chosen"]).strip(),
            "original_response_rejected": str(record["response_rejected"]).strip(),
            "label_correct": normalized["label_correct"],
            "swap_pair": normalized["swap_pair"],
            "rewrite_strength": normalized["rewrite_strength"],
            "issue_tags": normalized["issue_tags"],
            "target_tags": normalized["target_tags"],
            "rationale": normalized["rationale"],
            "reconstruction_model": normalized["actual_model"] or client.model,
        }
        audit_row = dict(raw_row)
        audit_row.update(
            {
                "raw_response": normalized["raw_response"],
                "split": split,
                "index": index_by_sample_id[sample_id],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        normalized_results.append(
            {
                "sample_id": sample_id,
                "raw_row": raw_row,
                "audit_row": audit_row,
            }
        )
    missing = set(record_by_sample_id) - seen
    if missing:
        raise ValueError(f"Batch response missing sample_ids: {sorted(missing)[:5]}")
    normalized_results.sort(key=lambda row: row["sample_id"])
    return normalized_results


def summarize_audit_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    issue_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    swap_count = 0
    rewrite_counter: Counter[str] = Counter()
    changed_count = 0
    for row in rows:
        issue_counter.update(row.get("issue_tags", []))
        target_counter.update(row.get("target_tags", []))
        rewrite_counter.update([row.get("rewrite_strength", "unknown")])
        if row.get("swap_pair"):
            swap_count += 1
        if (
            row.get("response_chosen") != row.get("original_response_chosen")
            or row.get("response_rejected") != row.get("original_response_rejected")
        ):
            changed_count += 1
    return {
        "total_rows": len(rows),
        "swap_count": swap_count,
        "changed_count": changed_count,
        "changed_ratio": round(changed_count / len(rows), 4) if rows else 0.0,
        "rewrite_strength_counts": dict(rewrite_counter.most_common()),
        "issue_tag_counts": dict(issue_counter.most_common()),
        "target_tag_counts": dict(target_counter.most_common()),
    }


def format_eta(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.strict_serial:
        args.max_workers = 1
        args.batch_size = 1

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required")
    if not base_url:
        raise EnvironmentError("OPENAI_BASE_URL is required")

    input_path = Path(args.input_file).resolve()
    output_root = Path(args.output_root).resolve()
    raw_output_path = output_root / "raw" / args.split / f"{args.output_name}.jsonl"
    audit_output_path = output_root / "audits" / args.split / f"{args.output_name}.audit.jsonl"
    report_output_path = output_root / "reports" / f"{args.output_name}.{args.split}.summary.json"
    central_log_path = Path("/home/qjh/llm_learning/my_medical_gpt/outputs/logs/dpo_reconstruct") / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.output_name}_{args.split}.log"
    )
    logger = setup_logger([output_root / "logs" / f"{args.output_name}.{args.split}.log", central_log_path])

    if args.overwrite:
        for path in [raw_output_path, audit_output_path, report_output_path]:
            if path.exists():
                path.unlink()

    rows = read_jsonl(input_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    existing_audits = load_existing_by_id(audit_output_path) if args.resume and not args.overwrite else {}
    logger.info("Loaded %d input rows from %s", len(rows), input_path)
    logger.info("Existing reconstructed rows: %d", len(existing_audits))

    client = OpenAICompatibleJsonClient(
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        temperature=args.temperature,
        min_request_interval=args.min_request_interval,
        retry_base_sleep=args.retry_base_sleep,
        retry_max_sleep=args.retry_max_sleep,
    )

    write_lock = threading.Lock()
    completed = 0
    skipped = 0
    start_time = time.monotonic()

    pending_items = []
    for index, record in enumerate(rows):
        sample_id = build_sample_id(args.split, index, str(record.get("question", "")))
        if sample_id in existing_audits:
            skipped += 1
            continue
        pending_items.append((index, record))
    logger.info("Pending rows to process: %d", len(pending_items))
    logger.info(
        "Reconstruction mode: strict_serial=%s max_workers=%d batch_size=%d min_request_interval=%.2fs",
        args.strict_serial,
        args.max_workers,
        args.batch_size,
        args.min_request_interval,
    )

    def make_batches(items: list[tuple[int, dict[str, Any]]], size: int) -> list[list[tuple[int, dict[str, Any]]]]:
        batches = []
        for start in range(0, len(items), size):
            batches.append(items[start : start + size])
        return batches

    batch_size = max(1, args.batch_size)
    pending_batches = make_batches(pending_items, batch_size)

    def task(batch_items: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
        try:
            if len(batch_items) == 1:
                index, record = batch_items[0]
                return [reconstruct_one(client, args.split, index, record)]
            return reconstruct_batch(client, args.split, batch_items)
        except Exception:
            if len(batch_items) == 1:
                raise
            fallback_results = []
            for index, record in batch_items:
                fallback_results.append(reconstruct_one(client, args.split, index, record))
            return fallback_results

    def log_progress() -> None:
        if completed <= 0:
            return
        elapsed = max(time.monotonic() - start_time, 1e-6)
        avg_seconds = elapsed / completed
        remaining = max(len(pending_items) - completed, 0)
        eta_seconds = avg_seconds * remaining
        logger.info(
            "Reconstructed %d/%d samples (skipped=%d, avg=%.2fs/sample, eta=%s)",
            completed,
            len(pending_items),
            skipped,
            avg_seconds,
            format_eta(eta_seconds),
        )

    if args.strict_serial:
        for batch in pending_batches:
            index = batch[0][0]
            try:
                results = task(batch)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed at index=%s: %s", index, exc)
                raise
            for result in results:
                append_jsonl(raw_output_path, result["raw_row"])
                append_jsonl(audit_output_path, result["audit_row"])
                completed += 1
            if completed % max(1, args.progress_log_interval) == 0 or completed == len(pending_items):
                log_progress()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_map = {executor.submit(task, batch): batch[0][0] for batch in pending_batches}
            for future in concurrent.futures.as_completed(future_map):
                index = future_map[future]
                try:
                    results = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed at index=%s: %s", index, exc)
                    raise
                with write_lock:
                    for result in results:
                        append_jsonl(raw_output_path, result["raw_row"])
                        append_jsonl(audit_output_path, result["audit_row"])
                        completed += 1
                    if completed % max(1, args.progress_log_interval) == 0 or completed == len(pending_items):
                        log_progress()

    all_audit_rows = [row for _, row in sorted(load_existing_by_id(audit_output_path).items())]
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path),
        "split": args.split,
        "output_name": args.output_name,
        "raw_output_path": str(raw_output_path),
        "audit_output_path": str(audit_output_path),
        "model": args.model,
        "base_url": normalize_chat_completions_url(base_url),
        "max_workers": args.max_workers,
        "num_rows": len(all_audit_rows),
        "skipped_existing": skipped,
        "summary": summarize_audit_rows(all_audit_rows),
    }
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Reconstruction complete. Summary written to %s", report_output_path)


if __name__ == "__main__":
    main()
