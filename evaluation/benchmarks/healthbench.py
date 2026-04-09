from __future__ import annotations

import json
import random
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DATASET_URLS = {
    "full": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
    "hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
}


@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: list[str]


@dataclass
class HealthBenchExample:
    prompt_id: str
    prompt: list[dict[str, str]]
    rubrics: list[RubricItem]
    example_tags: list[str]
    canary: str | None = None

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "example_tags": self.example_tags,
            "num_rubrics": len(self.rubrics),
        }


def _download_file(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path

    with urllib.request.urlopen(url) as response:
        data = response.read()
    target_path.write_bytes(data)
    return target_path


def _cache_path(cache_root: Path, subset_name: str) -> Path:
    filename = {
        "full": "healthbench_full.jsonl",
        "consensus": "healthbench_consensus.jsonl",
        "hard": "healthbench_hard.jsonl",
    }[subset_name]
    return cache_root / "evals" / "healthbench" / filename


def load_healthbench_examples(
    subset_name: str,
    cache_root: Path,
    max_examples: int = -1,
    seed: int = 42,
    shuffle: bool = False,
) -> tuple[list[HealthBenchExample], Path]:
    if subset_name not in DATASET_URLS:
        raise ValueError(f"Unsupported HealthBench subset: {subset_name}")

    cache_path = _cache_path(cache_root, subset_name)
    local_path = _download_file(DATASET_URLS[subset_name], cache_path)

    examples: list[HealthBenchExample] = []
    with local_path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            examples.append(
                HealthBenchExample(
                    prompt_id=str(payload["prompt_id"]),
                    prompt=payload["prompt"],
                    rubrics=[RubricItem(**rubric) for rubric in payload["rubrics"]],
                    example_tags=payload.get("example_tags", []),
                    canary=payload.get("canary"),
                )
            )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)
    if max_examples is not None and max_examples >= 0:
        examples = examples[:max_examples]
    return examples, local_path


def render_conversation_for_judge(
    prompt_messages: list[dict[str, str]],
    assistant_response: str,
) -> str:
    rendered_messages = prompt_messages + [{"role": "assistant", "content": assistant_response}]
    lines = [f"{message['role']}: {message['content']}" for message in rendered_messages]
    return "\n".join(lines)


def calculate_example_score(judgments: list[dict[str, Any]]) -> float | None:
    positive_points = sum(item["points"] for item in judgments if item["points"] > 0)
    if positive_points <= 0:
        return None
    achieved_points = sum(
        item["points"] for item in judgments if item.get("criteria_met")
    )
    return achieved_points / positive_points


def calculate_axis_scores(judgments: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in judgments:
        for tag in item.get("tags", []):
            if tag.startswith("axis:"):
                grouped[tag].append(item)

    scores: dict[str, float] = {}
    for axis_tag, axis_judgments in grouped.items():
        axis_score = calculate_example_score(axis_judgments)
        if axis_score is not None:
            scores[axis_tag] = axis_score
    return scores


def aggregate_healthbench_results(
    judged_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    scores: list[float] = []
    axis_values: dict[str, list[float]] = defaultdict(list)
    theme_values: dict[str, list[float]] = defaultdict(list)
    physician_values: dict[str, list[float]] = defaultdict(list)

    for row in judged_rows:
        score = row.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))

        for axis_tag, axis_score in row.get("axis_scores", {}).items():
            axis_values[axis_tag].append(float(axis_score))

        for tag in row.get("example_tags", []):
            if tag.startswith("theme:") and isinstance(score, (int, float)):
                theme_values[tag].append(float(score))
            if tag.startswith("physician_agreed_category:") and isinstance(score, (int, float)):
                physician_values[tag].append(float(score))

    def summarize(values: list[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        raw_mean = sum(values) / len(values)
        clipped_mean = min(max(raw_mean, 0.0), 1.0)
        return {
            "raw_mean": raw_mean,
            "clipped_mean": clipped_mean,
            "n": len(values),
        }

    by_axis = {
        key: summary(value_list)
        for key, value_list in sorted(axis_values.items())
        if summary(value_list) is not None
    }
    by_theme = {
        key: summary(value_list)
        for key, value_list in sorted(theme_values.items())
        if summary(value_list) is not None
    }
    by_physician_category = {
        key: summary(value_list)
        for key, value_list in sorted(physician_values.items())
        if summary(value_list) is not None
    }

    return {
        "overall": summarize(scores),
        "by_axis": by_axis,
        "by_theme": by_theme,
        "by_physician_category": by_physician_category,
        "num_judged_examples": len(judged_rows),
    }


def build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# HealthBench Summary", ""]

    overall = summary.get("overall") or {}
    if overall:
        lines.extend(
            [
                "## Overall",
                "",
                f"- clipped_mean: {overall['clipped_mean']:.4f}",
                f"- raw_mean: {overall['raw_mean']:.4f}",
                f"- n: {overall['n']}",
                "",
            ]
        )

    for section_name in ("by_axis", "by_theme", "by_physician_category"):
        section = summary.get(section_name, {})
        if not section:
            continue
        lines.append(f"## {section_name}")
        lines.append("")
        lines.append("| key | clipped_mean | raw_mean | n |")
        lines.append("| --- | ---: | ---: | ---: |")
        for key, values in section.items():
            lines.append(
                f"| {key} | {values['clipped_mean']:.4f} | {values['raw_mean']:.4f} | {values['n']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def serialize_example(example: HealthBenchExample) -> dict[str, Any]:
    return {
        "prompt_id": example.prompt_id,
        "prompt": example.prompt,
        "rubrics": [asdict(rubric) for rubric in example.rubrics],
        "example_tags": example.example_tags,
        "canary": example.canary,
    }

