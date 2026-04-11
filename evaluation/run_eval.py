#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmarks.healthbench import (
    aggregate_healthbench_results,
    build_summary_markdown,
    calculate_axis_scores,
    calculate_example_score,
    load_healthbench_examples,
    render_conversation_for_judge,
    serialize_example,
)
from evaluation.common import append_jsonl, load_jsonl, save_json, setup_logger, timestamp
from evaluation.generators.hf_chat import HFChatGenerator
from evaluation.judges.openai_healthbench import OpenAIHealthBenchJudge


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Star Medical GPT evaluations")
    parser.add_argument("--config", default=None, help="Optional JSON config file")
    parser.add_argument("--benchmark", default="healthbench", choices=["healthbench"])
    parser.add_argument("--subset-name", default="full", choices=["full", "consensus", "hard"])
    parser.add_argument("--mode", default="full", choices=["full", "generate_only", "judge_only"])
    parser.add_argument("--judge-mode", default="openai", choices=["openai", "none"])
    parser.add_argument("--judge-model", default="gpt-5.2")
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--model-alias", default=None)
    parser.add_argument("--cache-dir", default="/home/qjh/llm_learning/my_medical_gpt/cache")
    parser.add_argument("--output-root", default="/home/qjh/llm_learning/my_medical_gpt/outputs/eval")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling-mode",
        default="sequential",
        choices=["sequential", "stratified_theme"],
    )
    parser.add_argument(
        "--per-theme-examples",
        type=int,
        default=-1,
        help="When sampling_mode=stratified_theme, sample this many examples per theme.",
    )
    parser.add_argument("--generator-device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--responses-path", default=None)
    parser.add_argument("--judgments-path", default=None)
    parser.add_argument("--overwrite-responses", action="store_true", default=False)
    parser.add_argument("--overwrite-judgments", action="store_true", default=False)
    return parser


def slugify(text: str) -> str:
    cleaned = []
    for ch in text.lower():
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "model"


def infer_model_alias(model_name_or_path: str | None, adapter_path: str | None) -> str:
    if adapter_path:
        return slugify(Path(adapter_path).parent.name or Path(adapter_path).name)
    if model_name_or_path:
        return slugify(Path(model_name_or_path).name)
    return "unknown-model"


def default_run_name(args: argparse.Namespace) -> str:
    model_alias = args.model_alias or infer_model_alias(args.model_name_or_path, args.adapter_path)
    return f"{timestamp()}_{args.benchmark}_{args.subset_name}_{model_alias}_{args.mode}"


def load_responses(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    return {str(row["prompt_id"]): row for row in rows}


def load_judgments(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    return {str(row["prompt_id"]): row for row in rows}


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None, help="Optional JSON config file")
    bootstrap_args, remaining_argv = bootstrap.parse_known_args()

    parser = build_arg_parser()
    if bootstrap_args.config:
        config_path = Path(bootstrap_args.config)
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        parser_defaults = {}
        valid_dests = {action.dest for action in parser._actions}
        for key, value in config_payload.items():
            if key in valid_dests:
                parser_defaults[key] = value
        if parser_defaults:
            parser.set_defaults(**parser_defaults)
    args = parser.parse_args(remaining_argv)
    args.run_name = args.run_name or default_run_name(args)
    args.model_alias = args.model_alias or infer_model_alias(args.model_name_or_path, args.adapter_path)

    run_dir = Path(args.output_root) / args.run_name
    central_log_path = (
        PROJECT_ROOT / "evaluation" / "logs" / f"{timestamp()}_{args.run_name}.log"
    )
    logger = setup_logger(
        run_dir / "logs" / "eval.log",
        "my_medical_gpt.eval",
        extra_log_paths=[central_log_path],
    )
    save_json(vars(args), run_dir / "artifacts" / "run_args.json")

    if args.mode in {"full", "judge_only"} and args.judge_mode == "openai":
        if not OpenAIHealthBenchJudge(model_name=args.judge_model).api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required when judge_mode=openai and mode is full/judge_only."
            )

    if args.benchmark != "healthbench":
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    examples, local_dataset_path = load_healthbench_examples(
        subset_name=args.subset_name,
        cache_root=Path(args.cache_dir),
        max_examples=args.max_examples,
        seed=args.seed,
        shuffle=args.shuffle,
        sampling_mode=args.sampling_mode,
        per_theme_examples=args.per_theme_examples,
    )
    logger.info(
        "Loaded %d %s examples from %s with sampling_mode=%s per_theme_examples=%s shuffle=%s seed=%s",
        len(examples),
        args.subset_name,
        local_dataset_path,
        args.sampling_mode,
        args.per_theme_examples,
        args.shuffle,
        args.seed,
    )

    theme_counts: dict[str, int] = {}
    for example in examples:
        for tag in example.example_tags:
            if tag.startswith("theme:"):
                theme_counts[tag] = theme_counts.get(tag, 0) + 1

    if theme_counts:
        logger.info("Selected theme distribution: %s", json.dumps(theme_counts, ensure_ascii=False, sort_keys=True))

    save_json(
        {
            "benchmark": args.benchmark,
            "subset_name": args.subset_name,
            "num_examples": len(examples),
            "dataset_path": str(local_dataset_path),
            "sampling_mode": args.sampling_mode,
            "per_theme_examples": args.per_theme_examples,
            "theme_distribution": theme_counts,
            "manifest": [example.to_manifest_row() for example in examples],
        },
        run_dir / "artifacts" / "dataset_manifest.json",
    )

    responses_path = Path(args.responses_path) if args.responses_path else run_dir / "responses.jsonl"
    judgments_path = Path(args.judgments_path) if args.judgments_path else run_dir / "judgments.jsonl"

    if args.overwrite_responses and responses_path.exists():
        logger.info("Removing existing responses file before generation: %s", responses_path)
        responses_path.unlink()
    if args.overwrite_judgments and judgments_path.exists():
        logger.info("Removing existing judgments file before judging: %s", judgments_path)
        judgments_path.unlink()

    if args.mode in {"full", "generate_only"}:
        generator = HFChatGenerator(
            model_name_or_path=args.model_name_or_path,
            adapter_path=args.adapter_path,
            cache_dir=args.cache_dir,
            dtype_name=args.dtype,
            device=args.generator_device,
            enable_thinking=args.enable_thinking,
        )
        logger.info(
            "Loaded generator base=%s adapter=%s device=%s enable_thinking=%s",
            generator.base_model_name_or_path,
            args.adapter_path,
            args.generator_device,
            args.enable_thinking,
        )
        existing_responses = load_responses(responses_path)
        logger.info(
            "Loaded %d existing responses from %s",
            len(existing_responses),
            responses_path,
        )
        for idx, example in enumerate(examples, start=1):
            if example.prompt_id in existing_responses:
                logger.info(
                    "Skipping generation %d/%d for prompt_id=%s because it already exists",
                    idx,
                    len(examples),
                    example.prompt_id,
                )
                continue
            logger.info("Generating response %d/%d for prompt_id=%s", idx, len(examples), example.prompt_id)
            output_text = generator.generate(
                messages=example.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            append_jsonl(
                responses_path,
                {
                    "prompt_id": example.prompt_id,
                    "prompt": example.prompt,
                    "example_tags": example.example_tags,
                    "response": output_text,
                    "model_alias": args.model_alias,
                    "adapter_path": args.adapter_path,
                    "base_model_name_or_path": generator.base_model_name_or_path,
                },
            )

    if args.mode == "generate_only":
        response_rows = load_responses(responses_path)
        save_json(
            {
                "benchmark": args.benchmark,
                "subset_name": args.subset_name,
                "mode": args.mode,
                "judge_mode": args.judge_mode,
                "model_alias": args.model_alias,
                "model_name_or_path": args.model_name_or_path,
                "adapter_path": args.adapter_path,
                "num_examples": len(examples),
                "num_generated_responses": len(response_rows),
                "responses_path": str(responses_path),
            },
            run_dir / "summary.json",
        )
        logger.info("Generation-only mode complete. Responses saved to %s", responses_path)
        return

    response_rows = load_responses(responses_path)
    missing_prompt_ids = [example.prompt_id for example in examples if example.prompt_id not in response_rows]
    if missing_prompt_ids:
        raise ValueError(f"Missing responses for prompt_ids: {missing_prompt_ids[:10]}")

    if args.judge_mode == "none":
        save_json(
            {
                "benchmark": args.benchmark,
                "subset_name": args.subset_name,
                "mode": args.mode,
                "judge_mode": args.judge_mode,
                "model_alias": args.model_alias,
                "model_name_or_path": args.model_name_or_path,
                "adapter_path": args.adapter_path,
                "num_examples": len(examples),
                "responses_path": str(responses_path),
            },
            run_dir / "summary.json",
        )
        logger.info("Judge mode is none. Skipping rubric scoring.")
        return

    judge = OpenAIHealthBenchJudge(model_name=args.judge_model)
    judged_rows: list[dict[str, Any]] = []
    existing_judgments = load_judgments(judgments_path)
    logger.info(
        "Loaded %d existing judgments from %s",
        len(existing_judgments),
        judgments_path,
    )

    for idx, example in enumerate(examples, start=1):
        if example.prompt_id in existing_judgments:
            logger.info(
                "Skipping judging %d/%d for prompt_id=%s because it already exists",
                idx,
                len(examples),
                example.prompt_id,
            )
            judged_rows.append(existing_judgments[example.prompt_id])
            continue
        response_row = response_rows[example.prompt_id]
        response_text = str(response_row["response"])
        conversation_text = render_conversation_for_judge(example.prompt, response_text)
        logger.info(
            "Judging example %d/%d prompt_id=%s with %d rubrics",
            idx,
            len(examples),
            example.prompt_id,
            len(example.rubrics),
        )

        judgments: list[dict[str, Any]] = []
        for rubric_index, rubric in enumerate(example.rubrics, start=1):
            logger.info(
                "Judging rubric %d/%d for prompt_id=%s",
                rubric_index,
                len(example.rubrics),
                example.prompt_id,
            )
            judge_result = judge.grade(conversation_text=conversation_text, rubric_item=rubric.criterion)
            judgments.append(
                {
                    "criterion": rubric.criterion,
                    "points": rubric.points,
                    "tags": rubric.tags,
                    **judge_result,
                }
            )

        example_score = calculate_example_score(judgments)
        axis_scores = calculate_axis_scores(judgments)
        judged_row = {
            "prompt_id": example.prompt_id,
            "example_tags": example.example_tags,
            "prompt": example.prompt,
            "response": response_text,
            "judgments": judgments,
            "score": example_score,
            "axis_scores": axis_scores,
        }
        append_jsonl(judgments_path, judged_row)
        judged_rows.append(judged_row)

    summary = aggregate_healthbench_results(judged_rows)
    summary.update(
        {
            "benchmark": args.benchmark,
            "subset_name": args.subset_name,
            "judge_model": args.judge_model,
            "model_alias": args.model_alias,
            "model_name_or_path": args.model_name_or_path,
            "adapter_path": args.adapter_path,
            "responses_path": str(responses_path),
            "judgments_path": str(judgments_path),
        }
    )
    save_json(summary, run_dir / "summary.json")
    (run_dir / "summary.md").write_text(build_summary_markdown(summary), encoding="utf-8")
    logger.info("Evaluation complete. Summary written to %s", run_dir / "summary.json")


if __name__ == "__main__":
    main()
