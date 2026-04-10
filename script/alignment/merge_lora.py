#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "merged_models" / "sft"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "outputs" / "logs" / "merge"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(text: str) -> str:
    safe = []
    for ch in text.lower():
        if ch.isalnum():
            safe.append(ch)
        else:
            safe.append("-")
    slug = "".join(safe)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "run"


def setup_logging(*log_paths: Path) -> logging.Logger:
    for log_path in log_paths:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("my_medical_gpt.merge_lora")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    for log_path in log_paths:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "auto": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def infer_run_name(adapter_path: Path) -> str:
    parent_slug = slugify(adapter_path.parent.name)
    leaf_slug = slugify(adapter_path.name)
    return f"{timestamp()}_{parent_slug}_{leaf_slug}_merged"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--safe-serialization", action="store_true", default=True)
    parser.add_argument("--no-safe-serialization", dest="safe_serialization", action="store_false")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser


def choose_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise EnvironmentError("CUDA was requested but is not available.")
    return device


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path).resolve()
    base_model_path = Path(args.base_model_path).resolve()
    run_name = args.run_name or infer_run_name(adapter_path)
    run_dir = Path(args.output_root).resolve() / run_name
    model_output_dir = run_dir / "model"
    artifact_dir = run_dir / "artifacts"
    run_log_path = run_dir / "logs" / "merge.log"
    central_log_path = Path(args.log_root).resolve() / f"{timestamp()}_{run_name}.log"
    logger = setup_logging(run_log_path, central_log_path)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_model_path}")
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite_output:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {run_dir}. "
            "Use --overwrite-output to reuse it."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    dtype = resolve_dtype(args.dtype)
    flash_attn_available = importlib.util.find_spec("flash_attn") is not None

    start_ts = datetime.now()
    start_perf = time.perf_counter()

    logger.info("Starting LoRA merge")
    logger.info("Run directory: %s", run_dir)
    logger.info("Base model path: %s", base_model_path)
    logger.info("Adapter path: %s", adapter_path)
    logger.info("Output model path: %s", model_output_dir)
    logger.info("Device: %s", device)
    logger.info("Requested dtype: %s", args.dtype)
    logger.info("Resolved dtype: %s", dtype)
    logger.info("flash_attn available: %s", flash_attn_available)
    logger.info("Start time: %s", start_ts.strftime("%Y-%m-%d %H:%M:%S"))

    save_json(
        {
            "run_name": run_name,
            "base_model_path": str(base_model_path),
            "adapter_path": str(adapter_path),
            "run_dir": str(run_dir),
            "model_output_dir": str(model_output_dir),
            "run_log_path": str(run_log_path),
            "central_log_path": str(central_log_path),
            "device": device,
            "requested_dtype": args.dtype,
            "resolved_dtype": str(dtype),
            "safe_serialization": args.safe_serialization,
            "max_shard_size": args.max_shard_size,
            "start_time": start_ts.isoformat(),
        },
        artifact_dir / "merge_args.json",
    )

    model_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": str(base_model_path),
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": args.cache_dir,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    if flash_attn_available:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["attn_implementation"] = "eager"

    logger.info("Loading tokenizer from %s", base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        use_fast=False,
    )

    logger.info("Loading base model")
    base_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    logger.info("Loading LoRA adapter")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merging adapter into base model")
    merged_model = peft_model.merge_and_unload()

    logger.info("Saving merged model to %s", model_output_dir)
    merged_model.save_pretrained(
        str(model_output_dir),
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(model_output_dir))

    end_ts = datetime.now()
    elapsed_seconds = time.perf_counter() - start_perf
    summary = {
        "run_name": run_name,
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "model_output_dir": str(model_output_dir),
        "run_log_path": str(run_log_path),
        "central_log_path": str(central_log_path),
        "device": device,
        "requested_dtype": args.dtype,
        "resolved_dtype": str(dtype),
        "flash_attn_available": flash_attn_available,
        "start_time": start_ts.isoformat(),
        "end_time": end_ts.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_minutes": round(elapsed_seconds / 60.0, 3),
        "safe_serialization": args.safe_serialization,
        "max_shard_size": args.max_shard_size,
    }
    save_json(summary, artifact_dir / "merge_summary.json")

    logger.info("End time: %s", end_ts.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Elapsed time: %.3f seconds (%.3f minutes)", elapsed_seconds, elapsed_seconds / 60.0)
    logger.info("Merge complete")
    logger.info("Merged model ready at %s", model_output_dir)
    logger.info("Summary saved to %s", artifact_dir / "merge_summary.json")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
