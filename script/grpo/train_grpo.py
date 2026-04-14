#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, TrainerControl, TrainerState, set_seed
from trl import GRPOConfig, GRPOTrainer

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT = CURRENT_DIR.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from grpo.reward_functions import DEFAULT_REWARD_DESCRIPTIONS, DEFAULT_REWARD_FUNCS


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def is_local_main_process() -> bool:
    return get_local_rank() in (-1, 0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GRPO training for medical alignment prompts")
    parser.add_argument("--base-model-name-or-path", required=True, help="Base merged model path")
    parser.add_argument("--init-adapter-path", default=None, help="Optional adapter checkpoint used as GRPO init")
    parser.add_argument("--tokenizer-name-or-path", default=None, help="Optional tokenizer path override")
    parser.add_argument("--train-data", nargs="+", required=True, help="GRPO prompt json/jsonl files or directories")
    parser.add_argument("--valid-data", nargs="*", default=None, help="Validation prompt files or directories")
    parser.add_argument("--output-root", default="/home/qjh/llm_learning/my_medical_gpt/outputs/grpo")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cache-dir", default="/home/qjh/llm_learning/my_medical_gpt/cache")
    parser.add_argument("--wandb-project", default="my-medical-gpt-grpo")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--max-eval-samples", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--num-generations-eval", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--logging-steps", "--logging-interval", dest="logging_steps", type=int, default=2)
    parser.add_argument("--eval-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--eval-steps", "--eval-interval", dest="eval_steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--save-steps", "--save-interval", dest="save_steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--metric-for-best-model", default="eval_reward")
    parser.add_argument("--greater-is-better", action="store_true", default=True)
    parser.add_argument(
        "--loss-type",
        default="dapo",
        choices=["grpo", "dr_grpo", "dapo", "bnpo", "cispo", "sapo"],
    )
    parser.add_argument(
        "--multi-objective-aggregation",
        default="sum_then_normalize",
        choices=["sum_then_normalize", "normalize_then_sum"],
    )
    parser.add_argument("--scale-rewards", default="group", choices=["group", "batch", "none"])
    parser.add_argument("--mask-truncated-completions", action="store_true", default=True)
    parser.add_argument("--no-mask-truncated-completions", dest="mask_truncated_completions", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--use-cpu", action="store_true", default=False)
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--flash-attn", action="store_true", default=True)
    parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--log-completions", action="store_true", default=True)
    parser.add_argument("--num-completions-to-print", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", default=False)
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
    return slug or "run"


def default_run_name(base_model_name_or_path: str, train_data: Sequence[str], init_adapter_path: Optional[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_slug = slugify(Path(base_model_name_or_path).name)
    data_slug = slugify(Path(train_data[0]).stem)
    init_slug = slugify(Path(init_adapter_path).name) if init_adapter_path else "fresh"
    return f"{timestamp}_{base_slug}_{data_slug}_{init_slug}_grpo"


def setup_logging(*log_paths: Path) -> logging.Logger:
    logger = logging.getLogger("my_medical_gpt.grpo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    if not is_local_main_process():
        logger.addHandler(logging.NullHandler())
        return logger

    for log_path in log_paths:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    for log_path in log_paths:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def resolve_json_files(items: Optional[Sequence[str]]) -> List[str]:
    if not items:
        return []
    resolved: List[str] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            resolved.extend(sorted(glob(str(path / "**/*.json"), recursive=True)))
            resolved.extend(sorted(glob(str(path / "**/*.jsonl"), recursive=True)))
        elif path.is_file():
            resolved.append(str(path))
    return sorted(dict.fromkeys(resolved))


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def truncate_dataset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples is None or max_samples < 0 or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def prompt_preview(dataset: Dataset, logger: logging.Logger, name: str) -> None:
    if len(dataset) == 0:
        logger.warning("%s dataset is empty", name)
        return
    sample = dataset[0]
    logger.info("%s sample keys: %s", name, list(sample.keys()))
    logger.info("%s primary_slice=%s risk=%s source=%s", name, sample.get("primary_slice"), sample.get("risk_level"), sample.get("source_dataset"))
    prompt = sample.get("prompt") or []
    if isinstance(prompt, list) and prompt:
        logger.info("%s prompt preview: %s", name, json.dumps(prompt[-1], ensure_ascii=False))


def configure_wandb(args: argparse.Namespace, run_dir: Path) -> List[str]:
    if args.wandb_mode == "disabled":
        return []
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_DIR"] = str(run_dir / "wandb")
    return ["wandb"]


def choose_torch_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.use_cpu:
        return torch.float32
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def build_quantization_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if not args.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=choose_torch_dtype(args),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(args: argparse.Namespace):
    tokenizer_source = args.tokenizer_name_or_path or args.init_adapter_path or args.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_model(args: argparse.Namespace, logger: logging.Logger):
    quantization_config = build_quantization_config(args)
    torch_dtype = choose_torch_dtype(args)
    flash_attn_available = importlib.util.find_spec("flash_attn") is not None
    use_flash_attn = args.flash_attn and flash_attn_available
    if args.flash_attn and not flash_attn_available:
        logger.warning("flash_attn is not installed, fallback to standard attention")

    model_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": args.base_model_name_or_path,
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
        "dtype": torch_dtype,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif args.use_cpu:
        model_kwargs["device_map"] = None
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    logger.info("Loading GRPO base model from %s", args.base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False

    if args.init_adapter_path:
        logger.info("Loading GRPO init adapter from %s", args.init_adapter_path)
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)
    else:
        logger.info("No init adapter provided, creating a fresh LoRA adapter")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    if is_local_main_process():
        model.print_trainable_parameters()
    return model


def render_prompt_text(prompt: Any, tokenizer, max_length: int) -> List[int]:
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, max_length=max_length, truncation=False)
    return tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]


def filter_by_prompt_length(dataset: Dataset, tokenizer, max_prompt_length: int, logger: logging.Logger, name: str) -> Dataset:
    if max_prompt_length <= 0:
        return dataset

    def annotate(example: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = render_prompt_text(example["prompt"], tokenizer, max_prompt_length)
        return {"prompt_token_count": len(input_ids)}

    annotated = dataset.map(annotate, num_proc=1, desc=f"annotate_{name}_prompt_length")
    before = len(annotated)
    filtered = annotated.filter(lambda x: x["prompt_token_count"] <= max_prompt_length, num_proc=1, desc=f"filter_{name}_prompt_length")
    after = len(filtered)
    removed = before - after
    if removed > 0:
        logger.info("%s prompt length filter removed %s samples over %s tokens", name, removed, max_prompt_length)
    if "prompt_token_count" in filtered.column_names:
        filtered = filtered.remove_columns(["prompt_token_count"])
    return filtered


def load_prompt_dataset(files: Sequence[str], cache_dir: str) -> Dataset:
    if not files:
        raise FileNotFoundError("No GRPO prompt dataset files found")
    dataset = load_dataset("json", data_files={"train": list(files)}, cache_dir=cache_dir)["train"]
    if "prompt" not in dataset.column_names:
        raise ValueError(f"GRPO dataset must contain 'prompt', got {dataset.column_names}")
    return dataset


def compute_warmup_steps(args: argparse.Namespace, train_samples: int) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batch = max(1, args.per_device_train_batch_size * world_size)
    optimizer_batch = max(1, micro_batch * args.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(train_samples / optimizer_batch))
    total_steps = max(1, math.ceil(steps_per_epoch * args.num_train_epochs)) if args.max_steps < 0 else args.max_steps
    return max(0, int(total_steps * args.warmup_ratio))


def find_divisible_per_device_batch_size(initial_size: int, world_size: int, num_generations: int) -> int:
    size = max(1, initial_size)
    while (size * world_size) % num_generations != 0:
        size += 1
    return size


def build_training_arguments(
    args: argparse.Namespace,
    run_dir: Path,
    report_to: Sequence[str],
    train_samples: int,
    has_validation: bool,
) -> GRPOConfig:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_train_batch = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
    if global_train_batch % args.num_generations != 0:
        raise ValueError(
            "The global train batch size "
            f"({args.per_device_train_batch_size} * {world_size} * {args.gradient_accumulation_steps} = {global_train_batch}) "
            f"must be divisible by num_generations ({args.num_generations})."
        )

    per_device_eval_batch_size = args.per_device_eval_batch_size
    if has_validation:
        per_device_eval_batch_size = find_divisible_per_device_batch_size(
            initial_size=args.per_device_eval_batch_size,
            world_size=world_size,
            num_generations=args.num_generations_eval,
        )

    warmup_steps = compute_warmup_steps(args, train_samples)
    do_eval = has_validation and args.eval_strategy != "no"
    eval_strategy = args.eval_strategy if do_eval else "no"
    save_strategy = args.save_strategy
    if save_strategy != "no" and eval_strategy == "epoch" and save_strategy == "steps":
        save_strategy = "epoch"
    # GRPOTrainer can expose eval_reward and other rich evaluation signals through logging, but
    # Transformers' built-in best-checkpoint selection only inspects the raw evaluate() return.
    # In practice this can crash if metric_for_best_model points to eval_reward. We therefore keep
    # best-checkpoint tracking in our own callback instead of relying on Trainer's internal
    # metric_for_best_model / best-checkpoint selection.
    load_best_model_at_end = False

    return GRPOConfig(
        output_dir=str(run_dir / "checkpoints"),
        run_name=args.run_name,
        do_train=True,
        do_eval=do_eval,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_steps=args.save_steps if save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        use_cpu=args.use_cpu,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=min(args.num_proc, 8),
        dataloader_pin_memory=True,
        report_to=list(report_to),
        remove_unused_columns=False,
        seed=args.seed,
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=None,
        greater_is_better=None,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
        loss_type=args.loss_type,
        multi_objective_aggregation=args.multi_objective_aggregation,
        scale_rewards=args.scale_rewards,
        mask_truncated_completions=args.mask_truncated_completions,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
    )


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, metrics_path: Path) -> None:
        self.metrics_path = metrics_path
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: GRPOConfig,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs or not is_local_main_process():
            return
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": state.global_step,
            "epoch": state.epoch,
            "logs": logs,
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class BestMetricCallback(TrainerCallback):
    def __init__(self, metric_name: str, output_path: Path, logger: logging.Logger) -> None:
        self.metric_name = metric_name
        self.output_path = output_path
        self.logger = logger
        self.best_value: Optional[float] = None
        self.best_payload: Optional[Dict[str, Any]] = load_json(output_path)
        if self.best_payload is not None:
            metric_value = self.best_payload.get("metric_value")
            if isinstance(metric_value, (int, float)):
                self.best_value = float(metric_value)

    def on_log(
        self,
        args: GRPOConfig,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if not is_local_main_process() or not logs or self.metric_name not in logs:
            return
        value = logs[self.metric_name]
        candidate_checkpoint = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        should_update = self.best_value is None or (value > self.best_value if args.greater_is_better else value < self.best_value)
        if not should_update:
            return
        self.best_value = value
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metric_name": self.metric_name,
            "metric_value": value,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "candidate_checkpoint": candidate_checkpoint,
            "best_model_checkpoint": candidate_checkpoint,
        }
        self.best_payload = payload
        save_json(payload, self.output_path)
        self.logger.info(
            "Best metric updated: %s=%.6f at step=%s checkpoint=%s",
            self.metric_name,
            value,
            state.global_step,
            candidate_checkpoint,
        )


def summarize_dataset(dataset: Dataset, name: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "name": name,
        "count": len(dataset),
    }
    for column in ["primary_slice", "source_dataset", "risk_level"]:
        if column in dataset.column_names:
            counts: Dict[str, int] = {}
            for value in dataset[column]:
                key = str(value)
                counts[key] = counts.get(key, 0) + 1
            summary[f"{column}_counts"] = counts
    return summary


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_files = resolve_json_files(args.train_data)
    valid_files = resolve_json_files(args.valid_data)
    if not train_files:
        raise FileNotFoundError("No GRPO train files found")
    if args.run_name is None:
        args.run_name = default_run_name(args.base_model_name_or_path, train_files, args.init_adapter_path)

    run_dir = Path(args.output_root) / args.run_name
    log_dir = run_dir / "logs"
    logger = setup_logging(log_dir / "train.log", log_dir / "console.log")
    set_seed(args.seed)

    if is_local_main_process():
        logger.info("run_name=%s", args.run_name)
        logger.info("base_model=%s", args.base_model_name_or_path)
        logger.info("init_adapter=%s", args.init_adapter_path)
        logger.info("train_files=%s", train_files)
        logger.info("valid_files=%s", valid_files)

    tokenizer = load_tokenizer(args)
    train_dataset = load_prompt_dataset(train_files, args.cache_dir)
    valid_dataset = load_prompt_dataset(valid_files, args.cache_dir) if valid_files else None

    train_dataset = filter_by_prompt_length(train_dataset, tokenizer, args.max_prompt_length, logger, "train")
    if valid_dataset is not None:
        valid_dataset = filter_by_prompt_length(valid_dataset, tokenizer, args.max_prompt_length, logger, "valid")

    train_dataset = truncate_dataset(train_dataset, args.max_train_samples)
    if valid_dataset is not None:
        valid_dataset = truncate_dataset(valid_dataset, args.max_eval_samples)

    prompt_preview(train_dataset, logger, "train")
    if valid_dataset is not None:
        prompt_preview(valid_dataset, logger, "valid")

    run_dir.mkdir(parents=True, exist_ok=True)
    if is_local_main_process():
        save_json(vars(args), run_dir / "artifacts" / "run_args.json")
        save_json(
            {
                "train_files": train_files,
                "valid_files": valid_files,
                "train_summary": summarize_dataset(train_dataset, "train"),
                "valid_summary": summarize_dataset(valid_dataset, "valid") if valid_dataset is not None else None,
            },
            run_dir / "artifacts" / "dataset_summary.json",
        )
        save_json(
            {
                "reward_functions": [
                    {
                        "name": func.__name__,
                        "description": DEFAULT_REWARD_DESCRIPTIONS.get(func.__name__, ""),
                    }
                    for func in DEFAULT_REWARD_FUNCS
                ]
            },
            run_dir / "artifacts" / "reward_manifest.json",
        )

    if args.dry_run:
        logger.info("Dry run enabled, skip training startup")
        return

    report_to = configure_wandb(args, run_dir)
    model = build_model(args, logger)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    adjusted_eval_batch_size = find_divisible_per_device_batch_size(
        initial_size=args.per_device_eval_batch_size,
        world_size=world_size,
        num_generations=args.num_generations_eval,
    )
    if valid_dataset is not None and adjusted_eval_batch_size != args.per_device_eval_batch_size:
        logger.info(
            "Adjust per_device_eval_batch_size from %s to %s so global eval batch is divisible by num_generations_eval=%s",
            args.per_device_eval_batch_size,
            adjusted_eval_batch_size,
            args.num_generations_eval,
        )
        args.per_device_eval_batch_size = adjusted_eval_batch_size
    training_args = build_training_arguments(
        args=args,
        run_dir=run_dir,
        report_to=report_to,
        train_samples=len(train_dataset),
        has_validation=valid_dataset is not None and len(valid_dataset) > 0,
    )

    if is_local_main_process():
        save_json(training_args.to_dict(), run_dir / "artifacts" / "training_args.json")

    metrics_callback = JsonlMetricsCallback(log_dir / "metrics.jsonl")
    best_callback = BestMetricCallback(args.metric_for_best_model, run_dir / "artifacts" / "best_checkpoint.json", logger)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=DEFAULT_REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        callbacks=[metrics_callback, best_callback],
    )

    logger.info("Starting GRPO training")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info("GRPO training finished")

    trainer.save_model(str(run_dir / "final_model"))
    save_json(train_result.metrics, run_dir / "artifacts" / "train_result.json")

    final_best_payload = best_callback.best_payload or {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metric_name": args.metric_for_best_model,
        "metric_value": None,
        "global_step": trainer.state.global_step,
        "candidate_checkpoint": None,
        "best_model_checkpoint": None,
    }
    if is_local_main_process():
        save_json(final_best_payload, run_dir / "artifacts" / "best_checkpoint.json")
        save_json(
            {
                "global_step": trainer.state.global_step,
                "best_metric": final_best_payload.get("metric_value"),
                "best_model_checkpoint": final_best_payload.get("best_model_checkpoint"),
                "train_runtime": train_result.metrics.get("train_runtime"),
                "train_samples": len(train_dataset),
                "eval_samples": len(valid_dataset) if valid_dataset is not None else 0,
            },
            run_dir / "artifacts" / "summary.json",
        )
    logger.info("Best checkpoint: %s", final_best_payload.get("best_model_checkpoint"))


if __name__ == "__main__":
    main()
