#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)


IGNORE_INDEX = -100


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LoRA SFT for Qwen3-style chat models")
    parser.add_argument("--model-name-or-path", required=True, help="Base model path")
    parser.add_argument("--train-data", nargs="+", required=True, help="Train json/jsonl files or directories")
    parser.add_argument(
        "--valid-data",
        "--eval-data",
        dest="valid_data",
        nargs="*",
        default=None,
        help="Validation/Eval json/jsonl files or directories",
    )
    parser.add_argument(
        "--validation-split-ratio",
        type=float,
        default=0.05,
        help="Used to auto-split train data when explicit valid data is absent. Set 0 to disable auto-split.",
    )
    parser.add_argument("--output-root", default="/home/qjh/llm_learning/my_medical_gpt/outputs/sft")
    parser.add_argument("--run-name", default=None, help="Optional run name. If omitted, a timestamped name is used")
    parser.add_argument("--cache-dir", default="/home/qjh/llm_learning/my_medical_gpt/cache")
    parser.add_argument("--wandb-project", default="my-medical-gpt-sft")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", "--logging-interval", dest="logging_steps", type=int, default=10)
    parser.add_argument("--eval-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--eval-steps", "--eval-interval", dest="eval_steps", type=int, default=50)
    parser.add_argument("--save-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--save-steps", "--save-interval", dest="save_steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
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
    parser.add_argument("--train-on-inputs", action="store_true", default=False)
    parser.add_argument("--disable-assistant-only-loss", action="store_true", default=False)
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


def default_run_name(model_name_or_path: str, train_data: Sequence[str]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(Path(model_name_or_path).name)
    data_slug = slugify(Path(train_data[0]).stem)
    return f"{timestamp}_{model_slug}_{data_slug}"


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("my_medical_gpt.sft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
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


def load_raw_datasets(train_files: Sequence[str], valid_files: Sequence[str], cache_dir: str, seed: int, ratio: float) -> DatasetDict:
    data_files: Dict[str, List[str]] = {"train": list(train_files)}
    if valid_files:
        data_files["validation"] = list(valid_files)

    dataset_dict = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
    if "validation" not in dataset_dict and ratio > 0:
        split = dataset_dict["train"].train_test_split(test_size=ratio, seed=seed)
        dataset_dict = DatasetDict(train=split["train"], validation=split["test"])
    return dataset_dict


def truncate_dataset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples is None or max_samples < 0 or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sample_preview(dataset: Dataset, logger: logging.Logger, name: str) -> None:
    if len(dataset) == 0:
        logger.warning("%s dataset is empty", name)
        return
    sample = dataset[0]
    logger.info("%s sample keys: %s", name, list(sample.keys()))
    if "conversations" in sample:
        preview = sample["conversations"][:2]
        logger.info("%s sample preview: %s", name, json.dumps(preview, ensure_ascii=False))


def infer_validation_source(valid_files: Sequence[str], ratio: float, has_validation: bool) -> str:
    if valid_files:
        return "external_validation"
    if has_validation and ratio > 0:
        return "auto_holdout_from_train"
    return "disabled"


def normalize_conversations(record: Dict[str, Any]) -> List[Dict[str, str]]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Missing conversations")
    normalized: List[Dict[str, str]] = []
    for item in conversations:
        role = str(item.get("from", "")).strip().lower()
        value = str(item.get("value", "")).strip()
        if not value:
            continue
        if role not in {"system", "human", "gpt"}:
            raise ValueError(f"Unsupported role: {role}")
        normalized.append({"from": role, "value": value})
    return normalized


def build_messages(conversations: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for item in conversations:
        role = item["from"]
        if role == "human":
            messages.append({"role": "user", "content": item["value"]})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": item["value"]})
        elif role == "system":
            messages.append({"role": "system", "content": item["value"]})
    return messages


def render_chat_text(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Dict[str, str]],
) -> str:
    return tokenizer.apply_chat_template(
        conversation=list(messages),
        tokenize=False,
        add_generation_prompt=False,
    )


def build_assistant_spans(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Dict[str, str]],
) -> List[Tuple[int, int]]:
    assistant_header = "<|im_start|>assistant\n"
    spans: List[Tuple[int, int]] = []

    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        prev_messages = messages[:idx]
        curr_messages = messages[: idx + 1]

        prev_text = render_chat_text(tokenizer, prev_messages) if prev_messages else ""
        curr_text = render_chat_text(tokenizer, curr_messages)
        if not curr_text.startswith(prev_text):
            raise ValueError("Chat template rendering is not prefix-stable for assistant span extraction")

        delta_text = curr_text[len(prev_text):]
        header_pos = delta_text.find(assistant_header)
        if header_pos < 0:
            raise ValueError("Assistant header not found in rendered chat segment")

        content_start = len(prev_text) + header_pos + len(assistant_header)
        content_end = len(curr_text)
        if content_end > content_start:
            spans.append((content_start, content_end))

    return spans


def token_overlaps_any_span(
    token_start: int,
    token_end: int,
    spans: Sequence[Tuple[int, int]],
) -> bool:
    for span_start, span_end in spans:
        if token_end <= span_start:
            continue
        if token_start >= span_end:
            continue
        return True
    return False


def tokenize_example(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    model_max_length: int,
    train_on_inputs: bool,
) -> Dict[str, List[int]]:
    conversations = normalize_conversations(example)
    messages = build_messages(conversations)
    if not messages:
        raise ValueError("No valid messages found")

    rendered_text = render_chat_text(tokenizer, messages)
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=False,
        truncation=True,
        max_length=model_max_length,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )

    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    offset_mapping = list(encoded["offset_mapping"])

    if train_on_inputs:
        labels = list(input_ids)
    else:
        assistant_spans = build_assistant_spans(tokenizer, messages)
        labels = []
        for token_id, (token_start, token_end) in zip(input_ids, offset_mapping):
            if token_overlaps_any_span(token_start, token_end, assistant_spans):
                labels.append(token_id)
            else:
                labels.append(IGNORE_INDEX)

        if all(label == IGNORE_INDEX for label in labels):
            raise ValueError("No assistant tokens were selected for loss computation")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SupervisedDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [feature["input_ids"] for feature in features]
        batch_attention_mask = [feature["attention_mask"] for feature in features]
        batch_labels = [feature["labels"] for feature in features]

        padded = self.tokenizer.pad(
            {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask},
            padding=True,
            return_tensors="pt",
        )

        max_len = padded["input_ids"].shape[1]
        labels = torch.full((len(batch_labels), max_len), IGNORE_INDEX, dtype=torch.long)
        for row_idx, row in enumerate(batch_labels):
            labels[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)

        padded["labels"] = labels
        return padded


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, metrics_path: Path) -> None:
        self.metrics_path = metrics_path
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": state.global_step,
            "epoch": state.epoch,
            "logs": logs,
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def build_model(args: argparse.Namespace, logger: logging.Logger):
    quantization_config = build_quantization_config(args)
    torch_dtype = choose_torch_dtype(args)
    flash_attn_available = importlib.util.find_spec("flash_attn") is not None
    use_flash_attn = args.flash_attn and flash_attn_available
    if args.flash_attn and not flash_attn_available:
        logger.warning("flash_attn is not installed, fallback to standard attention")

    model_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": args.model_name_or_path,
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

    logger.info("Loading model from %s", args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def compute_warmup_steps(args: argparse.Namespace, train_samples: int) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batch = max(1, args.per_device_train_batch_size * world_size)
    optimizer_batch = max(1, micro_batch * args.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(train_samples / optimizer_batch))
    total_steps = max(1, math.ceil(steps_per_epoch * args.num_train_epochs))
    return max(0, int(total_steps * args.warmup_ratio))


def build_training_arguments(
    args: argparse.Namespace,
    run_dir: Path,
    report_to: Sequence[str],
    train_samples: int,
    has_validation: bool,
) -> TrainingArguments:
    warmup_steps = compute_warmup_steps(args, train_samples)
    do_eval = has_validation and args.eval_strategy != "no"
    eval_strategy = args.eval_strategy if do_eval else "no"
    save_strategy = args.save_strategy
    if save_strategy != "no" and eval_strategy == "epoch" and save_strategy == "steps":
        save_strategy = "epoch"
    if not do_eval and save_strategy == "no":
        load_best_model_at_end = False
    else:
        load_best_model_at_end = do_eval and eval_strategy == save_strategy and eval_strategy != "no"

    return TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        run_name=args.run_name,
        do_train=True,
        do_eval=do_eval,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
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
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_first_step=True,
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
    )


def tokenize_dataset_dict(
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> DatasetDict:
    preprocess_kwargs = {
        "tokenizer": tokenizer,
        "model_max_length": args.model_max_length,
        "train_on_inputs": args.train_on_inputs,
    }

    def map_fn(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        processed = []
        for i in range(len(batch["conversations"])):
            example = {key: value[i] for key, value in batch.items()}
            processed.append(tokenize_example(example, **preprocess_kwargs))
        return {
            "input_ids": [item["input_ids"] for item in processed],
            "attention_mask": [item["attention_mask"] for item in processed],
            "labels": [item["labels"] for item in processed],
        }

    map_kwargs = {
        "batched": True,
        "remove_columns": raw_datasets["train"].column_names,
        "desc": "Tokenizing conversations",
    }

    if args.num_proc > 1:
        try:
            return raw_datasets.map(map_fn, num_proc=args.num_proc, **map_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Multiprocess tokenization failed, fallback to single process: %s", exc)

    return raw_datasets.map(map_fn, **map_kwargs)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.run_name = args.run_name or default_run_name(args.model_name_or_path, args.train_data)

    run_dir = Path(args.output_root) / args.run_name
    log_dir = run_dir / "logs"
    logger = setup_logging(log_dir / "train.log")
    set_seed(args.seed)

    train_files = resolve_json_files(args.train_data)
    valid_files = resolve_json_files(args.valid_data)
    if not train_files:
        raise FileNotFoundError("No train files found")

    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("artifacts", "wandb"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info("Run directory: %s", run_dir)
    logger.info("Train files (%d): %s", len(train_files), train_files)
    logger.info("Validation files (%d): %s", len(valid_files), valid_files)
    save_json(vars(args), run_dir / "artifacts" / "run_args.json")

    raw_datasets = load_raw_datasets(
        train_files=train_files,
        valid_files=valid_files,
        cache_dir=args.cache_dir,
        seed=args.seed,
        ratio=args.validation_split_ratio,
    )

    has_validation = "validation" in raw_datasets and len(raw_datasets["validation"]) > 0
    validation_source = infer_validation_source(valid_files, args.validation_split_ratio, has_validation)
    logger.info("Validation source: %s", validation_source)

    raw_datasets["train"] = truncate_dataset(raw_datasets["train"], args.max_train_samples)
    if has_validation:
        raw_datasets["validation"] = truncate_dataset(raw_datasets["validation"], args.max_eval_samples)
    else:
        raw_datasets["validation"] = raw_datasets["train"].select([])

    sample_preview(raw_datasets["train"], logger, "train")
    sample_preview(raw_datasets["validation"], logger, "validation")

    logger.info("Loading tokenizer from %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized = tokenize_dataset_dict(raw_datasets, tokenizer, args, logger)

    logger.info("Tokenized train samples: %d", len(tokenized["train"]))
    logger.info("Tokenized validation samples: %d", len(tokenized["validation"]))
    if len(tokenized["train"]) == 0:
        raise ValueError("No train samples left after tokenization")

    tokenized_preview = tokenized["train"][0]
    logger.info(
        "First tokenized sample lengths: input_ids=%d attention_mask=%d labels=%d",
        len(tokenized_preview["input_ids"]),
        len(tokenized_preview["attention_mask"]),
        len(tokenized_preview["labels"]),
    )
    supervised_tokens = sum(1 for token in tokenized_preview["labels"] if token != IGNORE_INDEX)
    logger.info(
        "First tokenized sample supervised tokens: %d/%d",
        supervised_tokens,
        len(tokenized_preview["labels"]),
    )

    save_json(
        {
            "train_samples": len(tokenized["train"]),
            "valid_samples": len(tokenized["validation"]),
            "validation_source": validation_source,
        },
        run_dir / "artifacts" / "dataset_stats.json",
    )

    if args.dry_run:
        logger.info("Dry run completed successfully")
        return

    report_to = configure_wandb(args, run_dir)
    training_args = build_training_arguments(args, run_dir, report_to, len(tokenized["train"]), len(tokenized["validation"]) > 0)
    save_json(training_args.to_dict(), run_dir / "artifacts" / "training_args.json")

    model = build_model(args, logger)
    data_collator = SupervisedDataCollator(tokenizer)
    metrics_callback = JsonlMetricsCallback(log_dir / "metrics.jsonl")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if len(tokenized["validation"]) > 0 else None,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[metrics_callback],
    )

    logger.info("Starting training")
    train_result = trainer.train()
    trainer.save_model(str(run_dir / "final_model"))
    tokenizer.save_pretrained(str(run_dir / "final_model"))

    train_metrics = dict(train_result.metrics)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    if len(tokenized["validation"]) > 0:
        logger.info("Running final evaluation")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    else:
        logger.info("Skipping final evaluation because no validation dataset is available")
    logger.info("Training complete. Final model saved to %s", run_dir / "final_model")


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
