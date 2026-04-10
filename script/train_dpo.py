#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
from contextlib import nullcontext
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, TrainerControl, TrainerState, set_seed
from trl import DPOConfig, DPOTrainer

from train_sft import IGNORE_INDEX, SupervisedDataCollator, tokenize_example


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def is_local_main_process() -> bool:
    return get_local_rank() in (-1, 0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DPO training for Qwen3-style chat models")
    parser.add_argument("--model-name-or-path", required=True, help="Merged SFT model path used as DPO init/reference base")
    parser.add_argument("--train-data", nargs="+", required=True, help="Processed pairwise json/jsonl files or directories")
    parser.add_argument("--valid-data", nargs="*", default=None, help="Processed pairwise validation files or directories")
    parser.add_argument(
        "--aux-valid-data",
        nargs="*",
        default=None,
        help="Optional heterogeneous SFT-style validation files used for auxiliary LM-loss monitoring",
    )
    parser.add_argument("--output-root", default="/home/qjh/llm_learning/my_medical_gpt/outputs/dpo")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--cache-dir", default="/home/qjh/llm_learning/my_medical_gpt/cache")
    parser.add_argument("--wandb-project", default="my-medical-gpt-dpo")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--model-max-length", type=int, default=2048, help="Auxiliary SFT eval tokenization length")
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--max-eval-samples", type=int, default=-1)
    parser.add_argument("--max-aux-eval-samples", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--aux-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss-type", default="sigmoid", help="TRL DPO loss type")
    parser.add_argument("--logging-steps", "--logging-interval", dest="logging_steps", type=int, default=5)
    parser.add_argument("--eval-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--eval-steps", "--eval-interval", dest="eval_steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--save-steps", "--save-interval", dest="save_steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=20)
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
    parser.add_argument("--metric-for-best-model", default="eval_rewards/accuracies")
    parser.add_argument("--greater-is-better", action="store_true", default=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
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
    return f"{timestamp}_{model_slug}_{data_slug}_dpo"


def setup_logging(*log_paths: Path) -> logging.Logger:
    logger = logging.getLogger("my_medical_gpt.dpo")
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


def truncate_dataset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples is None or max_samples < 0 or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def load_pairwise_dataset(files: Sequence[str], cache_dir: str) -> Dataset:
    if not files:
        raise FileNotFoundError("No pairwise dataset files found")
    dataset = load_dataset("json", data_files={"train": list(files)}, cache_dir=cache_dir)["train"]
    required_columns = {"prompt", "chosen", "rejected"}
    if not required_columns.issubset(set(dataset.column_names)):
        raise ValueError(f"Pairwise dataset must contain columns: {required_columns}, got {dataset.column_names}")
    return dataset


def pairwise_preview(dataset: Dataset, logger: logging.Logger, name: str) -> None:
    if len(dataset) == 0:
        logger.warning("%s dataset is empty", name)
        return
    sample = dataset[0]
    logger.info("%s sample keys: %s", name, list(sample.keys()))
    logger.info("%s prompt preview: %s", name, json.dumps(sample["prompt"][-2:], ensure_ascii=False))
    logger.info("%s chosen preview: %s", name, json.dumps(sample["chosen"], ensure_ascii=False))
    logger.info("%s rejected preview: %s", name, json.dumps(sample["rejected"], ensure_ascii=False))


def load_aux_eval_dataset(
    files: Sequence[str],
    tokenizer,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Optional[Dataset]:
    if not files:
        return None
    dataset = load_dataset("json", data_files={"validation": list(files)}, cache_dir=args.cache_dir)["validation"]
    dataset = truncate_dataset(dataset, args.max_aux_eval_samples)
    if len(dataset) == 0:
        return None

    def map_fn(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        processed = []
        for i in range(len(batch["conversations"])):
            example = {key: value[i] for key, value in batch.items()}
            processed.append(
                tokenize_example(
                    example=example,
                    tokenizer=tokenizer,
                    model_max_length=args.model_max_length,
                    train_on_inputs=False,
                )
            )
        return {
            "input_ids": [item["input_ids"] for item in processed],
            "attention_mask": [item["attention_mask"] for item in processed],
            "labels": [item["labels"] for item in processed],
        }

    map_kwargs = {
        "batched": True,
        "remove_columns": dataset.column_names,
        "desc": "Tokenizing auxiliary SFT eval conversations",
    }
    if args.num_proc > 1:
        try:
            tokenized = dataset.map(map_fn, num_proc=args.num_proc, **map_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Multiprocess auxiliary tokenization failed, fallback to single process: %s", exc)
            tokenized = dataset.map(map_fn, **map_kwargs)
    else:
        tokenized = dataset.map(map_fn, **map_kwargs)
    return tokenized


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

    logger.info("Loading DPO init model from %s", args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False
    return model


def build_peft_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )


def compute_warmup_steps(args: argparse.Namespace, train_samples: int) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batch = max(1, args.per_device_train_batch_size * world_size)
    optimizer_batch = max(1, micro_batch * args.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(train_samples / optimizer_batch))
    total_steps = max(1, math.ceil(steps_per_epoch * args.num_train_epochs)) if args.max_steps < 0 else args.max_steps
    return max(0, int(total_steps * args.warmup_ratio))


def build_training_arguments(
    args: argparse.Namespace,
    run_dir: Path,
    report_to: Sequence[str],
    train_samples: int,
    has_validation: bool,
) -> DPOConfig:
    warmup_steps = compute_warmup_steps(args, train_samples)
    do_eval = has_validation and args.eval_strategy != "no"
    eval_strategy = args.eval_strategy if do_eval else "no"
    save_strategy = args.save_strategy
    if save_strategy != "no" and eval_strategy == "epoch" and save_strategy == "steps":
        save_strategy = "epoch"
    load_best_model_at_end = do_eval and eval_strategy == save_strategy and eval_strategy != "no"

    return DPOConfig(
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
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_length=args.max_length,
        dataset_num_proc=min(args.num_proc, 16),
        beta=args.beta,
        loss_type=[args.loss_type],
    )


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, metrics_path: Path) -> None:
        self.metrics_path = metrics_path
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: DPOConfig,
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


class BestMetricCallback(TrainerCallback):
    def __init__(self, metric_name: str, output_path: Path, logger: logging.Logger) -> None:
        self.metric_name = metric_name
        self.output_path = output_path
        self.logger = logger
        self.best_value: Optional[float] = None
        self.best_step: Optional[int] = None

    def on_evaluate(
        self,
        args: DPOConfig,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if not is_local_main_process() or not metrics or self.metric_name not in metrics:
            return
        value = metrics[self.metric_name]
        candidate_checkpoint = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        should_update = self.best_value is None
        if not should_update:
            if args.greater_is_better:
                should_update = value > self.best_value
            else:
                should_update = value < self.best_value
        if not should_update:
            return
        self.best_value = value
        self.best_step = state.global_step
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metric_name": self.metric_name,
            "metric_value": value,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "candidate_checkpoint": candidate_checkpoint,
            "best_model_checkpoint": state.best_model_checkpoint or candidate_checkpoint,
        }
        save_json(payload, self.output_path)
        self.logger.info(
            "Best metric updated: %s=%.6f at step=%s checkpoint=%s",
            self.metric_name,
            value,
            state.global_step,
            state.best_model_checkpoint or candidate_checkpoint,
        )


class AuxiliarySupervisedEvalCallback(TrainerCallback):
    def __init__(
        self,
        dataset: Optional[Dataset],
        tokenizer,
        batch_size: int,
        metrics_path: Path,
        logger: logging.Logger,
        enabled: bool,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.metrics_path = metrics_path
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.enabled = enabled and dataset is not None and len(dataset) > 0
        self.best_loss: Optional[float] = None

    def on_evaluate(
        self,
        args: DPOConfig,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs: Any,
    ) -> None:
        if not self.enabled or model is None or not is_local_main_process():
            return

        collator = SupervisedDataCollator(self.tokenizer)
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collator)
        device = next(model.parameters()).device
        use_autocast = device.type == "cuda" and (args.bf16 or args.fp16)
        autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

        total_loss = 0.0
        total_tokens = 0
        total_samples = 0

        previous_mode = model.training
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                supervised_tokens = int((batch["labels"] != IGNORE_INDEX).sum().item())
                if supervised_tokens == 0:
                    continue
                context = (
                    torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else nullcontext()
                )
                with context:
                    outputs = model(**batch)
                batch_loss = float(outputs.loss.detach().float().item())
                total_loss += batch_loss * supervised_tokens
                total_tokens += supervised_tokens
                total_samples += int(batch["input_ids"].shape[0])
        if previous_mode:
            model.train()

        if total_tokens == 0:
            return

        aux_loss = total_loss / total_tokens
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": state.global_step,
            "epoch": state.epoch,
            "aux_eval/valid_zh_loss": aux_loss,
            "aux_eval/samples": total_samples,
            "aux_eval/supervised_tokens": total_tokens,
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self.logger.info(
            "Auxiliary heterogeneous eval complete: step=%s valid_zh_loss=%.6f samples=%s supervised_tokens=%s",
            state.global_step,
            aux_loss,
            total_samples,
            total_tokens,
        )
        if self.best_loss is None or aux_loss < self.best_loss:
            self.best_loss = aux_loss
            self.logger.info("Auxiliary heterogeneous eval best updated: valid_zh_loss=%.6f at step=%s", aux_loss, state.global_step)

        if args.report_to and "wandb" in args.report_to:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "aux_eval/valid_zh_loss": aux_loss,
                            "aux_eval/samples": total_samples,
                            "aux_eval/supervised_tokens": total_tokens,
                        },
                        step=state.global_step,
                    )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to push auxiliary eval metrics to W&B: %s", exc)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.run_name = args.run_name or default_run_name(args.model_name_or_path, args.train_data)

    run_dir = Path(args.output_root) / args.run_name
    log_dir = run_dir / "logs"
    central_log_path = Path("/home/qjh/llm_learning/my_medical_gpt/outputs/logs/dpo") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}.log"
    logger = setup_logging(log_dir / "train.log", central_log_path)
    set_seed(args.seed)

    train_files = resolve_json_files(args.train_data)
    valid_files = resolve_json_files(args.valid_data)
    aux_valid_files = resolve_json_files(args.aux_valid_data)
    if not train_files:
        raise FileNotFoundError("No DPO train files found")

    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("artifacts", "wandb"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info("Run directory: %s", run_dir)
    logger.info("Train files (%d): %s", len(train_files), train_files)
    logger.info("Validation files (%d): %s", len(valid_files), valid_files)
    logger.info("Aux validation files (%d): %s", len(aux_valid_files), aux_valid_files)
    save_json(vars(args), run_dir / "artifacts" / "run_args.json")

    train_dataset = truncate_dataset(load_pairwise_dataset(train_files, args.cache_dir), args.max_train_samples)
    valid_dataset = truncate_dataset(load_pairwise_dataset(valid_files, args.cache_dir), args.max_eval_samples) if valid_files else None
    has_validation = valid_dataset is not None and len(valid_dataset) > 0

    pairwise_preview(train_dataset, logger, "train")
    if has_validation:
        pairwise_preview(valid_dataset, logger, "validation")

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

    aux_dataset = load_aux_eval_dataset(aux_valid_files, tokenizer, args, logger) if is_local_main_process() else None
    if aux_dataset is not None:
        logger.info("Auxiliary heterogeneous eval samples: %d", len(aux_dataset))
    else:
        logger.info("Auxiliary heterogeneous eval disabled")

    save_json(
        {
            "train_samples": len(train_dataset),
            "valid_samples": len(valid_dataset) if has_validation else 0,
            "aux_valid_samples": len(aux_dataset) if aux_dataset is not None else 0,
        },
        run_dir / "artifacts" / "dataset_stats.json",
    )

    if args.dry_run:
        logger.info("Dry run completed successfully")
        return

    report_to = configure_wandb(args, run_dir)
    training_args = build_training_arguments(args, run_dir, report_to, len(train_dataset), has_validation)
    save_json(training_args.to_dict(), run_dir / "artifacts" / "training_args.json")

    model = build_model(args, logger)
    peft_config = build_peft_config(args)
    metrics_callback = JsonlMetricsCallback(log_dir / "metrics.jsonl")
    best_callback = BestMetricCallback(args.metric_for_best_model, run_dir / "artifacts" / "best_checkpoint.json", logger)
    aux_callback = AuxiliarySupervisedEvalCallback(
        dataset=aux_dataset,
        tokenizer=tokenizer,
        batch_size=args.aux_eval_batch_size,
        metrics_path=log_dir / "aux_eval.jsonl",
        logger=logger,
        enabled=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if has_validation else None,
        processing_class=tokenizer,
        callbacks=[metrics_callback, best_callback, aux_callback],
        peft_config=peft_config,
    )

    logger.info("Starting DPO training")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(run_dir / "final_model"))
    tokenizer.save_pretrained(str(run_dir / "final_model"))

    train_metrics = dict(train_result.metrics)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    if has_validation:
        logger.info("Running final DPO evaluation")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    else:
        logger.info("Skipping final DPO evaluation because no validation dataset is available")

    final_best_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metric_name": args.metric_for_best_model,
        "metric_value": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "candidate_checkpoint": trainer.state.best_model_checkpoint,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    save_json(final_best_payload, run_dir / "artifacts" / "best_checkpoint.json")
    save_json(
        {
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "metric_for_best_model": args.metric_for_best_model,
            "global_step": trainer.state.global_step,
        },
        run_dir / "artifacts" / "training_summary.json",
    )
    logger.info("DPO training complete. Final model saved to %s", run_dir / "final_model")
    logger.info("Best checkpoint: %s", trainer.state.best_model_checkpoint)
    logger.info("Best metric (%s): %s", args.metric_for_best_model, trainer.state.best_metric)


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
