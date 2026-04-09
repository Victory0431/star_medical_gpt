#!/usr/bin/env python3
"""Export lightweight experiment records from outputs/ into a git-tracked directory."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export reproducible experiment records without large model artifacts."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=repo_root / "outputs" / "sft",
        help="Directory that contains SFT run folders.",
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        default=repo_root / "experiment_records" / "sft",
        help="Directory used to store exported lightweight run records.",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=[],
        help="Specific run name to export. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all eligible runs under outputs-root.",
    )
    parser.add_argument(
        "--include-dryrun",
        action="store_true",
        help="Include run names that contain 'dryrun'. By default they are skipped.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include runs classified as obvious failures. By default they are skipped.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing exported records.",
    )
    parser.add_argument(
        "--max-log-mb",
        type=float,
        default=2.0,
        help="Copy full log if below this size; otherwise store a truncated version.",
    )
    return parser


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def sanitize_text(text: str) -> str:
    patterns = [
        (r"wandb_v1_[A-Za-z0-9_\-]+", "<REDACTED_WANDB_KEY>"),
        (r"hf_[A-Za-z0-9]+", "<REDACTED_HF_TOKEN>"),
        (r"(WANDB_API_KEY\s*=\s*)(\S+)", r"\1<REDACTED>"),
        (r"(HF_TOKEN\s*=\s*)(\S+)", r"\1<REDACTED>"),
        (r"(HUGGING_FACE_HUB_TOKEN\s*=\s*)(\S+)", r"\1<REDACTED>"),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


def unique_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()
    for row in rows:
        step = row.get("step")
        logs = row.get("logs", {})
        metric_key = ",".join(sorted(logs.keys()))
        marker = (step, metric_key)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(row)
    return deduped


def extract_latest_metrics(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest_train: dict[str, Any] | None = None
    latest_eval: dict[str, Any] | None = None
    best_eval_loss: float | None = None

    for row in metrics_rows:
        logs = row.get("logs", {})
        if "loss" in logs:
            latest_train = row
        if "eval_loss" in logs:
            latest_eval = row
            eval_loss = logs.get("eval_loss")
            if isinstance(eval_loss, (int, float)):
                if best_eval_loss is None or eval_loss < best_eval_loss:
                    best_eval_loss = float(eval_loss)

    result: dict[str, Any] = {}
    if latest_train is not None:
        result["latest_train"] = latest_train
    if latest_eval is not None:
        result["latest_eval"] = latest_eval
    if best_eval_loss is not None:
        result["best_eval_loss"] = best_eval_loss
    return result


def count_metrics_rows(run_dir: Path) -> int:
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    return len(unique_metric_rows(read_metrics(metrics_path)))


def detect_failure_signatures(run_dir: Path) -> list[str]:
    candidates = [
        run_dir / "logs" / "console.log",
        run_dir / "logs" / "train.log",
        run_dir / "launcher.log",
    ]
    patterns = {
        "traceback": re.compile(r"Traceback \(most recent call last\):"),
        "child_failed": re.compile(r"ChildFailedError"),
        "import_error": re.compile(r"ImportError:"),
        "type_error": re.compile(r"TypeError:"),
        "runtime_error": re.compile(r"RuntimeError:"),
        "value_error": re.compile(r"ValueError:"),
        "torchrun_failed": re.compile(r"FAILED"),
    }

    hits: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in patterns.items():
            if pattern.search(text):
                hits.append(name)
    return sorted(set(hits))


def classify_run(run_dir: Path) -> tuple[str, list[str]]:
    run_name = run_dir.name
    reasons: list[str] = []

    if "dryrun" in run_name.lower():
        reasons.append("name_contains_dryrun")

    has_artifacts = (run_dir / "artifacts").exists()
    has_all_results = (run_dir / "checkpoints" / "all_results.json").exists()
    has_train_results = (run_dir / "checkpoints" / "train_results.json").exists()
    has_eval_results = (run_dir / "checkpoints" / "eval_results.json").exists()
    metrics_count = count_metrics_rows(run_dir)
    failure_hits = detect_failure_signatures(run_dir)

    has_progress = has_all_results or has_train_results or has_eval_results or metrics_count > 0

    if not has_artifacts:
        reasons.append("missing_artifacts_dir")

    if not has_progress:
        reasons.append("no_metrics_or_results")

    if failure_hits and not has_all_results and not has_train_results and not has_eval_results:
        reasons.extend(f"failure_signature:{hit}" for hit in failure_hits)

    if any(reason == "name_contains_dryrun" for reason in reasons):
        return "dryrun", reasons

    if "no_metrics_or_results" in reasons or any(
        reason.startswith("failure_signature:") for reason in reasons
    ):
        return "failed", reasons

    if has_all_results:
        return "completed", ["has_all_results"]

    if has_progress:
        progress_reason = "has_metrics" if metrics_count > 0 else "has_partial_results"
        return "partial", [progress_reason]

    return "failed", reasons or ["unclassified_failure"]


def find_wandb_run(run_dir: Path) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
    wandb_root = run_dir / "wandb" / "wandb"
    if not wandb_root.exists():
        return None, None, None

    run_dirs = sorted(p for p in wandb_root.glob("run-*") if p.is_dir())
    if not run_dirs:
        return None, None, None

    run_path = run_dirs[-1]
    run_id = run_path.name.rsplit("-", 1)[-1]
    metadata = read_json(run_path / "files" / "wandb-metadata.json")
    summary = read_json(run_path / "files" / "wandb-summary.json")
    return run_id, metadata, summary


def extract_wandb_urls(run_dir: Path) -> dict[str, str]:
    console_log = run_dir / "logs" / "console.log"
    if not console_log.exists():
        return {}

    text = console_log.read_text(encoding="utf-8", errors="replace")
    urls: dict[str, str] = {}
    project_match = re.search(r"View project at (https://\S+)", text)
    run_match = re.search(r"View run at (https://\S+)", text)
    if project_match:
        urls["project_url"] = project_match.group(1)
    if run_match:
        urls["run_url"] = run_match.group(1)
    return urls


def sanitize_wandb(metadata: dict[str, Any] | None, summary: dict[str, Any] | None, run_id: str | None, run_args: dict[str, Any], urls: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "project": run_args.get("wandb_project"),
        "mode": run_args.get("wandb_mode"),
        "run_id": run_id,
    }
    result.update(urls)
    if metadata:
        result["system"] = {
            "host": metadata.get("host"),
            "gpu": metadata.get("gpu"),
            "gpu_count": metadata.get("gpu_count"),
            "cpu_count": metadata.get("cpu_count"),
            "cpu_count_logical": metadata.get("cpu_count_logical"),
            "memory_total": metadata.get("memory", {}).get("total"),
            "cuda_version": metadata.get("cudaVersion"),
            "started_at": metadata.get("startedAt"),
            "python": metadata.get("python"),
        }
    if summary:
        result["summary_metrics"] = summary
    return result


def copy_small_or_truncated(src: Path, dst: Path, max_bytes: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    if size <= max_bytes:
        content = src.read_text(encoding="utf-8", errors="replace")
        dst.write_text(sanitize_text(content), encoding="utf-8")
        return

    lines = sanitize_text(src.read_text(encoding="utf-8", errors="replace")).splitlines()
    head = lines[:200]
    tail = lines[-200:] if len(lines) > 200 else []
    content = [
        f"# Truncated export generated from: {src}",
        f"# Original size bytes: {size}",
        f"# Exported at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "===== HEAD (first 200 lines) =====",
        *head,
    ]
    if tail:
        content.extend(["", "===== TAIL (last 200 lines) =====", *tail])
    dst.write_text("\n".join(content) + "\n", encoding="utf-8")


def export_run(run_dir: Path, records_root: Path, max_log_bytes: int, force: bool) -> Path:
    run_name = run_dir.name
    export_dir = records_root / run_name
    if export_dir.exists():
        if not force:
            raise FileExistsError(f"Export already exists: {export_dir}")
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "artifacts/run_args.json",
        "artifacts/training_args.json",
        "artifacts/dataset_stats.json",
        "logs/metrics.jsonl",
        "logs/train.log",
        "logs/console.log",
        "checkpoints/train_results.json",
        "checkpoints/eval_results.json",
        "checkpoints/all_results.json",
        "checkpoints/trainer_state.json",
    ]

    for relative_path in files_to_copy:
        src = run_dir / relative_path
        if not src.exists():
            continue
        dst = export_dir / relative_path
        if src.suffix == ".log":
            copy_small_or_truncated(src, dst, max_log_bytes)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    run_args = read_json(run_dir / "artifacts" / "run_args.json") or {}
    training_args = read_json(run_dir / "artifacts" / "training_args.json") or {}
    dataset_stats = read_json(run_dir / "artifacts" / "dataset_stats.json") or {}
    all_results = read_json(run_dir / "checkpoints" / "all_results.json")
    train_results = read_json(run_dir / "checkpoints" / "train_results.json")
    eval_results = read_json(run_dir / "checkpoints" / "eval_results.json")
    metrics_rows = unique_metric_rows(read_metrics(run_dir / "logs" / "metrics.jsonl"))
    latest_metrics = extract_latest_metrics(metrics_rows)
    wandb_run_id, wandb_metadata, wandb_summary = find_wandb_run(run_dir)
    wandb_urls = extract_wandb_urls(run_dir)

    classification, classification_reasons = classify_run(run_dir)

    summary = {
        "run_name": run_name,
        "source_run_dir": str(run_dir),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed" if all_results else "partial",
        "run_classification": classification,
        "classification_reasons": classification_reasons,
        "model_name_or_path": run_args.get("model_name_or_path"),
        "datasets": {
            "train_data": run_args.get("train_data", []),
            "valid_data": run_args.get("valid_data", []),
        },
        "key_hparams": {
            "num_train_epochs": run_args.get("num_train_epochs"),
            "max_steps": run_args.get("max_steps"),
            "learning_rate": run_args.get("learning_rate"),
            "per_device_train_batch_size": run_args.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": run_args.get("gradient_accumulation_steps"),
            "model_max_length": run_args.get("model_max_length"),
            "lora_r": run_args.get("lora_r"),
            "lora_alpha": run_args.get("lora_alpha"),
            "lora_dropout": run_args.get("lora_dropout"),
        },
        "dataset_stats": dataset_stats,
        "final_results": all_results or {},
        "train_results": train_results or {},
        "eval_results": eval_results or {},
        "latest_metrics": latest_metrics,
        "wandb": sanitize_wandb(wandb_metadata, wandb_summary, wandb_run_id, run_args, wandb_urls),
        "artifacts_present": {
            "training_args": bool(training_args),
            "metrics_jsonl": (run_dir / "logs" / "metrics.jsonl").exists(),
            "train_log": (run_dir / "logs" / "train.log").exists(),
            "console_log": (run_dir / "logs" / "console.log").exists(),
        },
    }

    (export_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return export_dir


def discover_runs(outputs_root: Path) -> list[Path]:
    if not outputs_root.exists():
        return []
    return sorted(
        path for path in outputs_root.iterdir() if path.is_dir() and (path / "artifacts").exists()
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    outputs_root = args.outputs_root.resolve()
    records_root = args.records_root.resolve()
    max_log_bytes = int(args.max_log_mb * 1024 * 1024)

    if args.run_name:
        run_dirs = [outputs_root / run_name for run_name in args.run_name]
    elif args.all:
        run_dirs = discover_runs(outputs_root)
    else:
        parser.error("Specify --run-name ... or --all.")

    selected_run_dirs: list[Path] = []
    skipped_runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        classification, reasons = classify_run(run_dir)
        if classification == "dryrun" and not args.include_dryrun:
            skipped_runs.append(
                {"run_name": run_dir.name, "classification": classification, "reasons": reasons}
            )
            continue
        if classification == "failed" and not args.include_failed:
            skipped_runs.append(
                {"run_name": run_dir.name, "classification": classification, "reasons": reasons}
            )
            continue
        selected_run_dirs.append(run_dir)

    exported: list[str] = []
    for run_dir in selected_run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        export_dir = export_run(run_dir, records_root, max_log_bytes, args.force)
        exported.append(str(export_dir))

    print(
        json.dumps(
            {"exported_runs": exported, "skipped_runs": skipped_runs},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
