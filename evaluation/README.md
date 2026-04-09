# Evaluation

This directory contains the benchmark-facing evaluation code for Star Medical GPT.

## Design goals

- keep model generation separate from scoring
- support base model and LoRA adapter evaluation through the same entrypoint
- make it easy to reuse the same evaluation harness for `SFT`, `DPO`, and `GRPO`
- keep outputs structured and easy to compare across runs

## Current benchmark

- `HealthBench`

HealthBench is more suitable than medical multiple-choice sets such as `C-EVAL` when the goal is to evaluate:

- open-ended medical response quality
- communication style
- context seeking
- instruction following
- hedging and safety behavior

This makes it a better fit for later `DPO` and `GRPO` stages, where improvements often appear in response style and clinical communication quality rather than pure multiple-choice accuracy.

## Supported modes

- `full`: local generation plus official rubric judging
- `generate_only`: only generate model responses and save them for later reuse
- `judge_only`: score an existing `responses.jsonl` file without regenerating answers

## Recovery behavior

- if `responses.jsonl` already contains a prompt, generation skips that prompt
- if `judgments.jsonl` already contains a prompt, judging skips that prompt
- use `--overwrite-responses` or `--overwrite-judgments` when you want a clean rerun

## Layout

```text
evaluation/
  benchmarks/
    healthbench.py
  generators/
    hf_chat.py
  judges/
    openai_healthbench.py
  run_eval.py
```

## Outputs

Evaluation runs are written under:

```text
outputs/eval/<run_name>/
  artifacts/
  logs/
  responses.jsonl
  judgments.jsonl
  summary.json
  summary.md
```

## Important note

Official HealthBench scoring requires `OPENAI_API_KEY`, because the rubric-based judge is itself an OpenAI model call.

Without that key, you can still run:

- `generate_only` mode for local model outputs
- `judge_only` later after exporting `OPENAI_API_KEY`

But you cannot obtain official rubric scores until the judge key is available.
