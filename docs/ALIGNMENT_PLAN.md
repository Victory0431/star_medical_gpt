# Alignment Data And Evaluation Loop Plan

This document explains:

1. what `DPO / RM / GRPO` data is currently available,
2. what each dataset is actually good for,
3. how to turn the next alignment stages into a clean train-eval iteration loop.

## Executive summary

The strongest next-step dataset is:

- medical `pairwise preference` data: `3800/100/100`
- directly usable for: `DPO`, `ORPO`, `KTO`, and `Reward Model`
- best suited to improve: medical answer preference, factual correction, safety style, and recommendation quality

The current `GRPO` sample is **not** suitable as your formal GRPO training set:

- it is too small,
- it is closer to exact-answer style supervision,
- and its reward target is not aligned with open-ended medical dialogue quality.

The practical roadmap is:

1. run `DPO v1` on the medical pairwise data,
2. train `RM v1` on the same pairwise data,
3. build a new medical prompt-only dataset for real `GRPO`,
4. keep using the current `HealthBench` setup as the main external evaluation loop.

## Available datasets

### 1. Medical pairwise preference data

Local paths in the new repo:

- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/medical_pairwise_train.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/medical_pairwise_valid.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/medical_pairwise_test.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/rm/medical_pairwise_train.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/rm/medical_pairwise_valid.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/rm/medical_pairwise_test.jsonl`

Source:

- original repo `data/finetune/reward/*.json`

Counts:

- train: `3800`
- valid: `100`
- test: `100`

Schema:

```json
{"question": "...", "response_chosen": "...", "response_rejected": "..."}
```

Best use cases:

- `DPO`
- `ORPO`
- `KTO`
- `RM`

What it improves:

- preference toward better medical answers
- reduced probability of selecting obviously wrong or harmful answers
- more realistic alignment behavior than plain SFT alone

### 2. General pairwise control dataset

Local path:

- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/general_pairwise_dpo_zh_500.jsonl`

Source:

- original repo `data/reward/dpo_zh_500.jsonl`

Count:

- `500`

Schema:

```json
{"system": "...", "history": [], "question": "...", "response_chosen": "...", "response_rejected": "..."}
```

Best use case:

- a small auxiliary dataset for DPO mixing experiments

What it improves:

- general instruction following
- non-medical style robustness
- compatibility with system/history-aware pairwise templates

Recommended role:

- not the main dataset
- use only as a small ablation such as `90% medical + 10% general`

### 3. GRPO reference sample

Local path:

- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/grpo_reference/grpo_sample_reference.jsonl`

Source:

- original repo `data/grpo/sample.jsonl`

Count:

- `32`

Schema:

```json
{"question": "...", "answer": "..."}
```

Why it is not enough for your formal GRPO stage:

- too small
- too close to exact-answer supervision
- not aligned with open-ended medical interaction quality

What it is still useful for:

- minimal format reference
- understanding how the old GRPO pipeline consumes prompt-answer data

## What should not be treated as alignment data

These are useful, but they are not your DPO/GRPO core data:

- `data/sft/...`
  used for `SFT` or for generating alignment candidates
- `data/finetune/pretrain/...`
  used for continued pretraining
- `data/rag/...`
  used for retrieval, not preference optimization

## Recommended roadmap

### Phase 1: DPO v1

Policy init:

- current best SFT checkpoint: `checkpoint-75`

Dataset:

- medical pairwise train/valid/test

Main expected gains:

- `communication_quality`
- `context_awareness`
- `hedging`
- safer answer preference

### Phase 2: DPO v2 or ORPO v1

Dataset mix:

- `90% medical_pairwise`
- `10% general_pairwise_dpo_zh_500`

Purpose:

- test whether a small amount of general preference data helps preserve generic instruction-following ability

### Phase 3: RM v1

Dataset:

- the same medical pairwise split

Purpose:

- upgrade the project from “I can run DPO” to “I can also build the reward model stage”

Note:

- `3800` samples are enough for a small RM baseline
- but a stronger production-style RM would eventually need more domain-specific pairwise data

### Phase 4: Real GRPO

Do **not** directly use the old sample as the main GRPO dataset.

Instead:

1. collect medical prompt-only questions,
2. generate multiple responses online,
3. score them with reward functions or an LLM judge,
4. optimize the policy with GRPO.

## What your GRPO data should really look like

Your future GRPO stage should focus on open-ended medical behavior, not exact-answer matching.

Good prompt sources:

- `question` fields from the medical pairwise data
- your existing SFT validation/test user questions
- additional triage, safety, context-seeking, and communication prompts

Good reward dimensions:

- medical correctness
- context seeking when information is missing
- emergency referral behavior
- appropriate hedging under uncertainty
- clarity and actionable structure

That is much better aligned with your current `HealthBench` slices.

## Evaluation loop

### DPO stage

Main benchmark:

- `HealthBench consensus`

Primary slices:

- `axis:communication_quality`
- `axis:context_awareness`
- `axis:instruction_following`
- `theme:hedging`
- `theme:communication`

Supplement:

- pairwise holdout win-rate
- medical preference accuracy on held-out pairs

### RM stage

Main metric:

- ranking accuracy on chosen vs rejected in `valid/test`

Supplement:

- compare RM ranking trends against `HealthBench` outcomes for base / SFT / DPO outputs

### GRPO stage

Main benchmark:

- still `HealthBench`

Primary slices:

- `theme:emergency_referrals`
- `theme:context_seeking`
- `theme:hedging`
- `axis:communication_quality`

Reason:

- GRPO is most meaningful when it improves open-ended behavior quality, not just static answer selection

## Best next-step loop

The most practical next sequence is:

1. use `SFT checkpoint-75` as policy init
2. run `DPO v1` on the medical pairwise data
3. evaluate with the current `HealthBench theme15x7` setup
4. run a small `DPO v2 / ORPO v1` mixed-data ablation
5. build `RM v1`
6. only then move into a real GRPO stage

## Best interview framing

A stronger answer to “how did you design DPO/GRPO data?” is:

- “I did not blindly run GRPO on a tiny demo dataset. I first used domain-specific medical pairwise data for DPO and reward modeling so the preference loop was credible. For GRPO, I planned a separate prompt-only dataset and reward design aligned with the same open-ended medical evaluation slices I use in HealthBench.”

That sounds much more like a real engineering workflow.
