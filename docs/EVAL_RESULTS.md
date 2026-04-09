# Evaluation Results

This document records the current benchmark snapshots that are worth discussing in interviews.

## Current smoke comparison

Comparison setting:

- benchmark: `HealthBench consensus`
- sample count: `10`
- judge model requested: `gpt-5.2`
- judge model actually returned: `gpt-5.2`
- judge API style: OpenAI-compatible chat completions

Compared runs:

- base: `20260409_healthbench_base_gpt52_full_10`
- sft: `20260409_healthbench_huatuo1k_gpt52_full_10`

## Headline result

| model | overall clipped mean | note |
| --- | ---: | --- |
| `Qwen3-8B base` | `0.4167` | current smoke baseline |
| `Qwen3-8B + huatuo_1k LoRA` | `0.2833` | current smoke SFT baseline |

Initial conclusion:

- on this `10`-example smoke set, the `1k` LoRA version underperforms the base model overall
- this is a valid and useful result, not a failure of the evaluation setup
- it suggests the first small SFT stage has not yet produced a robust open-ended medical gain on HealthBench

## Axis comparison

| axis | base | 1k LoRA | delta |
| --- | ---: | ---: | ---: |
| `axis:accuracy` | `0.5000` | `0.1429` | `-0.3571` |
| `axis:communication_quality` | `0.6667` | `0.6667` | `0.0000` |
| `axis:context_awareness` | `0.2500` | `0.1250` | `-0.1250` |
| `axis:instruction_following` | `0.5000` | `1.0000` | `+0.5000` |

Reading:

- the `1k` LoRA run improved `instruction_following`
- but it lost ground on `accuracy`
- it also did not improve `context_awareness`, which is one of the most important later-stage alignment targets

## Theme comparison

| theme | base | 1k LoRA | delta |
| --- | ---: | ---: | ---: |
| `theme:complex_responses` | `1.0000` | `0.5000` | `-0.5000` |
| `theme:context_seeking` | `0.0000` | `0.0000` | `0.0000` |
| `theme:emergency_referrals` | `0.5000` | `0.2500` | `-0.2500` |
| `theme:global_health` | `0.5000` | `0.2500` | `-0.2500` |
| `theme:health_data_tasks` | `0.5000` | `1.0000` | `+0.5000` |
| `theme:hedging` | `0.2222` | `0.1111` | `-0.1111` |

Reading:

- the `1k` LoRA run seems to help on `health_data_tasks`
- but it does not yet help on `hedging` or `context_seeking`
- for a medical assistant project, that means the current first-stage SFT is still not enough to claim stronger real-world interaction quality

## Why this is actually a strong interview point

A weak project story is:

- “I fine-tuned the model and the score improved.”

A stronger project story is:

- “I used open-ended medical evaluation to verify whether the first small SFT stage actually improved the behavior I cared about. It improved some narrow slices, but not the overall open-ended benchmark, so I treated that as evidence that the next stage needed better data or alignment.”

That second story sounds much closer to real model development.

## Practical next interpretation

Based on the current smoke comparison, the most reasonable next steps are:

1. Keep this `1k` run as a pipeline-validation SFT checkpoint, not a final quality claim.
2. Run the same benchmark on `huatuo_5w`.
3. Use `HealthBench` axis and theme slices to check whether larger SFT restores `accuracy` and `context_awareness`.
4. Only after that start discussing `DPO` or `GRPO` gains on style-sensitive slices such as `hedging`, `context_seeking`, and `communication_quality`.

## Source records

- [`experiment_records/eval/20260409_healthbench_base_gpt52_full_10/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_10/summary.json)
- [`experiment_records/eval/20260409_healthbench_huatuo1k_gpt52_full_10/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo1k_gpt52_full_10/summary.json)
- [`experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo1k_gpt52_10/README.md`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo1k_gpt52_10/README.md)
