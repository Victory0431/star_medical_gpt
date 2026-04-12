# Evaluation Results

This document records the current benchmark snapshots that are worth discussing in interviews.

## Formal stratified matrix

Comparison setting:

- benchmark: `HealthBench consensus`
- sampling mode: `stratified_theme`
- sampling plan: `7` themes, `15` examples per theme, `105` total
- random seed: `42`
- judge model requested: `gpt-5.2`
- judge model actually returned: `gpt-5.2`
- judge API style: OpenAI-compatible chat completions

Compared runs:

- base: `20260409_healthbench_base_gpt52_full_theme15x7`
- sft 1k: `20260410_healthbench_huatuo1k_gpt52_full_theme15x7`
- sft 5w best checkpoint: `20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7`
- sft 5w later checkpoint: `20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7`
- dpo v1: `20260410_healthbench_dpo_medpair_ckpt100_gpt52_full_theme15x7`

## Headline result

| model | overall clipped mean | note |
| --- | ---: | --- |
| `Qwen3-8B base` | `0.2206` | current formal stratified baseline |
| `Qwen3-8B + huatuo_1k LoRA` | `0.2508` | current lightweight formal SFT baseline |
| `Qwen3-8B + huatuo_5w LoRA (ckpt-75)` | `0.2889` | current best formal SFT baseline |
| `Qwen3-8B + huatuo_5w LoRA (ckpt-925)` | `0.2587` | later checkpoint from the same run |
| `Qwen3-8B + DPO medical_pairwise (ckpt-100)` | `0.2111` | current formal DPO v1 result |

Headline reading:

- `huatuo_5w checkpoint-75` remains the strongest formal baseline
- `huatuo_1k` beats the base model on the formal `105`-example slice, so it is a valid lightweight SFT baseline even if it is not the final mainline model
- `DPO v1` falls below both the best SFT run and the base model, which means the alignment loop is working but the current preference data and objective are not yet good enough
- `checkpoint-75` is clearly better than `checkpoint-925`, which makes early stopping and best-checkpoint selection benchmark-backed defaults rather than optional polish

## Axis comparison

| axis | base | 1k LoRA | 5w ckpt-75 | 5w ckpt-925 | DPO ckpt-100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `axis:accuracy` | `0.1711` | `0.2039` | `0.2368` | `0.1974` | `0.1447` |
| `axis:communication_quality` | `0.2000` | `0.2444` | `0.2889` | `0.2444` | `0.1778` |
| `axis:completeness` | `0.0000` | `0.0833` | `0.2500` | `0.0000` | `0.0833` |
| `axis:context_awareness` | `0.2407` | `0.2778` | `0.3704` | `0.3333` | `0.2778` |
| `axis:instruction_following` | `0.5000` | `0.4667` | `0.3667` | `0.5333` | `0.4667` |

Reading:

- `checkpoint-75` is the strongest and most balanced run on `accuracy`, `completeness`, and `context_awareness`
- `huatuo_1k` already beats base on `accuracy`, `communication_quality`, and `context_awareness`, which explains why it now clears base on the formal slice
- `DPO v1` keeps some structure on `context_awareness` and `instruction_following`, but gives back too much on `accuracy` and `communication_quality`
- `checkpoint-925` is higher on `instruction_following`, but that did not translate into a better overall result

## Theme comparison

| theme | base | 1k LoRA | 5w ckpt-75 | 5w ckpt-925 | DPO ckpt-100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Expertise-tailored communication` (`theme:communication`) | `0.0333` | `0.1333` | `0.0667` | `0.0667` | `0.0333` |
| `Response depth` (`theme:complex_responses`) | `0.2333` | `0.2333` | `0.2000` | `0.2333` | `0.2333` |
| `Context seeking` (`theme:context_seeking`) | `0.0333` | `0.0333` | `0.0333` | `0.1667` | `0.0333` |
| `Emergency referrals` (`theme:emergency_referrals`) | `0.2667` | `0.3000` | `0.4333` | `0.2000` | `0.2000` |
| `Global health` (`theme:global_health`) | `0.2667` | `0.2667` | `0.5333` | `0.3333` | `0.2667` |
| `Health data tasks` (`theme:health_data_tasks`) | `0.4667` | `0.4333` | `0.3333` | `0.4333` | `0.4000` |
| `Responding under uncertainty` (`theme:hedging`) | `0.2444` | `0.3556` | `0.4222` | `0.3778` | `0.3111` |

Reading:

- `checkpoint-75` shows its biggest gains on `Emergency referrals`, `Global health`, and `Responding under uncertainty`
- `huatuo_1k` already improves `Expertise-tailored communication`, `Emergency referrals`, and `Responding under uncertainty` over base
- `DPO v1` only retains a mild `Responding under uncertainty` gain over base and does not produce a strong formal win on the more important medical interaction themes
- `Health data tasks` is already a relatively strong theme for the base model, so it should not dominate the project story by itself

## Strong interview narrative

A weak project story is:

- “I ran SFT and then DPO, so the score naturally kept going up.”

A stronger project story is:

- “I evaluated base -> 1k SFT -> 5w SFT -> DPO on `HealthBench consensus` using stratified theme sampling: `15` examples per theme, `105` total. The best `5w` checkpoint beat base clearly, a later checkpoint regressed, and the first DPO recipe underperformed the best SFT and even slipped below base. That told me to treat checkpoint selection and reward/data alignment as first-class engineering problems instead of assuming every later stage automatically improves the benchmark.”

That framing is much closer to real model development work.

## Current alignment-stage interpretation

Based on the formal matrix, the current engineering judgment is:

1. Keep `huatuo_5w checkpoint-75` as the main model baseline.
2. Keep `huatuo_1k` as the lightweight SFT baseline for smoke tests and fast pipeline validation.
3. Treat `DPO v1` as a successful infrastructure milestone, not yet as a better model.
4. Focus the next alignment step on better medical preference data and reward targets that match slices such as `Expertise-tailored communication`, `Responding under uncertainty`, and `Emergency referrals`.

## Earlier smoke matrix (reference only)

Comparison setting:

- benchmark: `HealthBench consensus`
- sample count: `10`
- judge model requested: `gpt-5.2`
- judge model actually returned: `gpt-5.2`
- judge API style: OpenAI-compatible chat completions

Compared runs:

- base: `20260409_healthbench_base_gpt52_full_10`
- sft 1k: `20260409_healthbench_huatuo1k_gpt52_full_10`
- sft 5w best checkpoint: `20260409_healthbench_huatuo5w_ckpt75_gpt52_full_10`
- sft 5w latest checkpoint: `20260409_healthbench_huatuo5w_ckpt925_gpt52_full_10`

## Headline result

| model | overall clipped mean | note |
| --- | ---: | --- |
| `Qwen3-8B base` | `0.4167` | current smoke baseline |
| `Qwen3-8B + huatuo_1k LoRA` | `0.2833` | current smoke SFT baseline |
| `Qwen3-8B + huatuo_5w LoRA (ckpt-75)` | `0.4167` | best validation checkpoint in current 5w run |
| `Qwen3-8B + huatuo_5w LoRA (ckpt-925)` | `0.4000` | later checkpoint from the same run |

Headline reading:

- `huatuo_5w` clearly recovers from the weak `1k` result
- the best observed `5w` checkpoint matches the base model overall on this smoke set
- the later `5w` checkpoint is slightly worse than the early best checkpoint, which is a meaningful overtraining signal
- the current evidence supports keeping `5w` and dropping `1k` as the main SFT line, but it does **not** yet support a claim that `5w` robustly beats base

## Axis comparison

| axis | base | 1k LoRA | 5w ckpt-75 | 5w ckpt-925 |
| --- | ---: | ---: | ---: | ---: |
| `axis:accuracy` | `0.5000` | `0.1429` | `0.3571` | `0.2857` |
| `axis:communication_quality` | `0.6667` | `0.6667` | `0.6667` | `0.6667` |
| `axis:context_awareness` | `0.2500` | `0.1250` | `0.2500` | `0.2500` |
| `axis:instruction_following` | `0.5000` | `1.0000` | `1.0000` | `1.0000` |

Reading:

- `huatuo_5w` restores `context_awareness` back to the base level
- `huatuo_5w` keeps the `instruction_following` gain seen in the `1k` run
- `accuracy` improves substantially from `1k` to `5w`, but still does not surpass the base model in this smoke set
- `ckpt-75` is better than `ckpt-925` on `accuracy`, which aligns with the training-side `eval_loss` signal

## Theme comparison

| theme | base | 1k LoRA | 5w ckpt-75 | 5w ckpt-925 |
| --- | ---: | ---: | ---: | ---: |
| `Response depth` (`theme:complex_responses`) | `1.0000` | `0.5000` | `1.0000` | `1.0000` |
| `Context seeking` (`theme:context_seeking`) | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `Emergency referrals` (`theme:emergency_referrals`) | `0.5000` | `0.2500` | `0.0000` | `0.0000` |
| `Global health` (`theme:global_health`) | `0.5000` | `0.2500` | `0.2500` | `0.5000` |
| `Health data tasks` (`theme:health_data_tasks`) | `0.5000` | `1.0000` | `1.0000` | `1.0000` |
| `Responding under uncertainty` (`theme:hedging`) | `0.2222` | `0.1111` | `0.5556` | `0.3333` |

Reading:

- `huatuo_5w` shows a meaningful improvement on `Responding under uncertainty`, especially at `ckpt-75`
- `huatuo_5w` preserves the strong `Health data tasks` behavior
- `Emergency referrals` is weak for both `5w` checkpoints and is currently worse than base on this smoke slice
- this suggests the next stage should not only chase more generic medical SFT scale, but should explicitly target safety-sensitive medical interaction behavior

## Training signal from the 5w run

From the training artifacts of `20260409_121822_qwen3-8b_huatuo-5w_lora_eval`:

- best validation checkpoint: `checkpoint-75`
- best validation `eval_loss`: `2.5285`
- later checkpoint evaluated here: `checkpoint-925`
- later validation `eval_loss`: `2.8470`

This matters because:

- the external benchmark agrees with the training-time validation signal
- the best early checkpoint is better than the later checkpoint
- this is exactly the kind of evidence you want when arguing for early stopping, better checkpoint selection, or a different curriculum rather than simply training longer

## Why this is actually a strong interview point

A weak project story is:

- “I fine-tuned the model and the score improved.”

A stronger project story is:

- “I compared base, 1k SFT, and 5w SFT under the same open-ended medical benchmark. The 5w run clearly fixed most of the 1k degradation, but the best checkpoint appeared early and the later checkpoints regressed. I used both training-time `eval_loss` and benchmark slices to decide that the next step was better checkpoint selection and targeted data/alignment, not just longer training.”

That is much closer to real model development.

## Practical next interpretation

Based on the current smoke matrix, the most reasonable next steps are:

1. Keep `1k` only as a pipeline-validation checkpoint. It is no longer the main SFT candidate.
2. Use `huatuo_5w checkpoint-75` as the current main SFT baseline, not `checkpoint-925`.
3. Add explicit early-stopping or best-checkpoint selection to the standard workflow, because this run already shows that later is not better.
4. Before jumping into `DPO` or `GRPO`, strengthen safety-sensitive and triage-sensitive data slices, especially around `Emergency referrals`.
5. For the next formal comparison, increase the benchmark sample count from `10` to `50` or `100` to reduce smoke-test variance.
6. If the goal is to beat base rather than merely match it, the next likely lever is not much longer generic SFT, but either:
   - better-distributed SFT data
   - a stronger validation split strategy
   - targeted preference/reward optimization on communication and safety slices

## Source records

- [`experiment_records/eval/20260409_healthbench_base_gpt52_full_theme15x7/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_theme15x7/summary.json)
- [`experiment_records/eval/20260410_healthbench_huatuo1k_gpt52_full_theme15x7/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260410_healthbench_huatuo1k_gpt52_full_theme15x7/summary.json)
- [`experiment_records/eval/20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7/summary.json)
- [`experiment_records/eval/20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7/summary.json)
- [`experiment_records/eval/20260410_healthbench_dpo_medpair_ckpt100_gpt52_full_theme15x7/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260410_healthbench_dpo_medpair_ckpt100_gpt52_full_theme15x7/summary.json)
- [`experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo5w_gpt52_theme15x7/README.md`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo5w_gpt52_theme15x7/README.md)
- [`experiment_records/eval/20260410_healthbench_compare_base_vs_1k_vs_5w_vs_dpo_gpt52_theme15x7/README.md`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260410_healthbench_compare_base_vs_1k_vs_5w_vs_dpo_gpt52_theme15x7/README.md)
- [`experiment_records/eval/20260409_healthbench_base_gpt52_full_10/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_10/summary.json)
- [`experiment_records/eval/20260409_healthbench_huatuo1k_gpt52_full_10/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo1k_gpt52_full_10/summary.json)
- [`experiment_records/eval/20260409_healthbench_huatuo5w_ckpt75_gpt52_full_10/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt75_gpt52_full_10/summary.json)
- [`experiment_records/eval/20260409_healthbench_huatuo5w_ckpt925_gpt52_full_10/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_huatuo5w_ckpt925_gpt52_full_10/summary.json)
- [`experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo1k_gpt52_10/README.md`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo1k_gpt52_10/README.md)
- [`experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo5w_gpt52_10/README.md`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_compare_base_vs_huatuo5w_gpt52_10/README.md)
