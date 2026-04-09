# Evaluation Design

This document explains the evaluation design for this repository and why `HealthBench` is introduced as the primary open-ended benchmark.

## Why not rely only on medical multiple-choice benchmarks

Benchmarks such as the medical subset of `C-EVAL` are still useful, but they mainly measure:

- factual recall
- option discrimination
- exam-style accuracy

They do **not** sufficiently evaluate:

- response completeness in open-ended medical answers
- communication quality
- whether the model asks for missing context
- whether the model hedges correctly under uncertainty
- whether the model follows task framing and answer style constraints

These are exactly the areas where later-stage alignment methods like `DPO` and `GRPO` often change model behavior.

## Why HealthBench is a better fit here

`HealthBench` is an open-ended medical benchmark with rubric-based scoring. The workflow is:

1. your model generates a free-form medical response
2. an OpenAI judge model scores that response against detailed rubric items
3. the final metrics are aggregated across themes and axes

This makes it much more suitable for:

- `SFT` evaluation on accuracy and completeness
- `DPO` evaluation on answer style and communication
- `GRPO` evaluation on instruction following, hedging, and context-seeking behavior

## What HealthBench can measure for this project

From the released rubric tags, the benchmark naturally exposes these axes:

- `axis:accuracy`
- `axis:completeness`
- `axis:context_awareness`
- `axis:communication_quality`
- `axis:instruction_following`

And the released example tags naturally expose themes such as:

- `theme:global_health`
- `theme:hedging`
- `theme:communication`
- `theme:context_seeking`
- `theme:emergency_referrals`
- `theme:health_data_tasks`
- `theme:complex_responses`

## Recommended mapping to training stages

### SFT stage

Primary focus:

- overall score
- `axis:accuracy`
- `axis:completeness`
- `theme:complex_responses`
- `theme:health_data_tasks`

Reason:

- this stage should first establish medical knowledge coverage and coherent response structure

### DPO stage

Primary focus:

- `axis:communication_quality`
- `axis:instruction_following`
- `axis:context_awareness`
- `theme:communication`
- `theme:hedging`
- `theme:context_seeking`

Reason:

- DPO is often used to improve style preference, answer helpfulness, and user-facing behavior

### GRPO stage

Primary focus:

- same style-oriented axes as DPO
- plus safety-sensitive themes such as `theme:emergency_referrals`

Reason:

- GRPO is useful when you want the reward signal to shape nuanced behavior under open-ended prompts

## Practical conclusion

For this project, the best interview-ready evaluation story is not:

- “I only report one overall benchmark number”

But rather:

- “I use HealthBench as the main open-ended medical benchmark, then I read the result by axis and theme so I can distinguish knowledge gains from alignment gains.”

That is a much more industrial and credible evaluation narrative.

## Current implementation in this repo

Runtime code:

- [`evaluation/run_eval.py`](/home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py)
- [`evaluation/benchmarks/healthbench.py`](/home/qjh/llm_learning/my_medical_gpt/evaluation/benchmarks/healthbench.py)
- [`evaluation/generators/hf_chat.py`](/home/qjh/llm_learning/my_medical_gpt/evaluation/generators/hf_chat.py)
- [`evaluation/judges/openai_healthbench.py`](/home/qjh/llm_learning/my_medical_gpt/evaluation/judges/openai_healthbench.py)

Reference snapshot:

- [`references/openai/simple-evals/healthbench_eval.py`](/home/qjh/llm_learning/my_medical_gpt/references/openai/simple-evals/healthbench_eval.py)

## Engineering choices in this repo

The local evaluation implementation is intentionally split into three layers:

- benchmark loading and aggregation
- local Hugging Face generation
- external judge scoring

This is important for later extension:

- `SFT` can use the exact same benchmark and generator
- `DPO` and `GRPO` can plug in different model checkpoints or adapters without changing the benchmark code
- official judge scoring can be rerun independently from generation, which saves time and API cost

The practical run modes are:

- `full`: generate then judge
- `generate_only`: generate once, keep `responses.jsonl`
- `judge_only`: reuse an existing `responses.jsonl` and only call the judge

The last two modes are especially useful for industrial workflows, because API judging is often retried separately from local inference.

## Score construction

This repo computes scores in four layers.

### 1. Rubric layer

Each HealthBench example contains several rubric items.

For each rubric item, the judge must return structured JSON:

- `criteria_met`
- `explanation`

If the rubric item is positive, meeting it should contribute positive points.
If the rubric item is undesirable and has negative points, meeting it should count against the response.

### 2. Example layer

For one prompt, the example score is:

- achieved rubric points divided by total positive rubric points

This means:

- a strong answer can approach `1.0`
- harmful or undesirable behavior can reduce the raw score

### 3. Slice layer

After example scoring, the repo aggregates by:

- `axis:*`
- `theme:*`
- `physician_agreed_category:*`

This is what makes the benchmark useful for training-stage diagnosis rather than only leaderboard-style reporting.

### 4. Summary layer

The final summary stores both:

- `raw_mean`
- `clipped_mean`

`clipped_mean` is mainly for easier comparison and dashboard reading.

## Judge configuration in this repo

The judge implementation supports both standard OpenAI and OpenAI-compatible APIs.

Current defaults:

- judge backend: chat completions compatible API
- default judge model: `gpt-5.2`
- JSON-only judge output with `temperature=0`

Supported environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `JUDGE_API_KEY`
- `JUDGE_BASE_URL`

For proxy or gateway providers, the code also records:

- requested model: `judge_model`
- actual returned model: `judge_actual_model`
- actual API URL: `judge_api_base_url`

This is useful for catching provider-side model routing mismatches.

## Current baseline result

Current committed base-model smoke baseline:

- run name: `20260409_healthbench_base_gpt52_full_10`
- benchmark: `HealthBench consensus`
- sample count: `10`
- evaluated model: `Qwen3-8B base`
- judge model requested: `gpt-5.2`
- actual returned judge model: `gpt-5.2`

Main result:

- overall clipped mean: `0.4167`
- overall raw mean: `0.4167`

Axis slice:

- `axis:accuracy = 0.5000`
- `axis:communication_quality = 0.6667`
- `axis:context_awareness = 0.2500`
- `axis:instruction_following = 0.5000`

Initial reading:

- the base model is not terrible at surface communication
- the weak point in this smoke run is `context_awareness`
- this matches the project expectation that later `DPO` or `GRPO` should improve context-seeking, hedging, and user-facing medical interaction quality

Important caveat:

- this is a smoke baseline on `10` examples, not a full benchmark report
- it is suitable for pipeline verification and early comparison, but not for a final headline claim

The corresponding lightweight record is stored under:

- [`experiment_records/eval/20260409_healthbench_base_gpt52_full_10/summary.json`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_10/summary.json)
- [`experiment_records/eval/20260409_healthbench_base_gpt52_full_10/README.md`](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_10/README.md)

## Important limitation

Official HealthBench scoring requires `OPENAI_API_KEY`.

At the moment, without that key the repo can already run:

- local generation on HealthBench prompts
- base model evaluation entry
- LoRA model evaluation entry
- structured logging and response export

But it cannot yet produce official rubric scores until the judge API key is provided.
