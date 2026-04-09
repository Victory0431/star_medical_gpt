# HealthBench Base Formal Baseline (Theme 15 x 7)

This record stores the formal small-scale stratified evaluation result for `Qwen3-8B base` on `HealthBench consensus`.

Run details:

- run name: `20260409_healthbench_base_gpt52_full_theme15x7`
- benchmark: `HealthBench consensus`
- sampling mode: `stratified_theme`
- sampling plan: `7` themes, `15` examples per theme, `105` total
- evaluated model: `Qwen3-8B base`
- judge API style: OpenAI-compatible chat completions
- requested judge model: `gpt-5.2`
- actual returned judge model: `gpt-5.2`

Main result:

- overall clipped mean: `0.2206`
- `axis:accuracy`: `0.1711`
- `axis:context_awareness`: `0.2407`
- `theme:health_data_tasks`: `0.4667`

Reading:

- this result is more stable than the earlier `10`-example smoke run because it covers all `7` themes
- the base model retains some ability on `health_data_tasks`
- but interaction-heavy themes such as `communication` and `context_seeking` remain weak
