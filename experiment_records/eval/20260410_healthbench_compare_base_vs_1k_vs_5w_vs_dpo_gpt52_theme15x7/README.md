# HealthBench Formal Comparison: base vs 1k vs 5w vs DPO (Theme 15 x 7)

Comparison setting:

- benchmark: `HealthBench consensus`
- sampling mode: `stratified_theme`
- sampling plan: `7` themes, `15` examples per theme, `105` total
- judge model: `gpt-5.2`

Compared runs:

- base: `20260409_healthbench_base_gpt52_full_theme15x7`
- 1k: `20260410_healthbench_huatuo1k_gpt52_full_theme15x7`
- 5w best checkpoint: `20260409_healthbench_huatuo5w_ckpt75_gpt52_full_theme15x7`
- 5w later checkpoint: `20260409_healthbench_huatuo5w_ckpt925_gpt52_full_theme15x7`
- DPO v1: `20260410_healthbench_dpo_medpair_ckpt100_gpt52_full_theme15x7`

Core results:

- base overall clipped mean: `0.2206`
- 1k overall clipped mean: `0.2508`
- 5w checkpoint-75 overall clipped mean: `0.2889`
- 5w checkpoint-925 overall clipped mean: `0.2587`
- DPO checkpoint-100 overall clipped mean: `0.2111`

Reading:

- the current best model is still `huatuo_5w checkpoint-75`
- `huatuo_1k` is a real lightweight SFT baseline, not just a smoke-only artifact
- later SFT is not always better, which keeps checkpoint selection and early stopping important
- DPO v1 is a meaningful alignment-stage infrastructure milestone, but not yet a stronger benchmark winner
