# HealthBench DPO v1 Formal Result (medical_pairwise, checkpoint-100)

This record stores the formal stratified evaluation for DPO v1 trained on `medical_pairwise`.

Run details:

- run name: `20260410_healthbench_dpo_medpair_ckpt100_gpt52_full_theme15x7`
- benchmark: `HealthBench consensus`
- sampling mode: `stratified_theme`
- sampling plan: `7` themes, `15` examples per theme, `105` total
- evaluated model: `Qwen3-8B + DPO medical_pairwise`
- adapter checkpoint: `checkpoint-100`
- generator device: `cuda:0`
- judge model: `gpt-5.2`

Key results:

- overall clipped mean: `0.2111`
- `axis:accuracy`: `0.1447`
- `axis:communication_quality`: `0.1778`
- `theme:emergency_referrals`: `0.2000`
- `theme:hedging`: `0.3111`

Reading:

- this is a successful infrastructure milestone for the DPO stage, but not yet a better model than the best SFT baseline
- the current preference data and objective likely do not match the formal medical benchmark well enough
