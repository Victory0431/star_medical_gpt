# DPO Metrics Explained

Chinese primary version:

- [DPO_METRICS.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_METRICS.zh-CN.md)

This document's detailed explanation currently lives in Chinese first, because the project documentation is now maintained in Chinese-first mode.

Short summary:

- `chosen_reward = beta * (logpi_chosen - logpi_ref_chosen)`
- `rejected_reward = beta * (logpi_rejected - logpi_ref_rejected)`
- `margin = chosen_reward - rejected_reward`
- `accuracy = 1[chosen_reward > rejected_reward]`
- for sigmoid DPO: `loss = -log(sigmoid(margin))`

Key references:

- [train_dpo.py](/home/qjh/llm_learning/my_medical_gpt/script/alignment/train_dpo.py)
- [DPO_WORKFLOW.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_WORKFLOW.md)
- [TRL dpo_trainer.py](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py)
