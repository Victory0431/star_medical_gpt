# DPO V2 数据重构总结

这份文档记录 `medical_pairwise_*` 这批 `DPO v2` 数据的重构结果、去重结果、正式 processed 产物，以及它和昨晚其他训练任务之间的关系。

## 1. 先说结论

昨晚完成的是 `3800` 条医疗 pairwise 数据的 `DPO v2` 重构，不是 `DPO` 训练本身。

并行存在的另一条任务是 `HQ-50k SFT` 训练，它是独立 run，不要和 `DPO v2` 数据重构混在一起讲。

你后面面试里可以直接这样表述：

- `SFT` 线：继续做模型能力底座强化。
- `DPO v2` 线：先重构偏好数据，把旧 pairwise 数据改造成更贴近 `HealthBench` 优化目标的偏好样本。
- 这两条线是并行推进的，不是“昨晚已经把 DPO 训练也跑完了”。

## 2. 为什么要做 DPO v2 重构

原始 `medical_pairwise` 数据存在几个典型问题：

- `chosen / rejected` 方向并不总是可靠。
- `chosen` 有时只是更长，不一定更安全、更准确。
- 不少样本缺少 `context awareness`、`hedging`、`emergency referral` 这类医疗对齐中更关键的行为特征。
- 如果直接拿原始 pairwise 做 `DPO`，模型很可能学到“更像旧论坛回答”而不是“更像合格医疗助手”。

所以这次 `DPO v2` 的目标不是简单润色，而是把偏好方向重置到更接近 `HealthBench` 的目标轴上：

- `accuracy`
- `communication_quality`
- `context_awareness`
- `emergency_referrals`
- `hedging`

## 3. 数据重构产物

### 原始重构结果

路径：

- `train`
  - `data/alignment/reconstructed/dpo_v2/raw/train/medical_pairwise_train_v2.jsonl`
- `valid`
  - `data/alignment/reconstructed/dpo_v2/raw/valid/medical_pairwise_valid_v2.jsonl`
- `test`
  - `data/alignment/reconstructed/dpo_v2/raw/test/medical_pairwise_test_v2.jsonl`

对应审计文件：

- `data/alignment/reconstructed/dpo_v2/audits/train/medical_pairwise_train_v2.audit.jsonl`
- `data/alignment/reconstructed/dpo_v2/audits/valid/medical_pairwise_valid_v2.audit.jsonl`
- `data/alignment/reconstructed/dpo_v2/audits/test/medical_pairwise_test_v2.audit.jsonl`

最终条数：

- `train = 3800`
- `valid = 100`
- `test = 100`

### 去重结果

`test` 初始重构产物里出现了 `2` 条重复 `sample_id`，已经做了定向去重。

去重报告：

- `data/alignment/reconstructed/dpo_v2/reports/medical_pairwise_test_v2.dedupe.report.json`

关键结果：

- `raw_before = 102`
- `raw_after = 100`
- `raw_duplicates_removed = 2`
- `audit_duplicates_removed = 2`

去重策略使用的是 `keep=last`，因为重复 `sample_id` 对应的是后一次更完整的重构结果。

## 4. train 重构摘要

训练集摘要文件：

- `data/alignment/reconstructed/dpo_v2/reports/medical_pairwise_train_v2.train.summary.json`

核心统计：

- `total_rows = 3800`
- `swap_count = 2198`
- `changed_count = 3800`
- `changed_ratio = 1.0`

重写强度分布：

- `moderate = 2039`
- `heavy = 1732`
- `light = 29`

高频问题标签：

- `poor_communication = 3250`
- `missing_context_awareness = 3237`
- `factual_risk = 2833`
- `too_generic = 2114`
- `overconfidence = 1963`
- `label_direction_wrong = 1910`

目标标签覆盖：

- `axis:accuracy = 3800`
- `axis:communication_quality = 3800`
- `theme:hedging = 3724`
- `axis:context_awareness = 3671`
- `theme:emergency_referrals = 2470`

这组统计很有价值，因为它说明这不是“表面润色”，而是系统性修正了偏好方向。

## 5. 正式 processed 数据

重构后的 `raw` 已经继续转换成训练脚本直接可读的 `processed` 版本。

路径：

- `data/alignment/processed/dpo_v2/train/medical_pairwise_train_v2.processed.jsonl`
- `data/alignment/processed/dpo_v2/valid/medical_pairwise_valid_v2.processed.jsonl`
- `data/alignment/processed/dpo_v2/test/medical_pairwise_test_v2.processed.jsonl`

对应条数：

- `train = 3800`
- `valid = 100`
- `test = 100`

处理报告：

- `data/alignment/processed/dpo_v2/reports/medical_pairwise_train_v2.train.report.json`
- `data/alignment/processed/dpo_v2/reports/medical_pairwise_valid_v2.valid.report.json`
- `data/alignment/processed/dpo_v2/reports/medical_pairwise_test_v2.test.report.json`

处理后格式是标准偏好学习输入：

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "source": "medical_pairwise_train_v2"
}
```

## 6. 用到的脚本

### 数据重构

- [script/alignment/reconstruct_dpo_dataset.py](/home/qjh/llm_learning/my_medical_gpt/script/alignment/reconstruct_dpo_dataset.py)

作用：

- 调用 `OpenAI-compatible API`
- 逐条重构 `chosen / rejected`
- 输出最终 pairwise 数据和 audit 审计记录
- 支持串行、断点续跑、重试、进度日志

### 串行启动脚本

- [script/alignment/run_reconstruct_dpo_v2_serial.sh](/home/qjh/llm_learning/my_medical_gpt/script/alignment/run_reconstruct_dpo_v2_serial.sh)

作用：

- 用串行模式稳定跑 `DPO v2` 数据重构
- 每行日志自动带时间戳
- 输出到 `outputs/logs/dpo_reconstruct/`

### 去重脚本

- [script/alignment/dedupe_reconstructed_pairwise.py](/home/qjh/llm_learning/my_medical_gpt/script/alignment/dedupe_reconstructed_pairwise.py)

作用：

- 按 `sample_id` 对重构产物和 audit 文件同步去重
- 生成去重报告，保证后续训练/评测数据规整

### processed 转换脚本

- [script/alignment/dpo_data_prepare.py](/home/qjh/llm_learning/my_medical_gpt/script/alignment/dpo_data_prepare.py)

作用：

- 把重构后的 `raw` pairwise 转成 `DPO` 训练直接可读格式

## 7. 复现命令

### 先做 train 重构

```bash
OPENAI_BASE_URL="https://big-model.smart-agi.com/v1" \
OPENAI_API_KEY="你的 key" \
SPLIT=train \
OUTPUT_NAME=medical_pairwise_train_v2 \
bash /home/qjh/llm_learning/my_medical_gpt/script/alignment/run_reconstruct_dpo_v2_serial.sh
```

### 对 test 做去重

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/script/alignment/dedupe_reconstructed_pairwise.py \
  --input-file /home/qjh/llm_learning/my_medical_gpt/data/alignment/reconstructed/dpo_v2/raw/test/medical_pairwise_test_v2.jsonl \
  --audit-file /home/qjh/llm_learning/my_medical_gpt/data/alignment/reconstructed/dpo_v2/audits/test/medical_pairwise_test_v2.audit.jsonl \
  --keep last \
  --report-file /home/qjh/llm_learning/my_medical_gpt/data/alignment/reconstructed/dpo_v2/reports/medical_pairwise_test_v2.dedupe.report.json
```

### 转成 processed

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/script/alignment/dpo_data_prepare.py \
  --input-files /home/qjh/llm_learning/my_medical_gpt/data/alignment/reconstructed/dpo_v2/raw/train/medical_pairwise_train_v2.jsonl \
  --split train \
  --output-root /home/qjh/llm_learning/my_medical_gpt/data/alignment/processed/dpo_v2 \
  --output-name medical_pairwise_train_v2
```

## 8. Git 提交策略

当前仓库默认不提交 `data/`，所以：

- 数据处理结果保留在本地
- 代码、脚本、文档推送到 GitHub
- 后续别人要复现时，通过脚本再生成本地数据

这套策略更符合真实工程习惯，也避免把大体量数据直接塞进代码仓库。
