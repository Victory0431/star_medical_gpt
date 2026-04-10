# Star Medical GPT 中文文档

英文原版见 [README.md](/home/qjh/llm_learning/my_medical_gpt/README.md)。

这是一个面向简历项目与工业化表达的医疗大模型微调仓库，围绕 `Qwen3-8B`、中文医疗 `SFT` 数据、`LoRA` 微调、训练中评测、`W&B` 追踪，以及轻量级实验留档构建。

## 项目重点

- 基座模型：`Qwen3-8B`
- 微调方式：`LoRA`
- 损失设计：默认采用标准 `SFT`，只对 assistant token 计算 loss
- 运行方式：单机双卡 `torchrun`
- 实验追踪：`Weights & Biases`
- 评测方式：训练期 `eval_loss` + 训练后开放式基准评测
- 可追溯性：轻量实验快照导出到 `experiment_records/`

## 仓库结构

```text
evaluation/
  benchmarks/
  generators/
  judges/
  configs/
script/
  sft/
    sft_data_prepare.py
    train_sft.py
    run_sft_qwen3_8b_medical_1k.sh
    run_sft_qwen3_8b_huatuo_5w.sh
  alignment/
    merge_lora.py
    dpo_data_prepare.py
    train_dpo.py
    run_dpo_qwen3_8b_ckpt75_medical_pairwise.sh
  eval/
    run_eval_healthbench_qwen3_8b_base.sh
    run_eval_healthbench_qwen3_8b_huatuo_1k_lora.sh
  ops/
    export_experiment_records.py
  grpo/
data/
  sft/
    raw/
    processed/
outputs/
  sft/
experiment_records/
```

## 中文文档入口

- [脚本使用指南](/home/qjh/llm_learning/my_medical_gpt/docs/SCRIPT_GUIDE.zh-CN.md)
  介绍核心脚本的用途、命令、参数、输出与常见用法。
- [脚本目录说明](/home/qjh/llm_learning/my_medical_gpt/script/README.zh-CN.md)
  说明 `script/sft`、`script/alignment`、`script/eval`、`script/ops`、`script/grpo` 的职责边界。
- [项目工作流](/home/qjh/llm_learning/my_medical_gpt/docs/WORKFLOW.zh-CN.md)
  从原始数据、数据处理、Smoke Test、正式训练到实验归档的推荐流程。
- [评测设计](/home/qjh/llm_learning/my_medical_gpt/docs/EVALUATION.zh-CN.md)
  说明为什么主评测选择 `HealthBench`，以及它如何服务于 `SFT / DPO / GRPO`。
- [评测结果解读](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_RESULTS.zh-CN.md)
  总结当前 `base vs huatuo_1k` 的 smoke 对比结果，以及如何面试里解释。
- [评测接入指南](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_INTEGRATION.zh-CN.md)
  说明后续新的 `SFT / DPO / GRPO` checkpoint 如何复用同一套评测框架。
- [评测模块说明](/home/qjh/llm_learning/my_medical_gpt/evaluation/README.zh-CN.md)
  介绍评测模式、输出目录、恢复行为和模块分层。
- [DPO 工作流说明](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_WORKFLOW.zh-CN.md)
  说明 merge 后的 `SFT checkpoint` 如何接到 `DPO`，以及为什么同时看 pairwise 和 `valid_zh` 两类训练期评估。
- [实验记录说明](/home/qjh/llm_learning/my_medical_gpt/experiment_records/README.zh-CN.md)
  说明哪些内容会被导出到可提交 Git 的实验快照中。

## 快速开始

先激活环境：

```bash
conda activate medicalgpt
```

准备数据集：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft/sft_data_prepare.py \
  --input-files /path/to/raw.jsonl \
  --split train \
  --input-format auto
```

运行 `1k` 的 smoke test：

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/sft/run_sft_qwen3_8b_medical_1k.sh
```

运行 `5w` 的首个正式小版本：

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/sft/run_sft_qwen3_8b_huatuo_5w.sh
```

把最佳 `SFT checkpoint` merge 成后续对齐的稳定起点：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/alignment/merge_lora.py \
  --base-model-path /home/qjh/llm_learning/base_model/qwen3_8B \
  --adapter-path /home/qjh/llm_learning/my_medical_gpt/outputs/sft/20260409_121822_qwen3-8b_huatuo-5w_lora_eval/checkpoints/checkpoint-75 \
  --output-root /home/qjh/llm_learning/my_medical_gpt/outputs/merged_models/sft \
  --run-name 20260410_qwen3-8b_huatuo-5w_ckpt75_merged \
  --log-root /home/qjh/llm_learning/my_medical_gpt/outputs/logs/merge \
  --device cuda \
  --dtype bfloat16
```

准备 `DPO` pairwise 数据：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/alignment/dpo_data_prepare.py \
  --input-files /home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/medical_pairwise_train.jsonl \
  --split train \
  --output-name medical_pairwise_train
```

启动 `DPO`：

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/alignment/run_dpo_qwen3_8b_ckpt75_medical_pairwise.sh
```

导出轻量实验记录：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/ops/export_experiment_records.py --all --force
```

先跑一版不带官方打分的 HealthBench 基座评测：

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/eval/run_eval_healthbench_qwen3_8b_base.sh
```

如果已经配置好 judge API，可以跑官方 rubric 打分：

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
MODE=full JUDGE_MODE=openai \
bash /home/qjh/llm_learning/my_medical_gpt/script/eval/run_eval_healthbench_qwen3_8b_base.sh
```

## 数据与提交策略

仓库默认不提交以下大文件：

- 原始数据集
- 处理后的大规模数据集
- 模型权重
- LoRA 权重
- optimizer state
- tokenizer dump
- 本地 `W&B` 二进制产物

仓库会保留适合 GitHub 的轻量可复现实验信息：

- run 参数
- 数据统计
- metrics 历史
- 最终 train/eval 结果
- 规整后的日志
- benchmark summary

## 当前使用建议

- 没有安装 `flash_attn` 时，训练脚本会自动回退到标准 attention。
- 验证集既可以来自外部验证集，也可以来自训练集切分；这两种方案都值得做成对比实验。
- `export_experiment_records.py --all` 默认会跳过 `dryrun` 和明显失败的 run。
- 当前仓库已经按“后续可扩展到更大规模医疗 `SFT`、评测与对齐实验”的方式组织。
- 评测 judge 支持 OpenAI 兼容接口，默认 judge 模型为 `gpt-5.2`。
- `DPO` 训练默认同时记录两类训练期指标：pairwise `reward accuracy` 用于挑选最佳 checkpoint，`valid_zh` 的辅助 `LM loss` 用于监控异构分布下的稳定性。
