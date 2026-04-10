# 评测接入指南

英文原版见 [EVAL_INTEGRATION.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_INTEGRATION.md)。

这份文档说明后续新的 checkpoint 如何接入当前这套评测框架。

## 当前支持的模型接入方式

现有评测代码支持三种常见模式。

### 1. 只评测基座模型

适用于：

- 原始预训练模型
- 已经 merge 好的完整模型目录

```bash
conda activate medicalgpt
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

MODEL_PATH=/path/to/base_or_merged_model \
MODE=full JUDGE_MODE=openai JUDGE_MODEL=gpt-5.2 \
bash /home/qjh/llm_learning/my_medical_gpt/script/eval/run_eval_healthbench_qwen3_8b_base.sh
```

### 2. 基座模型 + LoRA 适配器

适用于：

- `SFT` LoRA
- `DPO` LoRA
- `GRPO` LoRA

```bash
conda activate medicalgpt
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

MODEL_PATH=/path/to/base_model \
ADAPTER_PATH=/path/to/adapter_or_final_model \
MODE=full JUDGE_MODE=openai JUDGE_MODEL=gpt-5.2 \
bash /home/qjh/llm_learning/my_medical_gpt/script/eval/run_eval_healthbench_qwen3_8b_huatuo_1k_lora.sh
```

### 3. 复用已有 responses，只重跑 judge

适用于：

- 本地生成成本较高
- judge provider 改了
- 想比较不同 judge 设置而不重复生成

```bash
conda activate medicalgpt
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_smoke_base.json \
  --mode judge_only \
  --judge-mode openai \
  --judge-model gpt-5.2 \
  --responses-path /path/to/responses.jsonl \
  --run-name my_judge_only_run
```

## 与后续训练阶段的映射

同一套 harness 可以直接服务于：

- `SFT`：把 `ADAPTER_PATH` 指向 `SFT` LoRA 输出
- `DPO`：把 `ADAPTER_PATH` 指向 `DPO` LoRA 输出
- `GRPO`：把 `ADAPTER_PATH` 指向 `GRPO` LoRA 输出

这意味着 benchmark 层保持不变，变的只是模型 checkpoint。
这正是做干净实验比较时最想要的结构。

## 命名规范建议

建议 run name 至少暴露这些信息：

- 日期
- benchmark 名称
- 模型阶段
- judge 模型
- 样本数

例如：

- `20260409_healthbench_huatuo5w_gpt52_full_50`
- `20260409_healthbench_dpo_v1_gpt52_full_50`
- `20260409_healthbench_grpo_v1_gpt52_full_50`

## 推荐工作流

正式评测时，建议按这个顺序走：

1. `generate_only`
2. 人工抽查几条生成结果
3. `judge_only`
4. 比较不同阶段的 `summary.json`

原因：

- 本地生成和外部 judge 的故障模式不同
- 两段式流程更容易复现和重跑
- judge 出问题时不必重新生成，节省很多成本

## Judge 设置建议

当前建议：

- 模型：`gpt-5.2`
- `temperature=0`
- 严格 JSON 输出
- 不同模型对比时保持同一 benchmark 子集和样本数

除非你明确要做 judge ablation，否则不要轻易改这套设置。
