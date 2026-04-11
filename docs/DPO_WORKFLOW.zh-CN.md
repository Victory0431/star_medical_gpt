# DPO 工作流说明

英文原版见 [DPO_WORKFLOW.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_WORKFLOW.md)。

这份文档专门说明当前仓库里的 `DPO` 训练链路，包括为什么先 merge `SFT checkpoint-75`，pairwise 数据如何组织，训练期看哪些指标，以及为什么额外引入 `valid_zh` 作为异构分布辅助评估。

如果你想看 `chosen / rejected / margin / accuracy / loss` 的底层公式和源码级解释，可以直接看 [DPO_METRICS.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_METRICS.zh-CN.md)。

## 1. 当前链路

推荐链路：

1. `Base model`
2. `SFT LoRA`
3. 把最佳 `SFT checkpoint` merge 回基座
4. 用 merge 后的 `SFT model` 作为 `DPO` 初始化策略
5. `DPO` 再训练一层新的 `LoRA`

当前已经完成的 merge 产物路径：

- [outputs/merged_models/sft/20260410_qwen3-8b_huatuo-5w_ckpt75_merged/model](/home/qjh/llm_learning/my_medical_gpt/outputs/merged_models/sft/20260410_qwen3-8b_huatuo-5w_ckpt75_merged/model)

这样做的好处是：

- 阶段边界清晰：`Base -> SFT -> DPO`
- DPO 的 reference 很自然：关闭 DPO adapter 后就是 merge 后的 `SFT policy`
- 面试时容易解释，不会陷入“LoRA 套 LoRA 套 LoRA”的混乱描述

## 2. DPO 数据

当前建议优先使用 `DPO v2` 重构版本，而不是直接使用原始 pairwise 数据。

原始数据：

- `medical_pairwise_train.jsonl`：`3800`
- `medical_pairwise_valid.jsonl`：`100`
- `medical_pairwise_test.jsonl`：`100`

原始字段：

```json
{
  "question": "...",
  "response_chosen": "...",
  "response_rejected": "..."
}
```

处理后字段：

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "source": "medical_pairwise_train"
}
```

`DPO v1` 处理后路径：

- `data/alignment/processed/dpo/train/medical_pairwise_train.processed.jsonl`
- `data/alignment/processed/dpo/valid/medical_pairwise_valid.processed.jsonl`
- `data/alignment/processed/dpo/test/medical_pairwise_test.processed.jsonl`

`DPO v2` 重构后再处理的路径：

- `data/alignment/processed/dpo_v2/train/medical_pairwise_train_v2.processed.jsonl`
- `data/alignment/processed/dpo_v2/valid/medical_pairwise_valid_v2.processed.jsonl`
- `data/alignment/processed/dpo_v2/test/medical_pairwise_test_v2.processed.jsonl`

处理脚本：

- [script/alignment/dpo_data_prepare.py](/home/qjh/llm_learning/my_medical_gpt/script/alignment/dpo_data_prepare.py)
- [script/alignment/reconstruct_dpo_dataset.py](/home/qjh/llm_learning/my_medical_gpt/script/alignment/reconstruct_dpo_dataset.py)
- [script/alignment/dedupe_reconstructed_pairwise.py](/home/qjh/llm_learning/my_medical_gpt/script/alignment/dedupe_reconstructed_pairwise.py)

如果你想看这次 `v2` 数据重构的详细统计、去重结果和正式产物，可以直接看 [DPO_V2_RECONSTRUCTION.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_RECONSTRUCTION.zh-CN.md)。

如果你想看 `DPO v2` 这次正式训练跑完后的指标变化、best checkpoint 选择问题和训练复盘，可以直接看 [DPO_V2_TRAINING_REPORT.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_TRAINING_REPORT.zh-CN.md)。

## 3. 训练期看什么指标

### 主评估：pairwise valid

这是 `DPO` 的主评估集，直接对应偏好优化目标。

重点看：

- `eval_rewards/accuracies`
  代表模型对 chosen/rejected 的相对偏好排序是否正确，当前默认也用它做最佳 checkpoint 选择。
- `eval_rewards/margins`
  chosen 和 rejected 之间的 reward 差距，越大说明偏好区分更明显。
- `eval_loss`
  训练损失，可以辅助看是否出现明显异常。

### 辅助评估：`valid_zh`

这是一个和 pairwise 分布不同的 `SFT` 风格验证集，不参与 `DPO` 最佳 checkpoint 选择，只作为辅助观察。

它的作用是看：

- 做偏好优化以后，模型在普通医疗问答上的语言建模质量是否明显恶化
- 是否出现“pairwise 指标上升，但开放式问答稳定性下降”的现象

这里记录的是：

- `aux_eval/valid_zh_loss`

这个值越低通常越好，但它不是 DPO 主目标，所以不能替代 pairwise 指标。

## 4. 为什么这样更有说服力

如果只看 pairwise valid：

- 你能证明模型更偏好 chosen 胜过 rejected
- 但不容易说明开放式医疗问答是否保持稳定

如果只看 `valid_zh`：

- 你只能证明语言建模没有明显坏掉
- 但不能证明偏好优化真的成功

把两者一起看，会更像工业里的双视角：

- 一个看目标任务是否被优化到
- 一个看泛化能力有没有被破坏

## 5. 标准命令

### 先准备 DPO v1 数据

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/alignment/dpo_data_prepare.py \
  --input-files /home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/medical_pairwise_train.jsonl \
  --split train \
  --output-name medical_pairwise_train
```

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/alignment/dpo_data_prepare.py \
  --input-files /home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/medical_pairwise_valid.jsonl \
  --split valid \
  --output-name medical_pairwise_valid
```

### 准备当前推荐的 DPO v2 数据

先串行重构：

```bash
OPENAI_BASE_URL="https://big-model.smart-agi.com/v1" \
OPENAI_API_KEY="你的 key" \
SPLIT=train \
OUTPUT_NAME=medical_pairwise_train_v2 \
bash /home/qjh/llm_learning/my_medical_gpt/script/alignment/run_reconstruct_dpo_v2_serial.sh
```

再转成训练可读格式：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/alignment/dpo_data_prepare.py \
  --input-files /home/qjh/llm_learning/my_medical_gpt/data/alignment/reconstructed/dpo_v2/raw/train/medical_pairwise_train_v2.jsonl \
  --split train \
  --output-root /home/qjh/llm_learning/my_medical_gpt/data/alignment/processed/dpo_v2 \
  --output-name medical_pairwise_train_v2
```

### 正式双卡启动

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/alignment/run_dpo_qwen3_8b_ckpt75_medical_pairwise.sh
```

### 当前推荐的 DPO v2 双卡启动

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/alignment/run_dpo_qwen3_8b_ckpt75_medical_pairwise_v2.sh
```

### 直接双卡命令

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/home/qjh/miniconda3/envs/medicalgpt/bin/torchrun \
  --nproc_per_node 2 \
  --master_port 29531 \
  /home/qjh/llm_learning/my_medical_gpt/script/alignment/train_dpo.py \
  --model-name-or-path /home/qjh/llm_learning/my_medical_gpt/outputs/merged_models/sft/20260410_qwen3-8b_huatuo-5w_ckpt75_merged/model \
  --train-data /home/qjh/llm_learning/my_medical_gpt/data/alignment/processed/dpo/train/medical_pairwise_train.processed.jsonl \
  --valid-data /home/qjh/llm_learning/my_medical_gpt/data/alignment/processed/dpo/valid/medical_pairwise_valid.processed.jsonl \
  --aux-valid-data /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/valid/valid_zh_500.processed.jsonl \
  --output-root /home/qjh/llm_learning/my_medical_gpt/outputs/dpo \
  --run-name 20260410_qwen3-8b_ckpt75_medical_pairwise_dpo \
  --wandb-project my-medical-gpt-dpo \
  --wandb-mode online \
  --max-prompt-length 1536 \
  --max-completion-length 512 \
  --max-length 2048 \
  --model-max-length 2048 \
  --num-proc 16 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --aux-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 3 \
  --learning-rate 5e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --beta 0.1 \
  --logging-steps 5 \
  --eval-strategy steps \
  --eval-steps 10 \
  --save-strategy steps \
  --save-steps 10 \
  --save-total-limit 20 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules all-linear \
  --metric-for-best-model eval_rewards/accuracies \
  --greater-is-better \
  --bf16 \
  --gradient-checkpointing
```

## 6. 产物与日志

单次 `DPO` run 会写出：

- `outputs/dpo/<run_name>/checkpoints/`
- `outputs/dpo/<run_name>/final_model/`
- `outputs/dpo/<run_name>/logs/train.log`
- `outputs/dpo/<run_name>/logs/metrics.jsonl`
- `outputs/dpo/<run_name>/logs/aux_eval.jsonl`
- `outputs/dpo/<run_name>/artifacts/best_checkpoint.json`
- `outputs/dpo/<run_name>/artifacts/training_summary.json`

中央时间戳日志额外写到：

- `outputs/logs/dpo/<timestamp>_<run_name>.log`

## 7. 面试可直接说的话

你可以这样讲：

- “我的 DPO 不是直接从 base model 起训，而是先把最佳 SFT checkpoint merge 成稳定的 SFT policy，再基于它训练新的 DPO adapter。”
- “训练期我同时看两类评估：主评估是 pairwise valid 上的 reward accuracy，用来选最佳 checkpoint；辅助评估是异构分布的 `valid_zh` assistant-only loss，用来监控 DPO 是否损伤通用医疗问答能力。”
- “这样做的好处是，既能证明偏好优化有效，也能证明没有把开放式问答质量训坏。” 
