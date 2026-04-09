# 脚本使用指南

英文原版见 [SCRIPT_GUIDE.md](/home/qjh/llm_learning/my_medical_gpt/docs/SCRIPT_GUIDE.md)。

这份文档说明仓库内核心脚本的用途、命令、参数和输出。训练脚本与 benchmark 评测脚本统一放在一份文档里，方便日常使用。

## 1. `sft_data_prepare.py`

路径：

- [script/sft_data_prepare.py](/home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py)

### 作用

- 读取原始 `SFT` 数据
- 支持 `sharegpt` 和 `instruction/input/output` 两类输入
- 统一转换成 `{"conversations": [...]}` JSONL
- 可选做严格重复样本去重
- 生成处理报告，方便后续审计

### 支持的输入格式

`sharegpt`

```json
{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
```

`instruction`

```json
{"instruction":"...","input":"...","output":"..."}
```

### 默认输出目录

```text
data/sft/processed/
  train/
  valid/
  test/
  reports/
```

### 基础命令

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /path/to/raw.jsonl \
  --split train \
  --input-format auto
```

### 常见示例

处理 `1k` smoke 数据：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /home/qjh/llm_learning/medicalgpt/MedicalGPT-main/data/finetune/finetune/medical_sft_1K_format.jsonl \
  --split train \
  --input-format sharegpt \
  --output-name huatuo_1k
```

处理 `5w` ShareGPT 数据：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /home/qjh/llm_learning/medicalgpt/MedicalGPT-main/data/finetune/finetune/HuatuoGPT2_sft_instruct_GPT4_sharegpt.jsonl \
  --split train \
  --input-format sharegpt \
  --workers 32 \
  --output-name huatuo_5w
```

把 instruction 格式转成 conversations：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /path/to/train_zh_0.json \
  --split train \
  --input-format instruction \
  --workers 32 \
  --output-name train_zh_195w
```

处理验证集：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/sft_data_prepare.py \
  --input-files /path/to/valid.json \
  --split valid \
  --input-format instruction \
  --output-name valid_zh_500
```

### 参数说明

| 参数 | 含义 | 常见值 |
| --- | --- | --- |
| `--input-files` | 一个或多个原始 `.json/.jsonl` 文件 | 必填 |
| `--split` | 输出到哪个 split 目录 | `train` `valid` `test` |
| `--input-format` | 输入解析方式 | `auto` `sharegpt` `instruction` |
| `--output-root` | 处理后数据根目录 | `data/sft/processed` |
| `--workers` | 预处理线程/进程数 | `16` `32` |
| `--no-deduplicate` | 关闭严格重复去重 | 可选 |
| `--output-name` | 自定义输出文件名前缀 | `huatuo_5w` |

### 输出文件

单个处理后数据集通常会生成：

- `data/sft/processed/train/xxx.processed.jsonl`
- `data/sft/processed/reports/xxx.processed.report.json`
- `data/sft/processed/reports/summary.json`

### 说明

- 原始数据来源不稳定时，推荐 `--input-format auto`
- 输出格式已经直接兼容 `train_sft.py`
- 对于大体量 JSONL，这个脚本按行流式处理，更适合大数据

## 2. `train_sft.py`

路径：

- [script/train_sft.py](/home/qjh/llm_learning/my_medical_gpt/script/train_sft.py)

### 作用

- 读取处理好的 `conversations` 数据
- 应用 Qwen chat template
- 默认只对 assistant token 计算标准 `SFT loss`
- 执行 `LoRA` 微调
- 周期性做 eval 与 checkpoint 保存
- 同时写本地日志与 `W&B`

### 标准命令

单进程直接启动：

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/script/train_sft.py \
  --model-name-or-path /home/qjh/llm_learning/base_model/qwen3_8B \
  --train-data /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/train/huatuo_1k.processed.jsonl \
  --valid-data /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/valid/valid_zh_500.processed.jsonl \
  --output-root /home/qjh/llm_learning/my_medical_gpt/outputs/sft \
  --run-name 20260409_demo_run \
  --wandb-project my-medical-gpt-sft \
  --wandb-mode online \
  --model-max-length 2048 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --eval-strategy steps \
  --eval-steps 25 \
  --save-strategy steps \
  --save-steps 25 \
  --bf16 \
  --gradient-checkpointing \
  --flash-attn
```

双卡正式启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
/home/qjh/miniconda3/envs/medicalgpt/bin/torchrun \
  --nproc_per_node 2 \
  --master_port 29521 \
  /home/qjh/llm_learning/my_medical_gpt/script/train_sft.py \
  --model-name-or-path /home/qjh/llm_learning/base_model/qwen3_8B \
  --train-data /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/train/huatuo_5w.processed.jsonl \
  --valid-data /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/valid/valid_zh_500.processed.jsonl \
  --output-root /home/qjh/llm_learning/my_medical_gpt/outputs/sft \
  --run-name qwen3_8b_huatuo_5w_manual \
  --cache-dir /home/qjh/llm_learning/my_medical_gpt/cache \
  --wandb-project my-medical-gpt-sft \
  --wandb-mode online \
  --model-max-length 2048 \
  --num-proc 16 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --logging-steps 5 \
  --eval-strategy steps \
  --eval-steps 25 \
  --save-strategy steps \
  --save-steps 25 \
  --save-total-limit 3 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules all-linear \
  --bf16 \
  --gradient-checkpointing \
  --flash-attn
```

### 什么时候直接调它

日常推荐用 launcher 脚本；以下场景再直接调 `train_sft.py`：

- 想快速切数据集
- 想自定义 run name
- 想改 batch size、eval 频率、精度、LoRA 参数
- 想先 dry run 验证 tokenization 和数据加载
- 想 debug 完再固化到 launcher

### 参数说明

#### 数据与路径

| 参数 | 含义 | 常见值 |
| --- | --- | --- |
| `--model-name-or-path` | 基座模型目录 | `/home/qjh/llm_learning/base_model/qwen3_8B` |
| `--train-data` | 一个或多个训练文件/目录 | `.../huatuo_5w.processed.jsonl` |
| `--valid-data` | 一个或多个验证文件/目录 | `.../valid_zh_500.processed.jsonl` |
| `--validation-split-ratio` | 未提供验证集时的自动切分比例 | `0.05` |
| `--output-root` | 训练输出根目录 | `outputs/sft` |
| `--run-name` | 当前实验名 | 时间戳风格 |
| `--cache-dir` | Hugging Face / datasets cache | `cache` |

#### 追踪与复现

| 参数 | 含义 | 常见值 |
| --- | --- | --- |
| `--wandb-project` | W&B 项目名 | `my-medical-gpt-sft` |
| `--wandb-mode` | W&B 模式 | `online` `offline` `disabled` |
| `--seed` | 随机种子 | `42` |

#### 训练计划

| 参数 | 含义 | 常见值 |
| --- | --- | --- |
| `--model-max-length` | 最大序列长度 | `2048` |
| `--max-train-samples` | 训练样本上限 | `-1` |
| `--max-eval-samples` | 验证样本上限 | `-1` |
| `--max-steps` | 最大优化步数，`-1` 表示按 epoch | `10` `-1` |
| `--num-proc` | tokenization 处理进程数 | `16` |
| `--per-device-train-batch-size` | 每卡训练 micro-batch | `4` |
| `--per-device-eval-batch-size` | 每卡验证 batch | `4` |
| `--gradient-accumulation-steps` | 梯度累积步数 | `8` |
| `--num-train-epochs` | 训练 epoch 数 | `1` `2` |
| `--learning-rate` | 学习率 | `2e-5` |
| `--weight-decay` | 权重衰减 | `0.01` |
| `--warmup-ratio` | warmup 比例 | `0.03` |

#### 日志、评估、checkpoint

| 参数 | 含义 | 常见值 |
| --- | --- | --- |
| `--logging-steps` | 本地/W&B 记录频率 | `5` `10` |
| `--eval-strategy` | eval 频率策略 | `steps` `epoch` `no` |
| `--eval-steps` | `steps` 模式下的 eval 间隔 | `25` `50` |
| `--save-strategy` | checkpoint 保存策略 | `steps` `epoch` `no` |
| `--save-steps` | `steps` 模式下的保存间隔 | `25` `50` |
| `--save-total-limit` | 最多保留多少个 checkpoint | `3` |

#### LoRA 与显存

| 参数 | 含义 | 常见值 |
| --- | --- | --- |
| `--lora-r` | LoRA rank | `16` |
| `--lora-alpha` | LoRA alpha | `32` |
| `--lora-dropout` | LoRA dropout | `0.05` |
| `--target-modules` | 目标模块选择 | `all-linear` |
| `--gradient-checkpointing` | 开启激活检查点 | 常开 |
| `--bf16` | 使用 bfloat16 | 常开 |
| `--fp16` | 使用 fp16 | 可选 |
| `--use-cpu` | 强制 CPU | 只用于 debug |
| `--load-in-4bit` | QLoRA 风格 4bit 加载 | 可选 |
| `--flash-attn` | 优先启用 FlashAttention2 | 常开 |

#### 行为开关

| 参数 | 含义 | 常见值 |
| --- | --- | --- |
| `--train-on-inputs` | 对整段序列算 loss，而不是只对 assistant token | 一般关闭 |
| `--dry-run` | 只做 tokenization / 统计，不真正训练 | 很适合冒烟检查 |

### 重要行为说明

- 默认是标准 `SFT`：只对 assistant token 算 loss。
- 如果没提供 `--valid-data`，脚本可通过 `--validation-split-ratio` 自动切验证集。
- 如果没有 `flash_attn`，脚本会自动回退到标准 attention。
- `--disable-assistant-only-loss` 目前保留为兼容参数，不是推荐工作流的一部分。

### 输出目录

每个 run 会在 `outputs/sft/<run_name>/` 下生成：

```text
artifacts/
  run_args.json
  dataset_stats.json
  training_args.json
logs/
  train.log
  console.log
  metrics.jsonl
checkpoints/
final_model/
wandb/
```

## 3. `evaluation/run_eval.py`

路径：

- [evaluation/run_eval.py](/home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py)

### 作用

- 加载 `HealthBench`
- 用本地 base 模型或 base + LoRA 做生成
- 按需调用官方/兼容 judge 打分
- 保存可复用的 `responses`、`judgments`、summary 和日志

### 标准命令

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --benchmark healthbench \
  --subset-name consensus \
  --mode full \
  --judge-mode openai \
  --judge-model gpt-5.2 \
  --model-name-or-path /home/qjh/llm_learning/base_model/qwen3_8B \
  --adapter-path /home/qjh/llm_learning/my_medical_gpt/outputs/sft/20260408_222930_qwen3-8b_medical-sft-1k_lora_clean/final_model \
  --run-name 20260409_healthbench_huatuo1k \
  --max-examples 10 \
  --generator-device cuda:0
```

### 常见示例

基座模型 `generate_only` 冒烟：

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_smoke_base.json \
  --mode generate_only \
  --judge-mode none \
  --run-name 20260409_healthbench_base_generate
```

LoRA 模型正式打分：

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_smoke_huatuo_1k_lora.json \
  --run-name 20260409_healthbench_huatuo1k_full
```

按主题分层抽样的正式对比：

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_base.json \
  --run-name 20260409_healthbench_base_gpt52_full_theme15x7
```

复用已有 responses 做 `judge_only`：

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_smoke_base.json \
  --mode judge_only \
  --responses-path /home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260409_healthbench_base_generate/responses.jsonl \
  --run-name 20260409_healthbench_base_judge
```

### 参数说明

| 参数 | 含义 | 常见值 |
| --- | --- | --- |
| `--config` | 可选 JSON 配置文件 | `evaluation/configs/*.json` |
| `--subset-name` | HealthBench 子集 | `consensus` `hard` `full` |
| `--mode` | 运行模式 | `full` `generate_only` `judge_only` |
| `--judge-mode` | judge 后端 | `openai` `none` |
| `--judge-model` | judge 模型名 | `gpt-5.2` |
| `--model-name-or-path` | 基座模型路径 | `/home/qjh/llm_learning/base_model/qwen3_8B` |
| `--adapter-path` | 可选 LoRA 路径 | `.../final_model` |
| `--max-examples` | 样本上限，适合做 smoke test | `1` `10` `50` |
| `--sampling-mode` | 采样方式 | `sequential` `stratified_theme` |
| `--per-theme-examples` | 分层抽样时每个主题取多少条 | `15` |
| `--generator-device` | 本地生成设备 | `cuda:0` |
| `--enable-thinking` | 是否启用 Qwen3 thinking mode | 一般关闭 |
| `--responses-path` | 复用已有 response 文件 | 可选 |
| `--judgments-path` | 复用已有 judgment 文件 | 可选 |
| `--overwrite-responses` | 强制重生 responses | 可选 |
| `--overwrite-judgments` | 强制重跑 judge | 可选 |

### 输出文件

- `outputs/eval/<run_name>/artifacts/run_args.json`
- `outputs/eval/<run_name>/artifacts/dataset_manifest.json`
- `outputs/eval/<run_name>/logs/eval.log`
- `outputs/eval/<run_name>/responses.jsonl`
- `outputs/eval/<run_name>/judgments.jsonl`
- `outputs/eval/<run_name>/summary.json`
- `outputs/eval/<run_name>/summary.md`
- `evaluation/logs/<timestamp>_<run_name>.log`

### 正式评测配置

当前仓库已经内置：

- [evaluation/configs/healthbench_smoke_base.json](/home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_smoke_base.json)
- [evaluation/configs/healthbench_smoke_huatuo_1k_lora.json](/home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_smoke_huatuo_1k_lora.json)
- [evaluation/configs/healthbench_theme15_base.json](/home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_base.json)
- [evaluation/configs/healthbench_theme15_huatuo_5w_ckpt75.json](/home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_huatuo_5w_ckpt75.json)
- [evaluation/configs/healthbench_theme15_huatuo_5w_ckpt925.json](/home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_huatuo_5w_ckpt925.json)

### 配套 launcher

- [run_eval_healthbench_qwen3_8b_base.sh](/home/qjh/llm_learning/my_medical_gpt/script/run_eval_healthbench_qwen3_8b_base.sh)
- [run_eval_healthbench_qwen3_8b_huatuo_1k_lora.sh](/home/qjh/llm_learning/my_medical_gpt/script/run_eval_healthbench_qwen3_8b_huatuo_1k_lora.sh)

launcher 默认行为：

- 默认 `mode=generate_only`
- 默认 `judge_mode=none`
- 要跑正式打分时，记得显式设置 `MODE=full JUDGE_MODE=openai`

## 4. `run_sft_qwen3_8b_medical_1k.sh`

路径：

- [script/run_sft_qwen3_8b_medical_1k.sh](/home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh)

### 作用

- 用 `torchrun` 包装双卡训练
- 预设模型路径、训练集、验证集、输出目录、`W&B`
- 自动加时间戳日志
- 适合作为第一版全链路 smoke test

### 默认训练目标

- train：`huatuo_1k.processed.jsonl`
- valid：`valid_zh_500.processed.jsonl`
- GPU：`CUDA_VISIBLE_DEVICES=0,1`
- 进程数：`NPROC_PER_NODE=2`

### 标准命令

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh
```

### `nohup` 版本

```bash
nohup bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh \
  > /home/qjh/llm_learning/my_medical_gpt/outputs/nohup_1k.out 2>&1 &
```

### 常见覆盖写法

```bash
RUN_NAME=demo_1k_lr1e5 \
LEARNING_RATE=1e-5 \
NUM_TRAIN_EPOCHS=1 \
CUDA_VISIBLE_DEVICES=0,1 \
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_medical_1k.sh
```

### 常见环境变量

| 变量 | 含义 | 默认值 |
| --- | --- | --- |
| `MODEL_PATH` | 基座模型路径 | `.../base_model/qwen3_8B` |
| `TRAIN_DATA` | 训练数据 | `huatuo_1k.processed.jsonl` |
| `VALID_DATA` | 验证数据 | `valid_zh_500.processed.jsonl` |
| `RUN_NAME` | 实验名 | 时间戳 |
| `OUTPUT_ROOT` | 输出根目录 | `outputs/sft` |
| `CUDA_VISIBLE_DEVICES` | 使用哪些卡 | `0,1` |
| `NPROC_PER_NODE` | `torchrun` 进程数 | `2` |
| `MASTER_PORT` | 分布式端口 | `29521` |
| `NUM_TRAIN_EPOCHS` | epoch 数 | `2` |
| `MAX_STEPS` | 最大步数 | `-1` |
| `EVAL_INTERVAL` | eval 间隔 | `25` |
| `SAVE_INTERVAL` | 保存间隔 | `25` |
| `WANDB_PROJECT` | W&B 项目名 | `my-medical-gpt-sft` |
| `WANDB_MODE` | W&B 模式 | `online` |

## 5. `run_sft_qwen3_8b_huatuo_5w.sh`

路径：

- [script/run_sft_qwen3_8b_huatuo_5w.sh](/home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh)

### 作用

- 复用 `1k` launcher
- 只覆盖 `TRAIN_DATA`、`VALID_DATA`、`RUN_NAME`
- 作为首个正式小版本训练入口

### 标准命令

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh
```

### `nohup` 版本

```bash
nohup bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh \
  > /home/qjh/llm_learning/my_medical_gpt/outputs/nohup_5w.out 2>&1 &
```

### 覆盖示例

```bash
RUN_NAME=huatuo_5w_epoch1_eval20 \
NUM_TRAIN_EPOCHS=1 \
EVAL_INTERVAL=20 \
SAVE_INTERVAL=20 \
bash /home/qjh/llm_learning/my_medical_gpt/script/run_sft_qwen3_8b_huatuo_5w.sh
```

## 6. `export_experiment_records.py`

路径：

- [script/export_experiment_records.py](/home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py)

### 作用

- 从 `outputs/sft/` 里复制轻量可复现产物到 Git 跟踪的 `experiment_records/sft/`
- 保持仓库轻量
- 方便在 GitHub 上回看实验历史
- `--all` 模式下默认跳过 `dryrun` 和明显失败的 run

### 标准命令

导出单个 run：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py \
  --run-name 20260409_1204_qwen3-8b_huatuo-1k_eval_smoke \
  --force
```

导出全部合格 run：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py --all --force
```

强制导出所有内容：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py \
  --all \
  --include-dryrun \
  --include-failed \
  --force
```

### 参数说明

| 参数 | 含义 | 常见值 |
| --- | --- | --- |
| `--outputs-root` | 原始 run 根目录 | `outputs/sft` |
| `--records-root` | 导出目标根目录 | `experiment_records/sft` |
| `--run-name` | 指定导出某个 run，可重复传入 | run 目录名 |
| `--all` | 导出 `outputs-root` 下全部合格 run | 常用 |
| `--include-dryrun` | 同时导出 `dryrun` | 较少用 |
| `--include-failed` | 同时导出明显失败的 run | 较少用 |
| `--force` | 覆盖已有导出结果 | 常用 |
| `--max-log-mb` | 超过这个大小的日志会截断 | `2.0` |

### 会导出的内容

- `artifacts/run_args.json`
- `artifacts/training_args.json`
- `artifacts/dataset_stats.json`
- `logs/metrics.jsonl`
- `logs/train.log`
- `logs/console.log`
- `checkpoints/train_results.json`
- `checkpoints/eval_results.json`
- `checkpoints/all_results.json`
- `summary.json`

### 不会导出的内容

- `adapter_model.safetensors`
- `optimizer.pt`
- tokenizer dump
- 本地 `W&B` 二进制产物

## 日常推荐流程

1. 用 `sft_data_prepare.py` 准备或更新处理后数据集。
2. 每次改代码或改数据后，先跑 `run_sft_qwen3_8b_medical_1k.sh` 做 smoke test。
3. 再跑 `run_sft_qwen3_8b_huatuo_5w.sh` 做首个正式训练版本。
4. 跑完以后执行 `export_experiment_records.py --all --force`。
5. 当 run 值得保留时，把代码和导出的实验记录一起提交。
