# HealthBench 评测提速与缓存说明

这份文档记录本次对 `evaluation/run_eval.py` 的两项关键增强：

- `batch generation`
- 同一采样点下可复用的共享回答缓存

目标不是单纯“跑快一点”，而是把评测做成一个更适合长期迭代的工程闭环：

- 样本量可以逐步扩大
- 已生成回答可以复用
- 评测耗时可预估
- 结果复现边界清晰

## 这次新增了什么

### 1. Batch generation

新增参数：

- `--generator-batch-size`

作用：

- 让本地 Hugging Face 生成阶段一次处理多条 prompt
- 主要节省的是 GPU 前向生成时间
- 不影响 judge 打分逻辑

### 2. 共享回答缓存

新增参数：

- `--shared-responses-root`
- `--disable-shared-response-cache`

默认情况下，生成出的 `responses.jsonl` 除了保存在本次 run 目录，还会额外写到共享缓存目录。

共享缓存 key 由以下信息共同决定：

- `benchmark`
- `subset`
- `model_alias`
- `sampling_mode`
- `seed`
- `shuffle`
- `enable_thinking`
- `max_new_tokens`
- `temperature`
- `top_p`

刻意没有把下面两个量放进共享缓存 key：

- `max_examples`
- `per_theme_examples`

原因是我们希望同一个采样点从小样本扩成大样本时，前面已经生成过的回答可以直接复用。

对当前 `HealthBench stratified_theme` 的实现来说，这样做是成立的，因为固定 `seed` 时，每个主题桶内部先打乱，再取前 `n` 条；所以 `n=1` 是 `n=2` 的前缀，`n=2` 又是 `n=3` 的前缀。

这正好满足你现在的需求：

- 先跑小样本 smoke
- 后面扩大样本量
- 老样本直接命中缓存
- 只为新增样本付生成成本

## 真实基准测试

测试环境：

- 生成模型：`Qwen3-8B + huatuo_5w checkpoint-75`
- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 每主题样本数：`2`
- 总样本数：`14`
- `seed=314`
- `temperature=0`
- `max_new_tokens=128`
- 单卡：`H200`

说明：

- 下表统计的是 `summary.json -> generation.generation_seconds`
- 这个时间只覆盖“真正的生成阶段”，不包含模型加载时间

| batch size | generation_seconds | 相对 `bs=1` 加速比 |
| --- | ---: | ---: |
| `1` | `75.166s` | `1.00x` |
| `2` | `41.421s` | `1.81x` |
| `4` | `25.527s` | `2.94x` |
| `8` | `12.814s` | `5.87x` |

这说明在你当前的硬件和模型规模下，`batch_size=8` 已经能带来非常明显的收益。

## 复现性验证

### 1. 同一 batch size 是否稳定

我额外复跑了一次 `batch_size=8`：

- `bench_bs8_20260411`
- `bench_bs8_repeat_20260411`

结果：

- `14 / 14` 条回答完全一致

这说明：

- 在相同 batch size
- 相同模型
- 相同采样点
- 相同生成参数

的前提下，当前实现是可稳定复现的。

### 2. 不同 batch size 是否逐字一致

把 `bs=1` 和 `bs=2/4/8` 逐条对比后，发现文本级别并不是完全一致：

- `bs=1 vs bs=2`：`9` 条不一致
- `bs=1 vs bs=4`：`10` 条不一致
- `bs=1 vs bs=8`：`9` 条不一致

这里不是缓存问题，也不是采样点变化，而更像是 batched GPU 推理下的数值路径差异导致 greedy 结果发生了分叉。

这类现象在大模型批量推理里并不罕见，所以工程上最重要的结论不是“不同 batch 一定逐字一致”，而是：

- 正式横向对比时必须固定 `generator_batch_size`
- 不能今天用 `bs=1` 测 base，明天用 `bs=8` 测 LoRA，再把分数直接硬比

换句话说，评测要稳定，关键不是追求“所有 batch 结果完全一样”，而是：

- 比较时口径完全一致
- 同一实验协议可重复

## 共享缓存验证

我专门做了一个两步缓存实验：

### 第一步

配置：

- `seed=314`
- `sampling_mode=stratified_theme`
- `per_theme_examples=1`

结果：

- 生成 `7` 条
- 写入共享缓存 `7` 条

### 第二步

保持同一个采样点，只把：

- `per_theme_examples=1`

扩成：

- `per_theme_examples=2`

结果：

- 从共享缓存恢复 `7` 条
- 新生成 `7` 条

也就是：

- 老样本全部命中缓存
- 只为新增样本付生成成本

这正是我们后面把 `7 x 15` 扩成更大样本时最需要的能力。

## 推荐使用规范

结合这次验证，当前最推荐的做法是：

### 1. 固定正式评测 batch size

建议先统一固定成：

- `--generator-batch-size 8`

原因：

- 当前单卡 H200 上吞吐收益非常明显
- 同 batch size 复跑稳定
- 足够适合后续大多数正式 HealthBench 生成任务

### 2. 正式对比时固定整套生成协议

除了模型本身，下面这些量也必须固定：

- `generator_batch_size`
- `seed`
- `sampling_mode`
- `per_theme_examples`
- `temperature`
- `max_new_tokens`
- `top_p`
- `enable_thinking`

### 3. 扩样时沿用同一个采样点

推荐做法：

- 先跑小样本
- 再逐步扩大 `per_theme_examples`
- 保持同一个 `seed`
- 保持同一套生成参数

这样就能最大化复用共享缓存。

### 4. judge 成本与生成成本分开看

当前 `summary.json` 已经单独记录：

- `generation.generation_seconds`
- `judge.judge_seconds`

后面分析评测耗时时，要把“本地生成”和“外部 judge”拆开看，避免笼统地说“评测很慢”。

## 常用命令

### 正式生成时启用 batch

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_base.json \
  --mode generate_only \
  --judge-mode none \
  --generator-batch-size 8 \
  --run-name 20260411_healthbench_base_theme15x7_generate_bs8
```

### 复用已有回答，只跑 judge

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_base.json \
  --mode judge_only \
  --responses-path /home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260411_healthbench_base_theme15x7_generate_bs8/responses.jsonl \
  --run-name 20260411_healthbench_base_theme15x7_judge
```

### 关闭共享缓存

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py \
  --config /home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_base.json \
  --mode generate_only \
  --judge-mode none \
  --generator-batch-size 8 \
  --disable-shared-response-cache \
  --run-name 20260411_healthbench_base_no_shared_cache
```

## 最终结论

这次改造后，评测部分已经具备下面三个关键能力：

- 能稳定地加速本地生成
- 能把同一采样点的历史回答沉淀为可复用缓存
- 能在扩大样本量时只支付增量生成成本

对你这个项目来说，这比单次跑出一个分数更重要，因为后面所有 `SFT / DPO / GRPO` 的优化决策，都要建立在“评测可持续、可复用、可复现”这个前提上。
