# 评测模块说明

英文原版见 [README.md](/home/qjh/llm_learning/my_medical_gpt/evaluation/README.md)。

这个目录存放的是 Star Medical GPT 的基准评测代码，目标是把“模型生成”和“外部打分”解耦，方便后续稳定比较不同训练阶段的 checkpoint。

## 设计目标

- 把本地生成和打分彻底拆开
- 基座模型与 LoRA 适配器共用同一个入口
- 后续 `SFT / DPO / GRPO` 都能复用同一套评测框架
- 输出结构清晰，方便多次实验横向比较

## 当前主评测基准

- `HealthBench`

相比 `C-EVAL` 这种医学选择题，`HealthBench` 更适合看：

- 开放式医学回答质量
- 医患沟通风格
- 是否会主动补问关键上下文
- 指令遵循能力
- 不确定场景下的 hedging 与安全性

这也是它更适合后续 `DPO` 和 `GRPO` 的原因，因为这两类对齐方法提升的往往不是选择题正确率，而是开放式回答质量。

当前 `HealthBench` 主题展示统一对齐 OpenAI 官方命名：

- `theme:communication` -> `Expertise-tailored communication`
- `theme:complex_responses` -> `Response depth`
- `theme:context_seeking` -> `Context seeking`
- `theme:emergency_referrals` -> `Emergency referrals`
- `theme:global_health` -> `Global health`
- `theme:health_data_tasks` -> `Health data tasks`
- `theme:hedging` -> `Responding under uncertainty`

## 支持的运行模式

- `full`：本地生成 + 外部 judge 打分
- `generate_only`：只生成模型回答，后面可以复用
- `judge_only`：只对现有 `responses.jsonl` 打分，不重新生成

## Judge 配置

judge 层同时支持标准 OpenAI 接口和 OpenAI 兼容接口。

支持的环境变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `JUDGE_API_KEY`
- `JUDGE_BASE_URL`

当前行为：

- 如果 `OPENAI_BASE_URL` 是 `https://.../v1`，代码会自动补成 `.../v1/chat/completions`
- 默认 judge 模型是 `gpt-5.2`
- 每条 judgment 都会保存 provider 实际返回的 `judge_actual_model`

这对于代理网关、路由平台特别有用，因为可以发现“请求模型”和“实际返回模型”是否一致。

## 断点恢复行为

- 如果 `responses.jsonl` 里已经有某条样本，对应生成会自动跳过
- 如果 `judgments.jsonl` 里已经有某条样本，对应打分会自动跳过
- 想强制重跑时，可加 `--overwrite-responses` 或 `--overwrite-judgments`

另外，当前评测入口还支持两项和大规模实验很相关的能力：

- `--generator-batch-size`：提升本地生成吞吐
- 共享回答缓存：同一采样点扩样时，旧回答可直接复用

这意味着你可以先跑小样本，再逐步扩大样本量，而不是每次都从零生成全部回答。

## 目录结构

```text
evaluation/
  benchmarks/
    healthbench.py
  generators/
    hf_chat.py
  judges/
    openai_healthbench.py
  run_eval.py
```

## 输出目录

每次评测都会写到：

```text
outputs/eval/<run_name>/
  artifacts/
  logs/
  responses.jsonl
  judgments.jsonl
  summary.json
  summary.md
```

## 打分机制

这套实现采用分层打分：

1. 每个 prompt 先由本地模型生成一条开放式回答。
2. 每条 rubric 交给 judge 输出严格 JSON：
   - `criteria_met`
   - `explanation`
3. 单条样本分数：
   - 满足的 rubric 正向分值之和 / 全部正向 rubric 分值之和
4. 最后再聚合出：
   - `overall`
   - `by_axis`
   - `by_theme`
   - `by_physician_category`

当前 summary 同时保留：

- `raw_mean`
- `clipped_mean`

其中 `clipped_mean` 更适合做 dashboard 和实验横向对比。

## 重要说明

如果要拿到官方 rubric 分数，必须配置 `OPENAI_API_KEY` 或兼容接口对应的 key。

没有 key 时，这套仓库仍然可以运行：

- 本地生成
- 基座评测入口
- LoRA 评测入口
- 结构化日志与 responses 导出

但不能得到正式的 rubric judge 分数。
