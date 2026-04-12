# 评测设计

英文原版见 [EVALUATION.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVALUATION.md)。

这份文档说明当前仓库的评测设计，以及为什么把 `HealthBench` 作为主要的开放式医疗 benchmark。

## 为什么不能只看医学选择题

像 `C-EVAL` 医学部分这样的 benchmark 仍然有价值，但它们更偏向衡量：

- 事实性记忆
- 选项判别能力
- 应试型准确率

它们不足以衡量：

- 开放式医疗回答是否完整
- 医患沟通质量
- 模型会不会主动索取缺失上下文
- 模型在不确定场景下是否合理 hedge
- 模型是否真正遵循了任务要求与回答风格

而这些恰好是后续 `DPO`、`GRPO` 常常会改变的地方。

## 为什么选择 HealthBench

`HealthBench` 是开放式医学 benchmark，采用 rubric-based scoring。它的流程是：

1. 你的模型先生成一段自由形式的医疗回答。
2. 外部 judge 模型根据多条 rubric 细则逐条打分。
3. 最终按整体、axis、theme 等维度聚合结果。

这使它非常适合：

- 用于 `SFT` 看准确性、完整性
- 用于 `DPO` 看回答风格与沟通质量
- 用于 `GRPO` 看指令遵循、补问能力与细粒度行为对齐

## 当前推荐采样方式

当前仓库已经支持两种采样模式：

- `sequential`：顺序取前 `N` 条，适合最早期 smoke test
- `stratified_theme`：按 `theme` 分层抽样，更适合正式对比实验

对 `HealthBench consensus` 来说，当前常用的正式小规模配置是：

- 7 个主题
- 每个主题抽 `15` 条
- 总计 `105` 条样本
- 固定 `seed=42`

这样做的好处是：

- 不会因为某一个主题恰好抽多了而让结果失真
- 更适合比较 `base / SFT / DPO / GRPO` 的阶段性差异
- 面试里更容易说明“我不是随手抽 10 条，而是按主题做了平衡采样”

## 这套 benchmark 能测什么

根据公开 rubric tag，目前比较核心的 axis 有：

- `axis:accuracy`
- `axis:completeness`
- `axis:context_awareness`
- `axis:communication_quality`
- `axis:instruction_following`

公开样本标签里常见的 theme 及其官方展示名称包括：

- `theme:communication` -> `Expertise-tailored communication`
- `theme:complex_responses` -> `Response depth`
- `theme:context_seeking` -> `Context seeking`
- `theme:emergency_referrals` -> `Emergency referrals`
- `theme:global_health` -> `Global health`
- `theme:health_data_tasks` -> `Health data tasks`
- `theme:hedging` -> `Responding under uncertainty`

## 与训练阶段的对应关系

### SFT 阶段

重点看：

- overall score
- `axis:accuracy`
- `axis:completeness`
- `theme:complex_responses`
- `theme:health_data_tasks`

原因：

- 这一阶段首先要建立医学知识覆盖和回答结构稳定性。

### DPO 阶段

重点看：

- `axis:communication_quality`
- `axis:instruction_following`
- `axis:context_awareness`
- `Expertise-tailored communication`（`theme:communication`）
- `Responding under uncertainty`（`theme:hedging`）
- `Context seeking`（`theme:context_seeking`）

原因：

- `DPO` 更常用于提升回答偏好、风格、帮助性和用户侧感知质量。

### GRPO 阶段

重点看：

- 与 `DPO` 类似的风格类指标
- 再加上 `Emergency referrals`（`theme:emergency_referrals`）这类安全敏感主题

原因：

- `GRPO` 适合用 reward 信号去塑造开放式行为细节。

## 更像工业项目的评测表达

一个偏弱的项目故事是：

- “我微调了模型，分数变高了。”

更强的故事是：

- “我使用开放式医疗 benchmark，不只看总分，还按 axis 和 theme 分析，从而区分知识层面的增益和对齐层面的增益。”

这种讲法更接近真实模型研发。

## 当前仓库里的实现

运行时代码：

- [evaluation/run_eval.py](/home/qjh/llm_learning/my_medical_gpt/evaluation/run_eval.py)
- [evaluation/benchmarks/healthbench.py](/home/qjh/llm_learning/my_medical_gpt/evaluation/benchmarks/healthbench.py)
- [evaluation/generators/hf_chat.py](/home/qjh/llm_learning/my_medical_gpt/evaluation/generators/hf_chat.py)
- [evaluation/judges/openai_healthbench.py](/home/qjh/llm_learning/my_medical_gpt/evaluation/judges/openai_healthbench.py)

参考快照：

- [references/openai/simple-evals/healthbench_eval.py](/home/qjh/llm_learning/my_medical_gpt/references/openai/simple-evals/healthbench_eval.py)

## 工程实现上的关键选择

当前本地实现故意拆成三层：

- benchmark 加载与聚合
- 本地 Hugging Face 生成
- 外部 judge 打分

这样做的好处是：

- `SFT` 可以直接复用 benchmark 与 generator
- `DPO / GRPO` 只换 checkpoint，不用改 benchmark 代码
- judge 可以独立重跑，节省时间与 API 成本

目前支持的模式：

- `full`：先生成，再打分
- `generate_only`：只生成，保留 `responses.jsonl`
- `judge_only`：复用已有 `responses.jsonl` 只打分

这两段式流程很符合真实工业使用方式。

另外，`run_eval.py` 现在还支持：

- 断点续跑：已有 `responses.jsonl` 或 `judgments.jsonl` 时会自动跳过已完成样本
- 中央日志：除了每个 run 自己的日志，还会在 `evaluation/logs/` 下额外写一份带时间戳的总日志
- 数据清单：每次运行都会把本次采样到的 prompt 清单落到 `artifacts/dataset_manifest.json`
- batch generation：可通过 `--generator-batch-size` 提升本地生成吞吐
- 共享回答缓存：同一采样点扩样时可直接复用旧 `responses`
- 计时统计：`summary.json` 会单独记录 `generation_seconds` 与 `judge_seconds`

关于提速、缓存设计和复现性边界，见：

- [EVAL_ACCELERATION_AND_CACHE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_ACCELERATION_AND_CACHE.zh-CN.md)

## 分数是怎么构成的

当前仓库把分数分成四层：

### 1. Rubric 层

每条样本下面有多条 rubric。

judge 对每条 rubric 返回结构化 JSON：

- `criteria_met`
- `explanation`

如果 rubric 是正向要求，满足则记正分。
如果 rubric 是负向要求，命中则应对回答产生扣分效果。

### 2. Example 层

单条 prompt 的 example score 计算方式是：

- 满足的正向 rubric 分值之和 / 全部正向 rubric 分值之和

因此：

- 好回答可接近 `1.0`
- 有害或不理想行为会拖低原始分数

### 3. Slice 层

单题打分后，再按以下维度聚合：

- `axis:*`
- `theme:*`
- `physician_agreed_category:*`

这也是它能拿来做训练阶段诊断，而不是只做排行榜展示的关键。

### 4. Summary 层

最终 summary 同时保留：

- `raw_mean`
- `clipped_mean`

`clipped_mean` 更适合做 dashboard 和稳定比较。

## Judge 配置

judge 支持标准 OpenAI 和 OpenAI 兼容接口。

当前默认设置：

- judge backend：chat completions 兼容接口
- 默认 judge model：`gpt-5.2`
- `temperature=0`
- 输出严格 JSON

支持环境变量：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `JUDGE_API_KEY`
- `JUDGE_BASE_URL`

为方便发现 provider 路由问题，当前记录还会保存：

- 请求模型：`judge_model`
- 实际返回模型：`judge_actual_model`
- 实际 API 地址：`judge_api_base_url`

## 当前基座 smoke baseline

当前已提交的基座 smoke baseline：

- run name：`20260409_healthbench_base_gpt52_full_10`
- benchmark：`HealthBench consensus`
- sample count：`10`
- evaluated model：`Qwen3-8B base`
- requested judge model：`gpt-5.2`
- actual returned judge model：`gpt-5.2`

主要结果：

- overall clipped mean：`0.4167`
- overall raw mean：`0.4167`

axis 结果：

- `axis:accuracy = 0.5000`
- `axis:communication_quality = 0.6667`
- `axis:context_awareness = 0.2500`
- `axis:instruction_following = 0.5000`

初步解读：

- 基座模型的表层沟通不算差
- 当前短板主要在 `context_awareness`
- 这也符合项目预期：后续 `DPO / GRPO` 更应该补上下文感知、hedging 和临床交互质量

重要前提：

- 这只是 `10` 条样本的 smoke baseline
- 它适合 pipeline 验证与早期对比，不适合作为最终 headline 结论

对应实验快照：

- [summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_10/summary.json)
- [README.md](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260409_healthbench_base_gpt52_full_10/README.md)

## 当前正式评测配置

为方便复现，仓库中已经补了三份 `theme15` 配置：

- [evaluation/configs/healthbench_theme15_base.json](/home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_base.json)
- [evaluation/configs/healthbench_theme15_huatuo_5w_ckpt75.json](/home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_huatuo_5w_ckpt75.json)
- [evaluation/configs/healthbench_theme15_huatuo_5w_ckpt925.json](/home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_huatuo_5w_ckpt925.json)

它们统一使用：

- `subset_name=consensus`
- `sampling_mode=stratified_theme`
- `per_theme_examples=15`
- `judge_model=gpt-5.2`
- `temperature=0`

## 现阶段的限制

官方 HealthBench rubric 打分需要 `OPENAI_API_KEY` 或兼容接口 key。

没有 key 时，仓库仍可完成：

- HealthBench prompt 上的本地生成
- 基座模型评测入口
- LoRA 模型评测入口
- 结构化日志和 responses 导出

但无法得到正式 rubric score。
