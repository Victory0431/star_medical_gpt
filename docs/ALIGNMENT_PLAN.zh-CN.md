# 对齐阶段数据与评测闭环规划

这份文档用于回答三个核心问题：

1. 现在手里有哪些 `DPO / RM / GRPO` 可用数据；
2. 它们各自适合提升什么能力；
3. 后续应该怎样把“数据 -> 训练 -> 评测 -> 迭代”串成闭环。

## 结论先看

当前最值得作为主线的数据是：

- 医疗 `pairwise preference` 数据：`3800/100/100`
- 可直接用于：`DPO`、`ORPO`、`KTO`、`Reward Model`
- 最适合提升：医学问答中的错误纠正、回答偏好、安全性、建议方式

当前不适合直接作为正式 `GRPO` 数据的是：

- `grpo_sample_reference.jsonl`
- 原因不是“格式不对”，而是它更像一个 demo 样例集，规模太小，而且奖励目标偏“标准答案匹配”，不够像真实医疗开放问答

所以更合理的路线是：

1. 先做 `DPO v1`，把医疗偏好数据跑通。
2. 再做 `RM v1`，把同一批 pairwise 数据用于奖励模型建模。
3. `GRPO` 不直接吃旧仓库这份 sample，而是重新构造“医疗 prompt-only 数据 + judge/reward 函数”。
4. 评测继续用当前已经搭好的 `HealthBench` 闭环，重点观察 `communication_quality / context_awareness / emergency_referrals / hedging`。

## 现有数据盘点

### 1. 医疗 pairwise 偏好主数据

新仓库本地路径：

- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/medical_pairwise_train.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/medical_pairwise_valid.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/medical_pairwise_test.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/rm/medical_pairwise_train.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/rm/medical_pairwise_valid.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/rm/medical_pairwise_test.jsonl`

来源：

- 原仓库 `data/finetune/reward/*.json`

样本量：

- train：`3800`
- valid：`100`
- test：`100`

字段格式：

```json
{"question": "...", "response_chosen": "...", "response_rejected": "..."}
```

这类数据最适合做什么：

- `DPO`
- `ORPO`
- `KTO`
- `RM`

这类数据能提升什么：

- 让模型更偏向“更好的医学回答”而不是“任何能答出来的回答”
- 降低明显错误、误导、过度自信回答被选中的概率
- 强化“什么叫更符合医患场景的回答偏好”

它的意义：

- 这是你后续对齐阶段最像工业数据的起点
- 因为它不是单纯问答，而是带有“优答/劣答”比较关系
- 面试里可以明确说：`SFT` 学“会答”，`DPO/RM` 学“更该怎么答”

### 2. 通用 pairwise 对照数据

新仓库本地路径：

- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/dpo/general_pairwise_dpo_zh_500.jsonl`

来源：

- 原仓库 `data/reward/dpo_zh_500.jsonl`

样本量：

- `500`

字段格式：

```json
{"system": "...", "history": [], "question": "...", "response_chosen": "...", "response_rejected": "..."}
```

这类数据最适合做什么：

- 作为 `DPO` 小比例混合数据
- 做一个“是否加入通用偏好数据”的对照实验

这类数据能提升什么：

- 通用指令遵循
- 非医疗场景下的回答风格稳定性
- 带 `system/history` 的多轮模板兼容性

它的意义：

- 不是主菜，是配菜
- 可以少量混入，防止模型对医疗语料过拟合后，日常指令跟随能力变差
- 但比例不能太大，否则会稀释你的医疗项目定位

建议：

- 第一版不要混
- 第二版可以试 `90% medical pairwise + 10% general pairwise`

### 3. GRPO 参考样例

新仓库本地路径：

- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/raw/grpo_reference/grpo_sample_reference.jsonl`

来源：

- 原仓库 `data/grpo/sample.jsonl`

样本量：

- `32`

字段格式：

```json
{"question": "...", "answer": "..."}
```

旧代码里它对应的奖励方式：

- `accuracy_reward`：预测答案和标准答案是否匹配
- `format_reward`：是否输出 `<think>...</think><answer>...</answer>`

为什么它不适合直接做你的正式 `GRPO`：

- 样本量太小
- 目标偏“标准答案匹配”
- 更像 reasoning / exact answer 任务，不像真实医疗开放问答
- 奖励目标和你当前 `HealthBench` 评测维度不对齐

它的意义：

- 只保留为格式参考
- 后续你如果重写 `GRPO` 训练脚本，可以参考它的最小输入格式

## 哪些旧数据不属于 DPO/GRPO 主线

以下数据仍然有价值，但不属于当前对齐主线：

- `data/sft/...`
  作用：`SFT` 或生成偏好候选答案
- `data/finetune/pretrain/...`
  作用：继续预训练或领域增量预训练
- `data/rag/...`
  作用：RAG 检索语料，不是偏好优化数据

面试里可以这样说：

- `SFT` 负责医学知识和回答能力打底
- `DPO / RM / GRPO` 负责把“回答风格、安全边界、上下文感知、是否该补问”往更符合医疗场景的方向推

## 建议训练路线

### Phase 1: DPO v1

目标：

- 在当前最佳 `SFT checkpoint-75` 基础上继续做偏好优化

数据：

- 主数据：`medical_pairwise_train/valid/test`

为什么先做 DPO：

- 实现最直接
- 风险最低
- 最容易和你当前的 `HealthBench` 评测体系形成前后对比

预期提升：

- `communication_quality`
- `context_awareness`
- `hedging`
- 部分安全表达

### Phase 2: DPO v2 或 ORPO v1

目标：

- 做更贴近工业实验风格的对照

数据：

- `90% medical_pairwise`
- `10% general_pairwise_dpo_zh_500`

意义：

- 看混入少量通用偏好数据后，是否能保住通用指令遵循能力
- 这类对照实验很适合面试讲

### Phase 3: RM v1

目标：

- 训练一个医疗偏好奖励模型

数据：

- 仍然用 `medical_pairwise_train/valid/test`

意义：

- 让你的项目故事从“只会 DPO”升级到“我也能做奖励建模”
- 更接近完整 RLHF pipeline

注意：

- `3800` 条能做一个小规模 RM baseline
- 但如果想让 RM 真正稳定，后续最好扩一版更大的中文医疗 pairwise 数据

### Phase 4: GRPO v1

目标：

- 不再优化“静态 pairwise 偏好”，而是优化“生成过程中的行为”

但不要直接用旧 sample。

更合理的数据构造方式是：

1. 收集一批医疗 prompt-only 问题集
2. 用当前 SFT/DPO 模型在线生成多个候选
3. 用 reward/judge 函数给每个候选打分
4. 用 `GRPO` 去优化策略

## 你真正需要的 GRPO 数据长什么样

你现在项目里的 `GRPO` 不应该以“有标准答案的短问答”为主，而应该以“开放式医疗问答行为”来设计 reward。

建议的 prompt 来源：

- `medical_pairwise_*` 里的 `question`
- 你已有 `SFT` 验证/测试集里的用户问题
- 后续你自己补的安全分诊题、补问题、沟通题

建议的 reward 维度：

- 医学正确性
- 是否主动索取缺失上下文
- 是否在高风险场景建议就医/急诊
- 是否避免过度确定性表达
- 是否回答清晰、结构化、可执行

这才和你当前 `HealthBench` 的评测方向一致。

## 评测闭环怎么设计

### DPO 阶段

主评测：

- `HealthBench consensus`

重点看：

- `axis:communication_quality`
- `axis:context_awareness`
- `axis:instruction_following`
- `theme:hedging`
- `theme:communication`

补充评测：

- pairwise dev win-rate
- medical preference holdout accuracy

### RM 阶段

主评测：

- 奖励模型在 `valid/test` 上对 chosen/rejected 的排序准确率

补充评测：

- 用 RM 给 base / SFT / DPO 输出打分，看排序是否与 `HealthBench` 大方向一致

### GRPO 阶段

主评测：

- 仍然回到 `HealthBench`

重点看：

- `theme:emergency_referrals`
- `theme:context_seeking`
- `theme:hedging`
- `axis:communication_quality`

这是因为：

- `GRPO` 最适合优化开放式行为细节
- 如果它没有把这些维度拉起来，那就说明 reward 设计还不够好

## 当前最合理的闭环版本

建议你下一阶段按这个顺序推进：

1. `SFT checkpoint-75` 作为 policy init
2. `medical_pairwise_train/valid/test` 做 `DPO v1`
3. 用现有 `HealthBench theme15x7` 跑正式 before/after
4. 再做一个 `DPO v2/ORPO v1` 混少量通用偏好数据的对照
5. 然后补 `RM v1`
6. 最后再进入真正有意义的 `GRPO`

## 一句话项目故事

如果面试官问你“DPO/GRPO 数据怎么设计”，更好的回答是：

- “我没有直接拿一个很小的 demo 数据去硬跑 GRPO。我的做法是先用医疗 pairwise 数据做 DPO 和奖励模型，先把偏好优化闭环做扎实；GRPO 则单独设计 prompt-only 数据和 reward 维度，让训练目标和 HealthBench 的开放式医疗评测切片保持一致。” 

这会比“我把所有算法都跑了一遍”更像真正能落地的人。
