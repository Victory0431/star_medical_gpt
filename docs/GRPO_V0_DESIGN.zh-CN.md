# GRPO v0 设计文档

这份文档记录当前 `GRPO` 路线的第一版数据设计、奖励设计和训练实现。文档采用原地持续更新的方式维护，不再额外拆很多零散说明。

## 1. 当前阶段

当前已完成的是 `GRPO prompt-only 数据集 v1`。

目标不是再做一份“标准答案监督集”，而是围绕 `DPO v2 checkpoint-330` 在 `HealthBench` 上已经稳定暴露出的短板，先把适合 `GRPO` 的训练 prompt 池组织出来，让后续 reward 可以对着这些短板做针对性优化。

## 2. 为什么先做 prompt-only 数据

`GRPO` 和 `SFT / DPO` 不一样，它不是直接拿一条固定答案做监督，而是：

1. 给模型一个 prompt
2. 让模型采样出多条候选回答
3. 用 reward 函数对这些回答打分
4. 在同组回答内部做相对优化

所以这一步最重要的不是“标准答案多漂亮”，而是：

- prompt 是否真的覆盖当前模型短板
- 每条 prompt 是否带有足够的 reward 设计线索
- 后续能不能围绕同一条 prompt 做稳定可复用的奖励打分

## 3. v1 数据来源

当前 `GRPO v1` 数据由两部分混合组成：

- `dpo_v2_train`
  - 来自重构后的 `3800` 条 `DPO v2` 训练集
  - 优点是已经带有明确的“医疗偏好差异”信息，适合补 `emergency / context / hedging`
- `hq_holdout`
  - 来自 `54,080` 条高分桶里未进入 `HQ-50k` 训练子集的剩余高质量样本
  - 优点是相对“新鲜”，没有被 `HQ-50k SFT` 直接见过，适合补 `communication / global_health`

## 4. v1 主切片

这版不是泛泛抽样，而是按 `HealthBench` 暴露出的短板定向组织了 5 个主切片：

- `communication`
- `global_health`
- `hedging`
- `context_seeking`
- `emergency`

它们分别对应后续 `GRPO` 想重点修复的方向：

- `communication`
  - 沟通质量、表达清晰度、安抚与可执行性
- `global_health`
  - 预防、传播、疫苗、复查、长期管理
- `hedging`
  - 不确定情形下避免过度断言
- `context_seeking`
  - 上下文不充分时先补信息、补检查
- `emergency`
  - 高风险场景下不能漏掉急诊 / 及时就医建议

## 5. 当前数据规模

本地产物已经生成完成：

- `train = 3000`
- `valid = 300`

候选池去重后总量：

- `7828`

来源构成：

- `dpo_v2_train = 3800`
- `hq_holdout = 4028`

正式训练集主切片分布：

- `communication = 900`
- `global_health = 700`
- `hedging = 600`
- `context_seeking = 450`
- `emergency = 350`

正式训练集来源分布：

- `dpo_v2_train = 1465`
- `hq_holdout = 1535`

## 6. 单条样本保留的信息

每条样本不是只有一个 prompt，而是把后续 reward 需要的上下文也一起留了下来：

- `primary_slice`
- `slice_tags`
- `slice_scores`
- `reward_profile`
- `penalty_profile`
- `hard_constraints`
- `risk_level`
- `selection_signals`
- `reference_answer`
- `negative_reference_answer`

这意味着后面的 `GRPO reward` 不需要从零“猜”这条数据该往哪里拉，而是能直接根据样本元信息做 prompt 级别的差异化打分。

## 7. 当前脚本与产物

数据构造脚本：

- [build_grpo_prompt_dataset.py](/home/qjh/llm_learning/my_medical_gpt/script/grpo/build_grpo_prompt_dataset.py)
- [run_build_grpo_prompt_dataset_v1.sh](/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_build_grpo_prompt_dataset_v1.sh)

本地产物：

- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1/train/medical_grpo_prompt_v1.train.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1/valid/medical_grpo_prompt_v1.valid.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1/reports/medical_grpo_prompt_v1.report.json`

说明：

- `data/` 目录在仓库中默认不纳入版本管理
- 仓库里保留的是“可复现脚本 + 设计文档”
- 数据可随脚本重新生成

## 8. 下一步

下一步会在这份文档里继续补充：

- `GRPO v0` 奖励函数设计
- `GRPO` 训练脚本与双卡 launcher
- 首轮训练配置与运行记录
