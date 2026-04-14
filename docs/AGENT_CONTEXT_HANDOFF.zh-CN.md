# Agent Handoff 全量上下文文档

这是一份给后续 agent 直接阅读的超大号上下文交接文档。

目标不是做一份“项目简介”，而是尽量把当前项目已经发生过的关键决策、实验脉络、代码实现、结果结论、工程坑点、用户偏好和下一步建议都压缩到一份文档里。后续如果开启新对话，优先让 agent 先阅读这份文档，再决定下一步动作。

这份文档默认比普通 README 更长、更细，服务对象是模型而不是人类快速浏览。

---

## 1. 文档用途与阅读顺序

如果你是后续 agent，建议阅读顺序：

1. 先读本文件，获得全局上下文和当前状态。
2. 再按任务需要跳转到本文档中列出的专题文档。
3. 遇到需要落地代码或复现实验时，再去对应脚本和产物路径。

这份文档主要解决的问题：

- 项目已经经历多轮 `SFT / DPO / HQ-50k / HealthBench / GRPO` 演化，单看某一份文档很难知道整个脉络。
- 用户非常在意“实验结论是否稳、是否可复现、是否是工程化而不是挑最好 checkpoint 讲故事”。
- 当前仓库已经有较多文档和产物，新 agent 如果没有总导航，容易重复踩坑或误判现状。

---

## 2. 项目一句话概括

这是一个围绕 `Qwen3-8B` 做医疗垂直对齐优化的个人项目，主线是：

- 先做医疗 `SFT`
- 再做医疗偏好对齐 `DPO`
- 再围绕 `HealthBench` 暴露出的稳定短板设计更针对性的 `GRPO`
- 全过程坚持“统一 benchmark 闭环 + 文档留痕 + 工程化复现”

注意：

- 这个项目不再是“简单跑 SFT/DPO/GRPO 三步就一定涨分”的叙事。
- 实际进展已经证明：数据质量、评测方差、参考策略定义、checkpoint 选择、主题短板拆解，都比“训练方法名词”本身更重要。

---

## 3. 用户画像与协作偏好

后续 agent 非常需要知道这些偏好，否则很容易给出不合拍的建议。

### 3.1 用户最看重什么

- 最看重“统一评测闭环”
- 很在意实验结论是否稳健，而不是一次性跑分
- 对“挑最好 checkpoint 讲故事”比较警惕
- 愿意做大量实验，但希望每一步都有明确工程意义
- 对项目叙事的要求很高，希望能支撑面试、简历和项目讲述

### 3.2 用户不喜欢什么

- 空泛规划，没有实际代码和运行
- 没有把文档同步更新到仓库
- 只讲概念，不结合当前真实代码和真实 run
- 只给理论，不核查本地产物和日志
- 把单次 benchmark 分数差异讲成过于确定的结论

### 3.3 用户偏好的工作方式

- 中文沟通
- 希望 agent 主动执行，不要总停在分析
- 有意义的阶段性完成后，希望提交并推送到仓库
- 文档希望“该原地更新就原地更新”，不要无节制新建碎片文档
- 喜欢详细、结构化、可复盘的说明，尤其是给后续 agent 的上下文整理

### 3.4 用户机器与工程习惯

- 双卡 `H200`
- 偏好后台 `nohup/setsid` 运行长任务
- 偏好 `W&B` 在线记录
- 希望日志带时间戳
- 经常会让 agent 直接后台启动任务，而不是只给命令

---

## 4. 仓库关键路径总览

项目根目录：

- `/home/qjh/llm_learning/my_medical_gpt`

核心目录：

- `docs/`
  - 主要设计文档、结果分析、工作流说明
- `script/`
  - 训练、数据处理、评测、导出、队列、GRPO 等脚本
- `outputs/`
  - 本地训练、评测、GRPO 运行产物
- `experiment_records/`
  - 轻量化导出的正式实验快照
- `data/`
  - 本地数据与中间数据
  - 注意：`data/` 默认不进 git

重要提醒：

- `data/` 是 `.gitignore` 的，后续 agent 不要误以为“仓库里没有数据就代表数据没做”。
- 代码和文档在 git 中，数据与大模型产物主要保留在本地路径。

---

## 5. 当前最重要的专题文档索引

### 5.1 总体工作流与脚本

- [WORKFLOW.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/WORKFLOW.zh-CN.md)
- [SCRIPT_GUIDE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/SCRIPT_GUIDE.zh-CN.md)

### 5.2 评测框架与结果

- [EVALUATION.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVALUATION.zh-CN.md)
- [EVAL_INTEGRATION.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_INTEGRATION.zh-CN.md)
- [EVAL_ACCELERATION_AND_CACHE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_ACCELERATION_AND_CACHE.zh-CN.md)
- [EVAL_RESULTS.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_RESULTS.zh-CN.md)
- [DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md)
- [HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md)
- [HQ50K_HEALTHBENCH_COMPARE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/HQ50K_HEALTHBENCH_COMPARE.zh-CN.md)

### 5.3 DPO

- [DPO_WORKFLOW.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_WORKFLOW.zh-CN.md)
- [DPO_METRICS.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_METRICS.zh-CN.md)
- [DPO_V2_RECONSTRUCTION.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_RECONSTRUCTION.zh-CN.md)
- [DPO_V2_TRAINING_REPORT.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_TRAINING_REPORT.zh-CN.md)

### 5.4 SFT 数据筛选

- [SFT_DATA_CURATION.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/SFT_DATA_CURATION.zh-CN.md)
- [HQ50K_HEALTHBENCH_COMPARE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/HQ50K_HEALTHBENCH_COMPARE.zh-CN.md)

### 5.5 当前 GRPO

- [GRPO_V0_DESIGN.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/GRPO_V0_DESIGN.zh-CN.md)

---

## 6. 当前代码演进时间线

最近关键提交如下，基本对应项目主线：

- `6ec2673`
  - 脚本目录重构
- `1d6ed7f`
  - 正式补齐 1k 与 DPO 的 HealthBench 评测
- `d161ed1`
  - DPO 指标文档细化
- `2a2d63f`
  - SFT 数据筛选管线
- `4c03a77`
  - curated subset builder
- `9950246`
  - DPO v2 reconstruction 工作流
- `f3ffc9b`
  - DPO v2 training report
- `18f262e`
  - DPO v2 vs HealthBench 比较
- `95f1f0b`
  - HQ-50k HealthBench 分析
- `380e42f`
  - Stability analysis
- `8350858`
  - 批量生成 + 共享缓存
- `a017520`
  - 加入官方 HealthBench 分数上下文
- `3c5e957`
  - overnight queue runner
- `52e8d8c`
  - HealthBench 队列重试加固
- `148eb3a`
  - 刷新 700-sample HealthBench 结果
- `3392280`
  - GRPO prompt dataset builder
- `151f110`
  - GRPO v0 training pipeline
- `8914b78`
  - 记录 GRPO 启动状态
- `921f755`
  - 扩写 GRPO FAQ

从时间线可以看出，这个项目已经不是一个“单一训练脚本仓库”，而是逐渐演化成：

- 数据工程
- 对齐训练
- 统一评测
- 实验记录导出
- 稳定性复测
- 后续强化学习路线

并行推进的工程体系。

---

## 7. 项目主线脉络总述

### 7.1 SFT 主线

最初先围绕医疗问答数据做 `SFT`，形成了几个重要 baseline：

- `Qwen3-8B base`
- `huatuo_1k`
- `huatuo_5w checkpoint-75`
- `huatuo_5w checkpoint-925`

后续又做了数据筛选路线：

- `HQ-50k`
- `HQ-54k`
- 规则粗筛
- 分布分析
- 轻量质量分桶

主结论：

- `huatuo_5w checkpoint-75` 是一个很强的 SFT baseline
- `HQ-50k` 在正式 `HealthBench consensus 7 x 15` 上略高于 `huatuo_5w checkpoint-75`
- 说明“数据筛选路线有效”，但收益不算巨大，属于“小幅但真实”的提升

### 7.2 DPO 主线

随后项目进入偏好对齐阶段。

#### DPO v1

- 使用原始医疗 pairwise 数据
- 外部 `HealthBench` 效果很差
- 正式结果约：
  - `DPO v1 checkpoint-100 = 0.2111`

这说明：

- 原始偏好数据质量、标签方向和优化目标之间存在较大偏差

#### DPO v2

针对上面的失败，项目没有简单放弃，而是做了较大幅度的数据重构：

- 使用 `gpt-5.2` 对约 `3800` 条 pairwise 数据做医疗偏好重构
- 核心重构方向：
  - 标签方向是否正确
  - `chosen / rejected` 是否更符合 `HealthBench` 关心的行为目标
  - 是否补齐：
    - 准确性
    - 沟通质量
    - 风险提示
    - 上下文感知
    - 急诊转诊

然后基于重构后的数据重新训练 `DPO v2`。

主结论：

- `DPO v2 checkpoint-30 = 0.2492`
- `DPO v2 checkpoint-330 = 0.2619`

相比：

- `DPO v1 checkpoint-100 = 0.2111`
- `base = 0.2206`

说明：

- DPO v2 是明显有效的
- 数据重构本身有价值
- 但 `DPO v2` 仍然没有在单轮结果上稳定超过最强 SFT

### 7.3 Stability 主线

用户非常关注评测方差，所以又做了 `checkpoint-75` vs `DPO v2 checkpoint-330` 的第二轮独立抽样复测。

稳定性文档见：

- [HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md)

关键结果：

- `huatuo_5w checkpoint-75`
  - 两轮均值 `0.2754`
  - 标准差 `0.0135`
- `DPO v2 checkpoint-330`
  - 两轮均值 `0.2603`
  - 标准差 `0.0016`

这说明：

- `huatuo_5w checkpoint-75` 单轮上限高
- `DPO v2 checkpoint-330` 更稳定
- 不能再用一次抽样的单次 gap 就下过强结论

这条稳定性线对项目叙事非常重要：

- 它说明项目不是“跑一次 benchmark 就宣布胜利”
- 而是主动验证 benchmark 方差与结论稳健性

### 7.4 Full / Theme100x7 评测主线

后面又做了更重的评测工程工作：

- 双卡队列调度
- 大样本主题分层抽样
- batched generation
- 缓存复用
- 失败重试

代表性产物目录包括：

- `/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260412_healthbench_theme100x7_resume_healthbench_qwen3_8b_base_gpt-52_consensus_theme100x7_seed42`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260412_healthbench_theme100x7_resume_healthbench_qwen3_8b_huatuo_5w_ckpt75_gpt-52_consensus_theme100x7_seed42`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260412_healthbench_theme100x7_resume_healthbench_qwen3_8b_dpo_v2_ckpt330_gpt-52_consensus_theme100x7_seed42`

以及 full generation / consensus-from-full 相关目录：

- `/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260413_healthbench_full_sft5w_dpo330_healthbench_qwen3_8b_huatuo_5w_ckpt75_gpt-52_full_all_seed42`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260413_healthbench_full_sft5w_dpo330_healthbench_qwen3_8b_dpo_v2_ckpt330_gpt-52_full_all_seed42`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260413_healthbench_consensus_from_full_sft5w_dpo330_healthbench_qwen3_8b_huatuo_5w_ckpt75_gpt-52_consensus_all_seed42`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260413_healthbench_consensus_from_full_sft5w_dpo330_healthbench_qwen3_8b_dpo_v2_ckpt330_gpt-52_consensus_all_seed42`

这一部分的主要意义不是单个分数，而是：

- 评测工程已经演进成一个能支撑大样本、缓存复用、后台队列、失败恢复的完整子系统

### 7.5 当前进入 GRPO 主线

现阶段项目已经从“做更好的 SFT / DPO”转向：

- 基于 `HealthBench` 暴露出来的稳定短板
- 设计更针对性的 `GRPO`

当前 GRPO 的定位不是泛化性的 RLHF，而是：

- 在当前最有潜力的 `DPO v2 checkpoint-330` 基础上
- 专门补：
  - `communication_quality`
  - `global_health`
  - `context_awareness`
  - `hedging`
  - `emergency_referrals`

---

## 8. 当前最重要的量化结果

这部分只放目前最关键、最常被引用的结果。

### 8.1 SFT / DPO 主干结果

来自 [DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md)：

| 模型 | HealthBench overall clipped mean |
| --- | --- |
| `Qwen3-8B base` | `0.2206` |
| `Qwen3-8B + huatuo_1k LoRA` | `0.2508` |
| `Qwen3-8B + huatuo_5w LoRA (checkpoint-75)` | `0.2889` |
| `Qwen3-8B + huatuo_5w LoRA (checkpoint-925)` | `0.2587` |
| `Qwen3-8B + DPO v1 medical_pairwise (checkpoint-100)` | `0.2111` |
| `Qwen3-8B + DPO v2 (checkpoint-30)` | `0.2492` |
| `Qwen3-8B + DPO v2 (checkpoint-330)` | `0.2619` |

关键解释：

- `huatuo_5w checkpoint-75` 是强 SFT baseline
- `DPO v1` 明显失败
- `DPO v2` 相对 `DPO v1` 有明显进步
- `DPO v2 checkpoint-330` 相比 `checkpoint-30` 更贴近 pairwise 偏好目标

### 8.2 HQ-50k 结果

来自 [HQ50K_HEALTHBENCH_COMPARE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/HQ50K_HEALTHBENCH_COMPARE.zh-CN.md)：

| 模型 | HealthBench overall clipped mean |
| --- | --- |
| `huatuo_5w checkpoint-75` | `0.2889` |
| `HQ-50k best` | `0.2905` |

关键解释：

- `HQ-50k best` 是当前所有 `SFT` 结果中最高的正式结果
- 但领先幅度只有 `+0.0016`
- 正确表述应该是“至少不弱于原始 5w，并出现了小幅正式优势”

### 8.3 稳定性复测结果

来自 [HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md)：

| 模型 | 第一轮 | 第二轮 | 两轮均值 | 标准差 |
| --- | --- | --- | --- | --- |
| `huatuo_5w checkpoint-75` | `0.2889` | `0.2619` | `0.2754` | `0.0135` |
| `DPO v2 checkpoint-330` | `0.2619` | `0.2587` | `0.2603` | `0.0016` |

关键解释：

- `huatuo_5w checkpoint-75` 单轮最好，但波动更大
- `DPO v2 checkpoint-330` 均值略低，但稳定性更高
- 因此不能简单讲成“DPO 明显弱于 SFT”

### 8.4 DPO v2 checkpoint 内部比较

来自 [DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md)：

- `checkpoint-330` 相比 `checkpoint-30` 更强的方向：
  - `accuracy`
  - `context_awareness`
  - `completeness`
  - `Context seeking`
  - `Emergency referrals`
  - `Responding under uncertainty`

- `checkpoint-330` 相比 `checkpoint-30` 付出的代价：
  - `communication_quality`
  - `instruction_following`
  - 部分 `global_health`
  - 部分 `health_data_tasks`

这也是后来为什么项目会把 `GRPO` 的主补齐方向聚焦到：

- `communication`
- `global_health`
- `context / hedging / emergency`

---

## 9. 评测工程能力现状

这一部分是本项目非常重要的工程亮点。

### 9.1 当前评测已具备的能力

- 基于 `HealthBench` 的统一评测入口
- `7 大主题` 分层抽样
- `15` 条 rubric 维度方案
- `consensus subset` 与更大样本评测
- 批量生成
- 回答缓存复用
- 打分失败重试
- 后台任务队列
- 双卡调度
- 结果轻量导出到 `experiment_records`

### 9.2 重要的评测工程文档

- [EVAL_ACCELERATION_AND_CACHE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_ACCELERATION_AND_CACHE.zh-CN.md)
- [EVALUATION.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVALUATION.zh-CN.md)
- [EVAL_RESULTS.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/EVAL_RESULTS.zh-CN.md)

### 9.3 重要的评测工程提交

- `8350858`
  - batched generation + shared response cache
- `3c5e957`
  - overnight queue runner
- `52e8d8c`
  - queue retry hardening
- `68f6b34`
  - overnight acceptance 文档

### 9.4 评测时最容易踩的坑

- 父进程没完全脱离终端，关页面后任务被杀
- 队列脚本中子任务失败但没有正确续跑
- judge API 可能出现 `429` 或其他失败，需要退避重试
- 抽样样本数太小，单次分差不稳定
- full 评测代价高，需要先想清楚实验意义，避免“随便测几个 checkpoint”

---

## 10. 数据工程主线现状

### 10.1 SFT 数据筛选

已经完成：

- 规则粗筛
- 来源分布分析
- embedding / 分布视角的分析尝试
- 轻量质量打分 `light quality score`
- 三桶分流思路，而不是直接硬删

重要脚本主要在：

- `/home/qjh/llm_learning/my_medical_gpt/script/sft/filter_sft_rules.py`
- `/home/qjh/llm_learning/my_medical_gpt/script/sft/light_quality_score.py`
- `/home/qjh/llm_learning/my_medical_gpt/script/sft/analyze_sft_distribution.py`
- `/home/qjh/llm_learning/my_medical_gpt/script/sft/build_curation_subset.py`

主要结论：

- 不是“数据越多越好”
- 规则粗筛 + 轻量质量分桶 + 分层抽样可以带来真实收益
- `HQ-50k` 已经给出小幅正式优势，证明这条线可行

### 10.2 DPO v2 数据重构

已经完成：

- 用 `gpt-5.2` 对 `3800` 条医疗 pairwise 数据做重构
- 支持串行稳定重构、去重、断点续跑
- 形成：
  - 重构审计产物
  - processed 数据
  - train/valid/test 正式分割
  - 汇总报告

专题文档：

- [DPO_V2_RECONSTRUCTION.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_RECONSTRUCTION.zh-CN.md)

### 10.3 当前 GRPO 数据

当前 `GRPO prompt-only v1` 已完成。

核心情况见：

- [GRPO_V0_DESIGN.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/GRPO_V0_DESIGN.zh-CN.md)

本地数据规模：

- `train = 3000`
- `valid = 300`

来源：

- `dpo_v2_train = 3800`
- `hq_holdout = 4028`

主切片：

- `communication = 900`
- `global_health = 700`
- `hedging = 600`
- `context_seeking = 450`
- `emergency = 350`

样本保留字段包括：

- `prompt_id`
- `prompt`
- `primary_slice`
- `slice_tags`
- `reward_profile`
- `penalty_profile`
- `hard_constraints`
- `risk_level`
- `reference_answer`
- `negative_reference_answer`

这为后续 reward 的 prompt 级差异化打分提供了基础。

---

## 11. DPO 阶段的关键认知沉淀

这是项目里非常重要的一层认知，不应丢失。

### 11.1 DPO 没起作用不是方法本身无效，更可能是数据目标不一致

项目中已经验证：

- 原始医学 pairwise 数据并不天然等于“你想优化的 HealthBench 行为目标”
- 很多 pairwise 样本实际更偏知识性对齐，未必是风格/沟通/风险行为对齐
- 如果 `chosen / rejected` 方向、措辞、标签有噪声，就会直接把 DPO 带偏

### 11.2 DPO 的“对齐税”在这里不能粗暴下结论

不能简单讲成：

- “DPO 一做就有对齐税”

更准确的说法是：

- `DPO v2 checkpoint-330` 在一些更像医疗行为对齐的主题上确实有收益
- 但同时牺牲了 `communication_quality`、部分 `global_health` 等维度
- 这更像“优化方向偏了 / 权重失衡了”，而不是方法名义上的绝对失败

### 11.3 DPO checkpoint 不能只看训练内 best

项目里已经明确形成结论：

- `best checkpoint` 不能只看训练内指标
- 必须放回统一外部 benchmark 闭环里看
- 否则会误把“pairwise 训练内最强”当成“对外医疗行为最优”

---

## 12. 当前 GRPO 设计与实现现状

这一部分是目前最需要后续 agent 接上的主线。

### 12.1 当前 GRPO 文档

- [GRPO_V0_DESIGN.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/GRPO_V0_DESIGN.zh-CN.md)

### 12.2 当前 GRPO 代码

- [build_grpo_prompt_dataset.py](/home/qjh/llm_learning/my_medical_gpt/script/grpo/build_grpo_prompt_dataset.py)
- [run_build_grpo_prompt_dataset_v1.sh](/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_build_grpo_prompt_dataset_v1.sh)
- [reward_functions.py](/home/qjh/llm_learning/my_medical_gpt/script/grpo/reward_functions.py)
- [train_grpo.py](/home/qjh/llm_learning/my_medical_gpt/script/grpo/train_grpo.py)
- [run_grpo_qwen3_8b_dpo330_v0.sh](/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_grpo_qwen3_8b_dpo330_v0.sh)

### 12.3 当前 GRPO 的设计定位

当前版本是：

- `GRPO v0`
- `rule + reference anchored reward`
- `prompt-only` 数据
- 以 `DPO330` 为初始化策略
- 先验证链路与方向，不宣称已经是最终 reward 方案

### 12.4 当前 GRPO reward

本地规则 reward 一共 6 个：

- `communication_quality_reward`
- `context_awareness_reward`
- `hedging_reward`
- `emergency_referral_reward`
- `reference_alignment_reward`
- `safety_penalty_reward`

不是在线 `LLM judge` 打分。

当前 reward 的本质是：

- 规则打分
- 参考答案 overlap 锚定
- 样本级动态加权

### 12.5 当前 GRPO 参数结构

这是最容易混淆的部分，必须保留给后续 agent。

当前结构不是：

- `base + SFT LoRA + DPO LoRA`

而是：

- `SFT 5w merged backbone`
- `DPO330 adapter`

所以：

- 当前在线训练时真正外挂、真正更新的 LoRA 只有 `DPO330`
- 当前 GRPO 本质是“继续微调 DPO330 这套 LoRA”

### 12.6 当前参考策略机制

当前不是没有参考策略，而是：

- 没有额外再加载一整份完整 8B reference model
- `trl` 在 `PEFT + beta != 0` 场景下，会复制当前 adapter 形成 `ref adapter`

可理解为：

- `backbone = SFT merged`
- `policy adapter = DPO330 default`
- `reference adapter = DPO330 ref`

### 12.7 为什么没有先 merge DPO 再新开 GRPO LoRA

当前版本选择“继续训 DPO330 adapter”，主要原因：

- 工程链路更短
- 更容易快速验证方向
- 更方便让 `trl` 复制 `ref adapter`
- 参考策略精确锁定为“训练开始时的 DPO330”

但这不是唯一正确路线。

一个很值得补做的后续对照是：

1. `continue DPO adapter`
2. `merge DPO -> fresh GRPO LoRA`

这个对照实验很重要，后续 agent 不要忘记。

### 12.8 当前运行中的 GRPO run

历史正式 run：

- `run_name = 20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid`

目录：

- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid`

关键文件：

- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid/artifacts/run_args.json`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid/artifacts/training_args.json`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid/artifacts/dataset_summary.json`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid/artifacts/reward_manifest.json`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid/logs/console.log`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid/logs/metrics.jsonl`

### 12.9 当前 GRPO 训练可见状态

这是 `GRPO v0` 首轮 run 在当时交接时的可见状态：

- `torchrun` 和两个 worker 进程仍然驻留
- GPU 仍有占用
- 但 `metrics.jsonl` 目前最新可见记录停在 `step 10`
- 暂时还没看到 `eval_*` 指标落盘

最近可见 step 信号：

- `step 10`
  - `reward = 0.3019`
  - `communication_quality_reward = 0.0981`
  - `context_awareness_reward = 0.1088`
  - `hedging_reward = 0.1050`
  - `emergency_referral_reward = 0.0231`
  - `reference_alignment_reward = 0.0460`
  - `safety_penalty_reward = -0.0791`
  - `kl = 0.000352`
  - `completions/clipped_ratio = 0.5625`

重要判断：

- 训练链路已经跑通
- 但当前 run 是否正在 eval、还是卡在某一步，需要后续 agent 继续检查
- 当前最大 completion 长度 `512` 可能偏紧，用户已经明确表达后续要尽量和 `SFT/DPO` 长度分布对齐，避免大量截断

### 12.10 2026-04-14 状态刷新：full consensus 打分与 GRPO v1

以下内容是 `2026-04-14` 新增核查结果，后续 agent 需要优先以这里为准，而不是继续沿用上面的“运行中”表述。

#### 12.10.1 完整 consensus-from-full 评测状态

这两个 full generation -> consensus scoring 任务都已经完成回答生成，但打分没有完成，且当前没有活跃评测进程：

1. `SFT 5w checkpoint-75`
   - 目录：
     - `/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260413_healthbench_consensus_from_full_sft5w_dpo330_healthbench_qwen3_8b_huatuo_5w_ckpt75_gpt-52_consensus_all_seed42`
   - 当前文件状态：
     - `responses.jsonl = 3671`
     - `judgments.jsonl = 1771`
     - `summary.json` 已存在，但由于 `judgments` 未完成，不能视为最终正式结果
   - 最后日志时间：
     - `2026-04-13 13:36:30`
   - 最后可见进度：
     - `Judging example 1772/3671`

2. `DPO v2 checkpoint-330`
   - 目录：
     - `/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260413_healthbench_consensus_from_full_sft5w_dpo330_healthbench_qwen3_8b_dpo_v2_ckpt330_gpt-52_consensus_all_seed42`
   - 当前文件状态：
     - `responses.jsonl = 3671`
     - `judgments.jsonl = 2193`
   - 最后日志时间：
     - `2026-04-13 13:42:01`
   - 最后可见进度：
     - `Judging example 2194/3671`

重要提醒：

- 当前 full consensus 任务的“生成部分”已经完成，不需要重生回答
- 只需要后续基于现有 `responses.jsonl` 继续打分
- 当前没有活跃 `run_eval.py` 进程，因此如果后续要继续，应该视为“续跑评分任务”，不是“重新启动整轮 full eval”

#### 12.10.2 GRPO v0 实际结局

`GRPO v0` 首轮正式 run 最终没有正常完整结束。

关键目录：

- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid.nohup.log`

最终定位到的问题是：

- 训练确实推进到了 `step 10`
- 也确实产出了 `eval_reward` 等评估日志
- 但在 best metric / best checkpoint 处理时，`metric_for_best_model=eval_reward` 与 trainer 默认逻辑不兼容，导致 `KeyError`

修复信息：

- 代码已修：
  - `/home/qjh/llm_learning/my_medical_gpt/script/grpo/train_grpo.py`
- 对应提交：
  - `ca62a7b`
  - `fix: prevent GRPO best-metric eval crash`

结论：

- 以后不要再把 `20260413_125800...` 视为“仍在运行”
- 它是一个已经暴露并定位完工程问题的失败 run

#### 12.10.3 GRPO v1 emergency/context 数据与脚本

针对用户后来明确要求的两类优化：

- 提高 `emergency_referral` reward 权重
- 提高 `missed_emergency_penalty`
- 提高 `emergency` prompt 占比
- 略提高 `context_seeking` 占比
- 批量提速、充分利用显存

已经新增的正式脚本与文档：

- 数据构造：
  - `/home/qjh/llm_learning/my_medical_gpt/script/grpo/build_grpo_prompt_dataset.py`
  - `/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_build_grpo_prompt_dataset_v1_emergency.sh`
- 训练 launcher：
  - `/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_grpo_qwen3_8b_dpo330_v1_emergency.sh`
- 设计文档：
  - `/home/qjh/llm_learning/my_medical_gpt/docs/GRPO_V0_DESIGN.zh-CN.md`

对应提交：

- `b6d8125`
  - `feat: add GRPO emergency-context v1 pipeline`

这版的核心配置：

- 数据：
  - `v1_emergency_context`
  - `train = 3000`
  - `valid = 300`
  - `emergency = 550`
  - `context_seeking = 500`
- 训练：
  - `per_device_train_batch_size = 4`
  - `gradient_accumulation_steps = 2`
  - 双卡每步实际消费 `16` 条 prompt
  - `max_completion_length = 768`
  - `model_max_length = 2560`

#### 12.10.4 GRPO v1 三次实际启动情况

截至 `2026-04-14` 已知有 3 个 `GRPO v1 emergency/context` 运行目录：

1. `20260413_173900_qwen3-8b_dpo330_grpo_v1_emergency`
   - 目录：
     - `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_173900_qwen3-8b_dpo330_grpo_v1_emergency`
   - 状态：
     - 只写了启动头
     - 没有进入真正训练
   - 原因：
     - 当时是沙箱内的后台尝试，不应被视为正式环境成功启动

2. `20260413_174104_qwen3-8b_dpo330_grpo_v1_emergency`
   - 目录：
     - `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_174104_qwen3-8b_dpo330_grpo_v1_emergency`
   - 状态：
     - 明确失败
   - 错误：
     - `torch.distributed.DistNetworkError`
     - `The server socket has failed to bind. port: 29581 ... EPERM`
   - 含义：
     - 这是沙箱/受限环境问题，不是训练代码本身问题

3. `20260413_174259_qwen3-8b_dpo330_grpo_v1_emergency`
   - 目录：
     - `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_174259_qwen3-8b_dpo330_grpo_v1_emergency`
   - 状态：
     - 真正进入了训练
     - 产出了 artifacts、`train.log`、`metrics.jsonl`
   - 日志关键时间：
     - `2026-04-13 17:43:29`
       - `Starting GRPO training`
     - `2026-04-13 17:44:35`
       - `metrics.jsonl` 写出 `step 1`
   - `step 1` 指标：
     - `reward = 0.05795`
     - `context_awareness_reward = 0.08531`
     - `emergency_referral_reward = 0.02437`
     - `reference_alignment_reward = 0.03607`
     - `safety_penalty_reward = -0.13784`
     - `completions/mean_length = 676.69`
     - `completions/clipped_ratio = 0.75`
     - `step_time = 62.13s`
   - 当前判断：
     - 它不是“没开始”
     - 而是“已经真正进训练并写出 step 1，但没有继续稳定跑完”

#### 12.10.5 截至 2026-04-14 的结论与后续动作

截至 `2026-04-14 10:59` 左右再次核查时：

- 没有活跃 `GRPO` 训练进程
- 没有活跃 full consensus 打分进程

因此后续 agent 的优先动作应理解为：

1. full consensus 打分：
   - 先不要重生回答
   - 只在需要时基于现有 `responses.jsonl` 续跑 judge
2. GRPO：
   - 应在正式环境重新拉起 `v1 emergency/context`
   - 不能再把前两次沙箱内失败尝试当成“已经挂上后台”

推荐启动文件：

- `/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_grpo_qwen3_8b_dpo330_v1_emergency.sh`

推荐注意事项：

- 使用正式环境，不要在受限沙箱里跑 `torchrun`
- 启动时显式给一个新 `RUN_NAME`
- `MASTER_PORT` 建议避开旧值，减少端口残留冲突风险
- 启动后必须立刻核查：
  - `ps`
  - `nvidia-smi`
  - `logs/console.log`
  - `logs/metrics.jsonl`

#### 12.10.6 2026-04-14 11:08 正式环境已重新拉起 GRPO v1

在正式环境中，已重新启动一轮新的 `GRPO v1 emergency/context`：

- `run_name = 20260414_110300_qwen3-8b_dpo330_grpo_v1_emergency`
- `master_port = 29641`
- launcher：
  - `/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_grpo_qwen3_8b_dpo330_v1_emergency.sh`
- nohup 外层日志：
  - `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_110300_qwen3-8b_dpo330_grpo_v1_emergency.nohup.log`
- run 目录：
  - `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_110300_qwen3-8b_dpo330_grpo_v1_emergency`

已确认的正式启动证据：

- `torchrun` 进程存在
- 双卡 worker 已加载到 GPU
- `console.log` 已进入：
  - `Loading GRPO init adapter`
  - `Starting GRPO training`
- `W&B` 已在线同步
  - run id: `ucujxfbd`
- `metrics.jsonl` 已写出 `step 1`

`step 1` 当前记录：

- 时间：
  - `2026-04-14 11:09:56`
- 核心指标：
  - `reward = 0.05795`
  - `context_awareness_reward = 0.08531`
  - `emergency_referral_reward = 0.02437`
  - `reference_alignment_reward = 0.03607`
  - `safety_penalty_reward = -0.13784`
  - `completions/mean_length = 676.69`
  - `completions/clipped_ratio = 0.75`
  - `step_time = 60.38s`

截至这次记录时，这一轮 run 应视为：

- 已在正式环境成功启动
- 已经过了假启动阶段
- 已经进入真实训练并完成首个 step

后续 agent 如果继续接手，优先检查这个目录，而不是旧的 `20260413_*` 失败尝试目录。

---

## 13. 当前最合理的项目结论

到当前为止，比较稳妥、也最适合面试/项目汇报的结论是：

### 13.1 SFT 路线已经被证明有效

- `huatuo_5w checkpoint-75` 是强 baseline
- `HQ-50k` 给出了 `0.2905` 的当前最佳 SFT 正式结果
- 说明数据筛选路线是有价值的

### 13.2 DPO 路线不是失败，而是“数据重构后有效，但还没在外部 benchmark 上稳定超过最强 SFT”

- `DPO v1` 失败
- `DPO v2` 有明显进步
- `DPO v2 checkpoint-330` 在更偏医疗行为的维度上有正式收益
- 稳定性还优于 `huatuo_5w checkpoint-75`
- 但单轮最好分还没稳定超过最强 SFT

### 13.3 项目最大的亮点之一是“没有把 benchmark 当一次性跑分”

已经有完整证据表明：

- 做了复测
- 做了稳定性分析
- 做了 full / larger-sample eval 工程化
- 做了缓存和提速
- 主动讨论方差与结论边界

这对项目质量和面试叙事都很加分。

### 13.4 下一步主线走 GRPO 是合理的

原因：

- `DPO330` 已经证明比 `DPO30` 更贴近医疗行为强化方向
- 剩下没补齐的短板很像“reward 更适合精细定向优化”的场景
- 当前 GRPO 数据与代码已经搭好，链路也跑起来了

---

## 14. 后续最值得做的事情

如果后续 agent 需要接着推进，优先级建议如下。

### 14.1 第一优先级：确认当前 GRPO run 状态

需要确认：

- 是不是仍在正常推进
- 是否卡在 eval / save / logging
- 是否已有 `checkpoint-*`
- 是否已有 `best_checkpoint.json`
- 是否已有 `eval_reward`

如果卡住，需要：

- 先读 `console.log`
- 再读 `metrics.jsonl`
- 看 GPU 利用和进程
- 再决定是否停掉重启

### 14.2 第二优先级：如果当前 run 能训完，先做小规模验收

建议：

- 优先看 `eval_reward`
- 再看 6 个子 reward 的变化方向
- 再选 1-2 个代表性 checkpoint 拉去跑统一 `HealthBench`

注意：

- 不要随便挑 checkpoint
- 要事先定义实验意义，例如：
  - early
  - best eval_reward
  - last

### 14.3 第三优先级：如果 GRPO v0 没明显提升，优先改这几个点

1. 提高 `max_completion_length`
2. 检查是否大量 `clipped`
3. 调整 reward 权重
4. 增强 `communication_quality` 与 `global_health`
5. 必要时做结构对照：
   - `continue DPO adapter`
   - `merge DPO -> fresh GRPO LoRA`

### 14.4 第四优先级：如果 GRPO v0 方向正确，再考虑 reward 升级

升级路线：

1. 离线 `LLM judge` 小批量验证 reward 方向
2. 对小批量数据做人工/LLM 对照
3. 再蒸馏成更稳的本地 scorer

不建议一开始就把在线训练改成重度 `LLM judge`。

---

## 15. 面试 / 简历层面的沉淀

这是用户非常在意的一部分，后续 agent 回答项目叙事问题时不要忘。

### 15.1 最适合强调的亮点

- 不只是做训练，还做了评测闭环和稳定性复测
- 不只是追求“单次涨分”，而是主动验证结论稳健性
- DPO 不是一帆风顺，而是发现问题后做数据重构并证明有效
- SFT 数据筛选路线有正式小幅收益
- 当前已经进入更针对性的 GRPO 设计阶段

### 15.2 最适合讲的困难

- 原始 DPO 数据与优化目标不一致，导致 DPO v1 失败
- 小样本 benchmark 方差大，单次分差不够稳
- 训练内 best checkpoint 不一定等于外部 benchmark best checkpoint
- 评测工程早期有后台队列、失败重试、缓存命中等实际工程问题

### 15.3 当前比较稳妥的项目叙事

比起只写“提升 xx%”，更适合写成：

- 建立了面向医疗大模型的全链路对齐与评测闭环
- 完成了 SFT、偏好对齐、数据重构、稳定性复测和评测加速
- 证明了数据筛选和偏好数据重构都能带来正式 benchmark 收益
- 当前正在进一步用 GRPO 定向补齐 `communication / global_health / context / emergency` 短板

---

## 16. 后续 agent 的操作建议

### 16.1 开始任何新任务前优先检查

- 当前是否有后台长任务在跑
- 最新文档是否已经覆盖该主题
- 当前分数结论是否来自单次 run 还是多次复测

### 16.2 做代码 / 文档后要遵守

- 重要阶段完成后提交并推送仓库
- 文档优先原地更新，不要到处散落新文档
- 如果新增大改动，至少补一份中文说明

### 16.3 回答用户时注意

- 多结合本地代码、日志、真实路径
- 少讲空泛理论
- 对评测差异要明确说清“单轮 / 多轮 / 均值 / 方差”
- 对任何“最新进展”要先核日志和产物

---

## 17. 当前最重要的本地路径索引

### 17.1 项目根目录

- `/home/qjh/llm_learning/my_medical_gpt`

### 17.2 当前最关键文档

- `/home/qjh/llm_learning/my_medical_gpt/docs/AGENT_CONTEXT_HANDOFF.zh-CN.md`
- `/home/qjh/llm_learning/my_medical_gpt/docs/GRPO_V0_DESIGN.zh-CN.md`
- `/home/qjh/llm_learning/my_medical_gpt/docs/HEALTHBENCH_STABILITY_ANALYSIS.zh-CN.md`
- `/home/qjh/llm_learning/my_medical_gpt/docs/DPO_V2_HEALTHBENCH_COMPARE.zh-CN.md`
- `/home/qjh/llm_learning/my_medical_gpt/docs/HQ50K_HEALTHBENCH_COMPARE.zh-CN.md`

### 17.3 当前最关键脚本

- `/home/qjh/llm_learning/my_medical_gpt/script/grpo/train_grpo.py`
- `/home/qjh/llm_learning/my_medical_gpt/script/grpo/reward_functions.py`
- `/home/qjh/llm_learning/my_medical_gpt/script/grpo/build_grpo_prompt_dataset.py`
- `/home/qjh/llm_learning/my_medical_gpt/script/alignment/train_dpo.py`
- `/home/qjh/llm_learning/my_medical_gpt/script/alignment/reconstruct_dpo_dataset.py`
- `/home/qjh/llm_learning/my_medical_gpt/script/sft/light_quality_score.py`

### 17.4 当前最关键 run / 产物

- 当前 GRPO run
  - `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid`
- DPO v2 训练目录
  - `/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260411_090021_qwen3-8b_ckpt75_medical_pairwise_v2_dpo`
- SFT merged base
  - `/home/qjh/llm_learning/my_medical_gpt/outputs/merged_models/sft/20260410_qwen3-8b_huatuo-5w_ckpt75_merged/model`
- 重要实验快照
  - `/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_dpo_v2_ckpt330_gpt52_consensus_theme15x7`
  - `/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_huatuo5w_ckpt75_gpt52_consensus_theme15x7_seed314`
  - `/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_hq50k_best_gpt52_consensus_theme15x7`

---

## 18. 交接时的最终提醒

后续 agent 最容易犯的错有三个：

1. 把当前项目理解成“只差跑一个 GRPO 就收工”
2. 忽视评测方差，只看单次分数
3. 不看现有文档和本地产物，重复做已经完成的工作

请始终记住：

- 这个项目的价值不只是分数本身
- 还包括：
  - 数据工程
  - 对齐失败后的问题定位
  - DPO 数据重构
  - 统一 benchmark 闭环
  - 稳定性复测
  - 评测加速与缓存工程
  - 当前进入更定向的 GRPO 路线

如果后续 agent 能在这个基础上继续推进，并最终让 `GRPO` 至少在某一套统一评测口径下稳定超过当前强基线，那么项目主线会非常完整。
