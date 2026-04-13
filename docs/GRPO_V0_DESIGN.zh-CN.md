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

## 9. GRPO v0 奖励函数设计

这版先不上“在线 LLM judge”，而是采用一版更稳、更便宜、能快速闭环的 `hybrid reward`：

- 主奖励：多维行为 reward
- 约束项：规则 penalty
- 方向锚点：`reference_answer / negative_reference_answer`

这不是最终形态，但非常适合你当前这个阶段：

- 先验证 `GRPO` 训练链路能不能稳定跑起来
- 先看它能不能把 `DPO v2 checkpoint-330` 的稳定短板往上拉
- 先把 reward 和 `HealthBench` 暴露出来的问题对齐

### 9.1 当前 6 个 reward 函数

#### `communication_quality_reward`

目标：

- 奖励更清晰、更有安抚性、更可执行的医学沟通

主要看：

- 回答长度是否过短
- 是否带有安抚/共情表达
- 是否给出明确下一步动作
- 是否具备一定结构化表达
- 是否只是空泛废话

#### `context_awareness_reward`

目标：

- 奖励在信息不足时先补上下文、补检查，而不是直接下结论

主要看：

- 是否主动询问关键补充信息
- 是否建议完善检查/复查
- 是否使用条件化表达
- 是否出现“没看全信息就直接断言”的情况

#### `hedging_reward`

目标：

- 奖励医疗不确定场景下的保守表达

主要看：

- 是否出现 `可能 / 需要结合 / 不能仅凭 / 还不能完全确定`
- 是否存在绝对化表达

#### `emergency_referral_reward`

目标：

- 奖励高风险样本里的急诊识别与及时转诊

主要看：

- 高风险 prompt 下是否保留 `尽快就医 / 急诊 / 立即就医 / 住院` 等建议
- 如果该转急诊却没转，会被明显扣分

#### `reference_alignment_reward`

目标：

- 用本地轻量方式把输出锚定在“更像正参考答案、远离负参考答案”的方向上

实现方式：

- 计算生成回答与 `reference_answer` 的 lexical overlap
- 计算生成回答与 `negative_reference_answer` 的 lexical overlap
- 做一个轻量 margin：
  - 靠近正例加分
  - 靠近负例扣分

它不是完整 RM，也不是 LLM judge，但对 `GRPO v0` 很有价值，因为它能防止 reward 完全漂到纯关键词投机。

#### `safety_penalty_reward`

目标：

- 对明显坏行为做负奖励

主要包括：

- 低信息、过短回答
- 过度绝对化表达
- 该给急诊建议却没给
- 明显重复、灌水

### 9.2 为什么这样设计

这版 reward 设计的核心思路是：

1. 不让 reward 完全依赖关键词
2. 也不让 reward 完全失去“方向锚点”
3. 用样本自带的 `reward_profile / penalty_profile / hard_constraints` 做 prompt 级别差异化加权

也就是说：

- `communication` 主切片会更看重沟通质量
- `emergency` 主切片会更看重急诊转诊
- `global_health` 主切片会更看重参考答案一致性与医学合理性

这比“一套固定 reward 打所有样本”更符合你这批数据的设计目标。

## 10. 当前训练实现

已经新增的训练相关脚本：

- [reward_functions.py](/home/qjh/llm_learning/my_medical_gpt/script/grpo/reward_functions.py)
- [train_grpo.py](/home/qjh/llm_learning/my_medical_gpt/script/grpo/train_grpo.py)
- [run_grpo_qwen3_8b_dpo330_v0.sh](/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_grpo_qwen3_8b_dpo330_v0.sh)

### 10.1 训练起点

当前 `GRPO v0` 默认不是从裸基座起跑，而是从：

- `huatuo_5w checkpoint-75` merged model
- 再加载 `DPO v2 checkpoint-330` adapter

也就是：

- `base = SFT 5w merged`
- `init policy = DPO v2 checkpoint-330`

这和你当前项目目标是一致的，因为我们不是要证明“GRPO 从零也能训”，而是要验证：

- 在当前最有潜力的 `DPO v2` 版本上
- 能不能进一步补齐 `communication / global_health` 等短板

### 10.2 当前默认配置

首轮配置偏保守，目标是先把链路稳定跑通：

- `2 GPU`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 8`
- `num_generations = 4`
- `max_steps = 120`
- `learning_rate = 1e-6`
- `beta = 0.02`
- `loss_type = dapo`
- `scale_rewards = group`
- `max_eval_samples = 120`

### 10.3 工程行为

当前训练脚本已经支持：

- `W&B` 记录
- 时间戳控制台日志
- `metrics.jsonl` 逐步日志落盘
- `best_checkpoint.json`
- `reward_manifest.json`
- `dataset_summary.json`
- 双卡 `torchrun` 启动

另外补了一个很实用的工程兜底：

- `GRPO` 要求全局 `train / eval batch` 与 `num_generations` 可整除
- 当前脚本会主动校验训练 batch
- 对 `eval batch` 会自动向上调整到可整除的最小值

这样可以避免夜里后台训练时，因为一个看似很小的 batch 配置细节在评估阶段直接报错。

### 10.4 当前 smoke 验证

已经做过一次超小样本 smoke：

- `train = 4`
- `valid = 4`
- `max_steps = 1`
- `num_generations = 2`

验证到的关键点：

- `SFT 5w merged + DPO v2 checkpoint-330` 可以作为 `GRPO` 初始化策略正常加载
- 自定义 `reward_functions.py` 能正常参与训练与评估
- `metrics.jsonl` 已经成功记录训练和评估 reward 指标

这说明当前最关键的链路已经打通：

- 模型加载
- adapter 接续
- group generation
- reward 计算
- 训练日志落盘

## 11. 当前定位

这版可以理解为：

- `GRPO v0`
- `rule + reference anchored reward`
- `先验证链路和方向，不宣称已经是最终 reward 方案`

如果首轮训练结果能把：

- `communication_quality`
- `global_health`

这两个当前最稳定短板明显拉上来，那这条路线就已经非常有价值。

后续再往上走，可以继续加两种更强版本：

1. `GRPO v1`
   - 在离线阶段引入 `LLM judge` 给一小批样本打多维分，验证 reward 与人工偏好方向是否一致
2. `GRPO v2`
   - 把 `LLM judge` 分数蒸馏成本地 reward scorer / RM，降低线上训练成本

## 12. 当前运行状态

`GRPO v0` 首轮正式训练已于 `2026-04-13 12:55` 启动，当前采用双卡后台运行。

本轮运行标识：

- `run_name = 20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid`
- `W&B project = my-medical-gpt-grpo`
- `W&B mode = online`

当前本地日志位置：

- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid.nohup.log`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid/logs/console.log`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid/logs/metrics.jsonl`

启动后已确认：

- 双卡 `torchrun` 进程已脱离终端独立运行
- `W&B` 已成功在线同步
- `metrics.jsonl` 已持续写入训练步指标
- 调试用 `grpo_online_probe` 已清理，当前显卡仅保留正式训练任务

当前已观测到的前两步训练信号：

- `step 1`
  - `reward = 0.3723`
  - `communication_quality_reward = 0.0813`
  - `context_awareness_reward = 0.1284`
  - `hedging_reward = 0.1578`
  - `safety_penalty_reward = -0.0468`
- `step 2`
  - `reward = 0.0934`
  - `communication_quality_reward = 0.0828`
  - `context_awareness_reward = 0.0794`
  - `reference_alignment_reward = 0.0398`
  - `safety_penalty_reward = -0.1307`

这两步样本还非常早期，暂时不能拿来判断最终效果，但已经足够说明：

- reward 函数在真实训练中已经开始分化不同 completion
- 日志、指标、W&B、双卡训练链路都已经进入稳定工作状态
- 这版 `GRPO v0` 可以继续完整跑完，再决定是否调奖励权重或扩大 prompt 集
