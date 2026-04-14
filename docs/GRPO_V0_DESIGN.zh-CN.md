# GRPO 设计文档

这份文档记录当前 `GRPO` 路线的第一版数据设计、奖励设计和训练实现。文档采用原地持续更新的方式维护，不再额外拆很多零散说明。

## 1. 当前阶段

当前文档同时维护两版信息：

- `GRPO v0`
  - 首轮平衡版数据、奖励和训练链路
- `GRPO v1 emergency/context`
  - 针对 `DPO v2 checkpoint-330` 的 `emergency / context` 短板做强化后的下一轮版本

当前最新准备进入正式训练的是 `GRPO v1 emergency/context`。

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

当前 `GRPO` 数据由两部分混合组成：

- `dpo_v2_train`
  - 来自重构后的 `3800` 条 `DPO v2` 训练集
  - 优点是已经带有明确的“医疗偏好差异”信息，适合补 `emergency / context / hedging`
- `hq_holdout`
  - 来自 `54,080` 条高分桶里未进入 `HQ-50k` 训练子集的剩余高质量样本
  - 优点是相对“新鲜”，没有被 `HQ-50k SFT` 直接见过，适合补 `communication / global_health`

## 4. 主切片

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

## 5. 数据规模与版本

### 5.1 v0 balanced

本地第一版平衡数据产物已经生成完成：

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

### 5.2 v1 emergency/context

这版是针对上一轮观察到的“`communication/context` 略有起色，但 `emergency_referral` 基本没起量”做的定向增强版。

正式训练集主切片分布：

- `communication = 780`
- `global_health = 620`
- `hedging = 550`
- `context_seeking = 500`
- `emergency = 550`

正式验证集主切片分布：

- `communication = 78`
- `global_health = 62`
- `hedging = 55`
- `context_seeking = 50`
- `emergency = 55`

正式训练集来源分布：

- `dpo_v2_train = 1682`
- `hq_holdout = 1318`

正式验证集来源分布：

- `dpo_v2_train = 168`
- `hq_holdout = 132`

风险等级分布：

- `train high = 1405`
- `train medium = 665`
- `train low = 930`
- `valid high = 147`
- `valid medium = 18`
- `valid low = 135`

这几个数字说明两件事：

- `emergency` 在主切片里已经被明显抬升，不再像 `v0` 一样只占 `350/3000`
- 数据来源也更偏向 `dpo_v2_train`，因为这一部分更集中携带“漏急诊 / 缺上下文 / 过度断言”的偏好错误信号

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
- [run_build_grpo_prompt_dataset_v1_emergency.sh](/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_build_grpo_prompt_dataset_v1_emergency.sh)

本地产物：

- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1/train/medical_grpo_prompt_v1.train.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1/valid/medical_grpo_prompt_v1.valid.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1/reports/medical_grpo_prompt_v1.report.json`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1_emergency_context/train/medical_grpo_prompt_v1_emergency_context.train.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1_emergency_context/valid/medical_grpo_prompt_v1_emergency_context.valid.jsonl`
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1_emergency_context/reports/medical_grpo_prompt_v1_emergency_context.report.json`

说明：

- `data/` 目录在仓库中默认不纳入版本管理
- 仓库里保留的是“可复现脚本 + 设计文档”
- 数据可随脚本重新生成

## 8. 下一步

下一步会在这份文档里继续补充：

- `GRPO v1 emergency/context` 训练记录
- 实测 `emergency_referral` 是否真正起量
- 是否需要进一步补强 `prompt/reference` 而不只是调权重

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
- [run_grpo_qwen3_8b_dpo330_v1_emergency.sh](/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_grpo_qwen3_8b_dpo330_v1_emergency.sh)

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

### 10.2.1 v1 emergency/context 配置升级

下一轮 `v1 emergency/context` 在不改变整体训练框架的前提下，主要做 4 类优化：

1. 数据切片重配
   - `emergency: 350 -> 550`
   - `context_seeking: 450 -> 500`
   - `communication/global_health` 适度回收，避免稀释高风险样本
2. reward / penalty 更激进
   - `emergency` 切片的 `emergency_referral` 权重提高到 `0.65`
   - `emergency` 切片的 `missed_emergency_penalty` 提高到 `1.8`
   - `context_seeking` 切片的 `context_awareness` 权重提高到 `0.50`
3. batch 提速
   - `per_device_train_batch_size = 4`
   - `gradient_accumulation_steps = 2`
   - 双卡下每个 optimizer step 实际消费 `2 x 4 x 2 = 16` 条 prompt
4. rollout 长度放宽
   - `max_completion_length = 768`
   - `model_max_length = 2560`

这版的意图很明确：

- 先不额外引入更复杂的 `LLM reward`
- 也先不额外补一大批“强 emergency reference”
- 先验证“仅靠样本配比 + reward 权重 + batch/长度优化”，能不能把 `emergency_referral` 真正拉起来

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

另外，`2026-04-13` 已补上一个关键工程修复：

- 之前首轮正式训练在评估保存最优模型时，因 `metric_for_best_model=eval_reward` 与 `Trainer` 默认 best-metric 行为不一致，导致在 `step 10` 后抛出 `KeyError`
- 当前 [train_grpo.py](/home/qjh/llm_learning/my_medical_gpt/script/grpo/train_grpo.py) 已改为：
  - 关闭 `load_best_model_at_end`
  - 用自定义 callback 从日志流里追踪 best metric

这意味着下一轮不会再因为这个评估收尾问题半路停掉。

### 10.5 当前参数结构到底是什么

这部分很容易让人误解，所以单独说明。

当前 `GRPO v0` 不是：

- `base + SFT LoRA + DPO LoRA`

而是：

- `SFT 5w merged backbone`
- `DPO v2 checkpoint-330 adapter`

也就是说：

- `SFT 5w` 已经先被 merge 进底座参数
- 当前在线训练时真正外挂、真正更新的 LoRA 只有一套，就是 `DPO v2 checkpoint-330`

因此这轮 `GRPO` 的本质不是“在 SFT 上重新开一套全新的 GRPO LoRA 从零学”，而是：

- 以 `DPO v2 checkpoint-330` 作为当前策略初始化
- 在这套 `DPO` adapter 上继续做 `GRPO` 优化

这么做的直接好处是：

- 起点精确等于当前最有潜力的 `DPO v2` 版本
- 不需要额外先产出一个新的 `DPO merged` 中间模型
- 工程链路更短，先验证 `GRPO` 能否补齐 `DPO v2` 暴露出的稳定短板

### 10.6 参考策略是怎么实现的

当前训练不是“没有参考策略”，而是“参考策略没有额外以一整份 8B 模型的形式重新加载到显存里”。

在当前 `PEFT + beta != 0` 的设置下，`trl` 的 `GRPOTrainer` 会：

1. 将当前可训练 adapter 视为 `default`
2. 在初始化时复制一份得到 `ref`
3. 训练过程中：
   - `default` 持续更新
   - `ref` 保持冻结
4. 计算 `KL` 时，使用同一套 backbone 上的 `ref adapter` 作为参考策略

所以当前结构可以理解为：

- `backbone = SFT merged`
- `policy adapter = DPO330 default`
- `reference adapter = DPO330 ref`

这就是为什么显存里不会再多出一整份“完整 reference model”。

### 10.7 为什么这次不先 merge DPO 再新开一套 GRPO LoRA

这是一个合理但不是唯一的设计选择。

当前版本选择“直接续训 DPO adapter”，主要是为了：

- 尽量少改工程链路，先把 `GRPO v0` 跑通
- 让参考策略精确锁定为“训练开始时的 `DPO330`”
- 直接利用 `trl` 对现有 adapter 复制 `ref` 的机制

但从实验设计上看，另一条路线同样成立：

1. 先把 `DPO330` merge 进 backbone
2. 再在 `DPO-merged backbone` 上新开一套 `GRPO LoRA`

两条路线的差异是：

- 当前路线：
  - 更像“继续微调 DPO policy”
  - 更适合快速验证方向
- 另一条路线：
  - 更像“在 DPO policy 上再叠一层 GRPO 修正量”
  - 参数职责会更清晰，更适合做后续严格 ablation

当前 `v0` 先选前者，是为了让首轮链路尽快闭环；如果 `GRPO` 证明有效，下一版完全可以补做“`merge DPO` 再新开 `GRPO LoRA`”的对照实验。

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

## 12. 运行状态

### 12.1 v0 运行回顾

`GRPO v0` 首轮正式训练已于 `2026-04-13 12:55` 启动，当前采用双卡后台运行。

本轮运行标识：

- `run_name = 20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid`
- `W&B project = my-medical-gpt-grpo`
- `W&B mode = online`

本地日志位置：

- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid.nohup.log`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid/logs/console.log`
- `/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260413_125800_qwen3-8b_dpo330_grpo_v0_train_setsid/logs/metrics.jsonl`

启动后已确认：

- 双卡 `torchrun` 进程已脱离终端独立运行
- `W&B` 已成功在线同步
- `metrics.jsonl` 已持续写入训练步指标
- 调试用 `grpo_online_probe` 已清理，当前显卡仅保留正式训练任务

已观测到的前两步训练信号：

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
- 这版 `GRPO v0` 的真正问题不在 reward 本身是否完全失效，而是在首轮配置下 `emergency` 信号仍然偏弱，且训练因为 best-metric 工程问题中断

### 12.2 v1 emergency/context 当前状态

`GRPO v1 emergency/context` 数据已于 `2026-04-13 17:34` 重建完成。

当前本地数据与脚本位置：

- [run_build_grpo_prompt_dataset_v1_emergency.sh](/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_build_grpo_prompt_dataset_v1_emergency.sh)
- [run_grpo_qwen3_8b_dpo330_v1_emergency.sh](/home/qjh/llm_learning/my_medical_gpt/script/grpo/run_grpo_qwen3_8b_dpo330_v1_emergency.sh)
- `/home/qjh/llm_learning/my_medical_gpt/data/alignment/grpo/v1_emergency_context/reports/medical_grpo_prompt_v1_emergency_context.report.json`

这版的核心优化点就是：

- 先从 `reward + data distribution` 两端同步增强 `emergency`
- 再用更大的 train batch 和更宽的 completion 上限提升训练吞吐
- 保持 `communication/global_health` 仍在数据里，但不再让它们过度稀释高风险 prompt 的训练信号

### 12.3 v1 emergency/context 正式训练结果

`GRPO v1 emergency/context` 正式双卡训练已于 `2026-04-14 20:32` 完成。

本轮运行：

- `run_name = 20260414_133600_qwen3-8b_dpo330_grpo_v1_emergency_fix1`
- [run dir](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_133600_qwen3-8b_dpo330_grpo_v1_emergency_fix1)
- [metrics.jsonl](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_133600_qwen3-8b_dpo330_grpo_v1_emergency_fix1/logs/metrics.jsonl)
- [summary.json](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_133600_qwen3-8b_dpo330_grpo_v1_emergency_fix1/artifacts/summary.json)

核心配置：

- `train prompts = 3000`
- `valid prompts = 300`
- `max_eval_samples = 120`
- `max_steps = 120`
- `per_device_train_batch_size = 4`
- `gradient_accumulation_steps = 2`
- `world_size = 2`
- `num_generations = 4`
- `max_completion_length = 768`

因此这轮的单个 `global_step` 对应：

- `2` 个 micro-batch
- `16` 个 train prompt
- 约 `64` 条 rollout completion

训练总耗时约：

- `25207.9s`
- 约 `7.0h`

#### 12.3.1 真实 best checkpoint

由于本轮训练前半段仍受旧版 callback 的 best-direction 问题影响，训练结束时自动写出的 `best_checkpoint.json` 一度错误指向较差 checkpoint。

训练结束后已按真实 `metrics.jsonl` 回填：

- `best checkpoint = checkpoint-60`
- `best eval_reward = 0.1629`

对应文件已修正为：

- [best_checkpoint.json](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_133600_qwen3-8b_dpo330_grpo_v1_emergency_fix1/artifacts/best_checkpoint.json)
- [summary.json](/home/qjh/llm_learning/my_medical_gpt/outputs/grpo/20260414_133600_qwen3-8b_dpo330_grpo_v1_emergency_fix1/artifacts/summary.json)

#### 12.3.2 eval 结果摘要

每 `10 step` 做一次 eval，关键 `eval_reward` 序列为：

- `step 10 = 0.1565`
- `step 20 = 0.1046`
- `step 30 = 0.1225`
- `step 40 = 0.1159`
- `step 50 = 0.1248`
- `step 60 = 0.1629`
- `step 70 = 0.1434`
- `step 80 = 0.1484`
- `step 90 = 0.0977`
- `step 100 = 0.1231`
- `step 110 = 0.0956`
- `step 120 = 0.1337`

从这组数据看：

- 最优点明确出现在 `step 60`
- `step 60` 之后没有继续稳定变好
- 后半程出现了明显回撤

这说明当前配置下：

- `GRPO` 不是完全学不到东西
- 但继续拉长训练并没有带来单调增益
- 首轮更像“中途能学到，后面开始过优化/漂移”

#### 12.3.3 reward 维度结论

按 `eval` 维度看，这轮最值得关注的变化是：

1. `emergency_referral`
   - 在 `step 60` 达到本轮最高：
     - `0.1366`
   - 说明这版数据与 reward 确实能把 `emergency` 信号拉起来
   - 但后期掉到 `0.1049 / 0.1014`，说明稳定保持还不够

2. `communication_quality`
   - 从早期约 `0.024~0.025` 小幅提升到：
     - `step 60 = 0.0278`
     - `step 120 = 0.0279`
   - 幅度不大，但方向偏正

3. `context_awareness`
   - 大部分时间落在 `0.040~0.045`
   - 基本横盘，没有形成明确上升趋势

4. `hedging`
   - 几乎全程稳定在 `0.040~0.041`
   - 训练端能看到 reward 抬升，但 eval 端几乎没外显收益

5. `safety_penalty`
   - `step 60 = -0.1346` 是相对较好的阶段
   - 后面恶化到 `step 90 / 110` 的 `-0.1678` 左右
   - 这和总分后半程回撤是对得上的

#### 12.3.4 这轮训练给下一步的指导

这轮首轮正式 run 最重要的结论不是“已经赢了”，而是把后续改法收窄了：

1. `checkpoint-60` 应该作为本轮正式评测入口
   - 不要默认拿 `final checkpoint-120`
   - 也不应该再把错误写出的 `checkpoint-110` 当 best

2. 下一轮不宜盲目继续拉长 step
   - 当前结果更支持：
     - 缩短到 `60~80 step`
     - 或引入更明确的 early stopping / best-checkpoint eval 策略

3. `emergency` 不是没学到，而是“拉起来了但没稳住”
   - 这更像 reward/data 强度不够稳定
   - 而不是方向完全错了

4. `context_awareness` 仍然是最顽固短板之一
   - 后续更值得加的是：
     - 更高占比的 `context_seeking` prompt
     - 更明确的“缺失关键信息时必须补问”正向 reward

5. `hedging` 已经不再是第一优先级
   - 训练端涨得最明显的是它
   - eval 端却没有明显继续受益
   - 后续不应再让它过度占 reward 预算

6. 真正需要外部验证的是：
   - `checkpoint-60`
   - `checkpoint-120`
   - 与 `DPO v2 checkpoint-330`
   - 与 `huatuo_5w checkpoint-75`
   - 在统一 `HealthBench` 口径下比较

如果 `checkpoint-60` 在外部评测上能稳定优于 `DPO330`，即使还没超过最强 `SFT`，这轮 `GRPO` 也已经证明：

- `reward` 设计不是空转
- `DPO -> GRPO` 这条链路能对开放式医疗行为继续做定向修正

#### 12.3.5 `checkpoint-60` 外部 HealthBench 105 条浅测

为尽快验证训练内最优点是否能兑现到外部 benchmark，已在 `2026-04-14 21:22 ~ 21:48` 完成一轮 `HealthBench consensus` 浅测：

- `run_name = 20260414_healthbench_grpo_v1_ckpt60_gpt52_consensus_theme15x7`
- [summary.json](/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260414_healthbench_grpo_v1_ckpt60_gpt52_consensus_theme15x7/summary.json)
- [eval.log](/home/qjh/llm_learning/my_medical_gpt/outputs/eval/20260414_healthbench_grpo_v1_ckpt60_gpt52_consensus_theme15x7/logs/eval.log)
- [config](/home/qjh/llm_learning/my_medical_gpt/evaluation/configs/healthbench_theme15_grpo_v1_ckpt60.json)

本轮口径：

- `subset = consensus`
- `sampling = stratified_theme`
- `7 themes x 15 = 105 examples`
- `judge_model = gpt-5.2`
- `generator_batch_size = 8`
- 生成阶段耗时约 `353s`
- 全流程完成时间约 `26min`

核心结果：

- `GRPO v1 checkpoint-60 = 0.3143`
- `HQ-50k best = 0.2905`
- `SFT 5w checkpoint-75 = 0.2619`
- `DPO v2 checkpoint-330 = 0.2619`
- `DPO v2 checkpoint-30 = 0.2492`
- `SFT 1k = 0.2508`
- `Base = 0.2206`

相对差值：

- 对 `HQ-50k best`：
  - `+0.0238`
- 对 `SFT 5w checkpoint-75`：
  - `+0.0524`
- 对 `DPO v2 checkpoint-330`：
  - `+0.0524`
- 对 `Base`：
  - `+0.0937`

这轮浅测里最亮眼的主题分数：

- `theme:global_health = 0.5000`
- `theme:health_data_tasks = 0.5000`
- `theme:hedging = 0.4000`
- `theme:emergency_referrals = 0.3000`

相对 `SFT 5w checkpoint-75` 的主题变化：

- `theme:emergency_referrals = +0.1333`
- `theme:communication = +0.1333`
- `theme:context_seeking = +0.0667`
- `theme:global_health = +0.0667`
- `theme:hedging = +0.0667`
- `theme:complex_responses = -0.1000`

相对 `DPO v2 checkpoint-330` 的主题变化：

- `theme:global_health = +0.2667`
- `theme:communication = +0.1000`
- `theme:health_data_tasks = +0.1000`
- `theme:context_seeking = +0.0667`
- `theme:complex_responses = -0.0333`
- `theme:emergency_referrals = -0.1333`

相对 `SFT 5w checkpoint-75` 的轴向变化：

- `axis:accuracy = +0.0899`
- `axis:context_awareness = +0.1032`
- `axis:completeness = +0.0833`
- `axis:communication_quality = -0.0444`
- `axis:instruction_following = -0.0333`

这轮外部浅测说明：

1. `checkpoint-60` 不是只在训练内 eval 看起来更好。
   它已经在一轮统一 `HealthBench` 口径下，实测超过了当前几组经典对比权重。

2. 这次 `GRPO v1` 确实补到了此前项目最想补的部分。
   特别是 `context / global_health / communication` 明显起量，说明“围绕暴露短板去组织数据与 reward”这件事是有效的。

3. `emergency` 仍不能算完全解决。
   它相对 `SFT 5w` 已经明显更强，但相对 `DPO330` 这轮 105 条浅测还没有继续抬升，说明下一轮仍应保留更高的 `emergency prompt` 占比和更强的 missed-emergency penalty。

4. 这还只是 `105` 条小样本结果。
   这轮非常适合拿来做“方向确认”和阶段结论，但还不能替代更大样本甚至 full consensus 的正式复核。

## 13. 常见问题

### 13.1 当前是规则打分还是 LLM judge 打分

当前 `GRPO v0` 是：

- `规则打分`
- `轻量 reference overlap`
- `样本级动态加权`

不是在线调用 `LLM judge API` 打分。

也就是说，当前 reward 来源于本地 `reward_functions.py` 中的 6 个 reward 函数，而不是 GPT 类评审器。

这样做的原因是：

- 先把 `GRPO` 训练链路稳定跑起来
- 控制训练成本和延迟
- 避免在线 `LLM judge` 成为训练吞吐瓶颈

后续如果 `v0` 方向成立，再升级到：

- 离线 `LLM judge` 标注小批量样本
- 蒸馏成本地 reward scorer / RM

### 13.2 动态权重是怎么实现的

动态权重不是一个额外的大模块，而是直接体现在每条样本自带的元信息里：

- `reward_profile`
- `penalty_profile`
- `slice_tags`
- `hard_constraints`
- `risk_level`

具体机制是：

1. reward 函数先判断这条样本属于什么类型
   - 例如是否属于 `context_seeking`
   - 是否属于 `emergency`
   - 是否带有“不能过度断言”“不能漏急诊”的硬约束
2. 再根据这条样本自己的 `reward_profile / penalty_profile` 调整对应 reward 的权重

例如：

- `communication` 切片会更重视 `communication_quality`
- `emergency` 切片会更重视 `emergency_referral`
- 高风险样本漏掉急诊建议时会被更重地处罚

这就是“按样本切片动态加权”的含义。

### 13.3 `max_completion_length` 为什么会导致截断

`GRPO` 和 `SFT / DPO` 的关键区别是：

- `SFT / DPO` 是离线吃现成回答
- `GRPO` 是在线生成多条 completion 再打分

因此 `GRPO` 必须给 rollout 设置 completion 上限，否则：

- 显存和训练时间会快速失控
- group generation 吞吐会明显下降

当前配置中：

- `max_completion_length = 512`

`v1 emergency/context` 正式 run 已提高到：

- `max_completion_length = 768`

如果模型在生成到当前 rollout 上限时还没有自然结束，就会被记为 `clipped`。

所以：

- `clipped_ratio` 高，不等于训练直接失效
- 但它说明当前上限可能偏紧，或者模型回答偏长

当前脚本已打开：

- `mask_truncated_completions = True`

这意味着训练时会尽量避免把“被硬截断的 completion”当成完整回答去直接强化。

后续版本如果继续观察到较高 `clipped_ratio`，优先会做两件事：

- 提高 `max_completion_length`
- 让 rollout 长度尽量和现有医疗回答分布更对齐

### 13.4 当前显卡上为什么会看到 3 个进程

这通常不是“3 个完整模型副本”，而是分布式训练的正常现象。

当前双卡训练大体由三类进程构成：

- `torchrun` 启动器父进程
- `rank 0` 训练 worker
- `rank 1` 训练 worker

在 `nvidia-smi` 里，有时还会看到某个 worker 在另一张卡上保留少量通信上下文显存，这会让进程显示得像“多了一条”。

所以看到 `3` 条 GPU 相关进程记录时，不能直接理解为“多启动了一份完整模型”。

### 13.5 DPO 为什么要先 merge SFT，而 GRPO 这里又没有先 merge DPO

这是因为两阶段想固定的“参考基线”不一样。

在当前项目里：

- `DPO` 阶段想固定的是 `SFT policy`
- `GRPO` 阶段想固定的是“训练开始时的 `DPO330 policy`”

因此：

- `DPO` 先把 `SFT` merge 到 backbone
  - 这样 `adapter off` 时就能自然回到 `SFT policy`
- `GRPO` 当前版本直接继续训练 `DPO330 adapter`
  - 这样 `ref adapter` 就能自然代表“初始 DPO330 policy”

两者并不矛盾，只是各自想保留的参考策略不同。

### 13.6 后续最值得补做的结构对照实验

如果首轮 `GRPO v0` 有正向结果，后面最值得补的结构性对照不是继续盲目调学习率，而是直接比较两条结构路线：

1. `continue DPO adapter`
   - 当前路线
2. `merge DPO -> fresh GRPO LoRA`
   - 更利于参数职责分离

这个对照实验的意义很大，因为它能直接回答：

- `GRPO` 更适合做“继续修正已有偏好策略”
- 还是更适合做“在固定 DPO 基线之上再加一层独立修正”

如果后续要把项目做得更完整，这会是一个很好的补充实验点。
