# DPO 指标详解

这份文档专门解释当前仓库里 `DPO` 训练阶段最容易被面试官追问的几个问题：

- `chosen / rejected / margins / accuracies / loss` 到底怎么计算
- `logps` 和 `rewards` 有什么区别
- 当前项目里的 `reference model` 到底是谁
- 为什么 `pairwise accuracy` 很高，外部医疗 benchmark 仍然可能下降

如果你只想先看结论，可以先记住下面这句话：

> DPO 的核心不是“让 chosen 的概率绝对更高”，而是“让 chosen 相对于 reference 的提升，显著高于 rejected 相对于 reference 的提升”。

英文原版见 [DPO_METRICS.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_METRICS.md)。

## 1. 这次项目里的 DPO 到底在优化什么

当前训练入口是 [train_dpo.py](/home/qjh/llm_learning/my_medical_gpt/script/alignment/train_dpo.py)。

这次 run 的核心配置见 [run_args.json](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/run_args.json)：

- DPO 初始化模型：
  [run_args.json:2](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/run_args.json#L2)
  也就是 merge 后的 `SFT checkpoint-75`
- pairwise train：
  [run_args.json:3-5](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/run_args.json#L3)
- pairwise valid：
  [run_args.json:6-8](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/run_args.json#L6)
- 异构辅助验证：
  [run_args.json:9-10](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/run_args.json#L9)
- `beta=0.1`：
  [run_args.json:35](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/run_args.json#L35)
- 主 best metric：
  [run_args.json:53-54](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/run_args.json#L53)

## 2. 一个 pairwise 样本长什么样

对 DPO 来说，一个样本是三元组：

- `prompt`
- `chosen`
- `rejected`

含义是：

- 对同一个 `prompt`
- 标注者认为 `chosen` 比 `rejected` 更好

模型不是做分类，而是在比较：

- 当前策略模型 `policy`
- 对 `chosen` 的相对偏好
- 是否强于它对 `rejected` 的相对偏好

## 3. 这次项目里的 reference model 是谁

这是面试里非常容易被追问的一点。

在我们的训练代码里，`DPOTrainer` 是这样初始化的：

- [train_dpo.py:658-667](/home/qjh/llm_learning/my_medical_gpt/script/alignment/train_dpo.py#L658)

你会看到：

```python
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    ...
    peft_config=peft_config,
)
```

这里 `ref_model=None` 不代表“没有 reference”。

在 `trl` 里，如果当前模型是 `PeftModel`，且 `ref_model=None`，那么它会把“同一个底座模型在关闭当前 adapter 时的输出”当作 reference。

对应源码：

- `ref_model=None` 时用 reference 的逻辑：
  [dpo_trainer.py:929-940](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L929)
- 关闭 adapter 作为 reference 的上下文：
  [dpo_trainer.py:915-927](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L915)

所以在我们这个项目里：

- `policy model = merged SFT model + 当前 DPO LoRA`
- `reference model = 同一个 merged SFT model，但临时关闭 DPO LoRA`

这也是为什么我一直强调：

- `DPO` 不是直接从 base model 起飞
- 而是在一个已经不错的 `SFT policy` 上继续做偏好对齐

## 4. `chosen_logps / rejected_logps` 到底是什么

这是最底层的关键。

在 `trl` 的 `concatenated_forward` 里，prompt token 不参与 DPO 的损失，只有 completion token 参与。

对应源码：

- prompt mask 为 0，completion mask 为 1：
  [dpo_trainer.py:1525-1535](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L1525)
- 对 completion token 计算逐 token log prob：
  [dpo_trainer.py:1635-1639](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L1635)
- 把逐 token log prob 求和，得到序列级 `all_logps`：
  [dpo_trainer.py:1650](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L1650)
- 再拆成 `chosen_logps` 和 `rejected_logps`：
  [dpo_trainer.py:1700-1701](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L1700)

也就是说：

```text
chosen_logps
= 当前 policy 对 chosen completion 所有监督 token 的 log probability 求和

rejected_logps
= 当前 policy 对 rejected completion 所有监督 token 的 log probability 求和
```

注意两点：

1. 这是“序列级总和”，不是单 token 均值。
2. 因为 log probability 通常是负数，所以 `logps` 常常也是负的，而且回答越长，数值往往越负。

所以：

- `logps/chosen` 不能简单按绝对值和别的样本直接横向比较
- 它更适合拿来和“同一条 completion 的 reference logps”做差

## 5. DPO 的四个核心指标怎么计算

这部分最重要。

### 5.1 先定义四个 log probability

对每个样本，记：

```text
pi_c   = log π_theta(chosen | prompt)
pi_r   = log π_theta(rejected | prompt)
ref_c  = log π_ref(chosen | prompt)
ref_r  = log π_ref(rejected | prompt)
```

其中：

- `π_theta` 是当前 DPO 正在训练的 policy
- `π_ref` 是 reference policy

### 5.2 chosen / rejected reward

`trl` 里的计算公式就是：

- [dpo_trainer.py:1239-1240](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L1239)

```python
chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
rejected_rewards = beta * (rejected_logps - ref_rejected_logps)
```

写成公式：

```text
r_c = beta * (pi_c - ref_c)
r_r = beta * (pi_r - ref_r)
```

解释：

- `r_c > 0` 表示当前 policy 相比 reference，更偏好 chosen
- `r_r < 0` 表示当前 policy 相比 reference，更压低 rejected

所以面试里千万不要把它讲成“奖励模型打分”。

这里的 `reward` 不是单独 reward model 给出来的分数，而是：

- “当前 policy 相对 reference 的对数概率提升量”

### 5.3 reward margin

`trl` 里的定义：

- [dpo_trainer.py:1786-1787](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L1786)

```text
margin = chosen_rewards - rejected_rewards
```

代入上面的定义可得：

```text
margin
= beta * [(pi_c - ref_c) - (pi_r - ref_r)]
= beta * [(pi_c - pi_r) - (ref_c - ref_r)]
```

这个量最重要，因为它表示：

- 当前 policy 相比 reference
- 对 chosen 胜过 rejected 的偏好
- 到底增强了多少

### 5.4 reward accuracy

`trl` 里的定义：

- [dpo_trainer.py:1771](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L1771)

```python
reward_accuracies = (chosen_rewards > rejected_rewards).float()
```

也就是：

```text
accuracy_i = 1[r_c > r_r]
```

一个 batch 的 `rewards/accuracies`，就是把这批样本的 `0/1` 取平均。

所以：

- `accuracy = 1.0` 表示这批验证样本上，每个样本都满足 `chosen` 比 `rejected` 更受当前 policy 偏好
- 但它只说明 pairwise 排序正确
- 不说明开放式问答 benchmark 一定更强

### 5.5 DPO loss

这次我们用的是 `sigmoid` DPO loss：

- 训练参数见 [run_args.json:35-36](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/run_args.json#L35)
- 源码见 [dpo_trainer.py:1111-1115](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L1111)

其中先定义：

- [dpo_trainer.py:1089-1097](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L1089)

```text
logits = (pi_c - pi_r) - (ref_c - ref_r)
```

然后损失为：

```text
loss = -log sigma(beta * logits)
```

因为前面已经有：

```text
margin = beta * logits
```

所以也可以直接记成：

```text
loss = -log sigma(margin)
```

含义非常直观：

- `margin` 越大，说明 chosen 胜过 rejected 的幅度越大
- `sigma(margin)` 越接近 1
- `loss` 越接近 0

## 6. 一个可以手算的最小例子

假设某个样本：

```text
pi_c  = -12.0
pi_r  = -15.0
ref_c = -13.0
ref_r = -14.0
beta  = 0.1
```

那就有：

```text
r_c = 0.1 * [(-12) - (-13)] = 0.1
r_r = 0.1 * [(-15) - (-14)] = -0.1
margin = r_c - r_r = 0.2
accuracy = 1, 因为 0.1 > -0.1
loss = -log(sigmoid(0.2)) ≈ 0.598
```

可以用下面这段最小代码直接演示：

```python
import math

pi_c = -12.0
pi_r = -15.0
ref_c = -13.0
ref_r = -14.0
beta = 0.1

r_c = beta * (pi_c - ref_c)
r_r = beta * (pi_r - ref_r)
margin = r_c - r_r
accuracy = float(r_c > r_r)
loss = -math.log(1 / (1 + math.exp(-margin)))

print("chosen_reward =", r_c)
print("rejected_reward =", r_r)
print("margin =", margin)
print("accuracy =", accuracy)
print("dpo_loss =", loss)
```

输出大致是：

```text
chosen_reward = 0.1
rejected_reward = -0.1
margin = 0.2
accuracy = 1.0
dpo_loss = 0.5981388693815918
```

## 7. 这些指标在 `trl` 里是如何汇总成日志的

源码位置：

- [dpo_trainer.py:1771-1787](/home/qjh/miniconda3/envs/medicalgpt/lib/python3.11/site-packages/trl/trainer/dpo_trainer.py#L1771)

你会看到：

```python
reward_accuracies = (chosen_rewards > rejected_rewards).float()

metrics["rewards/chosen"] = gather(chosen_rewards).mean()
metrics["rewards/rejected"] = gather(rejected_rewards).mean()
metrics["rewards/accuracies"] = gather(reward_accuracies).mean()
metrics["rewards/margins"] = gather(chosen_rewards - rejected_rewards).mean()
```

也就是说日志里的这些值都是：

- 先按样本算出单条指标
- 再跨卡 gather
- 最后取 mean

所以你在 W&B 上看到的：

- `train/rewards/chosen`
- `train/rewards/rejected`
- `train/rewards/margins`
- `eval/rewards/accuracies`

本质上都是“批级均值”或“验证集均值”，不是单条样本值。

## 8. 这次 run 里的真实走势应该怎么解释

这次 run 的手工选择理由见：

- [manual_selection.json:5-19](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/manual_selection.json#L5)

你可以概括为：

- `eval_rewards/accuracies` 在 step `60` 就达到 `1.0`
- 之后 `margin` 继续上升，说明模型还在继续拉大 chosen 和 rejected 的区分
- 但异构辅助验证 `valid_zh_loss` 在 step `100` 达到最好，之后开始反弹

对应辅助验证日志：

- [aux_eval.jsonl:1-12](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/logs/aux_eval.jsonl#L1)

关键点：

- step `100`：`aux_eval/valid_zh_loss = 2.4167`
- step `110`：`2.4183`
- step `120`：`2.4288`

这说明：

- pairwise 排序目标已经很早学会了
- 后面继续训练更多是在“加大间隔”
- 但这种加大间隔不一定会转化成开放式医疗问答质量提升

这就是为什么：

- `DPO pairwise accuracy` 可以非常漂亮
- 但 `HealthBench` 正式分数仍然可能下降

## 9. 面试里最容易被追问的坑

### 9.1 `rewards/chosen` 越大是不是一定越好

不一定。

它只表示：

- 当前 policy 相比 reference
- 更偏向 chosen

如果这个偏好数据本身分布很窄，或者偏好方向和最终 benchmark 不一致，那么：

- `chosen reward` 很大
- 也不代表外部任务更强

### 9.2 `accuracy = 1.0` 为什么还会掉分

因为这个 `accuracy` 的定义只是：

```text
chosen_reward > rejected_reward
```

它只回答：

- “pairwise 验证集上排序对不对”

而不回答：

- “模型在开放式医疗问答 benchmark 上是否更安全、更完整、更有 context awareness”

### 9.3 `logps/chosen` 和 `rewards/chosen` 是一回事吗

不是。

- `logps/chosen`：当前 policy 对 chosen completion 的序列 log probability
- `rewards/chosen`：`beta * (logps_chosen - ref_logps_chosen)`

所以：

- `reward` 是“相对于 reference 的变化量”
- `logps` 是“当前模型本身的绝对序列 log probability”

### 9.4 `margin` 很大是不是一定是好事

也不一定。

`margin` 很大说明：

- 模型在当前 pairwise 数据上，把 chosen 和 rejected 拉得很开

但如果已经早早 `accuracy=1.0`，`margin` 还在持续暴涨，往往说明：

- 模型在继续强化一种狭窄偏好
- 而不是持续提升泛化质量

## 10. 面试可直接说的话

你可以这样讲：

- “DPO 的底层单位不是一个分类标签，而是一对 completion 的相对偏好。”
- “在 TRL 里，`chosen_reward = beta * (logpi_chosen - logpi_ref_chosen)`，`rejected_reward = beta * (logpi_rejected - logpi_ref_rejected)`，`margin = chosen_reward - rejected_reward`，`accuracy = 1[margin > 0]`。”
- “如果是标准 sigmoid DPO，那么单样本损失就是 `-log(sigmoid(margin))`。”
- “这些指标只能证明 pairwise 排序是否学会，不能直接替代外部 benchmark，所以我额外用了异构 `valid_zh_loss` 和 `HealthBench` 做双重验收。”

## 11. 相关代码与文档

- DPO 训练脚本：
  [train_dpo.py](/home/qjh/llm_learning/my_medical_gpt/script/alignment/train_dpo.py)
- DPO 工作流：
  [DPO_WORKFLOW.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/DPO_WORKFLOW.zh-CN.md)
- 当前 DPO run 参数：
  [run_args.json](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/run_args.json)
- 当前 DPO 手工选点说明：
  [manual_selection.json](/home/qjh/llm_learning/my_medical_gpt/outputs/dpo/20260410_qwen3_8b_dpo_medical_pairwise_v1/artifacts/manual_selection.json)
