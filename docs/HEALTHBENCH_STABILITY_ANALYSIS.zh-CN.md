# HealthBench 稳定性分析

这份文档记录 `huatuo_5w checkpoint-75` 和 `DPO v2 checkpoint-330` 在 `HealthBench consensus` 上做第二次独立抽样复测后的结果，并回答一个更重要的问题：

- 当前能不能说 `SFT` 在外部 benchmark 上显著强于 `DPO v2`？

对应轻量评测快照见：

- [experiment_records/eval/20260411_healthbench_huatuo5w_ckpt75_gpt52_consensus_theme15x7_seed314/summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_huatuo5w_ckpt75_gpt52_consensus_theme15x7_seed314/summary.json)
- [experiment_records/eval/20260411_healthbench_dpo_v2_ckpt330_gpt52_consensus_theme15x7_seed314/summary.json](/home/qjh/llm_learning/my_medical_gpt/experiment_records/eval/20260411_healthbench_dpo_v2_ckpt330_gpt52_consensus_theme15x7_seed314/summary.json)

## 1. 复测设置

- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，共 `105` 条
- judge 模型：`gpt-5.2`
- judge API：OpenAI 兼容接口
- 复测变化：只更换抽样随机种子，从 `42` 改为 `314`
- 参与模型：
  - `Qwen3-8B + huatuo_5w LoRA (checkpoint-75)`
  - `Qwen3-8B + DPO v2 (checkpoint-330)`

这里最关键的是：

- 模型没变
- judge 没变
- 评测口径没变
- 只有抽样 seed 变了

所以这次实验不是“重新挑 checkpoint”，而是纯粹检查 `105` 条分层抽样结果到底稳不稳。

## 2. 先看两轮结果

| 模型 | seed=42 | seed=314 | 两轮均值 | 两轮标准差 |
| --- | ---: | ---: | ---: | ---: |
| `huatuo_5w checkpoint-75` | `0.2889` | `0.2619` | `0.2754` | `0.0135` |
| `DPO v2 checkpoint-330` | `0.2619` | `0.2587` | `0.2603` | `0.0016` |

两轮 gap：

- `seed=42`
  - `huatuo_5w - DPO v2 = +0.0270`
- `seed=314`
  - `huatuo_5w - DPO v2 = +0.0032`

这张表已经说明了最重要的一点：

- 第一轮看到的 `+0.0270` 差距，在第二轮并没有稳定复现
- 第二轮里两者已经接近到几乎可以视为“同一量级”

## 3. 能不能说 SFT 显著更强

当前更稳妥的结论是：

- 不能

更准确地说：

- 你仍然可以说 `huatuo_5w checkpoint-75` 保持了当前单次最好分数上限，也就是 `0.2889`
- 但你不能再把“`SFT` 明显强于 `DPO v2`”当成稳定结论来讲
- 因为在第二次独立抽样里，这个 gap 几乎收敛到了 `0.0032`

如果面试官追问“所以谁更强”，更专业的回答应该是：

- `SFT` 当前 best run 上限更高
- 但在 `theme 15 x 7 = 105` 的抽样规模下，模型排序存在明显波动
- 现阶段不能把单次 `0.02~0.03` 的差距直接当成显著优势

这其实是一个很好的工程亮点，因为它说明你没有把 benchmark 当成一次性的“跑分游戏”，而是在主动检查评测方差和结论稳健性。

## 4. 哪个模型更稳定

从两轮波动看：

- `huatuo_5w checkpoint-75`
  - 从 `0.2889` 降到 `0.2619`
  - 波动更大
- `DPO v2 checkpoint-330`
  - 从 `0.2619` 到 `0.2587`
  - 波动非常小

所以至少在这两轮结果里，可以得到一个有价值的观察：

- `DPO v2 checkpoint-330` 的外部 benchmark 表现更稳定
- `huatuo_5w checkpoint-75` 的单轮上限更高，但也更吃抽样切片

注意，这里只能说“当前观察到更稳定”，还不能说已经完成统计学意义上的稳定性证明，因为目前只有两个 seed。

## 5. 哪些维度最影响波动

### huatuo_5w 的主要波动来源

`huatuo_5w checkpoint-75` 从 `seed=42` 到 `seed=314` 的主要回撤来自：

- `Emergency referrals` (`theme:emergency_referrals`)
  - `0.4333 -> 0.1667`
- `Global health` (`theme:global_health`)
  - `0.5333 -> 0.4333`
- `axis:completeness`
  - `0.2500 -> 0.0000`
- `axis:context_awareness`
  - `0.3704 -> 0.2857`

这说明它原来领先的那部分优势，至少有一部分是建立在较小抽样切片上的。

### DPO v2 的主要波动来源

`DPO v2 checkpoint-330` 的变化明显更小，主要回撤来自：

- `Emergency referrals` (`theme:emergency_referrals`)
  - `0.4333 -> 0.2333`
- `axis:context_awareness`
  - `0.3889 -> 0.2679`

但整体总分只下降了 `0.0032`，说明它对这轮抽样变化更不敏感。

## 6. 两轮平均后怎么看

把两轮简单取平均后，可以看到更稳定的结构性信息。

### 两轮平均的 theme 对比

| theme | huatuo 两轮均值 | DPO v2 两轮均值 | 谁更高 |
| --- | ---: | ---: | --- |
| `Expertise-tailored communication` | `0.0333` | `0.0333` | 持平 |
| `Response depth` | `0.2500` | `0.2667` | `DPO v2` |
| `Context seeking` | `0.0667` | `0.1000` | `DPO v2` |
| `Emergency referrals` | `0.3000` | `0.3333` | `DPO v2` |
| `Global health` | `0.4833` | `0.2333` | `huatuo` |
| `Health data tasks` | `0.4167` | `0.4667` | `DPO v2` |
| `Responding under uncertainty` | `0.3778` | `0.3889` | `DPO v2` |

### 两轮平均的 axis 对比

| axis | huatuo 两轮均值 | DPO v2 两轮均值 | 谁更高 |
| --- | ---: | ---: | --- |
| `accuracy` | `0.2018` | `0.2053` | 接近，`DPO v2` 略高 |
| `communication_quality` | `0.3000` | `0.1889` | `huatuo` |
| `completeness` | `0.1250` | `0.1250` | 持平 |
| `context_awareness` | `0.3280` | `0.3284` | 几乎持平 |
| `instruction_following` | `0.4833` | `0.5500` | `DPO v2` |

这组平均值很值得在面试里展开，因为它说明：

- `DPO v2` 并不是“完全没打过 SFT”
- 它在 `context_seeking / emergency_referrals / health_data_tasks / instruction_following` 上已经有相当竞争力
- 当前最稳的短板其实更像是：
  - `communication_quality`
  - `Global health`

## 7. 当前最合理的工程结论

### 结论 1：不能再用“单次分差”下重结论

现在更严谨的说法应该是：

- `huatuo_5w checkpoint-75` 仍是当前单次最好 run
- 但在 `105` 条分层抽样口径下，`SFT` 对 `DPO v2` 的领先并不稳定
- 所以“`SFT` 显著强于 `DPO v2`”这个说法证据不够

### 结论 2：DPO v2 已经逼近主 SFT baseline

这次复测后，更准确的定位是：

- `DPO v2 checkpoint-330` 已经不是“明显落后”
- 它更像“已经追到主 baseline 附近，但优势/劣势分布不同”

### 结论 3：下一步主线仍然适合走 GRPO

这次稳定性分析反而让 `GRPO` 这条路更合理了，因为：

- 如果 `DPO v2` 还明显落后，主线就更像继续重构 `DPO` 数据
- 但现在它已经非常接近 `SFT`，说明偏好对齐方向本身是有潜力的
- 下一步最值得做的，是直接把 reward 设计对准当前稳定暴露的短板：
  - `communication_quality`
  - `Global health`
  - 保持并继续放大
    - `context_awareness`
    - `Responding under uncertainty`
    - `Emergency referrals`

也就是说，当前最合理的主线不是盲目继续拉长 `DPO`，而是：

- 保留 `huatuo_5w checkpoint-75` 作为强基线
- 把 `DPO v2 checkpoint-330` 视为当前最有潜力的偏好对齐版本
- 正式转入 `GRPO`，用更可控的 reward 去补它在 `Expertise-tailored communication` 侧的缺口

## 8. 后续评测该怎么升级

如果你后面想把“显著更强”讲得更有底气，最合理的升级路线是：

1. 固定同一套口径，再跑至少 `3~5` 个不同 seed
2. 报告 `mean ± std`，而不是只报单次最好分数
3. 如果预算允许，把每主题从 `15` 提高到 `20` 或 `25`
4. 把“单次 best run”与“多 seed 平均表现”分开汇报

这样你在面试里会非常像真正做过评测体系的人，而不是只会展示最好看的那一次分数。
