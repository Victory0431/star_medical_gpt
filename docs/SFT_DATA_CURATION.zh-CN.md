# SFT 数据精筛方案与当前进展

这份文档记录当前 `SFT` 数据筛选模块的目标、方法、已完成进展，以及接下来最值得推进的方向。

## 1. 为什么要做数据精筛

当前可用的核心医疗 `SFT` 数据主要来自两份：

- `huatuo_5w`
- `huatuo_v1_226k`

两者合并后总量达到 `274,806` 条，但“更多数据”并不等于“更好的 SFT 效果”。

当前项目的目标不是训练一个只会复述医疗知识的模型，而是训练一个更接近 `HealthBench` 风格的医疗助手，因此数据筛选重点不是单纯扩容，而是：

- 降低明显低质量、低信息、模板化样本的比例
- 保留更有医疗沟通价值的样本
- 避免单一来源或单一主题在训练集中主导风格

## 2. 当前筛选方法

当前采用的是三层思路，但目前实际已经落地前两层加一个轻量质量分层。

### 第一层：规则粗筛

脚本：

- [filter_sft_rules.py](/home/qjh/llm_learning/my_medical_gpt/script/sft/filter_sft_rules.py)

作用：

- 去掉明显短回答
- 去掉异常长问题/回答
- 去掉轮次异常
- 去掉泛化性极差的低信号回答
- 去掉部分明显泛化“去医院检查/咨询医生”但几乎不提供任何信息的样本

这一步的目标不是“选精品”，而是做最便宜的第一层脏数据清洗。

### 第二层：轻量 embedding 分布分析

脚本：

- [analyze_sft_distribution.py](/home/qjh/llm_learning/my_medical_gpt/script/sft/analyze_sft_distribution.py)

当前实现不是神经网络 embedding，而是更轻量、可本地快速跑通的方案：

- `char TF-IDF`
- `SVD` 降维
- `MiniBatchKMeans` 聚类

作用：

- 查看合并数据是否出现头部 cluster 过大
- 查看不同来源在 cluster 中是否严重失衡
- 输出代表样本，帮助人工判断分布差异

它不直接判断医学回答对不对，但很适合判断：

- 数据是否重复
- 数据是否被某类表达风格主导
- 是否需要 cluster 限额抽样

### 第三层：轻量 quality score

脚本：

- [light_quality_score.py](/home/qjh/llm_learning/my_medical_gpt/script/sft/light_quality_score.py)

当前没有训练专门打分模型，而是先采用启发式特征。

核心特征包括：

- 回答长度是否落在合理区间
- 问题长度是否异常
- 回答是否有结构化表达
- 是否有补充检查/就医/复查类提示
- 是否有适度 hedging 表达
- 是否存在低信息废话
- 是否存在过度绝对化用语
- 高风险问题是否缺少就医/急诊提示

输出不是“硬删”，而是三桶分流：

- `high_confidence_keep`
- `borderline_review`
- `high_risk_drop`

这一步的定位是：

- 先做便宜的高召回预筛
- 不直接替代更强 judge
- 为后续 `HQ-30k / HQ-50k` 构建提供候选池

## 3. 当前已完成结果

### 3.1 规则粗筛结果

输入：

- `huatuo_5w`
- `huatuo_v1_226k`

总计：

- 原始样本：`274,806`
- 粗筛保留：`274,073`
- 过滤掉：`733`
- 保留率：`99.73%`

结论：

- 当前规则粗筛主要在做“清洗脏样本”
- 还远远不是“精选高质量子集”

这说明当前数据本身并不是充满明显坏样本，而是更多存在“中等质量但风格和安全性不完全一致”的问题。

### 3.2 分布分析结果

采样分析规模：

- `30,000`

结果：

- 最大 cluster 占比：`7.49%`
- 前 5 个 cluster 占比：`30.73%`

这说明当前没有特别夸张的“单一主题大爆炸”现象。

但来源分布失衡明显：

- `huatuo_v1_226k`：`82.16%`
- `huatuo_5w`：`17.84%`

更重要的是，cluster 代表样本显示两份数据并不只是“文件不同”，而是回答风格和质量倾向存在差异：

- `5w` 更像较规整的 instruction 风格
- `22w` 中存在更多旧式问答站风格、模板化回答和不够严谨的医疗表达

因此当前更值得控制的是：

- 来源对整体训练风格的主导权

而不是机械要求文件比例对半。

### 3.3 轻量质量评分结果

总计：

- `high_confidence_keep`：`54,080`
- `borderline_review`：`216,968`
- `high_risk_drop`：`3,025`

占比：

- 高置信保留：`19.73%`
- 中间待复核：`79.16%`
- 高风险低优先级：`1.10%`

按来源看：

- `huatuo_5w`
  - 平均分：`0.6743`
  - 高分桶占比：`37.20%`
  - 低分桶占比：`0.12%`
- `huatuo_v1_226k`
  - 平均分：`0.6369`
  - 高分桶占比：`15.84%`
  - 低分桶占比：`1.32%`

这个结果非常关键：

- `5w` 不只是样本量小，它在当前启发式质量分上也明显更强
- `22w` 不适合直接视为“和 5w 同质，只是更大”
- 因此后续不能简单地把 `27w` 全量直接拿去当正式 SFT 主训练集

## 4. 当前方法的优点与边界

### 优点

- 完全本地可跑，不依赖额外 API
- 成本低，适合快速建立数据漏斗
- 对明显脏样本、低信息样本、结构差样本有较强过滤作用
- 已经能初步证明 `5w` 与 `22w` 的质量差异

### 边界

- 当前 quality score 不是医学事实核验器
- 不能替代更强 judge 对复杂医疗安全问题的细粒度判断
- 会误伤一部分“短但质量高”的回答
- 会保留大量“尚可但不够优”的中间样本

因此当前更适合的定位是：

- 第一版工业化预筛模块
- 不是最终精选器

## 5. 目前最推荐的下一步

### 方案 A：直接构造可训练子集

基于当前结果直接导出：

- `hq_30k`
- `hq_50k`
- `diverse_50k`

做法：

- 先从 `high_confidence_keep` 取主池
- 再从 `borderline_review` 中按 `source + cluster` 配额补齐

这条线最适合尽快进入 SFT 对比实验。

当前已经实际导出的可训练子集：

- [hq_50k_source_stratified.jsonl](/home/qjh/llm_learning/my_medical_gpt/data/sft/curation/subsets/hq_50k_source_stratified.jsonl)
  - 规模：`50,000`
  - 策略：从 `high_confidence_keep` 中按来源占比抽取
  - 来源构成：
    - `huatuo_v1_226k`：`32,809`
    - `huatuo_5w`：`17,191`
- [hq_54k_high_bucket_all.jsonl](/home/qjh/llm_learning/my_medical_gpt/data/sft/curation/subsets/hq_54k_high_bucket_all.jsonl)
  - 规模：`54,080`
  - 策略：直接使用 `high_confidence_keep` 全量
  - 来源构成：
    - `huatuo_v1_226k`：`35,487`
    - `huatuo_5w`：`18,593`

推荐 SFT 对比实验口径：

1. `huatuo_5w`
   - 当前正式 baseline
2. `hq_50k_source_stratified`
   - 更公平地检验“数据筛选是否优于原始 5w”
   - 样本数和 `5w` 基本一致，更适合做主结论
3. `hq_54k_high_bucket_all`
   - 检验“把当前高分桶全拿来训练，实际最佳效果能到哪里”
   - 这组更偏实用最佳实践，不适合作为最严格公平对照

### 第一轮正式结果已落地

`hq_50k_source_stratified` 已经完成一轮正式 `SFT + HealthBench` 验证：

- 训练 run：
  - [20260410_234458_qwen3-8b_hq-50k_lora_eval](/home/qjh/llm_learning/my_medical_gpt/outputs/sft/20260410_234458_qwen3-8b_hq-50k_lora_eval)
- 对应正式评测结果：
  - `HQ-50k best = 0.2905`
  - `huatuo_5w checkpoint-75 = 0.2889`

当前能下的最稳妥结论是：

- `HQ-50k` 已经显示出小幅正式优势
- 说明轻量数据筛选路线是有效的
- 但领先幅度不大，因此更适合说“略优于原始 5w”，而不是“显著优于原始 5w”

专项分析见：

- [HQ50K_HEALTHBENCH_COMPARE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/docs/HQ50K_HEALTHBENCH_COMPARE.zh-CN.md)

### 方案 B：引入更强 judge 做第二轮精筛

不是对 `27w` 全量打分，而是：

- 只对候选 `2~5 万` 样本做更强 judge 评分
- 构建更强版本 `HQ-30k / HQ-50k`

这条线更贵，但更接近正式精选数据集构建。

### 方案 C：训练小型质量打分器

先用更强模型给一小批样本打标签，再训练一个小 scorer：

- 用于全量 SFT 数据打分
- 后续也能复用于 DPO / RM / GRPO

这条线更有研究和工程亮点，但当前不是必须立刻走。

## 6. 复现命令

### 规则粗筛

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/script/sft/filter_sft_rules.py \
  --input-files \
  /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/train/huatuo_5w.processed.jsonl \
  /home/qjh/llm_learning/my_medical_gpt/data/sft/processed/train/huatuo_v1_226k.processed.jsonl \
  --output-root /home/qjh/llm_learning/my_medical_gpt/data/sft/curation \
  --output-name medical_sft_huatuo_27w_rule_filtered \
  --keep-rejected
```

### 分布分析

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/script/sft/analyze_sft_distribution.py \
  --input-file /home/qjh/llm_learning/my_medical_gpt/data/sft/curation/rule_filtered/medical_sft_huatuo_27w_rule_filtered.jsonl \
  --output-root /home/qjh/llm_learning/my_medical_gpt/data/sft/curation \
  --output-name medical_sft_huatuo_27w_rule_filtered \
  --sample-size 30000 \
  --max-features 12000 \
  --svd-dim 64 \
  --n-clusters 24 \
  --representatives-per-cluster 4
```

### 轻量质量打分

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/script/sft/light_quality_score.py \
  --input-file /home/qjh/llm_learning/my_medical_gpt/data/sft/curation/rule_filtered/medical_sft_huatuo_27w_rule_filtered.jsonl \
  --output-root /home/qjh/llm_learning/my_medical_gpt/data/sft/curation \
  --output-name medical_sft_huatuo_27w_rule_filtered \
  --high-threshold 0.72 \
  --low-threshold 0.45 \
  --sample-per-bucket 20
```

### 导出可训练子集

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/script/sft/build_curation_subset.py \
  --input-file /home/qjh/llm_learning/my_medical_gpt/data/sft/curation/quality/medical_sft_huatuo_27w_rule_filtered.scored_all.jsonl \
  --output-root /home/qjh/llm_learning/my_medical_gpt/data/sft/curation/subsets \
  --output-name hq_50k_source_stratified \
  --target-size 50000 \
  --strategy source_stratified_high
```

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/python \
  /home/qjh/llm_learning/my_medical_gpt/script/sft/build_curation_subset.py \
  --input-file /home/qjh/llm_learning/my_medical_gpt/data/sft/curation/quality/medical_sft_huatuo_27w_rule_filtered.scored_all.jsonl \
  --output-root /home/qjh/llm_learning/my_medical_gpt/data/sft/curation/subsets \
  --output-name hq_54k_high_bucket_all \
  --target-size 54080 \
  --strategy high_bucket_only
```

### 启动新的 SFT 对比实验

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/sft/run_sft_qwen3_8b_huatuo_5w.sh
```

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/sft/run_sft_qwen3_8b_hq_50k.sh
```

```bash
bash /home/qjh/llm_learning/my_medical_gpt/script/sft/run_sft_qwen3_8b_hq_54k.sh
```

## 7. 当前工程判断

当前最合理的主张不是：

- “我把 27 万数据全喂进去就行。”

而是：

- “我先搭建了规则粗筛、分布分析和轻量质量打分三层漏斗，确认 5w 与 22w 在质量与风格上并不完全同分布，因此下一步会基于 `source + cluster + quality score` 继续构造更高质量、更平衡的医疗 SFT 子集，再做正式训练对比。”

这会比单纯堆数据量更接近真实工业项目。
