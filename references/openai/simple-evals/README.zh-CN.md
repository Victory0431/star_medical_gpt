# simple-evals 参考说明（中文）

英文原版见 [README.md](/home/qjh/llm_learning/my_medical_gpt/references/openai/simple-evals/README.md)。

这份中文文档不是对上游 README 的逐字全文翻译，而是面向当前仓库使用场景做的中文导读。当前项目把这份 `simple-evals` 快照保存在 `references/` 下，主要是为了参考 `HealthBench` 的实现思路，而不是直接把它作为运行时代码。

## 上游仓库的核心信息

OpenAI 在 `simple-evals` 仓库中开源了一套轻量级评测库，用来提高其公开 benchmark 数字的透明度。根据上游 README 当前说明：

- `simple-evals` 自 `2025-07` 起不再继续为新模型和新榜单主动更新
- 仓库仍保留 `HealthBench`、`BrowseComp`、`SimpleQA` 等参考实现
- 该仓库不是 `openai/evals` 的替代品，而是一套更轻量、偏“参考实现 + 结果展示”的项目

## 对当前项目最有价值的部分

对 `my_medical_gpt` 来说，这个参考快照最关键的不是整张大 benchmark 榜单，而是：

- `HealthBench` 的评测思路
- rubric-based judge 的实现方式
- 基座模型生成与外部 judge 分离的工程结构

这也是为什么当前仓库把真正运行时代码放在：

- [evaluation/](/home/qjh/llm_learning/my_medical_gpt/evaluation)

而不是直接依赖 `references/` 中的文件。

## 上游仓库主要包含的评测

根据原 README，`simple-evals` 中包含或展示过的评测包括：

- `MMLU`
- `MATH`
- `GPQA`
- `DROP`
- `MGSM`
- `HumanEval`
- `SimpleQA`
- `BrowseComp`
- `HealthBench`

其中与你当前项目最相关的是：

- `HealthBench`

因为你现在要做的是医疗场景开放式回答评测，而不是纯选择题或代码题。

## 上游支持的采样接口

README 中提到他们实现过的采样接口包括：

- OpenAI API
- Claude API

使用时需要准备相应的 `*_API_KEY` 环境变量。

## 上游 README 的 setup / 运行方式

上游文档里的基本思路是：

- 按不同评测分别安装依赖
- 通过命令行枚举可评测模型
- 再指定模型名和样本数运行评测

例如上游 README 里给出的通用形式包括：

```bash
python -m simple-evals.simple_evals --list-models
python -m simple-evals.simple_evals --model <model_name> --examples <num_examples>
```

## 对你当前仓库的实际意义

你现在不需要直接照搬上游的整套工程，而是已经把最重要的部分抽出来并重构成了更适合你项目的本地版本，特点是：

- 适配本地 Hugging Face 生成
- 支持基座模型和 LoRA 统一评测
- 支持 OpenAI 兼容 judge API
- 支持 `generate_only / judge_only / full`
- 更适合后续接入 `SFT / DPO / GRPO`

## 建议如何阅读这个参考目录

推荐顺序：

1. 先看 [SOURCE.zh-CN.md](/home/qjh/llm_learning/my_medical_gpt/references/openai/simple-evals/SOURCE.zh-CN.md)
2. 再看项目自己的 [评测设计文档](/home/qjh/llm_learning/my_medical_gpt/docs/EVALUATION.zh-CN.md)
3. 如果要核对上游思路，再去看英文原始 [README.md](/home/qjh/llm_learning/my_medical_gpt/references/openai/simple-evals/README.md)
4. 最后按需查看 `healthbench_eval.py` 等参考实现

## 说明

- 由于上游 README 中包含大量模型榜单、脚注与非本项目必需内容，这里采用“中文导读”的形式，而不是全文逐字翻译。
- 如果后续你希望把参考目录也完全做成中英双语快照，可以再单独补一版完整版中文翻译。
