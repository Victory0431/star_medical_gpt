# DPO 实验记录

这个目录存放轻量、适合 Git 跟踪的 `DPO / ORPO / RM` 训练记录快照。

建议跟踪的内容：

- `summary.json`
- 关键 `artifacts`
- `metrics.jsonl`
- `aux_eval.jsonl`
- 截断后的文本日志

不建议放在这里的内容：

- LoRA 权重
- 完整 checkpoint
- optimizer state
- 大体积缓存文件
