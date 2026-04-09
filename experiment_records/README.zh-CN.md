# 实验记录说明

英文原版见 [README.md](/home/qjh/llm_learning/my_medical_gpt/experiment_records/README.md)。

这个目录用于存放从本地训练/评测结果里导出的轻量实验快照，适合直接纳入 Git 管理。

会跟踪的内容包括：

- run 参数
- training 参数
- 数据集统计
- metrics 历史
- 最终 train / eval 结果
- benchmark summary
- 截断后或完整的文本日志
- 规整过的 `W&B` / 系统元数据

不会放到这里的内容包括：

- 模型权重
- checkpoint 分片
- optimizer state
- tokenizer dump
- 本地 `W&B` 二进制文件

导出命令：

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py --all --force
```
