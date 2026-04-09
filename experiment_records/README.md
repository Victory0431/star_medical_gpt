# Experiment Records

This directory stores lightweight, git-friendly experiment snapshots exported from local training runs.

Tracked content:

- run arguments
- training arguments
- dataset statistics
- metrics history
- final train/eval results
- benchmark summaries
- truncated or full text logs
- sanitized W&B/system metadata

Not tracked here:

- model weights
- checkpoint shards
- optimizer states
- tokenizer dumps
- local W&B binary artifacts

Generate records with:

```bash
python /home/qjh/llm_learning/my_medical_gpt/script/export_experiment_records.py --all --force
```
