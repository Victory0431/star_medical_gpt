# HealthBench 基座正式基线（Theme 15 x 7）

这份记录保存了 `Qwen3-8B base` 在 `HealthBench consensus` 上的一次正式小规模分层评测结果。

运行信息：

- run name：`20260409_healthbench_base_gpt52_full_theme15x7`
- benchmark：`HealthBench consensus`
- 采样方式：`stratified_theme`
- 采样配置：`7` 个主题，每个主题 `15` 条，总计 `105` 条
- evaluated model：`Qwen3-8B base`
- judge API 形式：OpenAI 兼容 chat completions
- 请求 judge 模型：`gpt-5.2`
- 实际返回 judge 模型：`gpt-5.2`

主要结果：

- overall clipped mean：`0.2206`
- `axis:accuracy`：`0.1711`
- `axis:context_awareness`：`0.2407`
- `theme:health_data_tasks`：`0.4667`

解读：

- 这份结果比早期 `10` 条 smoke set 更稳，因为采样覆盖了全部 `7` 个主题
- 基座模型在 `health_data_tasks` 上还有一定能力
- 但在 `communication`、`context_seeking` 这类交互主题上仍明显偏弱
