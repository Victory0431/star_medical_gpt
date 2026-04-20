# Medical GPT Gradio WebUI

这个目录提供一个独立的 `gradio` 推理页面，用来和项目里已经训练完成的代表性权重直接对话。

## 页面能力

- 下拉选择代表性权重
- 页面内直接切换：
  - `Qwen3-8B Base`
  - `SFT 1K Final`
  - `SFT Huatuo-5W Checkpoint-75`
  - `SFT HQ-50K Best`
  - `DPO v2 Checkpoint-330`
  - `GRPO v1 Checkpoint-60`
- 推理超参数可调：
  - `max_new_tokens`
  - `temperature`
  - `top_p`
  - `repetition_penalty`
  - `dtype`
  - `device`
  - `enable_thinking`
- 页面只在真正发送消息时加载模型，避免打开网页就立即占满显存
- 同一时刻只保留一个已加载模型；切换权重时会自动卸载上一个模型

## 目录结构

```text
gradio_webui/
  app.py
  launch.sh
  model_presets.json
  README.md
```

## 启动方式

先确保项目环境里已经安装 `gradio`：

```bash
/home/qjh/miniconda3/envs/medicalgpt/bin/pip install gradio
```

然后启动页面：

```bash
cd /home/qjh/llm_learning/my_medical_gpt
bash gradio_webui/launch.sh
```

默认地址：

- `http://127.0.0.1:7860`

如果想自定义端口：

```bash
cd /home/qjh/llm_learning/my_medical_gpt
SERVER_PORT=7861 bash gradio_webui/launch.sh
```

## 权重清单

代表性权重定义在：

- [model_presets.json](/home/qjh/llm_learning/my_medical_gpt/gradio_webui/model_presets.json)

如果后续要新增新的 checkpoint，只需要往这个文件继续追加一条预设即可。

每个预设支持两种模式：

- 直接给 `model_name_or_path`，用于 base model
- 给 `model_name_or_path + adapter_path`，用于 LoRA adapter

## 实现说明

- 模型加载方式和评测模块保持一致，使用 `transformers + peft`
- LoRA 权重通过 `PeftModel.from_pretrained(...)` 挂载
- 聊天 prompt 使用 tokenizer 自带的 `apply_chat_template(...)`
- 历史消息会被整理成标准的 `system/user/assistant` 消息列表后再生成

## 联通测试

页面联通测试建议：

```bash
cd /home/qjh/llm_learning/my_medical_gpt
SERVER_PORT=7861 bash gradio_webui/launch.sh
```

另开一个终端：

```bash
curl -I http://127.0.0.1:7861
```

返回 `HTTP/1.1 200 OK` 或等价成功状态，说明网页已经正常启动。

## 注意事项

- 首次加载某个大权重时会比较慢，这是正常现象
- 切换不同权重时会释放上一个模型，但显存回收仍可能有短暂延迟
- 页面联通不代表模型回答质量最佳，最终仍以统一评测结果为准
