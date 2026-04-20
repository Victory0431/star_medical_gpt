#!/home/qjh/miniconda3/envs/medicalgpt/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import threading
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET_PATH = Path(__file__).resolve().with_name("model_presets.json")
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"
CUSTOM_CSS = """
:root {
  --page-bg: linear-gradient(135deg, #f4efe6 0%, #fbfaf7 45%, #eef3f7 100%);
  --panel-bg: rgba(255, 255, 255, 0.92);
  --panel-border: rgba(126, 98, 68, 0.14);
  --shadow: 0 12px 40px rgba(80, 62, 42, 0.10);
  --accent: #b96b2c;
  --accent-deep: #7f4b1f;
  --soft-text: #6e645a;
}

html, body, .gradio-container {
  height: 100%;
}

body {
  overflow: hidden;
  background: var(--page-bg);
}

.gradio-container {
  max-width: 100% !important;
  padding: 14px !important;
  background: transparent !important;
}

.app-shell {
  height: calc(100vh - 28px);
  gap: 14px;
}

.sidebar-panel,
.chat-panel {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 22px;
  box-shadow: var(--shadow);
}

.sidebar-panel {
  padding: 14px !important;
}

.chat-panel {
  padding: 12px !important;
}

.sidebar-scroll {
  height: calc(100vh - 56px);
  overflow: auto;
}

.chat-stack {
  height: calc(100vh - 56px);
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.title-card {
  padding: 6px 8px 2px 8px;
}

.title-card h1 {
  margin: 0;
  font-size: 1.5rem;
  color: #35291f;
}

.title-card p {
  margin: 6px 0 0 0;
  color: var(--soft-text);
  font-size: 0.95rem;
}

.status-card {
  border-radius: 16px;
  background: rgba(242, 235, 225, 0.82);
  border: 1px solid rgba(185, 107, 44, 0.16);
  padding: 2px 12px !important;
}

.status-card p {
  margin: 6px 0 !important;
  color: #5e4b39;
  font-size: 0.92rem;
}

.chatbot-panel {
  flex: 1 1 auto;
  min-height: 0;
  border-radius: 18px !important;
}

#chatbot {
  height: 100% !important;
}

.composer-card {
  padding-top: 2px;
}

.composer-card textarea {
  min-height: 88px !important;
}

.button-row {
  gap: 8px;
}

.button-row button {
  border-radius: 14px !important;
  min-height: 42px !important;
  font-weight: 600 !important;
}

.primary-btn button {
  background: linear-gradient(135deg, var(--accent) 0%, #d98943 100%) !important;
  color: white !important;
  border: none !important;
}

.primary-btn button:hover {
  background: linear-gradient(135deg, var(--accent-deep) 0%, #c67938 100%) !important;
}

.secondary-btn button {
  background: rgba(245, 242, 236, 0.95) !important;
  color: #4a4038 !important;
  border: 1px solid rgba(120, 103, 86, 0.18) !important;
}

.meta-card {
  border-radius: 16px;
  background: rgba(249, 247, 243, 0.90);
  border: 1px solid rgba(120, 103, 86, 0.14);
  padding: 6px 12px !important;
}

.meta-card h3 {
  margin-top: 0 !important;
}
"""


def choose_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def infer_base_model_name(model_name_or_path: str | None, adapter_path: str | None) -> str:
    if model_name_or_path:
        return model_name_or_path
    if not adapter_path:
        raise ValueError("Either model_name_or_path or adapter_path must be provided.")

    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json under {adapter_path}")
    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model_name = payload.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")
    return str(base_model_name)


def load_presets(preset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    presets: list[dict[str, Any]] = []
    for item in payload:
        model_path = item.get("model_name_or_path")
        adapter_path = item.get("adapter_path")
        if model_path and not Path(model_path).exists():
            continue
        if adapter_path and not Path(adapter_path).exists():
            continue
        presets.append(item)
    if not presets:
        raise ValueError(f"No valid presets found in {preset_path}")
    return presets


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return content_to_text(content.get("text"))
        if "content" in content:
            return content_to_text(content.get("content"))
        return ""
    if isinstance(content, list):
        parts = [content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part.strip()).strip()
    return str(content).strip()


def normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = content_to_text(item.get("content"))
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    return normalized


class ModelManager:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        self.lock = threading.Lock()
        self.loaded_key: tuple[str, str, str, bool] | None = None
        self.loaded_meta: dict[str, Any] | None = None
        self.model = None
        self.tokenizer = None
        self.device = None

    def _unload_locked(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.loaded_key = None
        self.loaded_meta = None
        self.device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> str:
        with self.lock:
            self._unload_locked()
        return "已卸载当前模型。"

    def load(
        self,
        preset: dict[str, Any],
        dtype_name: str,
        device: str,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        key = (preset["id"], dtype_name, device, enable_thinking)
        with self.lock:
            if self.loaded_key == key and self.model is not None and self.tokenizer is not None:
                return {
                    "status": f"复用已加载模型：{preset['label']}",
                    "meta": self.loaded_meta,
                }

            self._unload_locked()

            adapter_path = preset.get("adapter_path")
            model_name_or_path = preset.get("model_name_or_path")
            base_model_name_or_path = infer_base_model_name(model_name_or_path, adapter_path)

            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name_or_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                use_fast=False,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                dtype=choose_dtype(dtype_name),
            )
            if adapter_path:
                model = PeftModel.from_pretrained(model, adapter_path)
            model.to(device)
            model.eval()
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.top_k = None

            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.loaded_key = key
            self.loaded_meta = {
                "label": preset["label"],
                "description": preset.get("description", ""),
                "base_model_name_or_path": base_model_name_or_path,
                "adapter_path": adapter_path,
                "dtype_name": dtype_name,
                "device": device,
                "enable_thinking": enable_thinking,
            }
            return {
                "status": f"已加载模型：{preset['label']}",
                "meta": self.loaded_meta,
            }

    def generate(
        self,
        preset: dict[str, Any],
        dtype_name: str,
        device: str,
        enable_thinking: bool,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[str, dict[str, Any]]:
        payload = self.load(preset, dtype_name, device, enable_thinking)

        prompt_text = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_width = int(inputs["input_ids"].shape[1])
        do_sample = temperature > 0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[:, input_width:]
        response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        return response, payload["meta"]


def build_messages(system_prompt: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for item in normalize_history(history):
        role = str(item.get("role", "")).strip()
        content = content_to_text(item.get("content"))
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def format_model_info(preset: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
    runtime = meta or {}
    base_model_name_or_path = runtime.get("base_model_name_or_path") or preset.get("model_name_or_path") or "由 adapter 自动推断"
    adapter_path = runtime.get("adapter_path") or preset.get("adapter_path") or "无"
    dtype_name = runtime.get("dtype_name") or "-"
    device = runtime.get("device") or "-"
    enable_thinking = runtime.get("enable_thinking")

    lines = [
        f"### 当前权重",
        f"- 名称：`{preset['label']}`",
        f"- 说明：{preset.get('description', '无')}",
        f"- base：`{base_model_name_or_path}`",
        f"- adapter：`{adapter_path}`",
        f"- dtype：`{dtype_name}`",
        f"- device：`{device}`",
    ]
    if enable_thinking is not None:
        lines.append(f"- enable_thinking：`{enable_thinking}`")
    return "\n".join(lines)


def build_status_message(message: str, tone: str = "normal") -> str:
    prefix = {
        "normal": "状态",
        "working": "运行中",
        "error": "错误",
    }.get(tone, "状态")
    return f"**{prefix}**：{message}"


def build_demo(presets: list[dict[str, Any]], cache_dir: str) -> gr.Blocks:
    manager = ModelManager(cache_dir=cache_dir)
    preset_map = {item["id"]: item for item in presets}
    default_preset = presets[0]

    with gr.Blocks(title="Medical GPT Gradio WebUI", fill_height=True) as demo:
        with gr.Row(elem_classes=["app-shell"], equal_height=True):
            with gr.Column(scale=4, min_width=330, elem_classes=["sidebar-panel", "sidebar-scroll"]):
                gr.Markdown(
                    """
                    ## 配置面板
                    页面按需加载模型。切换权重后直接发下一条消息即可自动完成卸载与重载。
                    """
                )
                preset_dropdown = gr.Dropdown(
                    label="代表性权重",
                    choices=[(item["label"], item["id"]) for item in presets],
                    value=default_preset["id"],
                )
                dtype_dropdown = gr.Dropdown(
                    label="推理精度",
                    choices=["bfloat16", "float16", "float32"],
                    value="bfloat16",
                )
                with gr.Row():
                    device_box = gr.Textbox(label="设备", value="cuda:0", scale=3)
                    thinking_checkbox = gr.Checkbox(label="thinking", value=False, scale=1)

                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="你是一个专业、谨慎、清晰的医疗助手。请基于已有信息回答；信息不足时先说明不确定性并建议补充必要检查或及时就医。",
                    lines=5,
                )

                with gr.Accordion("推理超参数", open=True):
                    max_new_tokens = gr.Slider(label="max_new_tokens", minimum=32, maximum=1024, value=512, step=32)
                    temperature = gr.Slider(label="temperature", minimum=0.0, maximum=1.5, value=0.7, step=0.05)
                    top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.95, step=0.05)
                    repetition_penalty = gr.Slider(label="repetition_penalty", minimum=1.0, maximum=1.3, value=1.03, step=0.01)

                model_info = gr.Markdown(value=format_model_info(default_preset), elem_classes=["meta-card"])

            with gr.Column(scale=8, min_width=700, elem_classes=["chat-panel", "chat-stack"]):
                gr.Markdown(
                    """
                    # Medical GPT Demo
                    <p>紧凑展示版：左侧配置，右侧对话。发送后先展示你的消息，再加载模型并生成回复。</p>
                    """,
                    elem_classes=["title-card"],
                )
                status_box = gr.Markdown(
                    value=build_status_message("页面已启动，尚未加载模型。"),
                    elem_classes=["status-card"],
                )
                chatbot = gr.Chatbot(label="对话", height="100%", elem_id="chatbot", elem_classes=["chatbot-panel"])
                prompt_box = gr.Textbox(
                    label="用户输入",
                    lines=3,
                    max_lines=4,
                    placeholder="输入问题后发送。页面会先显示你的消息，再生成模型回复。",
                    elem_classes=["composer-card"],
                )
                with gr.Row(elem_classes=["button-row"]):
                    send_button = gr.Button("发送", variant="primary", elem_classes=["primary-btn"])
                    clear_button = gr.Button("清空对话", elem_classes=["secondary-btn"])
                    unload_button = gr.Button("卸载模型", elem_classes=["secondary-btn"])

        def on_preset_change(preset_id: str) -> str:
            return format_model_info(preset_map[preset_id])

        def on_unload() -> str:
            return manager.unload()

        def on_clear() -> tuple[list[dict[str, str]], str]:
            return [], build_status_message("已清空对话历史。")

        def on_submit(
            user_input: str,
            history: list[dict[str, str]] | None,
            preset_id: str,
            dtype_name: str,
            device: str,
            enable_thinking: bool,
            system_prompt_text: str,
            max_new_tokens_value: int,
            temperature_value: float,
            top_p_value: float,
            repetition_penalty_value: float,
        ):
            history = normalize_history(history)
            text = user_input.strip()
            if not text:
                yield history, "", build_status_message("请输入问题。", tone="error"), format_model_info(
                    preset_map[preset_id], manager.loaded_meta
                )
                return

            preset = preset_map[preset_id]
            working_history = history[:] + [{"role": "user", "content": text}]
            loading_status = build_status_message(
                f"已提交消息，正在准备 `{preset['label']}` 并生成回复……",
                tone="working",
            )
            yield working_history, "", loading_status, format_model_info(preset, manager.loaded_meta)

            messages = build_messages(system_prompt_text, working_history)

            try:
                answer, meta = manager.generate(
                    preset=preset,
                    dtype_name=dtype_name,
                    device=device.strip(),
                    enable_thinking=enable_thinking,
                    messages=messages,
                    max_new_tokens=int(max_new_tokens_value),
                    temperature=float(temperature_value),
                    top_p=float(top_p_value),
                    repetition_penalty=float(repetition_penalty_value),
                )
                working_history.append({"role": "assistant", "content": answer})
                status = build_status_message(
                    f"已完成一次推理：`{preset['label']}` | "
                    f"max_new_tokens={int(max_new_tokens_value)} | "
                    f"temperature={float(temperature_value):.2f} | "
                    f"top_p={float(top_p_value):.2f} | "
                    f"repetition_penalty={float(repetition_penalty_value):.2f}",
                    tone="normal",
                )
                yield working_history, "", status, format_model_info(preset, meta)
            except Exception as exc:  # noqa: BLE001
                working_history.append({"role": "assistant", "content": f"推理失败：{type(exc).__name__}: {exc}"})
                yield (
                    working_history,
                    "",
                    build_status_message(f"推理失败：`{type(exc).__name__}: {exc}`", tone="error"),
                    format_model_info(preset, manager.loaded_meta),
                )

        preset_dropdown.change(on_preset_change, inputs=[preset_dropdown], outputs=[model_info])
        unload_button.click(on_unload, outputs=[status_box])
        clear_button.click(on_clear, outputs=[chatbot, status_box])

        send_inputs = [
            prompt_box,
            chatbot,
            preset_dropdown,
            dtype_dropdown,
            device_box,
            thinking_checkbox,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
        ]
        send_outputs = [chatbot, prompt_box, status_box, model_info]
        send_button.click(on_submit, inputs=send_inputs, outputs=send_outputs)
        prompt_box.submit(on_submit, inputs=send_inputs, outputs=send_outputs)

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio chat UI for representative Medical GPT checkpoints.")
    parser.add_argument("--preset-path", default=str(DEFAULT_PRESET_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    presets = load_presets(Path(args.preset_path))
    demo = build_demo(presets=presets, cache_dir=args.cache_dir)
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share, css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
