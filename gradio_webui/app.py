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
    for item in history:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
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


def build_demo(presets: list[dict[str, Any]], cache_dir: str) -> gr.Blocks:
    manager = ModelManager(cache_dir=cache_dir)
    preset_map = {item["id"]: item for item in presets}
    default_preset = presets[0]

    with gr.Blocks(title="Medical GPT Gradio WebUI") as demo:
        gr.Markdown(
            """
            # Medical GPT Gradio WebUI
            用统一页面切换代表性权重，调整推理超参数，并直接和已训练模型对话。
            页面采用“按需加载”方式：只有真正发送消息时才会加载所选模型。
            """
        )

        with gr.Row():
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
            device_box = gr.Textbox(label="设备", value="cuda:0")
            thinking_checkbox = gr.Checkbox(label="启用 thinking", value=False)

        model_info = gr.Markdown(value=format_model_info(default_preset))
        status_box = gr.Markdown(value="页面已启动，尚未加载模型。")

        system_prompt = gr.Textbox(
            label="System Prompt",
            value="你是一个专业、谨慎、清晰的医疗助手。请基于已有信息回答；信息不足时先说明不确定性并建议补充必要检查或及时就医。",
            lines=3,
        )

        with gr.Accordion("推理超参数", open=True):
            max_new_tokens = gr.Slider(label="max_new_tokens", minimum=32, maximum=1024, value=512, step=32)
            temperature = gr.Slider(label="temperature", minimum=0.0, maximum=1.5, value=0.7, step=0.05)
            top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.95, step=0.05)
            repetition_penalty = gr.Slider(label="repetition_penalty", minimum=1.0, maximum=1.3, value=1.03, step=0.01)

        chatbot = gr.Chatbot(label="对话", height=560)
        prompt_box = gr.Textbox(label="用户输入", lines=4, placeholder="输入问题后按发送。")

        with gr.Row():
            send_button = gr.Button("发送", variant="primary")
            clear_button = gr.Button("清空对话")
            unload_button = gr.Button("卸载模型")

        def on_preset_change(preset_id: str) -> str:
            return format_model_info(preset_map[preset_id])

        def on_unload() -> str:
            return manager.unload()

        def on_clear() -> tuple[list[dict[str, str]], str]:
            return [], "已清空对话历史。"

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
        ) -> tuple[list[dict[str, str]], str, str, str]:
            history = history or []
            text = user_input.strip()
            if not text:
                return history, "", "请输入问题。", format_model_info(preset_map[preset_id], manager.loaded_meta)

            preset = preset_map[preset_id]
            working_history = history[:]
            messages = build_messages(system_prompt_text, working_history)
            messages.append({"role": "user", "content": text})

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
                working_history.append({"role": "user", "content": text})
                working_history.append({"role": "assistant", "content": answer})
                status = (
                    f"已完成一次推理：`{preset['label']}` | "
                    f"max_new_tokens={int(max_new_tokens_value)} | "
                    f"temperature={float(temperature_value):.2f} | "
                    f"top_p={float(top_p_value):.2f} | "
                    f"repetition_penalty={float(repetition_penalty_value):.2f}"
                )
                return working_history, "", status, format_model_info(preset, meta)
            except Exception as exc:  # noqa: BLE001
                return history, user_input, f"推理失败：`{type(exc).__name__}: {exc}`", format_model_info(preset, manager.loaded_meta)

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
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
