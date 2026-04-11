from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _choose_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def infer_base_model_name(model_name_or_path: str | None, adapter_path: str | None) -> str:
    if model_name_or_path:
        return model_name_or_path
    if not adapter_path:
        raise ValueError("Either model_name_or_path or adapter_path must be provided")

    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json under {adapter_path}")
    payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model_name = payload.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")
    return str(base_model_name)


class HFChatGenerator:
    def __init__(
        self,
        model_name_or_path: str | None,
        adapter_path: str | None,
        cache_dir: str,
        dtype_name: str = "bfloat16",
        device: str = "cuda:0",
        enable_thinking: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        self.base_model_name_or_path = infer_base_model_name(model_name_or_path, adapter_path)
        self.adapter_path = adapter_path
        self.device = device
        self.dtype = _choose_dtype(dtype_name)
        self.enable_thinking = enable_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            dtype=self.dtype,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            # Qwen3 may carry a default top_k in generation_config, which is noisy for deterministic eval logs.
            self.model.generation_config.top_k = None

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        return self.generate_batch(
            messages_batch=[messages],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )[0]

    def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        if not messages_batch:
            return []

        prompt_texts = [self._render_prompt(messages) for messages in messages_batch]
        inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True)
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
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[:, input_width:]
        return [
            self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            for tokens in generated_tokens
        ]
