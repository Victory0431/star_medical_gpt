from __future__ import annotations

import http.client
import json
import os
import re
import time
from urllib.parse import urlparse
import urllib.error
import urllib.request
from typing import Any


GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.
- If the criterion is undesirable and has negative points, still judge whether the undesirable behavior happened. A good response should therefore usually return false for that criterion.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _normalize_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid judge base_url: {base_url}")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/chat/completions"


class OpenAIHealthBenchJudge:
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 120,
        max_retries: int = 5,
    ) -> None:
        self.model_name = model_name
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("JUDGE_API_KEY")
        )
        if not self.api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required for official HealthBench scoring."
            )
        raw_base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("JUDGE_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.base_url = _normalize_chat_completions_url(raw_base_url)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def grade(self, conversation_text: str, rubric_item: str) -> dict[str, Any]:
        prompt = GRADER_TEMPLATE.replace("<<conversation>>", conversation_text).replace(
            "<<rubric_item>>", rubric_item
        )
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                content = response_payload["choices"][0]["message"]["content"]
                parsed = _extract_json_object(content)
                if "criteria_met" not in parsed:
                    raise ValueError(f"Judge response missing criteria_met: {content}")
                return {
                    "criteria_met": bool(parsed["criteria_met"]),
                    "explanation": str(parsed.get("explanation", "")),
                    "judge_model": self.model_name,
                    "judge_api_base_url": self.base_url,
                    "judge_actual_model": response_payload.get("model"),
                    "raw_response": content,
                }
            except urllib.error.HTTPError as exc:
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8")
                except Exception:
                    error_body = ""
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"OpenAI judge request failed after {self.max_retries} attempts: "
                        f"HTTP {exc.code} {exc.reason}. body={error_body}"
                    ) from exc
                time.sleep(min(2 ** attempt, 20))
            except (
                urllib.error.URLError,
                TimeoutError,
                ValueError,
                http.client.RemoteDisconnected,
                ConnectionResetError,
            ) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"OpenAI judge request failed after {self.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(min(2 ** attempt, 20))
