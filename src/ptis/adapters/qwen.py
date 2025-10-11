"""Adapter for interacting with DashScope Qwen OpenAI-compatible API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import QwenProviderConfig


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: Any

    def to_json(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class ChatCompletionRequest:
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [msg.to_json() for msg in self.messages],
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.tools is not None:
            payload["tools"] = self.tools
        if self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice
        if self.extra_body:
            payload.update(self.extra_body)
        return payload


class QwenAdapter:
    """Simple HTTP client with retry/backoff for Qwen DashScope endpoints."""

    def __init__(self, config: QwenProviderConfig) -> None:
        self._config = config

    def chat(self, request: ChatCompletionRequest, region: str = "cn") -> Dict[str, Any]:
        base_url = self._config.resolve_base_url(region)
        url = f"{base_url}/chat/completions"
        payload = json.dumps(request.to_payload()).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        attempt = 0
        backoff = 1.0
        last_error: Optional[Exception] = None
        while attempt <= self._config.max_retries:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._config.timeout_s) as response:
                    body = response.read().decode("utf-8")
                    return json.loads(body)
            except urllib.error.HTTPError as exc:  # pragma: no cover - depends on HTTP response
                if exc.code in {429, 500, 502, 503, 504}:
                    last_error = exc
                    time.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue
                raise
            except urllib.error.URLError as exc:
                last_error = exc
                time.sleep(backoff)
                backoff *= 2
                attempt += 1
        if last_error:
            raise last_error
        raise RuntimeError("Failed to execute chat request without specific error")
