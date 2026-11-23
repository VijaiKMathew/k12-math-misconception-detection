"""Unified API client for Gemini and DeepSeek models using keys from .env.

Supports:
- Gemini Flash (with 3-key round-robin rotation; retries the next key on
  429/5xx instead of failing the whole call)
- DeepSeek Chat / Reasoner (with native reasoning_content / thinking trace support)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


def load_env_vars() -> dict[str, str]:
    env_file = ROOT / ".env"
    env_vars: dict[str, str] = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip("\"'")
    return env_vars


class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        env = load_env_vars()
        self.keys: list[str] = [
            env[key]
            for key in ("GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3")
            if env.get(key)
        ]
        if not self.keys:
            raise ValueError("No GEMINI_API_KEY found in .env")
        self.model_name = model_name
        self._key_idx = 0

    def _next_key(self) -> str:
        key = self.keys[self._key_idx % len(self.keys)]
        self._key_idx += 1
        return key

    def generate(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        mime_type: str = "image/png",
        temperature: float = 0.0,
        max_attempts: int | None = None,
    ) -> str:
        parts: list[dict[str, Any]] = []
        if image_bytes is not None:
            import base64

            b64_data = base64.b64encode(image_bytes).decode("utf-8")
            parts.append({"inline_data": {"mime_type": mime_type, "data": b64_data}})
        parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": temperature},
        }
        body = json.dumps(payload).encode("utf-8")

        attempts = max_attempts if max_attempts is not None else len(self.keys)
        last_error: Exception | None = None
        for _ in range(attempts):
            key = self._next_key()
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                data=body,
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_HTTP_STATUS:
                    raise
                time.sleep(2)
        raise RuntimeError(
            f"Gemini request failed after {attempts} attempts across "
            f"{len(self.keys)} key(s): {last_error}"
        )


class DeepSeekClient:
    def __init__(self, model_name: str = "deepseek-chat") -> None:
        env = load_env_vars()
        self.api_key = env.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("No DEEPSEEK_API_KEY found in .env")
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_attempts: int = 3,
    ) -> tuple[str, str | None]:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=body,
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                msg = data["choices"][0]["message"]
                return msg.get("content", "").strip(), msg.get("reasoning_content")
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_HTTP_STATUS:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"DeepSeek request failed after {max_attempts} attempts: {last_error}")
