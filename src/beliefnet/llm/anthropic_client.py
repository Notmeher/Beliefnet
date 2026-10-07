"""Anthropic (Claude) client for the proxy-reviewer study.

Reads ``ANTHROPIC_API_KEY`` from the environment; the reviewer model (a Haiku
variant) is passed in. Reuses the defensive JSON extractor from the Azure client.
"""
from __future__ import annotations

import os

from beliefnet.llm.azure_client import _extract_json


class AnthropicClient:
    def __init__(self, model: str = "claude-haiku-4-5", api_key: str | None = None):
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("`pip install anthropic` to use AnthropicClient") from exc
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("set ANTHROPIC_API_KEY in the environment (.env)")
        self._client = Anthropic(api_key=api_key)
        self.model = model

    def complete_json(self, system: str, user: str, *, max_tokens: int = 512) -> dict:
        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        text = "".join(getattr(b, "text", "") for b in resp.content).strip()
        return _extract_json(text)
