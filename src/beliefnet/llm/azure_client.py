"""Azure OpenAI chat client (lazy import, environment-driven).

Reads credentials from the environment so no secret ever appears in code:

* ``AZURE_OPENAI_ENDPOINT``     - resource endpoint URL;
* ``AZURE_OPENAI_API_KEY``      - API key;
* ``AZURE_OPENAI_API_VERSION``  - optional (defaults to a recent GA version);

The deployment name (per model, e.g. a GPT-4o or GPT-5 deployment) is passed in
by the caller. Reasoning models (GPT-5 / o-series) reject ``temperature`` and
use ``max_completion_tokens``; set ``reasoning=True`` for those.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def load_dotenv() -> None:
    """Best-effort: populate os.environ from a nearby .env (no dependency)."""
    here = Path.cwd()
    for path in (here / ".env", *(p / ".env" for p in here.parents[:3])):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        return


class AzureChatClient:
    def __init__(self, deployment: str, *, api_version: str | None = None,
                 reasoning: bool = False, endpoint: str | None = None,
                 api_key: str | None = None):
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "the 'openai' package is required for AzureChatClient; "
                "install with `pip install -e \".[llm]\"`."
            ) from exc

        endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise RuntimeError(
                "set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in the "
                "environment (e.g. via .env) before creating AzureChatClient."
            )
        api_version = api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-10-21")

        self._client = AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self.deployment = deployment
        self.reasoning = reasoning

    def complete_json(self, system: str, user: str, *,
                      max_tokens: int = 1024, temperature: float = 0.0) -> dict:
        """Chat completion returning a parsed JSON object (robust to models
        that reject response_format or spend tokens on reasoning)."""
        def _call(use_response_format: bool):
            kwargs: dict = dict(
                model=self.deployment,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            if use_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            if self.reasoning:
                kwargs["max_completion_tokens"] = max(max_tokens, 2000)
            else:
                kwargs["max_tokens"] = max_tokens
                kwargs["temperature"] = temperature
            return self._client.chat.completions.create(**kwargs)

        try:
            resp = _call(True)
        except Exception:
            resp = _call(False)
        text = (resp.choices[0].message.content or "").strip()
        return _extract_json(text)


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {"_raw": text, "_parse_error": True}


# Per-model env-var registry (matches deployments on separate Azure resources).
MODEL_ENV = {
    "gpt5": dict(endpoint="AZURE_OPENAI_ENDPOINT", key="AZURE_OPENAI_API_KEY",
                 version="AZURE_OPENAI_API_VERSION",
                 deployment="AZURE_OPENAI_DEPLOYMENT", reasoning=True),
    "gpt4o": dict(endpoint="ENDPOINT", key="API", version="API_VERSION",
                  deployment="DEPLOYMENT", reasoning=False),
}


def model_available(model_key: str) -> bool:
    spec = MODEL_ENV.get(model_key)
    if not spec:
        return False
    return all(os.environ.get(spec[k]) for k in ("endpoint", "key", "deployment"))


def client_for(model_key: str) -> "AzureChatClient":
    spec = MODEL_ENV[model_key]
    return AzureChatClient(
        os.environ[spec["deployment"]],
        endpoint=os.environ[spec["endpoint"]],
        api_key=os.environ[spec["key"]],
        api_version=os.environ.get(spec["version"], "2024-12-01-preview"),
        reasoning=spec["reasoning"],
    )
