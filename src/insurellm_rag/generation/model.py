"""Single access point for local model calls (Ollama)."""

from __future__ import annotations

import logging
from typing import Protocol

from ..errors import GenerationError, ModelUnavailableError

log = logging.getLogger(__name__)


class LLMProtocol(Protocol):
    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False,
             max_tokens: int | None = None) -> str: ...


class OllamaChat:
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int,
                 num_ctx: int, timeout: float) -> None:
        self.base_url, self.model = base_url, model
        self.temperature, self.max_tokens, self.num_ctx = temperature, max_tokens, num_ctx
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.base_url, timeout=self.timeout)
        return self._client

    def installed_models(self) -> list[str]:
        try:
            resp = self.client.list()
        except Exception as exc:  # connection errors come from httpx/ollama; wrap uniformly
            raise ModelUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Start it (open the Ollama app or run "
                "'ollama serve') and try again."
            ) from exc
        models = getattr(resp, "models", None) or resp.get("models", [])
        names = []
        for m in models:
            name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
            if name:
                names.append(name)
        return names

    def check(self) -> None:
        """Raise :class:`ModelUnavailableError` with a fix-it command if unusable."""
        names = self.installed_models()
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            raise ModelUnavailableError(
                f"Model '{self.model}' is not installed in Ollama. Run: ollama pull {self.model}"
            )

    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False,
             max_tokens: int | None = None) -> str:
        try:
            resp = self.client.chat(
                model=self.model, messages=messages, format="json" if json_mode else None,
                options={"temperature": self.temperature if not json_mode else 0.0,
                         "num_predict": max_tokens or self.max_tokens, "num_ctx": self.num_ctx},
            )
        except ModelUnavailableError:
            raise
        except Exception as exc:  # ollama.ResponseError, httpx errors, timeouts
            text = str(exc).lower()
            if "not found" in text:
                raise ModelUnavailableError(
                    f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
                ) from exc
            if "timed out" in text or "timeout" in text:
                raise GenerationError(
                    f"The model did not answer within {self.timeout:.0f}s. "
                    "Try again or raise INSURELLM_REQUEST_TIMEOUT."
                ) from exc
            if "connect" in text or "refused" in text:
                raise ModelUnavailableError(
                    f"Cannot reach Ollama at {self.base_url}. Is it running?"
                ) from exc
            raise GenerationError(f"The model request failed ({exc.__class__.__name__}).") from exc
        content = getattr(getattr(resp, "message", None), "content", None)
        if content is None and isinstance(resp, dict):
            content = resp.get("message", {}).get("content")
        if not content or not content.strip():
            raise GenerationError("The model returned an empty response.")
        return content.strip()
