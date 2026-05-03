"""LLMReranker — uses any LLM (Ollama / OpenAI / Anthropic) to rerank."""
from __future__ import annotations

import logging
import re
from typing import Any

_logger = logging.getLogger(__name__)

_SCORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")

_SYSTEM_PROMPT = (
    "You are a relevance scoring assistant. "
    "Given a query and a document, output ONLY a single number between "
    "0 and 1 (inclusive) indicating how relevant the document is to the "
    "query.  0 means completely irrelevant, 1 means perfectly relevant."
)


class LLMReranker:
    """Reranker that asks an LLM to score query-document relevance.

    Supports three backends via the *provider* parameter:

    * ``"openai"``   — uses ``openai`` SDK
    * ``"anthropic"`` — uses ``anthropic`` SDK
    * ``"ollama"``    — uses ``requests`` to call a local Ollama server

    Requires the appropriate SDK to be installed.

    Usage::

        reranker = LLMReranker(
            provider="openai",
            api_key="sk-...",
            model="gpt-4o-mini",
        )
        scores = reranker.rerank("query", ["doc1", "doc2"], top_k=5)
    """

    def __init__(
        self,
        provider: str = "ollama",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> None:
        self._provider = provider.lower()
        self._api_key = api_key
        self._model = model or self._default_model()
        self._base_url = base_url
        self._extra = kwargs

    def _default_model(self) -> str:
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "ollama": "llama3",
        }
        return defaults.get(self._provider, "llama3")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Rerank *documents* against *query*.

        Returns:
            List of ``(original_index, relevance_score)`` sorted by score
            descending, limited to *top_k* results.
        """
        if not documents:
            return []

        scores: list[tuple[int, float]] = []
        for idx, doc in enumerate(documents):
            score = self._score_pair(query, doc)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _score_pair(self, query: str, doc: str) -> float:
        """Ask the LLM to score a single query-document pair."""
        user_msg = (
            f"Query: {query}\n\n"
            f"Document: {doc}\n\n"
            "Relevance score (0-1):"
        )

        try:
            raw = self._call_llm(user_msg)
            return self._parse_score(raw)
        except Exception as exc:
            _logger.warning(
                "LLM scoring failed for provider=%s: %s",
                self._provider,
                exc,
            )
            return 0.0

    def _call_llm(self, user_message: str) -> str:
        """Dispatch to the appropriate LLM provider."""
        if self._provider == "openai":
            return self._call_openai(user_message)
        if self._provider == "anthropic":
            return self._call_anthropic(user_message)
        if self._provider == "ollama":
            return self._call_ollama(user_message)
        raise ValueError(f"Unsupported provider: {self._provider}")

    def _call_openai(self, user_message: str) -> str:
        from openai import OpenAI  # type: ignore[import-untyped]

        client = OpenAI(
            api_key=self._api_key,
            **self._extra,
        )
        resp = client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            max_tokens=10,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(self, user_message: str) -> str:
        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(api_key=self._api_key)
        resp = client.messages.create(
            model=self._model,
            max_tokens=10,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text if resp.content else ""

    def _call_ollama(self, user_message: str) -> str:
        import requests  # type: ignore[import-untyped]

        url = self._base_url or "http://localhost:11434"
        resp = requests.post(
            f"{url}/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "options": {"temperature": 0.0},
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Extract a float 0–1 from raw LLM output."""
        match = _SCORE_PATTERN.search(raw)
        if match is None:
            return 0.0
        value = float(match.group(1))
        return max(0.0, min(1.0, value))
