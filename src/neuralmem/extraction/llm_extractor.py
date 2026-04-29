from __future__ import annotations
import json
import logging
from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ExtractionError
from neuralmem.core.types import Entity, Relation
from neuralmem.extraction.extractor import ExtractedItem, MemoryExtractor

_logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """Extract key facts and entities from the following text.
Return JSON with: {{"facts": ["fact1", "fact2"], "entities": [{{"name": "X", "type": "person|project|technology|concept"}}]}}
Text: {text}
JSON:"""


class LLMExtractor:
    """
    Ollama LLM 增强提取器（可选）。
    TODO(week-9): 扩展支持 OpenAI / Anthropic 提供商
    """

    def __init__(self, config: NeuralMemConfig):
        self._config = config
        self._rule_extractor = MemoryExtractor(config)
        self._available: bool | None = None  # None = 未检测

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import httpx
            resp = httpx.get(f"{self._config.ollama_url}/api/tags", timeout=2.0)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
            _logger.info("Ollama not available at %s, using rule extractor.", self._config.ollama_url)
        return self._available

    def extract(self, content: str, **kwargs: object) -> list[ExtractedItem]:
        if not self._config.enable_llm_extraction or not self._check_available():
            return self._rule_extractor.extract(content, **kwargs)  # type: ignore[arg-type]
        try:
            import httpx
            prompt = _EXTRACT_PROMPT.format(text=content[:1000])
            resp = httpx.post(
                f"{self._config.ollama_url}/api/generate",
                json={"model": self._config.ollama_model, "prompt": prompt, "stream": False},
                timeout=30.0,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "{}")
            # JSON 可能被包裹在 markdown 代码块中
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            items = self._rule_extractor.extract(content, **kwargs)  # type: ignore[arg-type]
            # 合并 LLM 补充的实体
            extra_entities = [
                Entity(name=e["name"], entity_type=e.get("type", "concept"))
                for e in data.get("entities", [])
            ]
            if items and extra_entities:
                first = items[0]
                merged_entities = list({e.name: e for e in first.entities + extra_entities}.values())
                items[0] = ExtractedItem(
                    content=first.content,
                    memory_type=first.memory_type,
                    entities=merged_entities,
                    relations=first.relations,
                    tags=first.tags,
                    importance=first.importance,
                )
            return items
        except Exception as e:
            _logger.warning("LLM extraction failed (%s), falling back to rules.", e)
            return self._rule_extractor.extract(content, **kwargs)  # type: ignore[arg-type]
