from __future__ import annotations

import logging
import re

from neuralmem.core.config import NeuralMemConfig
from neuralmem.core.exceptions import ExtractionError
from neuralmem.core.types import Entity, MemoryType, Relation

_logger = logging.getLogger(__name__)

# 实体类型关键词映射
_PERSON_PATTERNS = [
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # 英文大写人名
    r'(?:我|你|他|她|团队|组织|公司|老板|同事|朋友|用户)\s*[叫是：:]\s*([^\s，。！？,\.!?]{2,8})',
]
_PROJECT_PATTERNS = [
    r'(?:项目|project|repo|仓库|产品)\s*[叫是：:]\s*([^\s，。！？,\.!?]{2,20})',
    r'\b([A-Z][a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*)\b',  # CamelCase/kebab-case 技术名词
]
_TECH_KEYWORDS = {
    'python', 'typescript', 'javascript', 'go', 'rust', 'java', 'react', 'vue',
    'postgresql', 'mysql', 'redis', 'sqlite', 'docker', 'kubernetes', 'fastapi',
    'langchain', 'openai', 'claude', 'llm', 'ai', 'ml', 'api', 'mcp',
}


class ExtractedItem:
    def __init__(self, content: str, memory_type: MemoryType, entities: list[Entity],
                 relations: list[Relation], tags: list[str], importance: float,
                 event: str = "ADD"):
        self.content = content
        self.memory_type = memory_type
        self.entities = entities
        self.entity_ids = [e.id for e in entities]
        self.relations = relations
        self.tags = tags
        self.importance = importance
        self.event = event


class MemoryExtractor:
    """规则提取器 — 零 LLM 依赖，基于正则和启发式规则"""

    def __init__(self, config: NeuralMemConfig):
        self._config = config

    def extract(self, content: str, memory_type: MemoryType | None = None,
                existing_entities: list[Entity] | None = None) -> list[ExtractedItem]:
        """
        从文本中提取结构化记忆列表。
        一段文本可分割为多条独立记忆（按句号/换行分割）。
        """
        try:
            # 分割为句子（简单启发式）
            sentences = self._split_sentences(content)
            items = []
            for sent in sentences:
                if len(sent.strip()) < 5:
                    continue
                entities = self._extract_entities(sent, existing_entities or [])
                relations = self._extract_relations(sent, entities)
                tags = self._extract_tags(sent)
                mtype = memory_type or self._infer_type(sent)
                importance = self._score_importance(sent, entities)
                items.append(ExtractedItem(
                    content=sent.strip(),
                    memory_type=mtype,
                    entities=entities,
                    relations=relations,
                    tags=tags,
                    importance=importance,
                ))
            # 如果没有分割出任何有效句子，整段作为一条
            if not items:
                items.append(ExtractedItem(
                    content=content.strip(),
                    memory_type=memory_type or MemoryType.SEMANTIC,
                    entities=[],
                    relations=[],
                    tags=[],
                    importance=0.5,
                ))
            return items
        except Exception as e:
            raise ExtractionError(f"Extraction failed: {e}") from e

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r'[。！？\n.!?]', text)
        return [p.strip() for p in parts if p.strip()]

    def _extract_entities(self, text: str, existing: list[Entity]) -> list[Entity]:
        entities: list[Entity] = []
        seen_names: set[str] = {e.name.lower() for e in existing}

        # 技术关键词
        words = set(text.lower().split())
        for tech in _TECH_KEYWORDS:
            if tech in words and tech not in seen_names:
                entities.append(Entity(name=tech, entity_type="technology"))
                seen_names.add(tech)

        # 大写专有名词（英文）
        for m in re.finditer(r'\b([A-Z][a-zA-Z0-9]{2,})\b', text):
            name = m.group(1)
            if name.lower() not in seen_names and len(name) > 2:
                entities.append(Entity(name=name, entity_type="concept"))
                seen_names.add(name.lower())

        return entities[:5]  # 最多返回 5 个实体

    def _extract_relations(self, text: str, entities: list[Entity]) -> list[Relation]:
        if len(entities) < 2:
            return []
        # 简单：相邻实体在同一句中 → 建立 "co_occurs" 关系
        relations = []
        for i in range(len(entities) - 1):
            relations.append(Relation(
                source_id=entities[i].id,
                target_id=entities[i + 1].id,
                relation_type="co_occurs",
                weight=0.5,
            ))
        return relations

    def _extract_tags(self, text: str) -> list[str]:
        tags = []
        text_lower = text.lower()
        if any(w in text_lower for w in ['prefer', '偏好', '喜欢', '用', 'use']):
            tags.append('preference')
        if any(w in text_lower for w in ['project', '项目', 'repo', '仓库']):
            tags.append('project')
        if any(w in text_lower for w in ['fix', 'bug', '修复', 'issue']):
            tags.append('bug')
        if any(w in text_lower for w in ['todo', 'task', '任务', '待办']):
            tags.append('task')
        return tags

    def _infer_type(self, text: str) -> MemoryType:
        text_lower = text.lower()
        time_words = ['yesterday', 'today', 'ago', '昨天', '今天', '之前', 'happened']
        if any(w in text_lower for w in time_words):
            return MemoryType.EPISODIC
        if any(w in text_lower for w in ['how to', 'step', 'first', 'then', '步骤', '先', '然后']):
            return MemoryType.PROCEDURAL
        return MemoryType.SEMANTIC

    def _score_importance(self, text: str, entities: list[Entity]) -> float:
        score = 0.5
        score += min(len(entities) * 0.05, 0.2)
        important_words = ['important', 'critical', 'must', '重要', '关键', '必须']
        if any(w in text.lower() for w in important_words):
            score += 0.2
        return min(score, 1.0)
