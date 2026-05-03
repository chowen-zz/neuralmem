"""Prompt templates for LLM-driven memory management."""
from __future__ import annotations

EXTRACTION_PROMPT = """\
You are a memory extraction system. Given the following conversation, \
extract all user-related memories as standalone facts.

For each memory, provide:
- content: The fact stated as a standalone sentence
- type: One of "fact", "preference", "episodic", "procedural"
- confidence: A float between 0.0 and 1.0

Conversation messages:
{messages}

Return a JSON array of objects:
[
  {{"content": "...", "type": "fact", "confidence": 0.9}},
  ...
]

Only extract genuinely informative memories. Skip greetings and small talk.
JSON:"""


DEDUCTION_PROMPT = """\
You are a memory deduplication system. Compare new memory candidates \
with existing memories and decide what to do with each new memory.

Operations:
- ADD: This is genuinely new information
- UPDATE: This contradicts or supersedes an existing memory \
  (provide old_memory_id)
- DELETE: This explicitly negates an existing memory \
  (provide old_memory_id)
- NOOP: This is already known or redundant

New memory candidates:
{new_memories}

Existing memories (JSON list with id and content):
{existing_memories}

Return a JSON array of operations:
[
  {{"op_type": "ADD", "content": "...", "confidence": 0.9}},
  {{"op_type": "UPDATE", "content": "...", "old_memory_id": "...", \
    "confidence": 0.8}},
  {{"op_type": "DELETE", "old_memory_id": "...", "confidence": 0.95}},
  {{"op_type": "NOOP", "content": "...", "confidence": 0.7}}
]

JSON:"""


RELATION_PROMPT = """\
Classify the relationship type for each memory.

Relation types:
- semantic: Related by meaning or topic
- spatial: Related by location or place
- temporal: Related by time or sequence of events
- causal: Related by cause and effect

Memories to classify:
{memories}

Return a JSON array:
[
  {{"memory_content": "...", "relation_type": "semantic", \
    "related_entity": "...", "confidence": 0.85}},
  ...
]

JSON:"""


CONFLICT_PROMPT = """\
Detect conflicts between two sets of memories.

New memories:
{new_memories}

Existing memories:
{existing_memories}

For each conflict found, identify:
- new_memory: The new memory text
- existing_memory_id: ID of the conflicting existing memory
- conflict_type: "contradiction", "supersession", or "negation"
- resolution: Suggested resolution ("update" or "delete")

Return a JSON array:
[
  {{"new_memory": "...", "existing_memory_id": "...", \
    "conflict_type": "contradiction", "resolution": "update"}},
  ...
]

If no conflicts, return an empty array [].
JSON:"""
