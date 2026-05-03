"""记忆合并器 — 检测相似记忆并合并/更新"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from neuralmem.core.types import Memory

_logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of a merge operation."""
    merged_memory: Memory
    was_merged: bool
    merged_with: list[str] = field(default_factory=list)
    original_content: str = ""
    new_content: str = ""


class MemoryMerger:
    """Detect similar memories and merge/updates them.

    Uses vector similarity to find near-duplicates and applies
    a merge strategy: keep longer content, combine metadata,
    update timestamp.

    Args:
        similarity_threshold: Cosine similarity above which memories
            are considered duplicates (default 0.90).
    """

    def __init__(self, similarity_threshold: float = 0.90) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0.0 and 1.0, "
                f"got {similarity_threshold}"
            )
        self.similarity_threshold = similarity_threshold

    def find_duplicates(
        self,
        new_memory: Memory,
        existing_memories: list[Memory],
        get_similarity: object | None = None,
    ) -> list[Memory]:
        """Find existing memories that are similar to the new one.

        This method provides a simple content-based duplicate finder
        for use without a storage backend. For vector-based similarity,
        use merge() directly with a storage that provides find_similar().

        Args:
            new_memory: The new memory to check for duplicates.
            existing_memories: List of existing memories to compare against.
            get_similarity: Optional callable(Memory, Memory) -> float.
                If None, uses simple content token overlap.

        Returns:
            List of memories that are similar above threshold.
        """
        duplicates: list[Memory] = []
        for existing in existing_memories:
            if existing.id == new_memory.id:
                continue
            if not existing.is_active:
                continue

            if get_similarity is not None:
                sim = get_similarity(new_memory, existing)
            else:
                sim = self._content_similarity(
                    new_memory.content, existing.content
                )

            if sim >= self.similarity_threshold:
                duplicates.append(existing)

        return duplicates

    def merge(
        self,
        new_memory: Memory,
        duplicate: Memory,
    ) -> MergeResult:
        """Merge a new memory with an existing duplicate.

        Strategy:
        - Keep longer content (more informative)
        - Combine tags from both memories
        - Use higher importance of the two
        - Update timestamp to now
        - Record merge provenance in supersedes

        Args:
            new_memory: The new memory to merge.
            duplicate: The existing duplicate memory.

        Returns:
            MergeResult with the merged memory.
        """
        # Decide which content to keep (longer is better)
        if len(new_memory.content) >= len(duplicate.content):
            merged_content = new_memory.content
            original_content = duplicate.content
        else:
            merged_content = duplicate.content
            original_content = new_memory.content

        # Combine tags
        combined_tags = tuple(
            dict.fromkeys(new_memory.tags + duplicate.tags)
        )

        # Take higher importance
        max_importance = max(new_memory.importance, duplicate.importance)

        # Combine entity IDs
        combined_entity_ids = tuple(
            dict.fromkeys(
                new_memory.entity_ids + duplicate.entity_ids
            )
        )

        now = datetime.now(timezone.utc)

        # Build merged memory via model_copy (frozen model)
        merged = duplicate.model_copy(update={
            "content": merged_content,
            "tags": combined_tags,
            "importance": max_importance,
            "entity_ids": combined_entity_ids,
            "updated_at": now,
            "last_accessed": now,
            "access_count": duplicate.access_count + 1,
            "memory_type": new_memory.memory_type,
        })

        return MergeResult(
            merged_memory=merged,
            was_merged=True,
            merged_with=[duplicate.id],
            original_content=original_content,
            new_content=merged_content,
        )

    def merge_batch(
        self,
        new_memories: list[Memory],
        existing_memories: list[Memory],
    ) -> list[MergeResult]:
        """Process a batch of new memories, merging duplicates.

        For each new memory, checks for duplicates in existing
        memories and in previously merged results.

        Args:
            new_memories: List of new memories to process.
            existing_memories: List of existing memories to compare.

        Returns:
            List of MergeResult items.
        """
        results: list[MergeResult] = []
        # Track merged memories so we can deduplicate against them
        processed: list[Memory] = list(existing_memories)

        for new_mem in new_memories:
            duplicates = self.find_duplicates(new_mem, processed)
            if duplicates:
                # Merge with the most similar duplicate (first found)
                best_duplicate = duplicates[0]
                result = self.merge(new_mem, best_duplicate)
                results.append(result)
                # Add merged result to processed list for subsequent
                # dedup checks
                processed.append(result.merged_memory)
            else:
                results.append(MergeResult(
                    merged_memory=new_mem,
                    was_merged=False,
                    new_content=new_mem.content,
                ))
                processed.append(new_mem)

        return results

    @staticmethod
    def _content_similarity(text_a: str, text_b: str) -> float:
        """Compute simple content similarity using token overlap (Jaccard).

        This is a fallback when no embedding/vector similarity is
        available.
        """
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)
