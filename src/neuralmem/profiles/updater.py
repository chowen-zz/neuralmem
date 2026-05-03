"""画像更新器 — ProfileUpdater: 增量更新用户模型.

支持基于新记忆的增量画像更新，包括置信度衰减、冲突解决和版本管理。
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from neuralmem.core.types import Memory
from neuralmem.profiles.base import ProfileAttribute, UserProfile
from neuralmem.profiles.engine import ProfileEngine
from neuralmem.profiles.types import ProfileDimension

_logger = logging.getLogger(__name__)


class ProfileUpdater:
    """增量画像更新器.

    基于新到达的记忆增量更新用户画像，支持：
    - 属性置信度的时间衰减
    - 新旧属性冲突解决
    - 画像版本快照
    - 批量更新优化

    Args:
        engine: ProfileEngine 实例用于推断新属性.
        decay_half_life: 置信度衰减半衰期（天），默认 30 天.
        min_confidence: 属性保留的最小置信度，低于此值会被移除.
    """

    def __init__(
        self,
        engine: ProfileEngine,
        decay_half_life: float = 30.0,
        min_confidence: float = 0.15,
    ) -> None:
        self.engine = engine
        self.decay_half_life = max(1.0, decay_half_life)
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self._profiles: dict[str, dict[str, ProfileAttribute]] = {}
        self._history: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ #
    # Core update methods
    # ------------------------------------------------------------------ #

    def update(
        self,
        user_id: str,
        new_memories: Sequence[Memory],
        existing_profile: dict[str, ProfileAttribute] | None = None,
    ) -> dict[str, ProfileAttribute]:
        """Incrementally update a user profile with new memories.

        Args:
            user_id: The user identifier.
            new_memories: Newly arrived memories to analyze.
            existing_profile: Optional existing profile attributes to merge with.

        Returns:
            Updated profile attributes dict.

        Raises:
            ValueError: If new_memories is empty.
        """
        if not new_memories:
            raise ValueError("new_memories must not be empty for update")

        _logger.info(
            "Updating profile for user %s with %d new memories",
            user_id,
            len(new_memories),
        )

        # Infer new attributes from fresh memories
        try:
            new_attrs = self.engine.build_profile(user_id, new_memories)
        except Exception as exc:
            _logger.error("Failed to infer new attributes: %s", exc)
            raise RuntimeError(f"Profile inference failed: {exc}") from exc

        # Get existing profile
        current = existing_profile or self._profiles.get(user_id, {})

        # Apply decay to existing attributes
        decayed = self._apply_decay(current)

        # Merge new attributes with decayed existing ones
        merged = self._merge_attributes(decayed, new_attrs)

        # Store updated profile
        self._profiles[user_id] = merged

        # Record update history
        self._record_history(user_id, new_attrs, merged)

        _logger.info(
            "Profile updated for user %s: %d attributes",
            user_id,
            len(merged),
        )
        return merged

    def update_single_dimension(
        self,
        user_id: str,
        new_memories: Sequence[Memory],
        dimension: ProfileDimension,
        existing_profile: dict[str, ProfileAttribute] | None = None,
    ) -> dict[str, ProfileAttribute]:
        """Update only a specific dimension of the profile.

        Args:
            user_id: The user identifier.
            new_memories: New memories for analysis.
            dimension: The specific dimension to update.
            existing_profile: Optional existing profile.

        Returns:
            Updated attributes for the specified dimension.

        Raises:
            ValueError: If new_memories is empty or dimension is invalid.
        """
        if not new_memories:
            raise ValueError("new_memories must not be empty")

        _logger.info(
            "Updating dimension %s for user %s",
            dimension.value,
            user_id,
        )

        # Select inference method based on dimension
        inference_methods = {
            ProfileDimension.INTENT: self.engine.infer_intent,
            ProfileDimension.PREFERENCE: self.engine.infer_preferences,
            ProfileDimension.KNOWLEDGE: self.engine.infer_knowledge,
            ProfileDimension.INTERACTION_STYLE: self.engine.infer_interaction_style,
        }

        infer_fn = inference_methods.get(dimension)
        if infer_fn is None:
            raise ValueError(f"Unknown dimension: {dimension}")

        try:
            new_attrs = infer_fn(new_memories)
        except Exception as exc:
            _logger.error("Dimension inference failed for %s: %s", dimension.value, exc)
            raise RuntimeError(f"Dimension update failed: {exc}") from exc

        current = existing_profile or self._profiles.get(user_id, {})
        decayed = self._apply_decay(current)
        merged = self._merge_attributes(decayed, new_attrs)

        self._profiles[user_id] = merged
        self._record_history(user_id, new_attrs, merged, dimension=dimension.value)

        return {k: v for k, v in merged.items() if k in new_attrs or k in current}

    def batch_update(
        self,
        updates: Sequence[tuple[str, Sequence[Memory]]],
    ) -> dict[str, dict[str, ProfileAttribute]]:
        """Batch update profiles for multiple users.

        Args:
            updates: Sequence of (user_id, new_memories) tuples.

        Returns:
            Dict mapping user_id to updated profile attributes.

        Raises:
            ValueError: If updates is empty.
        """
        if not updates:
            raise ValueError("updates must not be empty")

        results: dict[str, dict[str, ProfileAttribute]] = {}
        for user_id, memories in updates:
            try:
                results[user_id] = self.update(user_id, memories)
            except Exception as exc:
                _logger.error("Batch update failed for user %s: %s", user_id, exc)
                # Continue with other users, store error marker
                results[user_id] = {}
        return results

    # ------------------------------------------------------------------ #
    # Decay and merge logic
    # ------------------------------------------------------------------ #

    def _apply_decay(
        self,
        profile: dict[str, ProfileAttribute],
    ) -> dict[str, ProfileAttribute]:
        """Apply time-based confidence decay to existing attributes.

        Args:
            profile: Current profile attributes.

        Returns:
            Decayed attributes (low-confidence ones may be removed).
        """
        now = datetime.now(timezone.utc)
        decayed: dict[str, ProfileAttribute] = {}

        for name, attr in profile.items():
            age_days = (now - attr.timestamp).total_seconds() / 86400.0
            if age_days <= 0:
                decayed[name] = attr
                continue

            # Exponential decay: confidence * (0.5 ^ (age / half_life))
            decay_factor = 0.5 ** (age_days / self.decay_half_life)
            new_confidence = attr.confidence * decay_factor

            if new_confidence >= self.min_confidence:
                decayed[name] = attr.with_confidence(round(new_confidence, 4))
            else:
                _logger.debug(
                    "Dropping attribute %s (confidence %.3f below threshold)",
                    name,
                    new_confidence,
                )

        return decayed

    def _merge_attributes(
        self,
        existing: dict[str, ProfileAttribute],
        new_attrs: dict[str, ProfileAttribute],
    ) -> dict[str, ProfileAttribute]:
        """Merge new attributes into existing profile with conflict resolution.

        Resolution rules:
        - If only one side has the attribute, keep it.
        - If both have it, weighted merge by confidence.
        - Higher confidence attribute gets more weight.

        Args:
            existing: Decayed existing attributes.
            new_attrs: Newly inferred attributes.

        Returns:
            Merged attributes dict.
        """
        merged = dict(existing)

        for name, new_attr in new_attrs.items():
            if name not in merged:
                merged[name] = new_attr
                continue

            old_attr = merged[name]

            # Weighted confidence merge
            total_conf = old_attr.confidence + new_attr.confidence
            if total_conf == 0:
                weight_old = 0.5
                weight_new = 0.5
            else:
                weight_old = old_attr.confidence / total_conf
                weight_new = new_attr.confidence / total_conf

            # Use the higher confidence value if values are different types
            # or if new attribute has significantly higher confidence
            if new_attr.confidence > old_attr.confidence + 0.2:
                # New attribute is significantly more confident
                merged_value = new_attr.value
                merged_conf = min(0.95, new_attr.confidence + 0.05)
            else:
                # Weighted merge - for simplicity keep the higher confidence value
                # but boost confidence slightly
                merged_value = new_attr.value if new_attr.confidence > old_attr.confidence else old_attr.value
                merged_conf = min(0.95, max(old_attr.confidence, new_attr.confidence) + 0.02)

            # Merge evidence
            merged_evidence = tuple(sorted(set(old_attr.evidence + new_attr.evidence)))[:10]

            merged[name] = ProfileAttribute(
                name=name,
                value=merged_value,
                confidence=round(merged_conf, 4),
                source="merged",
                timestamp=datetime.now(timezone.utc),
                evidence=merged_evidence,
            )

        return merged

    # ------------------------------------------------------------------ #
    # History and versioning
    # ------------------------------------------------------------------ #

    def _record_history(
        self,
        user_id: str,
        new_attrs: dict[str, ProfileAttribute],
        merged: dict[str, ProfileAttribute],
        dimension: str | None = None,
    ) -> None:
        """Record an update event in history.

        Args:
            user_id: The user identifier.
            new_attrs: New attributes added in this update.
            merged: Full merged profile after update.
            dimension: Optional dimension name if dimension-specific update.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dimension": dimension,
            "new_attributes": {k: v.to_dict() for k, v in new_attrs.items()},
            "total_attributes": len(merged),
        }
        self._history.setdefault(user_id, []).append(entry)

    def get_history(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get update history for a user.

        Args:
            user_id: The user identifier.
            limit: Maximum number of history entries.

        Returns:
            List of history entry dicts.
        """
        return self._history.get(user_id, [])[-limit:]

    def get_profile(self, user_id: str) -> dict[str, ProfileAttribute]:
        """Get current profile for a user.

        Args:
            user_id: The user identifier.

        Returns:
            Current profile attributes dict.
        """
        return dict(self._profiles.get(user_id, {}))

    def clear_profile(self, user_id: str) -> bool:
        """Clear a user's profile and history.

        Args:
            user_id: The user identifier.

        Returns:
            True if profile existed and was cleared.
        """
        existed = user_id in self._profiles
        self._profiles.pop(user_id, None)
        self._history.pop(user_id, None)
        return existed

    def snapshot(self, user_id: str) -> dict[str, Any]:
        """Create a full snapshot of a user's profile state.

        Args:
            user_id: The user identifier.

        Returns:
            Snapshot dict with profile, metadata, and history summary.
        """
        profile = self._profiles.get(user_id, {})
        history = self._history.get(user_id, [])

        return {
            "user_id": user_id,
            "snapshot_time": datetime.now(timezone.utc).isoformat(),
            "attribute_count": len(profile),
            "attributes": {k: v.to_dict() for k, v in profile.items()},
            "history_entries": len(history),
            "last_update": history[-1]["timestamp"] if history else None,
        }
