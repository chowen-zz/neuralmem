"""DeepProfileEngine V1.5 — LLM-based deep behavior modeling with continuous learning.

Extends ProfileEngine with:
  • Behavior pattern extraction from memory access logs
  • Preference inference with confidence scoring
  • Temporal preference drift detection
  • Cross-domain preference transfer
  • Continuous learning loop
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from neuralmem.core.types import Memory, MemoryType
from neuralmem.profiles.base import ProfileAttribute
from neuralmem.profiles.engine import ProfileEngine
from neuralmem.profiles.types import (
    CommunicationStyle,
    Intent,
    IntentCategory,
    InteractionStyle,
    Knowledge,
    KnowledgeLevel,
    Preference,
    PreferenceType,
    ProfileDimension,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuralmem.storage.base import StorageBackend

_logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Deep behavior pattern templates
# ------------------------------------------------------------------ #

_BEHAVIOR_PATTERNS: dict[str, dict[str, list[str]]] = {
    "query_depth": {
        "shallow": ["quick", "brief", "short", "summary", "tl;dr", "一句话", "简要"],
        "deep": ["deep dive", "internals", "原理", "源码", "底层", "深入", "详细", "thorough"],
        "exploratory": ["compare", "difference", "vs", "对比", "区别", "优劣", "alternatives"],
    },
    "engagement_style": {
        "passive": ["read", "browse", "look at", "看看", "了解一下"],
        "active": ["implement", "build", "write", "deploy", "实现", "写", "搭建", "开发"],
        "collaborative": ["review", "feedback", "together", "一起", "讨论", "评审"],
    },
    "learning_pattern": {
        "tutorial_first": ["tutorial", "getting started", "入门", "新手", "beginner", "hello world"],
        "doc_first": ["documentation", "reference", "api", "手册", "文档", "spec"],
        "code_first": ["example", "snippet", "sample", "代码", "示例", "demo"],
    },
}

# Cross-domain transfer rules: source domain -> transferable target domains
_CROSS_DOMAIN_TRANSFER: dict[str, list[str]] = {
    "machine_learning": ["data_science", "web_development", "mobile"],
    "web_development": ["mobile", "devops", "security"],
    "devops": ["web_development", "security", "data_science"],
    "data_science": ["machine_learning", "web_development"],
    "security": ["web_development", "devops", "mobile"],
    "mobile": ["web_development", "security"],
}

# ------------------------------------------------------------------ #
# Memory access log entry
# ------------------------------------------------------------------ #

class AccessLogEntry:
    """Single memory access event for behavior analysis."""

    def __init__(
        self,
        memory_id: str,
        user_id: str,
        action: str,  # "read", "write", "search", "update", "delete"
        timestamp: datetime | None = None,
        query_text: str | None = None,
        result_count: int = 0,
        dwell_time_ms: int = 0,
    ) -> None:
        self.memory_id = memory_id
        self.user_id = user_id
        self.action = action
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.query_text = query_text or ""
        self.result_count = result_count
        self.dwell_time_ms = dwell_time_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "query_text": self.query_text,
            "result_count": self.result_count,
            "dwell_time_ms": self.dwell_time_ms,
        }


# ------------------------------------------------------------------ #
# DeepProfileEngine
# ------------------------------------------------------------------ #

class DeepProfileEngine(ProfileEngine):
    """Deep user profile engine with LLM-based behavior modeling and continuous learning.

    Inherits all keyword-based inference from ProfileEngine and adds:
      - Deep behavior pattern extraction from access logs
      - Preference inference with confidence scoring and temporal tracking
      - Preference drift detection across time windows
      - Cross-domain knowledge/preference transfer
      - Continuous learning via feedback loop

    Args:
        storage: Optional storage backend for fetching user history.
        llm_client: Optional callable for LLM-based deep inference.
            Signature: ``llm_client(prompt: str, system: str | None = None) -> str``.
        learning_rate: How aggressively new evidence updates existing beliefs (0.0-1.0).
        drift_window_days: Time window for drift detection.
        min_samples_for_drift: Minimum memory count to trigger drift analysis.
    """

    def __init__(
        self,
        storage: StorageBackend | None = None,
        llm_client: Callable[[str, str | None], str] | None = None,
        learning_rate: float = 0.3,
        drift_window_days: float = 7.0,
        min_samples_for_drift: int = 5,
    ) -> None:
        super().__init__(storage=storage)
        self.llm_client = llm_client
        self.learning_rate = max(0.0, min(1.0, learning_rate))
        self.drift_window_days = max(1.0, drift_window_days)
        self.min_samples_for_drift = max(3, min_samples_for_drift)

        # Continuous learning state
        self._access_logs: list[AccessLogEntry] = []
        self._preference_history: dict[str, list[Preference]] = defaultdict(list)
        self._behavior_models: dict[str, dict[str, Any]] = {}
        self._drift_alerts: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # ------------------------------------------------------------------ #
    # 1. Behavior pattern extraction from memory access logs
    # ------------------------------------------------------------------ #

    def log_access(self, entry: AccessLogEntry) -> None:
        """Record a memory access event for behavior analysis.

        Args:
            entry: The access log entry to record.
        """
        self._access_logs.append(entry)
        _logger.debug(
            "Access logged: user=%s action=%s mem=%s",
            entry.user_id,
            entry.action,
            entry.memory_id,
        )

    def extract_behavior_patterns(
        self,
        user_id: str,
        access_logs: Sequence[AccessLogEntry] | None = None,
    ) -> dict[str, ProfileAttribute]:
        """Extract deep behavior patterns from access logs.

        Analyzes query depth, engagement style, and learning patterns
        from the user's memory access history.

        Args:
            user_id: The user identifier.
            access_logs: Optional sequence of access logs. If None, uses internal logs.

        Returns:
            Dict of behavior pattern attributes.
        """
        logs = access_logs or [log for log in self._access_logs if log.user_id == user_id]
        if not logs:
            return {}

        results: dict[str, ProfileAttribute] = {}
        query_texts = " ".join(log.query_text for log in logs if log.query_text)
        query_texts = query_texts.lower()

        # Query depth analysis
        depth_scores: Counter[str] = Counter()
        for depth, keywords in _BEHAVIOR_PATTERNS["query_depth"].items():
            depth_scores[depth] = sum(1 for kw in keywords if kw in query_texts)

        if depth_scores:
            primary_depth, dcount = depth_scores.most_common(1)[0]
            total_depth = sum(depth_scores.values())
            depth_conf = min(0.9, 0.4 + dcount / max(total_depth * 0.6, 1))
            results["behavior_query_depth"] = ProfileAttribute(
                name="behavior_query_depth",
                value={
                    "primary_depth": primary_depth,
                    "depth_distribution": dict(depth_scores),
                    "avg_dwell_ms": sum(log.dwell_time_ms for log in logs) / len(logs),
                },
                confidence=round(depth_conf, 3),
                source="behavior_analysis",
                evidence=tuple(log.memory_id for log in logs)[:5],
            )

        # Engagement style analysis
        engagement_scores: Counter[str] = Counter()
        for style, keywords in _BEHAVIOR_PATTERNS["engagement_style"].items():
            engagement_scores[style] = sum(1 for kw in keywords if kw in query_texts)

        # Also factor in action types
        action_counts = Counter(log.action for log in logs)
        if action_counts.get("write", 0) + action_counts.get("update", 0) > len(logs) * 0.3:
            engagement_scores["active"] += 2
        if action_counts.get("search", 0) > len(logs) * 0.5:
            engagement_scores["exploratory"] = engagement_scores.get("exploratory", 0) + 1

        if engagement_scores:
            primary_engagement, ecount = engagement_scores.most_common(1)[0]
            total_eng = sum(engagement_scores.values())
            eng_conf = min(0.85, 0.4 + ecount / max(total_eng * 0.6, 1))
            results["behavior_engagement"] = ProfileAttribute(
                name="behavior_engagement",
                value={
                    "primary_style": primary_engagement,
                    "action_distribution": dict(action_counts),
                    "search_ratio": action_counts.get("search", 0) / len(logs),
                },
                confidence=round(eng_conf, 3),
                source="behavior_analysis",
                evidence=tuple(log.memory_id for log in logs)[:5],
            )

        # Learning pattern analysis
        learning_scores: Counter[str] = Counter()
        for pattern, keywords in _BEHAVIOR_PATTERNS["learning_pattern"].items():
            learning_scores[pattern] = sum(1 for kw in keywords if kw in query_texts)

        if learning_scores:
            primary_learning, lcount = learning_scores.most_common(1)[0]
            total_learn = sum(learning_scores.values())
            learn_conf = min(0.85, 0.4 + lcount / max(total_learn * 0.6, 1))
            results["behavior_learning"] = ProfileAttribute(
                name="behavior_learning",
                value={
                    "primary_pattern": primary_learning,
                    "pattern_distribution": dict(learning_scores),
                },
                confidence=round(learn_conf, 3),
                source="behavior_analysis",
                evidence=tuple(log.memory_id for log in logs)[:5],
            )

        _logger.info(
            "Extracted %d behavior patterns for user %s from %d logs",
            len(results),
            user_id,
            len(logs),
        )
        return results

    # ------------------------------------------------------------------ #
    # 2. Preference inference with confidence scoring
    # ------------------------------------------------------------------ #

    def infer_preferences_with_confidence(
        self,
        memories: Sequence[Memory],
        prior_preferences: dict[str, Preference] | None = None,
    ) -> dict[str, ProfileAttribute]:
        """Infer preferences with Bayesian-style confidence scoring.

        Combines keyword-based signals with prior beliefs to produce
        robust preference estimates with calibrated confidence.

        Args:
            memories: Sequence of memories to analyze.
            prior_preferences: Optional prior preference beliefs for Bayesian update.

        Returns:
            Dict mapping preference attribute names to ProfileAttribute instances.

        Raises:
            ValueError: If memories is empty.
        """
        if not memories:
            raise ValueError("memories must not be empty for preference inference")

        # Start with base keyword inference
        base_attrs = self.infer_preferences(memories)

        results: dict[str, ProfileAttribute] = {}
        for attr_name, attr in base_attrs.items():
            pref_type_str = attr.value.get("type", "")
            pref_value = attr.value.get("value", "")
            base_conf = attr.confidence
            base_strength = attr.value.get("strength", 0.5)

            # Bayesian-style update with prior
            if prior_preferences and pref_type_str in prior_preferences:
                prior = prior_preferences[pref_type_str]
                # Weighted combination: prior + new evidence
                combined_strength = (
                    prior.strength * (1 - self.learning_rate)
                    + base_strength * self.learning_rate
                )
                # Confidence increases with corroborating evidence
                combined_conf = min(
                    0.95,
                    base_conf + prior.confidence * 0.3,
                )
            else:
                combined_strength = base_strength
                combined_conf = base_conf

            # Boost confidence with evidence multiplicity
            evidence_count = len(attr.evidence) if attr.evidence else 1
            boosted_conf = min(0.95, combined_conf + 0.05 * (evidence_count - 1))

            # Record in preference history
            pref = Preference(
                preference_type=PreferenceType(pref_type_str),
                value=pref_value,
                strength=round(combined_strength, 3),
                context=attr.value.get("context", "general"),
                confidence=round(boosted_conf, 3),
            )
            self._preference_history[pref_type_str].append(pref)

            results[attr_name] = ProfileAttribute(
                name=attr_name,
                value={
                    "type": pref_type_str,
                    "value": pref_value,
                    "strength": round(combined_strength, 3),
                    "context": attr.value.get("context", "general"),
                    "evidence_count": evidence_count,
                },
                confidence=round(boosted_conf, 3),
                source="deep_preference_inference",
                timestamp=datetime.now(timezone.utc),
                evidence=attr.evidence,
            )

        return results

    # ------------------------------------------------------------------ #
    # 3. Temporal preference drift detection
    # ------------------------------------------------------------------ #

    def detect_preference_drift(
        self,
        user_id: str,
        memories: Sequence[Memory],
        window_days: float | None = None,
    ) -> dict[str, Any]:
        """Detect temporal preference drift by comparing recent vs. historical preferences.

        Splits memories into recent and historical windows, infers preferences
        for each, and reports statistically significant shifts.

        Args:
            user_id: The user identifier.
            memories: Sequence of memories with timestamps.
            window_days: Override drift window size (default: self.drift_window_days).

        Returns:
            Drift report dict with detected shifts and confidence scores.
        """
        if len(memories) < self.min_samples_for_drift:
            return {
                "drift_detected": False,
                "reason": "insufficient_samples",
                "sample_count": len(memories),
                "required": self.min_samples_for_drift,
            }

        window = timedelta(days=window_days or self.drift_window_days)
        now = datetime.now(timezone.utc)
        cutoff = now - window

        recent_mems = [m for m in memories if m.created_at >= cutoff]
        historical_mems = [m for m in memories if m.created_at < cutoff]

        if len(recent_mems) < 3 or len(historical_mems) < 3:
            return {
                "drift_detected": False,
                "reason": "insufficient_window_samples",
                "recent_count": len(recent_mems),
                "historical_count": len(historical_mems),
            }

        recent_prefs = self.infer_preferences(recent_mems)
        historical_prefs = self.infer_preferences(historical_mems)

        drift_shifts: list[dict[str, Any]] = []

        # Compare all preference keys
        all_keys = set(recent_prefs.keys()) | set(historical_prefs.keys())
        for key in all_keys:
            recent = recent_prefs.get(key)
            historical = historical_prefs.get(key)

            if recent is None or historical is None:
                # New or disappeared preference
                if recent is not None:
                    drift_shifts.append({
                        "attribute": key,
                        "type": "emerged",
                        "new_value": recent.value,
                        "new_confidence": recent.confidence,
                        "severity": "medium",
                    })
                else:
                    drift_shifts.append({
                        "attribute": key,
                        "type": "faded",
                        "old_value": historical.value,
                        "old_confidence": historical.confidence,
                        "severity": "low",
                    })
                continue

            # Value change detection
            recent_val = recent.value.get("value") if isinstance(recent.value, dict) else recent.value
            hist_val = historical.value.get("value") if isinstance(historical.value, dict) else historical.value

            if recent_val != hist_val:
                # Significant value shift
                conf_drop = historical.confidence - recent.confidence
                severity = "high" if conf_drop > 0.3 else "medium" if conf_drop > 0.1 else "low"
                drift_shifts.append({
                    "attribute": key,
                    "type": "value_shift",
                    "old_value": hist_val,
                    "new_value": recent_val,
                    "old_confidence": historical.confidence,
                    "new_confidence": recent.confidence,
                    "severity": severity,
                })
            else:
                # Same value but confidence change
                conf_delta = abs(recent.confidence - historical.confidence)
                if conf_delta > 0.2:
                    drift_shifts.append({
                        "attribute": key,
                        "type": "confidence_shift",
                        "value": recent_val,
                        "old_confidence": historical.confidence,
                        "new_confidence": recent.confidence,
                        "severity": "low",
                    })

        drift_detected = len(drift_shifts) > 0
        report = {
            "drift_detected": drift_detected,
            "user_id": user_id,
            "window_days": window_days or self.drift_window_days,
            "recent_samples": len(recent_mems),
            "historical_samples": len(historical_mems),
            "shifts": drift_shifts,
            "shift_count": len(drift_shifts),
            "timestamp": now.isoformat(),
        }

        if drift_detected:
            self._drift_alerts[user_id].append(report)
            _logger.warning(
                "Preference drift detected for user %s: %d shifts",
                user_id,
                len(drift_shifts),
            )

        return report

    # ------------------------------------------------------------------ #
    # 4. Cross-domain preference transfer
    # ------------------------------------------------------------------ #

    def transfer_preferences(
        self,
        user_id: str,
        source_domain: str,
        target_domains: list[str] | None = None,
        memories: Sequence[Memory] | None = None,
    ) -> dict[str, ProfileAttribute]:
        """Transfer inferred preferences from a source domain to target domains.

        Uses domain similarity heuristics to propagate knowledge and preference
        signals across related domains.

        Args:
            user_id: The user identifier.
            source_domain: The domain with established preferences (e.g. "machine_learning").
            target_domains: Optional explicit target domains. If None, uses heuristic mapping.
            memories: Optional memories to analyze for target domain context.

        Returns:
            Dict of transferred preference attributes for target domains.
        """
        targets = target_domains or _CROSS_DOMAIN_TRANSFER.get(source_domain, [])
        if not targets:
            return {}

        # Get source domain preferences from history or infer from memories
        source_prefs: dict[str, Preference] = {}
        for ptype, hist in self._preference_history.items():
            for pref in hist:
                # Check if this preference is domain-relevant
                if pref.context == source_domain or pref.context == "general":
                    source_prefs[ptype] = pref
                    break

        transferred: dict[str, ProfileAttribute] = {}
        for target in targets:
            transfer_confidence = 0.5  # Base transfer confidence

            for ptype, pref in source_prefs.items():
                # Domain similarity boost
                similarity_boost = 0.1 if target in (_CROSS_DOMAIN_TRANSFER.get(source_domain, [])) else 0.0
                final_conf = min(0.7, pref.confidence * 0.6 + similarity_boost)

                attr_name = f"transferred_{target}_{ptype}"
                transferred[attr_name] = ProfileAttribute(
                    name=attr_name,
                    value={
                        "source_domain": source_domain,
                        "target_domain": target,
                        "preference_type": ptype,
                        "inferred_value": pref.value,
                        "transfer_confidence": round(final_conf, 3),
                        "original_strength": pref.strength,
                    },
                    confidence=round(final_conf, 3),
                    source="cross_domain_transfer",
                    timestamp=datetime.now(timezone.utc),
                )

        _logger.info(
            "Transferred %d preferences from %s to %d domains for user %s",
            len(source_prefs),
            source_domain,
            len(targets),
            user_id,
        )
        return transferred

    # ------------------------------------------------------------------ #
    # 5. LLM-based deep inference
    # ------------------------------------------------------------------ #

    def llm_deep_infer(
        self,
        memories: Sequence[Memory],
        dimension: ProfileDimension = ProfileDimension.INTENT,
    ) -> dict[str, ProfileAttribute]:
        """Use an LLM client for deep, nuanced profile inference.

        Falls back gracefully to keyword-based inference if no LLM client
        is configured or if the LLM call fails.

        Args:
            memories: Sequence of memories to analyze.
            dimension: The profile dimension to infer.

        Returns:
            Dict of inferred profile attributes.

        Raises:
            ValueError: If memories is empty.
        """
        if not memories:
            raise ValueError("memories must not be empty for LLM inference")

        if self.llm_client is None:
            # Graceful fallback to keyword-based inference
            _logger.debug("No LLM client configured; falling back to keyword inference")
            inference_methods = {
                ProfileDimension.INTENT: self.infer_intent,
                ProfileDimension.PREFERENCE: self.infer_preferences,
                ProfileDimension.KNOWLEDGE: self.infer_knowledge,
                ProfileDimension.INTERACTION_STYLE: self.infer_interaction_style,
            }
            infer_fn = inference_methods.get(dimension, self.infer_intent)
            return infer_fn(memories)

        # Build prompt from memory contents
        contents = [m.content for m in memories[:20]]  # Limit to 20 for token efficiency
        memory_text = "\n---\n".join(contents)

        prompts = {
            ProfileDimension.INTENT: (
                "Analyze the following user queries and identify the primary intent, "
                "subcategories, and key themes. Respond in JSON: "
                '{"category": "...", "subcategory": "...", "keywords": [...], "confidence": 0.0-1.0}',
                memory_text,
            ),
            ProfileDimension.PREFERENCE: (
                "Analyze the following user queries and infer preferences for "
                "technology, format, tone, depth, and language. "
                'Respond in JSON: [{"type": "...", "value": "...", "strength": 0.0-1.0, "confidence": 0.0-1.0}]',
                memory_text,
            ),
            ProfileDimension.KNOWLEDGE: (
                "Analyze the following user queries and infer knowledge domains, "
                "levels (beginner/intermediate/advanced/expert), and gaps. "
                'Respond in JSON: [{"domain": "...", "level": "...", "confidence": 0.0-1.0}]',
                memory_text,
            ),
            ProfileDimension.INTERACTION_STYLE: (
                "Analyze the following user queries and infer communication style, "
                "detail preference, initiative level, and follow-up tendency. "
                'Respond in JSON: {"communication": "...", "detail_level": 0.0-1.0, "initiative": 0.0-1.0, "follow_up": true/false, "confidence": 0.0-1.0}',
                memory_text,
            ),
        }

        system_msg, user_prompt = prompts.get(
            dimension,
            ("Analyze user behavior from queries.", memory_text),
        )

        try:
            raw_response = self.llm_client(user_prompt, system_msg)
            return self._parse_llm_response(raw_response, dimension)
        except Exception as exc:
            _logger.warning("LLM inference failed for %s: %s", dimension.value, exc)
            # Fallback
            inference_methods = {
                ProfileDimension.INTENT: self.infer_intent,
                ProfileDimension.PREFERENCE: self.infer_preferences,
                ProfileDimension.KNOWLEDGE: self.infer_knowledge,
                ProfileDimension.INTERACTION_STYLE: self.infer_interaction_style,
            }
            return inference_methods.get(dimension, self.infer_intent)(memories)

    def _parse_llm_response(
        self,
        response: str,
        dimension: ProfileDimension,
    ) -> dict[str, ProfileAttribute]:
        """Parse LLM JSON response into ProfileAttribute objects.

        Args:
            response: Raw LLM response string.
            dimension: The profile dimension being parsed.

        Returns:
            Dict of parsed profile attributes.
        """
        results: dict[str, ProfileAttribute] = {}

        # Extract JSON from potential markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            response = json_match.group(1)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            _logger.warning("Failed to parse LLM response as JSON: %s", response[:200])
            return results

        now = datetime.now(timezone.utc)

        if dimension == ProfileDimension.INTENT and isinstance(data, dict):
            category = data.get("category", "informational")
            subcategory = data.get("subcategory", "")
            keywords = tuple(data.get("keywords", []))
            confidence = max(0.0, min(1.0, data.get("confidence", 0.5)))

            intent = Intent(
                category=IntentCategory(category) if category in [e.value for e in IntentCategory] else IntentCategory.INFORMATIONAL,
                subcategory=subcategory,
                keywords=keywords,
                confidence=confidence,
            )
            results["primary_intent"] = intent.to_attribute()

        elif dimension == ProfileDimension.PREFERENCE and isinstance(data, list):
            for item in data:
                ptype_str = item.get("type", "topic")
                value = item.get("value", "")
                strength = max(0.0, min(1.0, item.get("strength", 0.5)))
                confidence = max(0.0, min(1.0, item.get("confidence", 0.5)))

                try:
                    ptype = PreferenceType(ptype_str)
                except ValueError:
                    ptype = PreferenceType.TOPIC

                pref = Preference(
                    preference_type=ptype,
                    value=value,
                    strength=strength,
                    context="general",
                    confidence=confidence,
                )
                attr = pref.to_attribute()
                results[attr.name] = attr

        elif dimension == ProfileDimension.KNOWLEDGE and isinstance(data, list):
            for item in data:
                domain = item.get("domain", "general")
                level_str = item.get("level", "intermediate")
                confidence = max(0.0, min(1.0, item.get("confidence", 0.5)))

                try:
                    level = KnowledgeLevel(level_str)
                except ValueError:
                    level = KnowledgeLevel.INTERMEDIATE

                knowledge = Knowledge(
                    domain=domain,
                    level=level,
                    confidence=confidence,
                )
                attr = knowledge.to_attribute()
                results[attr.name] = attr

        elif dimension == ProfileDimension.INTERACTION_STYLE and isinstance(data, dict):
            comm_str = data.get("communication", "conversational")
            try:
                comm = CommunicationStyle(comm_str)
            except ValueError:
                comm = CommunicationStyle.CONVERSATIONAL

            style = InteractionStyle(
                communication=comm,
                detail_level=max(0.0, min(1.0, data.get("detail_level", 0.5))),
                initiative=max(0.0, min(1.0, data.get("initiative", 0.5))),
                follow_up=bool(data.get("follow_up", True)),
                confidence=max(0.0, min(1.0, data.get("confidence", 0.5))),
            )
            results["interaction_style"] = style.to_attribute()

        return results

    # ------------------------------------------------------------------ #
    # 6. Continuous learning loop
    # ------------------------------------------------------------------ #

    def learn_from_feedback(
        self,
        user_id: str,
        attribute_name: str,
        feedback: float,  # -1.0 (wrong) to +1.0 (correct)
        memory_ids: tuple[str, ...] = (),
    ) -> ProfileAttribute | None:
        """Update an attribute's confidence based on explicit or implicit feedback.

        Positive feedback increases confidence; negative feedback decreases it.
        This forms the core of the continuous learning loop.

        Args:
            user_id: The user identifier.
            attribute_name: The profile attribute to update.
            feedback: Feedback score in [-1.0, 1.0].
            memory_ids: Memory IDs that triggered this feedback.

        Returns:
            Updated ProfileAttribute, or None if attribute not found.
        """
        # Look in behavior models first, then fall back to standard profiles
        profile = self._behavior_models.get(user_id, {})
        attr = profile.get(attribute_name)

        if attr is None:
            _logger.debug(
                "No attribute %s found for user %s to learn from feedback",
                attribute_name,
                user_id,
            )
            return None

        # Apply feedback delta
        delta = feedback * self.learning_rate
        new_confidence = max(0.0, min(1.0, attr.confidence + delta))

        # Merge evidence
        merged_evidence = tuple(sorted(set(attr.evidence + memory_ids)))[:10]

        updated = ProfileAttribute(
            name=attr.name,
            value=attr.value,
            confidence=round(new_confidence, 4),
            source=f"{attr.source}|feedback",
            timestamp=datetime.now(timezone.utc),
            evidence=merged_evidence,
        )

        self._behavior_models.setdefault(user_id, {})[attribute_name] = updated

        _logger.info(
            "Learned from feedback for user %s attr %s: conf %.3f -> %.3f",
            user_id,
            attribute_name,
            attr.confidence,
            new_confidence,
        )
        return updated

    def build_deep_profile(
        self,
        user_id: str,
        memories: Sequence[Memory],
        access_logs: Sequence[AccessLogEntry] | None = None,
        use_llm: bool = False,
    ) -> dict[str, ProfileAttribute]:
        """Build a comprehensive deep profile combining all inference methods.

        This is the main entry point for deep profile construction. It:
        1. Builds the base profile from keyword inference
        2. Extracts behavior patterns from access logs
        3. Runs preference inference with confidence scoring
        4. Optionally uses LLM for nuanced inference
        5. Detects preference drift

        Args:
            user_id: The user identifier.
            memories: All available memories for the user.
            access_logs: Optional memory access logs for behavior analysis.
            use_llm: Whether to use LLM-based deep inference.

        Returns:
            Combined dict of all inferred deep profile attributes.

        Raises:
            ValueError: If memories is empty.
        """
        if not memories:
            raise ValueError("memories must not be empty to build deep profile")

        _logger.info(
            "Building deep profile for user %s from %d memories",
            user_id,
            len(memories),
        )

        profile: dict[str, ProfileAttribute] = {}

        # 1. Base keyword-based inference
        try:
            base = self.build_profile(user_id, memories)
            profile.update(base)
        except Exception as exc:
            _logger.warning("Base profile build failed: %s", exc)

        # 2. Behavior patterns from access logs
        if access_logs:
            try:
                for log in access_logs:
                    self.log_access(log)
                behavior = self.extract_behavior_patterns(user_id, access_logs)
                profile.update(behavior)
            except Exception as exc:
                _logger.warning("Behavior extraction failed: %s", exc)

        # 3. Deep preference inference with confidence
        try:
            prefs = self.infer_preferences_with_confidence(memories)
            profile.update(prefs)
        except Exception as exc:
            _logger.warning("Deep preference inference failed: %s", exc)

        # 4. LLM deep inference (optional)
        if use_llm and self.llm_client is not None:
            for dimension in ProfileDimension:
                try:
                    llm_attrs = self.llm_deep_infer(memories, dimension)
                    profile.update(llm_attrs)
                except Exception as exc:
                    _logger.warning("LLM inference failed for %s: %s", dimension.value, exc)

        # 5. Drift detection
        try:
            drift_report = self.detect_preference_drift(user_id, memories)
            if drift_report.get("drift_detected"):
                profile["drift_report"] = ProfileAttribute(
                    name="drift_report",
                    value=drift_report,
                    confidence=0.8,
                    source="drift_detection",
                    timestamp=datetime.now(timezone.utc),
                )
        except Exception as exc:
            _logger.warning("Drift detection failed: %s", exc)

        # Store in behavior models
        self._behavior_models[user_id] = dict(profile)

        _logger.info(
            "Deep profile built with %d attributes for user %s",
            len(profile),
            user_id,
        )
        return profile

    # ------------------------------------------------------------------ #
    # Query / introspection helpers
    # ------------------------------------------------------------------ #

    def get_preference_history(self, preference_type: str) -> list[Preference]:
        """Return the full preference inference history for a type.

        Args:
            preference_type: The PreferenceType value string.

        Returns:
            List of Preference instances in chronological order.
        """
        return list(self._preference_history.get(preference_type, []))

    def get_drift_alerts(self, user_id: str) -> list[dict[str, Any]]:
        """Return all drift alerts for a user.

        Args:
            user_id: The user identifier.

        Returns:
            List of drift report dicts.
        """
        return list(self._drift_alerts.get(user_id, []))

    def get_behavior_model(self, user_id: str) -> dict[str, Any]:
        """Return the full behavior model for a user.

        Args:
            user_id: The user identifier.

        Returns:
            Behavior model dict with all tracked attributes.
        """
        model = self._behavior_models.get(user_id, {})
        return {
            "user_id": user_id,
            "attribute_count": len(model),
            "attributes": {
                k: {
                    "name": v.name,
                    "value": v.value,
                    "confidence": v.confidence,
                    "source": v.source,
                    "timestamp": v.timestamp.isoformat(),
                    "evidence": list(v.evidence),
                }
                for k, v in model.items()
            },
            "preference_history": {
                k: [
                    {
                        "type": p.preference_type.value,
                        "value": p.value,
                        "strength": p.strength,
                        "confidence": p.confidence,
                        "timestamp": p.timestamp.isoformat(),
                    }
                    for p in v
                ]
                for k, v in self._preference_history.items()
            },
            "drift_alert_count": len(self._drift_alerts.get(user_id, [])),
        }

    def clear_user_data(self, user_id: str) -> bool:
        """Clear all deep profile data for a user.

        Args:
            user_id: The user identifier.

        Returns:
            True if user had data and was cleared.
        """
        existed = user_id in self._behavior_models
        self._behavior_models.pop(user_id, None)
        self._drift_alerts.pop(user_id, None)
        self._access_logs = [log for log in self._access_logs if log.user_id != user_id]
        return existed
