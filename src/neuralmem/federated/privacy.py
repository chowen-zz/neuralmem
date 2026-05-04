"""Privacy budget manager for differential privacy tracking and noise calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PrivacyEvent:
    """A single privacy budget consumption event."""

    event_id: str
    timestamp: float
    epsilon_spent: float
    delta: float
    mechanism: str
    noise_scale: float
    description: str = ""


@dataclass
class PrivacyAuditReport:
    """Summary report of privacy budget consumption."""

    total_epsilon_spent: float
    remaining_epsilon: float
    total_events: int
    event_breakdown: dict[str, int]
    recommendations: list[str]
    risk_level: str  # "low", "medium", "high", "exhausted"


class PrivacyBudgetManager:
    """Manages epsilon-DP budget tracking, noise calibration, and privacy audits.

    Features:
    - Per-query epsilon tracking with cumulative budget enforcement
    - Automatic noise scale calibration based on remaining budget
    - Privacy consumption audit trail
    - Configurable delta parameter for (epsilon, delta)-DP
    """

    def __init__(
        self,
        epsilon_budget: float = 4.0,
        delta: float = 1e-5,
        min_noise_scale: float = 0.01,
        max_noise_scale: float = 10.0,
    ) -> None:
        self.epsilon_budget = epsilon_budget
        self.delta = delta
        self.min_noise_scale = min_noise_scale
        self.max_noise_scale = max_noise_scale
        self._spent: float = 0.0
        self._events: list[PrivacyEvent] = []
        self._event_counter = 0

    @property
    def remaining_epsilon(self) -> float:
        """Remaining privacy budget."""
        return max(0.0, self.epsilon_budget - self._spent)

    @property
    def budget_exhausted(self) -> bool:
        """True if privacy budget is fully consumed."""
        return self.remaining_epsilon <= 0.0

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"priv-{self._event_counter:06d}"

    def spend(
        self,
        epsilon: float,
        mechanism: str = "gaussian",
        noise_scale: float | None = None,
        description: str = "",
        timestamp: float = 0.0,
    ) -> dict[str, Any]:
        """Spend privacy budget for an operation.

        Args:
            epsilon: Amount of epsilon to spend.
            mechanism: DP mechanism name (e.g., "gaussian", "laplace").
            noise_scale: Actual noise scale used (optional).
            description: Human-readable description.
            timestamp: Unix timestamp of the event.

        Returns:
            Dict with success flag, spent amount, and remaining budget.
        """
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        if epsilon > self.remaining_epsilon:
            return {
                "success": False,
                "spent": 0.0,
                "remaining": self.remaining_epsilon,
                "reason": "budget_exceeded",
            }

        self._spent += epsilon
        event = PrivacyEvent(
            event_id=self._next_event_id(),
            timestamp=timestamp,
            epsilon_spent=epsilon,
            delta=self.delta,
            mechanism=mechanism,
            noise_scale=noise_scale or 0.0,
            description=description,
        )
        self._events.append(event)

        return {
            "success": True,
            "spent": epsilon,
            "remaining": self.remaining_epsilon,
            "event_id": event.event_id,
        }

    def calibrate_noise(
        self,
        sensitivity: float = 1.0,
        target_epsilon: float | None = None,
        mechanism: str = "gaussian",
    ) -> float:
        """Calibrate noise scale based on remaining budget and sensitivity.

        Uses the Gaussian mechanism formula: sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
        For Laplace: sigma = sensitivity / epsilon

        Args:
            sensitivity: Query sensitivity (L2 for Gaussian, L1 for Laplace).
            target_epsilon: Desired epsilon for this query; defaults to remaining budget.
            mechanism: "gaussian" or "laplace".

        Returns:
            Recommended noise scale.
        """
        eps = target_epsilon if target_epsilon is not None else self.remaining_epsilon
        if eps <= 0:
            return self.max_noise_scale

        if mechanism == "gaussian":
            # Gaussian mechanism: sigma = sqrt(2 ln(1.25/delta)) * sensitivity / epsilon
            sigma = math.sqrt(2.0 * math.log(1.25 / self.delta)) * sensitivity / eps
        elif mechanism == "laplace":
            sigma = sensitivity / eps
        else:
            sigma = sensitivity / eps

        return float(np.clip(sigma, self.min_noise_scale, self.max_noise_scale))

    def auto_spend_and_noise(
        self,
        sensitivity: float = 1.0,
        mechanism: str = "gaussian",
        description: str = "",
        timestamp: float = 0.0,
    ) -> dict[str, Any]:
        """Automatically calibrate noise, spend budget, and return configuration.

        Returns:
            Dict with noise_scale, epsilon_spent, success flag, and event_id.
        """
        # Use a fraction of remaining budget for this query
        target_epsilon = self.remaining_epsilon * 0.1
        if target_epsilon <= 0:
            return {
                "success": False,
                "noise_scale": self.max_noise_scale,
                "epsilon_spent": 0.0,
                "reason": "budget_exhausted",
            }

        noise_scale = self.calibrate_noise(sensitivity, target_epsilon, mechanism)
        spend_result = self.spend(
            epsilon=target_epsilon,
            mechanism=mechanism,
            noise_scale=noise_scale,
            description=description,
            timestamp=timestamp,
        )

        return {
            "success": spend_result["success"],
            "noise_scale": noise_scale,
            "epsilon_spent": spend_result["spent"],
            "remaining": spend_result["remaining"],
            "event_id": spend_result.get("event_id"),
        }

    def get_audit_trail(self) -> list[PrivacyEvent]:
        """Return the full privacy consumption audit trail."""
        return list(self._events)

    def generate_report(self) -> PrivacyAuditReport:
        """Generate a comprehensive privacy audit report."""
        breakdown: dict[str, int] = {}
        for ev in self._events:
            breakdown[ev.mechanism] = breakdown.get(ev.mechanism, 0) + 1

        spent_ratio = self._spent / max(self.epsilon_budget, 1e-9)
        if spent_ratio < 0.25:
            risk = "low"
        elif spent_ratio < 0.5:
            risk = "medium"
        elif spent_ratio < 0.9:
            risk = "high"
        else:
            risk = "exhausted"

        recommendations: list[str] = []
        if spent_ratio > 0.75:
            recommendations.append("Consider increasing epsilon budget or reducing query frequency.")
        if spent_ratio > 0.5 and "gaussian" in breakdown:
            recommendations.append("Switch to Laplace mechanism for lower sensitivity queries.")
        if not recommendations:
            recommendations.append("Privacy budget usage is healthy.")

        return PrivacyAuditReport(
            total_epsilon_spent=self._spent,
            remaining_epsilon=self.remaining_epsilon,
            total_events=len(self._events),
            event_breakdown=breakdown,
            recommendations=recommendations,
            risk_level=risk,
        )

    def reset(self) -> None:
        """Reset all privacy budget state."""
        self._spent = 0.0
        self._events.clear()
        self._event_counter = 0

    def clone(self) -> "PrivacyBudgetManager":
        """Create a copy with the same configuration but fresh state."""
        return PrivacyBudgetManager(
            epsilon_budget=self.epsilon_budget,
            delta=self.delta,
            min_noise_scale=self.min_noise_scale,
            max_noise_scale=self.max_noise_scale,
        )
