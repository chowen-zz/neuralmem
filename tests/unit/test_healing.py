"""Tests for NeuralMem V1.8 auto-healing system."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from neuralmem.diagnosis.healing import AutoHealingSystem, HealingAction, HealingResult


class TestDiagnoseAndHeal:
    def test_rebuild_index(self):
        rebuild = MagicMock()
        healing = AutoHealingSystem(rebuild_fn=rebuild)
        results = healing.diagnose_and_heal({"index_corrupt": True})
        assert len(results) == 1
        assert results[0].action == HealingAction.REBUILD_INDEX
        assert results[0].success is True
        rebuild.assert_called_once()

    def test_clear_cache(self):
        clear = MagicMock()
        healing = AutoHealingSystem(clear_cache_fn=clear)
        results = healing.diagnose_and_heal({"cache_stale": True})
        assert results[0].action == HealingAction.CLEAR_CACHE
        clear.assert_called_once()

    def test_restart_service(self):
        restart = MagicMock()
        healing = AutoHealingSystem(restart_fn=restart)
        results = healing.diagnose_and_heal({"service_unresponsive": True})
        assert results[0].action == HealingAction.RESTART_SERVICE
        restart.assert_called_once()

    def test_adjust_params(self):
        adjust = MagicMock()
        healing = AutoHealingSystem(adjust_fn=adjust)
        results = healing.diagnose_and_heal({"high_latency": True, "params": {"dim": 256}})
        assert results[0].action == HealingAction.ADJUST_PARAMS
        adjust.assert_called_once()

    def test_no_matching_symptoms(self):
        healing = AutoHealingSystem()
        results = healing.diagnose_and_heal({"unknown_issue": True})
        assert len(results) == 0


class TestCustomRules:
    def test_add_custom_rule(self):
        healing = AutoHealingSystem()
        healing.add_rule("custom_issue", HealingAction.NOTIFY_ADMIN)
        results = healing.diagnose_and_heal({"custom_issue": True})
        assert len(results) == 1
        assert results[0].action == HealingAction.NOTIFY_ADMIN


class TestHistory:
    def test_history_tracking(self):
        healing = AutoHealingSystem(rebuild_fn=lambda: None)
        healing.diagnose_and_heal({"index_corrupt": True})
        history = healing.get_history()
        assert len(history) == 1
        assert history[0].action == HealingAction.REBUILD_INDEX

    def test_reset(self):
        healing = AutoHealingSystem(rebuild_fn=lambda: None)
        healing.diagnose_and_heal({"index_corrupt": True})
        healing.reset()
        assert len(healing.get_history()) == 0
