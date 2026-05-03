"""Tests for config_hot_reload module."""
from __future__ import annotations

import json
import time

import pytest

from neuralmem.production.config_hot_reload import ConfigHotReload


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary JSON config file."""
    path = tmp_path / "config.json"
    data = {"db_host": "localhost", "port": 5432}
    path.write_text(json.dumps(data))
    return str(path)


class TestConfigHotReload:
    def test_initial_load(self, config_file):
        cr = ConfigHotReload(config_file)
        cfg = cr.get_current()
        assert cfg["db_host"] == "localhost"
        assert cfg["port"] == 5432

    def test_get_key(self, config_file):
        cr = ConfigHotReload(config_file)
        assert cr.get("port") == 5432
        assert cr.get("missing", "default") == "default"

    def test_start_stop(self, config_file):
        cr = ConfigHotReload(
            config_file, poll_interval=0.1
        )
        cr.start()
        assert cr._running
        cr.stop()
        assert not cr._running

    def test_detects_change(self, config_file):
        cr = ConfigHotReload(
            config_file, poll_interval=0.1
        )
        changes: list[tuple] = []

        def on_change(old, new):
            changes.append((old, new))

        cr.on_change(on_change)
        cr.start()

        # Write updated config
        time.sleep(0.05)
        new_data = {"db_host": "remote", "port": 3306}
        with open(config_file, "w") as f:
            json.dump(new_data, f)

        time.sleep(0.5)
        cr.stop()

        assert len(changes) >= 1
        assert changes[-1][1]["db_host"] == "remote"

    def test_no_change_no_callback(self, config_file):
        cr = ConfigHotReload(
            config_file, poll_interval=0.1
        )
        changes: list[tuple] = []

        def on_change(old, new):
            changes.append(True)

        cr.on_change(on_change)
        cr.start()
        time.sleep(0.3)
        cr.stop()

        assert len(changes) == 0

    def test_missing_file(self, tmp_path):
        path = str(tmp_path / "nope.json")
        cr = ConfigHotReload(path)
        assert cr.get_current() == {}

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid json")
        cr = ConfigHotReload(str(path))
        assert cr.get_current() == {}

    def test_callback_error_does_not_crash(self, config_file):
        cr = ConfigHotReload(
            config_file, poll_interval=0.1
        )

        def bad_cb(old, new):
            raise RuntimeError("oops")

        cr.on_change(bad_cb)
        cr.start()

        time.sleep(0.05)
        new_data = {"updated": True}
        with open(config_file, "w") as f:
            json.dump(new_data, f)

        time.sleep(0.5)
        cr.stop()
        # Should not raise
        assert True
