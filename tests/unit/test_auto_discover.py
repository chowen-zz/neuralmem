"""AutoDiscoveryEngine / ConnectorAutoDiscovery 单元测试 — 全部使用 mock.

覆盖要点:
- 环境变量扫描 (mock os.environ)
- 文件系统扫描 (mock Path / os.walk)
- 配置文件解析 (mock 文件内容)
- 置信度计算与排序
- 一键连接 (mock ConnectorRegistry.create)
- 环境变量占位符解析
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.connectors.auto_discover import (
    AutoDiscoveryEngine,
    ConnectorAutoDiscovery,
    ConnectorSuggestion,
    DiscoverySignal,
)
from neuralmem.connectors.base import ConnectorProtocol, ConnectorState, SyncItem
from neuralmem.connectors.registry import ConnectorRegistry


# --------------------------------------------------------------------------- #
# 模块级最小连接器实现 (供 ConnectorRegistry 通过模块路径实例化)
# --------------------------------------------------------------------------- #

class _MockConn(ConnectorProtocol):
    """用于注册表实例化的最小连接器."""

    def authenticate(self) -> bool:
        self.state = ConnectorState.CONNECTED
        return True

    def sync(self, **kwargs) -> list[SyncItem]:
        return []

    def disconnect(self) -> None:
        self.state = ConnectorState.DISCONNECTED


class _FailingConn(ConnectorProtocol):
    """认证失败的连接器."""

    def authenticate(self) -> bool:
        self._set_error("bad creds")
        return False

    def sync(self, **kwargs) -> list[SyncItem]:
        return []

    def disconnect(self) -> None:
        pass


class _FailAuth(ConnectorProtocol):
    """connect_best 回退测试用 — 第一个失败."""

    def authenticate(self) -> bool:
        return False

    def sync(self, **kwargs) -> list[SyncItem]:
        return []

    def disconnect(self) -> None:
        pass


class _OkAuth(ConnectorProtocol):
    """connect_best 回退测试用 — 第二个成功."""

    def authenticate(self) -> bool:
        self.state = ConnectorState.CONNECTED
        return True

    def sync(self, **kwargs) -> list[SyncItem]:
        return []

    def disconnect(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _preserve_registry():
    """备份并恢复 ConnectorRegistry 内置注册."""
    original = dict(ConnectorRegistry._registry)
    builtin = {"notion", "slack", "github", "filesystem", "gdrive", "s3"}
    for name in list(ConnectorRegistry._registry):
        if name in builtin:
            del ConnectorRegistry._registry[name]
    yield
    ConnectorRegistry._registry.clear()
    ConnectorRegistry._registry.update(original)


@pytest.fixture
def mock_connector_cls():
    """返回模块级 _MockConn."""
    return _MockConn


# --------------------------------------------------------------------------- #
# DiscoverySignal
# --------------------------------------------------------------------------- #

def test_discovery_signal_dataclass():
    sig = DiscoverySignal(
        source_type="env", path=None, detail="AWS key found", weight=0.9
    )
    assert sig.source_type == "env"
    assert sig.weight == 0.9


# --------------------------------------------------------------------------- #
# AutoDiscoveryEngine — env scan
# --------------------------------------------------------------------------- #

def test_scan_env_vars_detects_s3():
    with patch.dict(
        os.environ, {"AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE"}, clear=False
    ):
        engine = AutoDiscoveryEngine()
        results = engine.scan()
    names = [r.connector_name for r in results]
    assert "s3" in names
    s3 = next(r for r in results if r.connector_name == "s3")
    assert s3.confidence >= 0.9
    assert any(s.source_type == "env" for s in s3.signals)


def test_scan_env_vars_detects_notion():
    with patch.dict(os.environ, {"NOTION_TOKEN": "secret_123"}, clear=False):
        engine = AutoDiscoveryEngine()
        results = engine.scan()
    names = [r.connector_name for r in results]
    assert "notion" in names


def test_scan_env_vars_no_false_positives():
    """无目标环境变量时不应产生高置信度建议."""
    # 仅保留与目标模式无关的系统变量, 避免 HOME/PATH 等触发 filesystem
    keep = {k: v for k, v in os.environ.items() if not any(
        k.upper().startswith(prefix)
        for prefix in (
            "AWS", "NOTION", "SLACK", "GITHUB", "GH_", "GOOGLE", "GDRIVE",
            "NEURALMEM", "S3_",
        )
    )}
    with patch.dict(os.environ, keep, clear=True):
        engine = AutoDiscoveryEngine()
        # 同时屏蔽文件扫描避免 HOME 下的 .gitconfig 触发 github
        with patch.object(AutoDiscoveryEngine, "_walk_files", return_value=[]):
            results = engine.scan()
    assert all(r.confidence < 0.5 for r in results)


# --------------------------------------------------------------------------- #
# AutoDiscoveryEngine — file scan (mock filesystem)
# --------------------------------------------------------------------------- #

def test_scan_files_detects_gitconfig():
    """模拟 ~/.gitconfig 存在时应检测到 github 信号."""
    # 使用临时目录创建真实文件, 让 Path.exists() 通过
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        gitconfig = Path(tmpdir) / ".gitconfig"
        gitconfig.write_text("[user]\nname = test\n")
        engine = AutoDiscoveryEngine(scan_paths=[tmpdir], max_depth=1)
        with patch.dict(os.environ, {}, clear=True):
            results = engine.scan()
    names = [r.connector_name for r in results]
    assert "github" in names
    gh = next(r for r in results if r.connector_name == "github")
    assert any(s.source_type == "file" for s in gh.signals)


def test_scan_files_aws_credentials():
    """模拟 ~/.aws/credentials 存在时应检测到 s3 信号."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        aws_dir = Path(tmpdir) / ".aws"
        aws_dir.mkdir()
        creds = aws_dir / "credentials"
        creds.write_text("[default]\naws_access_key_id = AKIA\n")
        engine = AutoDiscoveryEngine(scan_paths=[tmpdir], max_depth=2)
        with patch.dict(os.environ, {}, clear=True):
            results = engine.scan()
    names = [r.connector_name for r in results]
    assert "s3" in names
    s3 = next(r for r in results if r.connector_name == "s3")
    assert s3.confidence >= 0.8


def test_scan_files_no_match():
    """模拟空目录不应产生 file 类信号."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = AutoDiscoveryEngine(scan_paths=[tmpdir], max_depth=1)
        with patch.dict(os.environ, {}, clear=True):
            results = engine.scan()
    assert not any(
        s.source_type == "file" for r in results for s in r.signals
    )


# --------------------------------------------------------------------------- #
# AutoDiscoveryEngine — config file scan
# --------------------------------------------------------------------------- #

def test_scan_config_json(tmp_path: Path):
    """扫描包含连接器键的 JSON 配置文件."""
    cfg = tmp_path / "neuralmem.json"
    cfg.write_text(json.dumps({"notion_token": "ntoken", "slack_token": "stoken"}))
    engine = AutoDiscoveryEngine(config_files=[str(cfg)], max_depth=0)
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(engine, "_walk_files", return_value=[]):
            results = engine.scan()
    names = [r.connector_name for r in results]
    assert "notion" in names
    assert "slack" in names
    notion = next(r for r in results if r.connector_name == "notion")
    assert notion.recommended_config.get("notion_token") == "ntoken"


def test_scan_dotenv(tmp_path: Path):
    """扫描 .env 文件中的键值对."""
    env_file = tmp_path / ".env"
    env_file.write_text("GITHUB_TOKEN=ghp_123\nAWS_ACCESS_KEY_ID=AKIA\n")
    engine = AutoDiscoveryEngine(config_files=[str(env_file)], max_depth=0)
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(engine, "_walk_files", return_value=[]):
            results = engine.scan()
    names = [r.connector_name for r in results]
    assert "github" in names
    assert "s3" in names


def test_scan_config_missing_file():
    """不存在的配置文件应被静默跳过."""
    engine = AutoDiscoveryEngine(
        config_files=["/nonexistent/config.json"], max_depth=0
    )
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(engine, "_walk_files", return_value=[]):
            results = engine.scan()
    # 无 env 无 file 无 config = 空结果或极低置信度
    assert all(r.confidence < 0.2 for r in results)


# --------------------------------------------------------------------------- #
# AutoDiscoveryEngine — confidence scoring
# --------------------------------------------------------------------------- #

def test_confidence_capped_at_1():
    """多个信号累加后置信度不应超过 1.0."""
    engine = AutoDiscoveryEngine()
    # 注入大量信号
    for _ in range(10):
        engine._add_signal("s3", "env", None, "fake", 0.3)
    suggestion = engine._suggestions["s3"]
    assert suggestion.confidence == 1.0


def test_results_sorted_by_confidence():
    """scan() 结果必须按置信度降序排列."""
    engine = AutoDiscoveryEngine()
    engine._add_signal("s3", "env", None, "low", 0.2)
    engine._add_signal("github", "env", None, "high", 0.9)
    engine._add_signal("notion", "env", None, "mid", 0.5)
    results = engine.scan()
    confidences = [r.confidence for r in results]
    assert confidences == sorted(confidences, reverse=True)


def test_low_confidence_filtered():
    """置信度低于 0.15 的建议应被过滤."""
    engine = AutoDiscoveryEngine()
    engine._add_signal("s3", "env", None, "tiny", 0.05)
    engine._add_signal("github", "env", None, "ok", 0.2)
    results = engine.scan()
    names = [r.connector_name for r in results]
    assert "github" in names
    assert "s3" not in names


# --------------------------------------------------------------------------- #
# AutoDiscoveryEngine — scan_single
# --------------------------------------------------------------------------- #

def test_scan_single_connector():
    """scan_single 应仅返回目标连接器的信号."""
    engine = AutoDiscoveryEngine()
    # 使用临时目录 + 真实文件让文件扫描产生可控信号
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        gitconfig = Path(tmpdir) / ".gitconfig"
        gitconfig.write_text("[user]\nname = test\n")
        with patch.dict(os.environ, {}, clear=True):
            result = engine.scan_single("github")
    assert result is not None
    assert result.connector_name == "github"
    # 仅有 .gitconfig 一个文件信号, 权重 0.4
    assert result.confidence == 0.4


def test_scan_single_no_signals():
    engine = AutoDiscoveryEngine()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {}, clear=True):
            result = engine.scan_single("notion")
    assert result is None


# --------------------------------------------------------------------------- #
# ConnectorAutoDiscovery — discover
# --------------------------------------------------------------------------- #

def test_cad_discover_returns_suggestions():
    engine = MagicMock()
    engine.scan.return_value = [
        ConnectorSuggestion(
            connector_name="s3", confidence=0.9, signals=[], description=""
        )
    ]
    cad = ConnectorAutoDiscovery(engine=engine)
    results = cad.discover()
    assert len(results) == 1
    assert results[0].connector_name == "s3"
    engine.scan.assert_called_once()


# --------------------------------------------------------------------------- #
# ConnectorAutoDiscovery — connect (mock registry)
# --------------------------------------------------------------------------- #

def test_cad_connect_success(mock_connector_cls):
    ConnectorRegistry.register(
        "mock", "tests.unit.test_auto_discover", "_MockConn", "mock"
    )
    suggestion = ConnectorSuggestion(
        connector_name="mock",
        confidence=0.9,
        recommended_config={"token": "abc"},
        description="",
    )
    cad = ConnectorAutoDiscovery()
    conn, sug = cad.connect(suggestion)
    assert isinstance(conn, mock_connector_cls)
    assert conn.state == ConnectorState.CONNECTED
    assert conn.config["token"] == "abc"


def test_cad_connect_missing_connector():
    suggestion = ConnectorSuggestion(
        connector_name="missing",
        confidence=0.9,
        recommended_config={},
        description="",
    )
    cad = ConnectorAutoDiscovery()
    with pytest.raises(ValueError, match="not registered"):
        cad.connect(suggestion)


def test_cad_connect_auth_failure():
    ConnectorRegistry.register(
        "failing", "tests.unit.test_auto_discover", "_FailingConn", "mock"
    )
    suggestion = ConnectorSuggestion(
        connector_name="failing",
        confidence=0.9,
        recommended_config={},
        description="",
    )
    cad = ConnectorAutoDiscovery()
    with pytest.raises(ValueError, match="authentication failed"):
        cad.connect(suggestion)


# --------------------------------------------------------------------------- #
# ConnectorAutoDiscovery — connect_best
# --------------------------------------------------------------------------- #

def test_cad_connect_best_picks_highest():
    ConnectorRegistry.register(
        "best", "tests.unit.test_auto_discover", "_MockConn", "mock"
    )
    cad = ConnectorAutoDiscovery(min_confidence=0.5)
    cad._last_suggestions = [
        ConnectorSuggestion("low", 0.2, [], {}, ""),
        ConnectorSuggestion("best", 0.9, [], {"token": "t"}, ""),
    ]
    conn, sug = cad.connect_best()
    assert conn is not None
    assert sug.connector_name == "best"


def test_cad_connect_best_none_available():
    cad = ConnectorAutoDiscovery(min_confidence=0.9)
    cad._last_suggestions = [
        ConnectorSuggestion("low", 0.2, [], {}, ""),
    ]
    conn, sug = cad.connect_best()
    assert conn is None
    assert sug is None


def test_cad_connect_best_falls_back():
    """第一个失败时自动尝试下一个."""
    ConnectorRegistry.register(
        "fail", "tests.unit.test_auto_discover", "_FailAuth", "mock"
    )
    ConnectorRegistry.register(
        "ok", "tests.unit.test_auto_discover", "_OkAuth", "mock"
    )

    cad = ConnectorAutoDiscovery(min_confidence=0.5)
    cad._last_suggestions = [
        ConnectorSuggestion("fail", 0.9, [], {}, ""),
        ConnectorSuggestion("ok", 0.8, [], {}, ""),
    ]
    conn, sug = cad.connect_best()
    assert conn is not None
    assert sug.connector_name == "ok"


# --------------------------------------------------------------------------- #
# ConnectorAutoDiscovery — env var resolution
# --------------------------------------------------------------------------- #

def test_resolve_env_vars():
    with patch.dict(os.environ, {"MY_TOKEN": "secret123"}, clear=False):
        result = ConnectorAutoDiscovery._resolve_env_vars("Bearer ${MY_TOKEN}")
    assert result == "Bearer secret123"


def test_resolve_env_vars_missing():
    result = ConnectorAutoDiscovery._resolve_env_vars("${MISSING_VAR}")
    assert result == "${MISSING_VAR}"


def test_resolve_config_with_lists():
    with patch.dict(
        os.environ, {"PATH_A": "/a", "PATH_B": "/b"}, clear=False
    ):
        resolved = ConnectorAutoDiscovery._resolve_config(
            {"paths": ["${PATH_A}", "${PATH_B}", 123]}
        )
    assert resolved["paths"] == ["/a", "/b", 123]


# --------------------------------------------------------------------------- #
# ConnectorAutoDiscovery — get_suggestions_for_connector
# --------------------------------------------------------------------------- #

def test_get_suggestions_from_cache():
    cad = ConnectorAutoDiscovery()
    cad._last_suggestions = [
        ConnectorSuggestion("s3", 0.7, [], {}, ""),
    ]
    result = cad.get_suggestions_for_connector("s3")
    assert result is not None
    assert result.confidence == 0.7


def test_get_suggestions_scan_fallback():
    """缓存无命中时触发单次扫描."""
    engine = MagicMock()
    engine.scan_single.return_value = ConnectorSuggestion(
        "github", 0.6, [], {}, ""
    )
    cad = ConnectorAutoDiscovery(engine=engine)
    result = cad.get_suggestions_for_connector("github")
    assert result is not None
    assert result.connector_name == "github"
    engine.scan_single.assert_called_once_with("github")


# --------------------------------------------------------------------------- #
# 边界与异常
# --------------------------------------------------------------------------- #

def test_walk_files_permission_error():
    """_walk_files 遇到权限错误时不应崩溃."""
    root = MagicMock()
    root.iterdir.side_effect = PermissionError("denied")
    # 生成器需遍历才会触发异常
    list(AutoDiscoveryEngine._walk_files(root, 1))


def test_parse_dotenv():
    content = "# comment\nKEY1=val1\nKEY2='val2'\nKEY3=\"val3\"\n\n"
    result = AutoDiscoveryEngine._parse_dotenv(content)
    assert result == {"KEY1": "val1", "KEY2": "val2", "KEY3": "val3"}


def test_scan_with_broken_json_config(tmp_path: Path):
    """损坏的 JSON 配置文件应被静默跳过."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json}")
    engine = AutoDiscoveryEngine(config_files=[str(bad_json)], max_depth=0)
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(engine, "_walk_files", return_value=[]):
            results = engine.scan()
    # 不应抛出异常, 结果可能为空或极低置信度
    assert isinstance(results, list)


# --------------------------------------------------------------------------- #
# 集成风格端到端 (全 mock)
# --------------------------------------------------------------------------- #

def test_end_to_end_discovery_and_connect(mock_connector_cls):
    """模拟完整流程: 发现 -> 连接 -> sync -> disconnect."""
    # 注册 filesystem 连接器 (使用模块级 _MockConn 作为实现占位)
    ConnectorRegistry.register(
        "filesystem", "tests.unit.test_auto_discover", "_MockConn", "mock"
    )
    # 使用临时目录 + 真实文件触发 filesystem 信号
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        note = Path(tmpdir) / "notes.txt"
        note.write_text("hello")
        engine = AutoDiscoveryEngine(scan_paths=[tmpdir], max_depth=1)
        with patch.dict(os.environ, {}, clear=True):
            suggestions = engine.scan()

    # 至少检测到 filesystem 连接器
    names = [s.connector_name for s in suggestions]
    assert "filesystem" in names

    fs_suggestion = next(s for s in suggestions if s.connector_name == "filesystem")
    cad = ConnectorAutoDiscovery(engine=engine)
    conn, _ = cad.connect(fs_suggestion)
    assert conn.authenticate() is True
    items = conn.sync()
    assert items == []
    conn.disconnect()
    assert conn.state == ConnectorState.DISCONNECTED
