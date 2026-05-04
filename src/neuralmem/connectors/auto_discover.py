"""NeuralMem V1.5 自动连接器发现 — 扫描用户环境并建议可用连接.

AutoDiscoveryEngine 扫描常见数据源模式（配置文件、环境变量、本地文件、
浏览器书签导出等），根据检测到的信号为每个连接器计算置信度分数，并返回
排序后的建议列表。支持一键连接设置（通过 ConnectorRegistry 实例化并认证）。
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neuralmem.connectors.base import ConnectorProtocol
from neuralmem.connectors.registry import ConnectorRegistry

_logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #

@dataclass
class DiscoverySignal:
    """单个检测到的数据源信号.

    Attributes
    ----------
    source_type : str
        信号来源类别, 如 'env', 'file', 'url', 'config'.
    path : str | None
        检测到的文件路径或 URL.
    detail : str
        人类可读描述.
    weight : float
        该信号对置信度的贡献权重 (0.0–1.0).
    """

    source_type: str
    path: str | None
    detail: str
    weight: float


@dataclass
class ConnectorSuggestion:
    """连接器建议结果.

    Attributes
    ----------
    connector_name : str
        建议的连接器名称 (与 ConnectorRegistry 注册名一致).
    confidence : float
        置信度分数 (0.0–1.0).
    signals : list[DiscoverySignal]
        支撑该建议的所有检测信号.
    recommended_config : dict[str, Any]
        推荐的初始配置字典 (可能包含占位符).
    description : str
        建议描述文案.
    """

    connector_name: str
    confidence: float
    signals: list[DiscoverySignal] = field(default_factory=list)
    recommended_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""


# --------------------------------------------------------------------------- #
# 检测规则
# --------------------------------------------------------------------------- #

# 连接器 -> 环境变量名 -> 权重
_ENV_PATTERNS: dict[str, dict[str, float]] = {
    "notion": {"NOTION_TOKEN": 0.9, "NOTION_API_KEY": 0.9, "NOTION_INTEGRATION_TOKEN": 0.85},
    "slack": {"SLACK_BOT_TOKEN": 0.9, "SLACK_API_TOKEN": 0.85, "SLACK_TOKEN": 0.8},
    "github": {"GITHUB_TOKEN": 0.9, "GH_TOKEN": 0.85, "GITHUB_API_TOKEN": 0.85},
    "gdrive": {"GOOGLE_APPLICATION_CREDENTIALS": 0.9, "GDRIVE_CREDENTIALS": 0.85},
    "s3": {"AWS_ACCESS_KEY_ID": 0.9, "AWS_SECRET_ACCESS_KEY": 0.9, "AWS_DEFAULT_REGION": 0.3},
    "filesystem": {"NEURALMEM_FS_PATHS": 0.7},
}

# 连接器 -> 文件路径 glob / 正则 -> 权重
_FILE_PATTERNS: dict[str, dict[str, float]] = {
    "notion": {
        r"notion.*export.*\.zip": 0.7,
        r"notion.*backup.*\.zip": 0.6,
        r"notion.*\.csv": 0.5,
    },
    "slack": {
        r"slack.*export.*\.zip": 0.7,
        r"slack.*backup.*\.zip": 0.6,
    },
    "github": {
        r"\.gitconfig": 0.4,
        r"\.git.*credentials": 0.5,
    },
    "gdrive": {
        r"credentials.*\.json": 0.6,
        r"client_secret.*\.json": 0.65,
        r"gdrive.*\.json": 0.6,
    },
    "s3": {
        r"\.aws[\\/]credentials": 0.8,
        r"\.aws[\\/]config": 0.5,
        r"s3.*\.json": 0.5,
    },
    "filesystem": {
        r"\.md$": 0.25,
        r"\.txt$": 0.2,
        r"notes.*": 0.3,
        r"documents.*": 0.25,
    },
}

# 连接器 -> URL 域名正则 -> 权重
_URL_PATTERNS: dict[str, dict[str, float]] = {
    "notion": {r"notion\.so": 0.8, r"notion\.site": 0.8},
    "slack": {r"slack\.com": 0.8, r"\.slack\.com": 0.85},
    "github": {r"github\.com": 0.8, r"github\.io": 0.6},
    "gdrive": {r"drive\.google\.com": 0.8, r"docs\.google\.com": 0.7},
    "s3": {r"s3\.amazonaws\.com": 0.8, r"\.s3\.": 0.75},
}

# 连接器 -> 配置文件中期望的键 -> 权重
_CONFIG_KEY_PATTERNS: dict[str, dict[str, float]] = {
    "notion": {"notion_token": 0.9, "notion_database_id": 0.7},
    "slack": {"slack_token": 0.9, "slack_channel": 0.6},
    "github": {"github_token": 0.9, "github_repo": 0.6},
    "gdrive": {"gdrive_credentials": 0.85, "google_credentials": 0.8},
    "s3": {"s3_bucket": 0.8, "aws_region": 0.5, "aws_access_key": 0.9},
    "filesystem": {"fs_paths": 0.7, "watch_paths": 0.6},
}

# 一键连接时使用的默认占位配置
_DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "notion": {"token": "${NOTION_TOKEN}"},
    "slack": {"token": "${SLACK_BOT_TOKEN}"},
    "github": {"token": "${GITHUB_TOKEN}"},
    "gdrive": {"credentials_path": "${GOOGLE_APPLICATION_CREDENTIALS}"},
    "s3": {
        "aws_access_key_id": "${AWS_ACCESS_KEY_ID}",
        "aws_secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
        "region_name": "${AWS_DEFAULT_REGION}",
    },
    "filesystem": {"paths": ["."]},
}


# --------------------------------------------------------------------------- #
# 核心引擎
# --------------------------------------------------------------------------- #

class AutoDiscoveryEngine:
    """自动发现引擎 — 扫描环境并生成连接器建议.

    扫描维度:
    1. 环境变量 (os.environ)
    2. 用户主目录常见文件 (~/.aws/credentials, ~/.gitconfig 等)
    3. 当前工作目录文件模式
    4. 可选配置文件 (neuralmem.json, .env 等)
    5. 已知 URL 模式 (从浏览器书签或历史记录导出中解析)

    用法::

        engine = AutoDiscoveryEngine()
        suggestions = engine.scan()
        for s in suggestions:
            print(s.connector_name, s.confidence)
    """

    def __init__(
        self,
        scan_paths: list[str] | None = None,
        config_files: list[str] | None = None,
        max_depth: int = 2,
    ) -> None:
        """初始化发现引擎.

        Parameters
        ----------
        scan_paths : list[str] | None
            额外扫描目录列表, 默认包含用户主目录和当前工作目录.
        config_files : list[str] | None
            配置文件路径列表, 默认扫描常见位置.
        max_depth : int
            目录递归扫描最大深度, 默认 2.
        """
        self.scan_paths = scan_paths or [str(Path.home()), "."]
        self.config_files = config_files or [
            "neuralmem.json",
            ".neuralmem.json",
            ".env",
            "config.json",
        ]
        self.max_depth = max_depth
        self._suggestions: dict[str, ConnectorSuggestion] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def scan(self) -> list[ConnectorSuggestion]:
        """执行完整扫描并返回排序后的建议列表.

        Returns
        -------
        list[ConnectorSuggestion]
            按置信度降序排列的建议.
        """
        self._suggestions.clear()

        self._scan_env_vars()
        self._scan_files()
        self._scan_config_files()

        # 按置信度降序排序, 过滤低置信度
        results = [
            s for s in self._suggestions.values() if s.confidence >= 0.15
        ]
        results.sort(key=lambda x: x.confidence, reverse=True)
        _logger.info("AutoDiscoveryEngine found %d suggestions", len(results))
        return results

    def scan_single(self, connector_name: str) -> ConnectorSuggestion | None:
        """仅扫描指定连接器的信号.

        Parameters
        ----------
        connector_name : str
            目标连接器名称.

        Returns
        -------
        ConnectorSuggestion | None
            若检测到信号则返回建议, 否则 None.
        """
        self._suggestions.clear()
        self._scan_env_vars(connector_name)
        self._scan_files(connector_name)
        self._scan_config_files(connector_name)
        return self._suggestions.get(connector_name)

    # ------------------------------------------------------------------ #
    # 内部扫描方法
    # ------------------------------------------------------------------ #

    def _add_signal(
        self,
        connector_name: str,
        source_type: str,
        path: str | None,
        detail: str,
        weight: float,
    ) -> None:
        """向建议累加信号并更新置信度."""
        if connector_name not in self._suggestions:
            self._suggestions[connector_name] = ConnectorSuggestion(
                connector_name=connector_name,
                confidence=0.0,
                recommended_config=_DEFAULT_CONFIGS.get(connector_name, {}),
                description=f"Detected potential {connector_name} data source.",
            )
        suggestion = self._suggestions[connector_name]
        suggestion.signals.append(
            DiscoverySignal(
                source_type=source_type,
                path=path,
                detail=detail,
                weight=weight,
            )
        )
        # 使用加权累加 + 上限 1.0 的简单模型
        suggestion.confidence = min(1.0, suggestion.confidence + weight)

    def _scan_env_vars(self, connector_name: str | None = None) -> None:
        """扫描环境变量中的连接器凭证信号."""
        targets = (
            [connector_name]
            if connector_name
            else list(_ENV_PATTERNS.keys())
        )
        for name in targets:
            patterns = _ENV_PATTERNS.get(name, {})
            for env_var, weight in patterns.items():
                if env_var in os.environ:
                    value = os.environ[env_var]
                    masked = value[:4] + "***" if len(value) > 4 else "***"
                    self._add_signal(
                        name,
                        "env",
                        None,
                        f"Environment variable {env_var}={masked} found",
                        weight,
                    )

    def _scan_files(self, connector_name: str | None = None) -> None:
        """扫描文件系统常见模式."""
        targets = (
            [connector_name]
            if connector_name
            else list(_FILE_PATTERNS.keys())
        )
        for name in targets:
            patterns = _FILE_PATTERNS.get(name, {})
            for scan_path in self.scan_paths:
                root = Path(scan_path).expanduser().resolve()
                if not root.exists():
                    continue
                for pattern, weight in patterns.items():
                    compiled = re.compile(pattern, re.IGNORECASE)
                    try:
                        for filepath in AutoDiscoveryEngine._walk_files(root, self.max_depth):
                            if compiled.search(str(filepath)):
                                self._add_signal(
                                    name,
                                    "file",
                                    str(filepath),
                                    f"File pattern '{pattern}' matched: {filepath.name}",
                                    weight,
                                )
                                # 每个 pattern 只计一次, 避免同目录大量文件刷分
                                break
                    except (OSError, PermissionError):
                        pass

    def _scan_config_files(self, connector_name: str | None = None) -> None:
        """扫描配置文件中的连接器相关键."""
        targets = (
            [connector_name]
            if connector_name
            else list(_CONFIG_KEY_PATTERNS.keys())
        )
        for cfg_file in self.config_files:
            path = Path(cfg_file)
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                data: dict[str, Any] = {}
                if path.suffix == ".json":
                    data = json.loads(content)
                else:
                    # 简单解析 KEY=VALUE 格式 (.env)
                    data = self._parse_dotenv(content)

                for name in targets:
                    patterns = _CONFIG_KEY_PATTERNS.get(name, {})
                    for key, weight in patterns.items():
                        if key in data or any(
                            key in str(k).lower() for k in data.keys()
                        ):
                            self._add_signal(
                                name,
                                "config",
                                str(path),
                                f"Config key '{key}' found in {path.name}",
                                weight,
                            )
                            # 更新推荐配置为实际值(若存在)
                            suggestion = self._suggestions.get(name)
                            if suggestion and key in data:
                                suggestion.recommended_config[key] = data[key]
            except (OSError, json.JSONDecodeError):
                pass

    # ------------------------------------------------------------------ #
    # 工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _walk_files(root: Path, max_depth: int) -> Any:
        """有限深度递归遍历文件."""
        if max_depth <= 0:
            return
        try:
            for entry in root.iterdir():
                if entry.is_file():
                    yield entry
                elif entry.is_dir() and entry.name not in (".git", "__pycache__", ".pytest_cache"):
                    yield from AutoDiscoveryEngine._walk_files(
                        entry, max_depth - 1
                    )
        except (OSError, PermissionError):
            pass

    @staticmethod
    def _parse_dotenv(content: str) -> dict[str, str]:
        """解析 .env 风格内容."""
        result: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip().strip('"').strip("'")
        return result


# --------------------------------------------------------------------------- #
# 一键连接设置
# --------------------------------------------------------------------------- #

class ConnectorAutoDiscovery:
    """高级封装 — 发现 + 一键连接.

    整合 AutoDiscoveryEngine 与 ConnectorRegistry, 提供高层 API:
    - discover(): 扫描并返回建议
    - connect(): 对指定建议执行一键连接 (实例化 + 认证)
    - connect_best(): 自动连接置信度最高的可用连接器

    用法::

        cad = ConnectorAutoDiscovery()
        suggestions = cad.discover()
        if suggestions:
            conn, suggestion = cad.connect_best()
            items = conn.sync()
    """

    def __init__(
        self,
        engine: AutoDiscoveryEngine | None = None,
        min_confidence: float = 0.3,
    ) -> None:
        """初始化.

        Parameters
        ----------
        engine : AutoDiscoveryEngine | None
            自定义扫描引擎, 默认新建实例.
        min_confidence : float
            可接受的最小置信度阈值.
        """
        self.engine = engine or AutoDiscoveryEngine()
        self.min_confidence = min_confidence
        self._last_suggestions: list[ConnectorSuggestion] = []

    def discover(self) -> list[ConnectorSuggestion]:
        """执行发现扫描.

        Returns
        -------
        list[ConnectorSuggestion]
            排序后的建议列表.
        """
        self._last_suggestions = self.engine.scan()
        return self._last_suggestions

    def connect(
        self, suggestion: ConnectorSuggestion
    ) -> tuple[ConnectorProtocol, ConnectorSuggestion]:
        """根据建议一键实例化并认证连接器.

        Parameters
        ----------
        suggestion : ConnectorSuggestion
            来自 discover() 的建议对象.

        Returns
        -------
        tuple[ConnectorProtocol, ConnectorSuggestion]
            已认证的连接器实例和对应建议.

        Raises
        ------
        ValueError
            连接器未注册或认证失败.
        """
        name = suggestion.connector_name
        if name not in ConnectorRegistry.list_connectors():
            raise ValueError(
                f"Connector '{name}' not registered in ConnectorRegistry. "
                f"Available: {ConnectorRegistry.list_connectors()}"
            )

        config = self._resolve_config(suggestion.recommended_config)
        instance = ConnectorRegistry.create(name, config)
        if not instance.authenticate():
            raise ValueError(
                f"Connector '{name}' authentication failed: "
                f"{instance.last_error() or 'unknown error'}"
            )
        return instance, suggestion

    def connect_best(
        self,
    ) -> tuple[ConnectorProtocol, ConnectorSuggestion] | tuple[None, None]:
        """自动连接置信度最高且超过阈值的建议.

        Returns
        -------
        tuple[ConnectorProtocol, ConnectorSuggestion] | tuple[None, None]
            成功返回 (connector, suggestion), 无可行建议返回 (None, None).
        """
        if not self._last_suggestions:
            self.discover()

        for suggestion in self._last_suggestions:
            if suggestion.confidence < self.min_confidence:
                continue
            try:
                return self.connect(suggestion)
            except Exception as exc:
                _logger.warning(
                    "Auto-connect '%s' failed: %s", suggestion.connector_name, exc
                )
                continue
        return None, None

    def get_suggestions_for_connector(
        self, connector_name: str
    ) -> ConnectorSuggestion | None:
        """获取指定连接器的最新建议.

        Parameters
        ----------
        connector_name : str
            连接器名称.

        Returns
        -------
        ConnectorSuggestion | None
            若存在则返回建议, 否则 None.
        """
        for s in self._last_suggestions:
            if s.connector_name == connector_name:
                return s
        # 若缓存中无, 执行单次扫描
        return self.engine.scan_single(connector_name)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_config(config: dict[str, Any]) -> dict[str, Any]:
        """解析配置中的环境变量占位符 ${VAR}."""
        resolved: dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, str):
                resolved[key] = ConnectorAutoDiscovery._resolve_env_vars(value)
            elif isinstance(value, list):
                resolved[key] = [
                    ConnectorAutoDiscovery._resolve_env_vars(v)
                    if isinstance(v, str)
                    else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved

    @staticmethod
    def _resolve_env_vars(value: str) -> str:
        """替换字符串中的 ${VAR} 为对应环境变量值."""
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return pattern.sub(replacer, value)
