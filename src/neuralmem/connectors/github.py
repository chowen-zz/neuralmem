"""GitHub Issue/PR 导入连接器 — 提取 Issues 和 Pull Requests."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from neuralmem.connectors.base import ConnectorProtocol, ConnectorState, SyncItem

_logger = logging.getLogger(__name__)

# Graceful degradation
try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class GitHubConnector(ConnectorProtocol):
    """GitHub API 连接器.

    Config keys
    -----------
    token : str | None
        GitHub Personal Access Token (可选, 用于私有仓库).
    owner : str
        仓库所有者.
    repo : str
        仓库名称.
    include_issues : bool
        是否同步 Issues, 默认 True.
    include_prs : bool
        是否同步 Pull Requests, 默认 True.
    state : str
        'open', 'closed', 'all' — 默认 'all'.
    """

    def __init__(self, name: str = "github", config: dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        if not _HAS_REQUESTS:
            self._available = False
            self._set_error("GitHubConnector requires 'requests'. Install: pip install requests")
            return
        self.token: str | None = self.config.get("token")
        self.owner: str = self.config.get("owner", "")
        self.repo: str = self.config.get("repo", "")
        self.include_issues: bool = self.config.get("include_issues", True)
        self.include_prs: bool = self.config.get("include_prs", True)
        self.state_filter: str = self.config.get("state", "all")
        self.base_url = "https://api.github.com"
        self.headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def authenticate(self) -> bool:
        """验证 token 或检查仓库可访问性."""
        if not self._available:
            return False
        if not self.owner or not self.repo:
            self._set_error("GitHub 'owner' and 'repo' required in config")
            return False
        try:
            import requests

            resp = requests.get(
                f"{self.base_url}/repos/{self.owner}/{self.repo}",
                headers=self.headers,
                timeout=30,
            )
            if resp.status_code == 200:
                self.state = ConnectorState.CONNECTED
                _logger.info("GitHub repo %s/%s accessible", self.owner, self.repo)
                return True
            if resp.status_code == 404:
                self._set_error(f"GitHub repo {self.owner}/{self.repo} not found or not accessible")
            else:
                self._set_error(f"GitHub auth failed: HTTP {resp.status_code}")
            return False
        except Exception as exc:
            self._set_error(f"GitHub auth error: {exc}")
            return False

    def sync(self, **kwargs: Any) -> list[SyncItem]:
        """同步 GitHub Issues 和 PRs.

        Parameters
        ----------
        since : str | None
            ISO 8601 时间戳, 仅同步此后更新的内容.
        limit : int
            最大提取数量, 默认 100.
        """
        if not self._available or self.state != ConnectorState.CONNECTED:
            _logger.warning("GitHubConnector not available or not connected")
            return []

        since: str | None = kwargs.get("since")
        limit: int = kwargs.get("limit", 100)
        items: list[SyncItem] = []

        try:
            self.state = ConnectorState.SYNCING

            if self.include_issues:
                issue_items = self._fetch_issues(since=since, limit=limit)
                items.extend(issue_items)

            if self.include_prs:
                pr_items = self._fetch_pull_requests(since=since, limit=limit)
                items.extend(pr_items)

            _logger.info("GitHub sync completed: %d items", len(items))
            self.state = ConnectorState.CONNECTED
            return items
        except Exception as exc:
            self._set_error(f"GitHub sync error: {exc}")
            return []

    def disconnect(self) -> None:
        """无状态 HTTP, 仅标记断开."""
        self.state = ConnectorState.DISCONNECTED

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_issues(self, since: str | None, limit: int) -> list[SyncItem]:
        import requests

        params: dict[str, Any] = {
            "state": self.state_filter,
            "per_page": min(limit, 100),
            "sort": "updated",
            "direction": "desc",
        }
        if since:
            params["since"] = since

        resp = requests.get(
            f"{self.base_url}/repos/{self.owner}/{self.repo}/issues",
            headers=self.headers,
            params=params,
            timeout=30,
        )
        if resp.status_code != 200:
            return []

        items: list[SyncItem] = []
        for issue in resp.json():
            # Skip PRs (they appear in /issues too)
            if "pull_request" in issue:
                continue
            item = self._issue_to_item(issue)
            if item:
                items.append(item)
        return items

    def _fetch_pull_requests(self, since: str | None, limit: int) -> list[SyncItem]:
        import requests

        params: dict[str, Any] = {
            "state": self.state_filter,
            "per_page": min(limit, 100),
            "sort": "updated",
            "direction": "desc",
        }

        resp = requests.get(
            f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls",
            headers=self.headers,
            params=params,
            timeout=30,
        )
        if resp.status_code != 200:
            return []

        items: list[SyncItem] = []
        for pr in resp.json():
            item = self._pr_to_item(pr)
            if item:
                items.append(item)
        return items

    def _issue_to_item(self, issue: dict[str, Any]) -> SyncItem | None:
        body = issue.get("body") or ""
        title = issue.get("title", "")
        content = f"{title}\n\n{body}".strip()
        if not content:
            return None

        return SyncItem(
            id=str(issue.get("number", "")),
            content=content,
            source="github",
            source_url=issue.get("html_url"),
            title=title or None,
            author=issue.get("user", {}).get("login") or None,
            created_at=self._parse_iso(issue.get("created_at", "")),
            updated_at=self._parse_iso(issue.get("updated_at", "")),
            metadata={
                "type": "issue",
                "state": issue.get("state"),
                "labels": [l["name"] for l in issue.get("labels", [])],
                "number": issue.get("number"),
            },
            tags=["github", "issue"],
        )

    def _pr_to_item(self, pr: dict[str, Any]) -> SyncItem | None:
        body = pr.get("body") or ""
        title = pr.get("title", "")
        content = f"{title}\n\n{body}".strip()
        if not content:
            return None

        return SyncItem(
            id=str(pr.get("number", "")),
            content=content,
            source="github",
            source_url=pr.get("html_url"),
            title=title or None,
            author=pr.get("user", {}).get("login") or None,
            created_at=self._parse_iso(pr.get("created_at", "")),
            updated_at=self._parse_iso(pr.get("updated_at", "")),
            metadata={
                "type": "pull_request",
                "state": pr.get("state"),
                "draft": pr.get("draft", False),
                "number": pr.get("number"),
                "base_branch": pr.get("base", {}).get("ref"),
                "head_branch": pr.get("head", {}).get("ref"),
            },
            tags=["github", "pull_request"],
        )

    @staticmethod
    def _parse_iso(ts: str) -> datetime:
        if not ts:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
