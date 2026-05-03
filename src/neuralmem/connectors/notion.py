"""Notion API 连接器 — 提取 Pages 和 Databases 内容."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from neuralmem.connectors.base import ConnectorProtocol, ConnectorState, SyncItem

_logger = logging.getLogger(__name__)

# Graceful degradation: optional dependency
try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class NotionConnector(ConnectorProtocol):
    """Notion API 连接器.

    Config keys
    -----------
    token : str
        Notion integration token.
    database_ids : list[str] | None
        指定同步的数据库 ID 列表; None 则同步所有可访问数据库.
    page_ids : list[str] | None
        指定同步的页面 ID 列表.
    """

    def __init__(self, name: str = "notion", config: dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        if not _HAS_REQUESTS:
            self._available = False
            self._set_error("NotionConnector requires 'requests'. Install: pip install requests")
            return
        self.token = self.config.get("token", "")
        self.database_ids: list[str] | None = self.config.get("database_ids")
        self.page_ids: list[str] | None = self.config.get("page_ids")
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        }
        self._session: Any | None = None

    def authenticate(self) -> bool:
        """验证 token 有效性."""
        if not self._available:
            return False
        if not self.token:
            self._set_error("Notion token not provided in config")
            return False
        try:
            import requests

            resp = requests.get(
                f"{self.base_url}/users/me",
                headers=self.headers,
                timeout=30,
            )
            if resp.status_code == 200:
                self.state = ConnectorState.CONNECTED
                _logger.info("Notion authenticated as %s", resp.json().get("name"))
                return True
            self._set_error(f"Notion auth failed: HTTP {resp.status_code} - {resp.text}")
            return False
        except Exception as exc:
            self._set_error(f"Notion auth error: {exc}")
            return False

    def sync(self, **kwargs: Any) -> list[SyncItem]:
        """同步 Notion 内容.

        Parameters
        ----------
        since : str | None
            ISO 8601 时间戳, 仅同步此后更新的内容.
        limit : int
            每类最大提取数量, 默认 100.
        """
        if not self._available or self.state != ConnectorState.CONNECTED:
            _logger.warning("NotionConnector not available or not connected")
            return []

        since: str | None = kwargs.get("since")
        limit: int = kwargs.get("limit", 100)
        items: list[SyncItem] = []

        try:
            import requests

            self.state = ConnectorState.SYNCING

            # 1) Sync databases
            db_ids = self.database_ids or self._list_databases(limit)
            for db_id in db_ids[:limit]:
                db_items = self._query_database(db_id, since=since, limit=limit)
                items.extend(db_items)

            # 2) Sync pages
            page_ids = self.page_ids or []
            for page_id in page_ids[:limit]:
                page_item = self._fetch_page(page_id)
                if page_item:
                    items.append(page_item)

            _logger.info("Notion sync completed: %d items", len(items))
            self.state = ConnectorState.CONNECTED
            return items
        except Exception as exc:
            self._set_error(f"Notion sync error: {exc}")
            return []

    def disconnect(self) -> None:
        """清理会话."""
        self._session = None
        self.state = ConnectorState.DISCONNECTED

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_databases(self, limit: int) -> list[str]:
        import requests

        resp = requests.post(
            f"{self.base_url}/databases",
            headers=self.headers,
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = data.get("results", [])
        return [db["id"] for db in results[:limit]]

    def _query_database(
        self, db_id: str, since: str | None, limit: int
    ) -> list[SyncItem]:
        import requests

        body: dict[str, Any] = {"page_size": limit}
        if since:
            body["filter"] = {
                "timestamp": "last_edited_time",
                "last_edited_time": {"after": since},
            }

        resp = requests.post(
            f"{self.base_url}/databases/{db_id}/query",
            headers=self.headers,
            json=body,
            timeout=30,
        )
        if resp.status_code != 200:
            return []

        items: list[SyncItem] = []
        for page in resp.json().get("results", []):
            item = self._page_to_item(page)
            if item:
                items.append(item)
        return items

    def _fetch_page(self, page_id: str) -> SyncItem | None:
        import requests

        resp = requests.get(
            f"{self.base_url}/pages/{page_id}",
            headers=self.headers,
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        return self._page_to_item(resp.json())

    def _page_to_item(self, page: dict[str, Any]) -> SyncItem | None:
        props = page.get("properties", {})
        title = ""
        for v in props.values():
            if v.get("type") == "title":
                title_parts = v.get("title", [])
                title = "".join(t.get("plain_text", "") for t in title_parts)
                break

        created = page.get("created_time", "")
        updated = page.get("last_edited_time", "")
        url = page.get("url", "")
        page_id = page.get("id", "")

        # Fetch block children for content text
        content_text = self._fetch_page_content(page_id)

        return SyncItem(
            id=page_id,
            content=content_text or title,
            source="notion",
            source_url=url,
            title=title or None,
            author=None,
            created_at=self._parse_iso(created),
            updated_at=self._parse_iso(updated),
            metadata={"notion_page_id": page_id, "properties": props},
            tags=["notion"],
        )

    def _fetch_page_content(self, page_id: str) -> str:
        import requests

        resp = requests.get(
            f"{self.base_url}/blocks/{page_id}/children",
            headers=self.headers,
            timeout=30,
        )
        if resp.status_code != 200:
            return ""
        texts: list[str] = []
        for block in resp.json().get("results", []):
            block_type = block.get("type", "")
            block_data = block.get(block_type, {})
            rich_text = block_data.get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in rich_text)
            if text:
                texts.append(text)
        return "\n".join(texts)

    @staticmethod
    def _parse_iso(ts: str) -> datetime:
        if not ts:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
