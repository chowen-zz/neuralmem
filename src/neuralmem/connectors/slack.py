"""Slack API 连接器 — 提取频道消息和线程."""
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


class SlackConnector(ConnectorProtocol):
    """Slack API 连接器.

    Config keys
    -----------
    token : str
        Slack Bot User OAuth Token (xoxb-...).
    channels : list[str] | None
        频道 ID 或名称列表; None 则同步所有可访问频道.
    include_threads : bool
        是否提取线程回复, 默认 True.
    """

    def __init__(self, name: str = "slack", config: dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        if not _HAS_REQUESTS:
            self._available = False
            self._set_error("SlackConnector requires 'requests'. Install: pip install requests")
            return
        self.token = self.config.get("token", "")
        self.channels: list[str] | None = self.config.get("channels")
        self.include_threads: bool = self.config.get("include_threads", True)
        self.base_url = "https://slack.com/api"
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def authenticate(self) -> bool:
        """验证 token 并获取 bot 信息."""
        if not self._available:
            return False
        if not self.token:
            self._set_error("Slack token not provided in config")
            return False
        try:
            import requests

            resp = requests.get(
                f"{self.base_url}/auth.test",
                headers=self.headers,
                timeout=30,
            )
            data = resp.json()
            if data.get("ok"):
                self.state = ConnectorState.CONNECTED
                _logger.info("Slack authenticated as %s", data.get("user"))
                return True
            self._set_error(f"Slack auth failed: {data.get('error')}")
            return False
        except Exception as exc:
            self._set_error(f"Slack auth error: {exc}")
            return False

    def sync(self, **kwargs: Any) -> list[SyncItem]:
        """同步 Slack 消息.

        Parameters
        ----------
        since : float | None
            Unix 时间戳, 仅同步此时间之后的消息.
        limit : int
            每频道最大消息数, 默认 100.
        """
        if not self._available or self.state != ConnectorState.CONNECTED:
            _logger.warning("SlackConnector not available or not connected")
            return []

        since: float | None = kwargs.get("since")
        limit: int = kwargs.get("limit", 100)
        items: list[SyncItem] = []

        try:
            import requests

            self.state = ConnectorState.SYNCING

            channel_ids = self.channels or self._list_channels()
            for ch_id in channel_ids:
                messages = self._fetch_messages(ch_id, since=since, limit=limit)
                for msg in messages:
                    item = self._message_to_item(ch_id, msg)
                    if item:
                        items.append(item)
                    # Thread replies
                    if self.include_threads and msg.get("thread_ts"):
                        thread_items = self._fetch_thread(
                            ch_id, msg["thread_ts"], since=since, limit=limit
                        )
                        items.extend(thread_items)

            _logger.info("Slack sync completed: %d items", len(items))
            self.state = ConnectorState.CONNECTED
            return items
        except Exception as exc:
            self._set_error(f"Slack sync error: {exc}")
            return []

    def disconnect(self) -> None:
        """无状态 HTTP, 仅标记断开."""
        self.state = ConnectorState.DISCONNECTED

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_channels(self) -> list[str]:
        import requests

        resp = requests.get(
            f"{self.base_url}/conversations.list",
            headers=self.headers,
            params={"types": "public_channel,private_channel", "limit": 200},
            timeout=30,
        )
        data = resp.json()
        if not data.get("ok"):
            return []
        return [ch["id"] for ch in data.get("channels", []) if not ch.get("is_archived")]

    def _fetch_messages(
        self, channel_id: str, since: float | None, limit: int
    ) -> list[dict[str, Any]]:
        import requests

        params: dict[str, Any] = {"channel": channel_id, "limit": min(limit, 200)}
        if since:
            params["oldest"] = str(since)

        resp = requests.get(
            f"{self.base_url}/conversations.history",
            headers=self.headers,
            params=params,
            timeout=30,
        )
        data = resp.json()
        if not data.get("ok"):
            return []
        return data.get("messages", [])

    def _fetch_thread(
        self, channel_id: str, thread_ts: str, since: float | None, limit: int
    ) -> list[SyncItem]:
        import requests

        params: dict[str, Any] = {
            "channel": channel_id,
            "ts": thread_ts,
            "limit": min(limit, 200),
        }
        resp = requests.get(
            f"{self.base_url}/conversations.replies",
            headers=self.headers,
            params=params,
            timeout=30,
        )
        data = resp.json()
        if not data.get("ok"):
            return []
        items: list[SyncItem] = []
        for msg in data.get("messages", []):
            # Skip the parent message (already counted)
            if msg.get("ts") == thread_ts:
                continue
            item = self._message_to_item(channel_id, msg, thread_ts=thread_ts)
            if item:
                items.append(item)
        return items

    def _message_to_item(
        self,
        channel_id: str,
        msg: dict[str, Any],
        thread_ts: str | None = None,
    ) -> SyncItem | None:
        text = msg.get("text", "").strip()
        if not text:
            return None

        ts = msg.get("ts", "")
        user = msg.get("user", "")
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc) if ts else datetime.now(timezone.utc)

        return SyncItem(
            id=f"{channel_id}-{ts}",
            content=text,
            source="slack",
            source_url=None,
            title=None,
            author=user or None,
            created_at=dt,
            updated_at=dt,
            metadata={
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "message_ts": ts,
                "slack_user": user,
            },
            tags=["slack"],
        )
