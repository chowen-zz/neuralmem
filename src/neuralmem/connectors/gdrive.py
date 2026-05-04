"""Google Drive 连接器 — OAuth2 认证、文件同步、文件夹监控."""
from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any

from neuralmem.connectors.base import ConnectorProtocol, ConnectorState, SyncItem

_logger = logging.getLogger(__name__)

# Graceful degradation for google-api-python-client
try:
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.http import MediaIoBaseDownload

    _HAS_GOOGLE_API = True
except ImportError:
    _HAS_GOOGLE_API = False
    build = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment,misc]
    Credentials = None  # type: ignore[assignment,misc]
    MediaIoBaseDownload = None  # type: ignore[assignment,misc]


class GoogleDriveConnector(ConnectorProtocol):
    """Google Drive API 连接器.

    支持 OAuth2 认证、文件列表、下载、文件夹同步与监控.

    Config keys
    -----------
    credentials : dict | None
        OAuth2 credentials dict (包含 token, refresh_token, client_id, client_secret).
    folder_id : str | None
        要同步的 Google Drive 文件夹 ID; None 则同步根目录.
    mime_types : list[str] | None
        只同步指定 MIME 类型, 如 ['text/plain', 'application/pdf'].
    include_trashed : bool
        是否包含回收站文件, 默认 False.
    max_file_size : int
        最大文件大小(字节), 默认 10MB.
    """

    def __init__(
        self, name: str = "gdrive", config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(name, config)
        if not _HAS_GOOGLE_API:
            self._available = False
            self._set_error(
                "GoogleDriveConnector requires 'google-api-python-client' and "
                "'google-auth-httplib2'. Install: pip install google-api-python-client google-auth-httplib2"
            )
            return

        self.credentials: dict[str, Any] | None = self.config.get("credentials")
        self.folder_id: str | None = self.config.get("folder_id")
        self.mime_types: list[str] | None = self.config.get("mime_types")
        self.include_trashed: bool = self.config.get("include_trashed", False)
        self.max_file_size: int = self.config.get("max_file_size", 10 * 1024 * 1024)

        self._service: Any | None = None
        self._creds: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """建立 OAuth2 认证并初始化 Drive API 服务.

        Returns
        -------
        bool
            True 表示认证成功, False 表示失败.
        """
        if not self._available:
            return False
        if not self.credentials:
            self._set_error("Google Drive 'credentials' required in config")
            return False

        try:
            self._creds = Credentials(
                token=self.credentials.get("token"),
                refresh_token=self.credentials.get("refresh_token"),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.credentials.get("client_id"),
                client_secret=self.credentials.get("client_secret"),
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )

            if self._creds.expired and self._creds.refresh_token:
                self._creds.refresh(Request())

            self._service = build("drive", "v3", credentials=self._creds, static_discovery=False)
            self.state = ConnectorState.CONNECTED
            _logger.info("Google Drive authenticated successfully")
            return True
        except Exception as exc:
            self._set_error(f"Google Drive auth error: {exc}")
            return False

    def sync(self, **kwargs: Any) -> list[SyncItem]:
        """同步 Google Drive 文件内容.

        Parameters
        ----------
        folder_id : str | None
            覆盖 config 中的 folder_id.
        limit : int
            最大提取数量, 默认 100.
        full_scan : bool
            强制全量扫描, 默认 True (Google Drive 无实时监听).
        """
        if not self._available or self.state != ConnectorState.CONNECTED:
            _logger.warning("GoogleDriveConnector not available or not connected")
            return []

        folder_id: str | None = kwargs.get("folder_id", self.folder_id)
        limit: int = kwargs.get("limit", 100)

        try:
            self.state = ConnectorState.SYNCING
            items = self._list_and_fetch_files(folder_id=folder_id, limit=limit)
            _logger.info("Google Drive sync completed: %d items", len(items))
            self.state = ConnectorState.CONNECTED
            return items
        except Exception as exc:
            self._set_error(f"Google Drive sync error: {exc}")
            return []

    def disconnect(self) -> None:
        """清理 OAuth2 凭证与服务对象."""
        self._service = None
        self._creds = None
        self.state = ConnectorState.DISCONNECTED

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_files(self, folder_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """列出 Google Drive 中的文件元数据.

        Parameters
        ----------
        folder_id : str | None
            文件夹 ID; None 则列出根目录或所有文件.
        limit : int
            最大返回数量.

        Returns
        -------
        list[dict]
            文件元数据列表, 每个 dict 包含 id, name, mimeType, size, modifiedTime 等.
        """
        if not self._available or self.state not in (ConnectorState.CONNECTED, ConnectorState.SYNCING) or not self._service:
            _logger.warning("GoogleDriveConnector not connected")
            return []

        query_parts = ["trashed = false"] if not self.include_trashed else []
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")
        if self.mime_types:
            mime_queries = " or ".join([f"mimeType='{mt}'" for mt in self.mime_types])
            query_parts.append(f"({mime_queries})")

        query = " and ".join(query_parts) if query_parts else ""

        try:
            results = (
                self._service.files()
                .list(
                    q=query,
                    pageSize=limit,
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, createdTime, owners, webViewLink, description)",
                )
                .execute()
            )
            return results.get("files", [])
        except Exception as exc:
            _logger.error("Google Drive list_files error: %s", exc)
            return []

    def download_file(self, file_id: str, mime_type: str | None = None) -> bytes | None:
        """下载 Google Drive 文件内容.

        Parameters
        ----------
        file_id : str
            文件 ID.
        mime_type : str | None
            文件 MIME 类型; Google Docs/Sheets 等需要导出为可下载格式.

        Returns
        -------
        bytes | None
            文件内容字节, 失败返回 None.
        """
        if not self._available or self.state not in (ConnectorState.CONNECTED, ConnectorState.SYNCING) or not self._service:
            _logger.warning("GoogleDriveConnector not connected")
            return None

        try:
            # Google Workspace 文件需要导出
            if mime_type and mime_type.startswith("application/vnd.google-apps"):
                export_mime = self._get_export_mime(mime_type)
                request = self._service.files().export_media(fileId=file_id, mimeType=export_mime)
            else:
                request = self._service.files().get_media(fileId=file_id)

            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                _logger.debug("Download %s: %d%%", file_id, int(status.progress() * 100))

            content = fh.getvalue()
            if len(content) > self.max_file_size:
                _logger.warning("File %s exceeds max size (%d > %d)", file_id, len(content), self.max_file_size)
                return None
            return content
        except Exception as exc:
            _logger.error("Google Drive download_file error: %s", exc)
            return None

    def sync_folder(self, folder_id: str | None = None, limit: int = 100) -> list[SyncItem]:
        """同步指定文件夹下的所有文件.

        与 sync() 行为一致, 但显式指定 folder_id.
        """
        return self.sync(folder_id=folder_id, limit=limit, full_scan=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_and_fetch_files(self, folder_id: str | None, limit: int) -> list[SyncItem]:
        files = self.list_files(folder_id=folder_id, limit=limit)
        items: list[SyncItem] = []
        for file_meta in files:
            item = self._file_meta_to_item(file_meta)
            if item:
                items.append(item)
        return items

    def _file_meta_to_item(self, file_meta: dict[str, Any]) -> SyncItem | None:
        file_id = file_meta.get("id", "")
        name = file_meta.get("name", "")
        mime_type = file_meta.get("mimeType", "")
        size_str = file_meta.get("size", "0")
        try:
            size = int(size_str)
        except ValueError:
            size = 0

        # Skip folders
        if mime_type == "application/vnd.google-apps.folder":
            return None

        # Download content
        content_bytes = self.download_file(file_id, mime_type)
        if content_bytes is None:
            return None

        # Try decode as text; fallback to base64 marker for binary
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = f"<binary file: {name}, size={size} bytes>"

        if not content:
            return None

        owners = file_meta.get("owners", [])
        author = owners[0].get("displayName") if owners else None

        return SyncItem(
            id=file_id,
            content=content,
            source="gdrive",
            source_url=file_meta.get("webViewLink"),
            title=name or None,
            author=author,
            created_at=self._parse_rfc3339(file_meta.get("createdTime", "")),
            updated_at=self._parse_rfc3339(file_meta.get("modifiedTime", "")),
            metadata={
                "mime_type": mime_type,
                "size": size,
                "folder_id": self.folder_id,
                "description": file_meta.get("description"),
            },
            tags=["gdrive", mime_type.split("/")[-1] if "/" in mime_type else "unknown"],
        )

    @staticmethod
    def _parse_rfc3339(ts: str) -> datetime:
        if not ts:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)

    @staticmethod
    def _get_export_mime(mime_type: str) -> str:
        """Google Workspace 文件导出 MIME 类型映射."""
        mapping = {
            "application/vnd.google-apps.document": "text/plain",
            "application/vnd.google-apps.spreadsheet": "text/csv",
            "application/vnd.google-apps.presentation": "text/plain",
            "application/vnd.google-apps.drawing": "image/png",
        }
        return mapping.get(mime_type, "text/plain")
