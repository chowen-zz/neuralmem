"""AWS S3 连接器 — bucket 同步、对象导入、前缀过滤."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from neuralmem.connectors.base import ConnectorProtocol, ConnectorState, SyncItem

_logger = logging.getLogger(__name__)

# Graceful degradation for boto3
try:
    import boto3
    from botocore.exceptions import ClientError

    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[misc,assignment]


class S3Connector(ConnectorProtocol):
    """AWS S3 连接器.

    支持 bucket 列表、对象下载、前缀过滤、全量/增量同步.

    Config keys
    -----------
    aws_access_key_id : str | None
        AWS Access Key ID.
    aws_secret_access_key : str | None
        AWS Secret Access Key.
    region_name : str
        AWS 区域, 默认 'us-east-1'.
    endpoint_url : str | None
        自定义 S3 endpoint (用于 MinIO 等兼容服务).
    bucket : str
        S3 bucket 名称.
    prefix : str
        对象前缀过滤器, 默认 '' (根目录).
    suffixes : list[str] | None
        只同步指定后缀, 如 ['.txt', '.md'].
    max_file_size : int
        最大文件大小(字节), 默认 10MB.
    """

    def __init__(
        self, name: str = "s3", config: dict[str, Any] | None = None
    ) -> None:
        super().__init__(name, config)
        if not _HAS_BOTO3:
            self._available = False
            self._set_error(
                "S3Connector requires 'boto3'. Install: pip install boto3"
            )
            return

        self.aws_access_key_id: str | None = self.config.get("aws_access_key_id")
        self.aws_secret_access_key: str | None = self.config.get("aws_secret_access_key")
        self.region_name: str = self.config.get("region_name", "us-east-1")
        self.endpoint_url: str | None = self.config.get("endpoint_url")
        self.bucket: str = self.config.get("bucket", "")
        self.prefix: str = self.config.get("prefix", "")
        self.suffixes: list[str] | None = self.config.get("suffixes")
        self.max_file_size: int = self.config.get("max_file_size", 10 * 1024 * 1024)

        self._client: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """建立与 S3 的连接并验证 bucket 可访问性.

        Returns
        -------
        bool
            True 表示认证成功, False 表示失败.
        """
        if not self._available:
            return False
        if not self.bucket:
            self._set_error("S3 'bucket' required in config")
            return False

        try:
            session_kwargs: dict[str, Any] = {"region_name": self.region_name}
            if self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs.update(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                )
            if self.endpoint_url:
                session_kwargs["endpoint_url"] = self.endpoint_url

            self._client = boto3.client("s3", **session_kwargs)
            # Verify bucket exists
            self._client.head_bucket(Bucket=self.bucket)
            self.state = ConnectorState.CONNECTED
            _logger.info("S3 bucket '%s' connected", self.bucket)
            return True
        except Exception as exc:
            if hasattr(exc, "response") and isinstance(exc.response, dict):
                code = exc.response.get("Error", {}).get("Code", "Unknown")
                self._set_error(f"S3 auth/bucket error: {code} — {exc}")
            else:
                self._set_error(f"S3 auth error: {exc}")
            return False

    def sync(self, **kwargs: Any) -> list[SyncItem]:
        """同步 S3 bucket 对象内容.

        Parameters
        ----------
        prefix : str | None
            覆盖 config 中的 prefix.
        limit : int
            最大提取数量, 默认 100.
        since : datetime | None
            仅同步此时间后修改的对象.
        """
        if not self._available or self.state != ConnectorState.CONNECTED:
            _logger.warning("S3Connector not available or not connected")
            return []

        prefix: str = kwargs.get("prefix", self.prefix)
        limit: int = kwargs.get("limit", 100)
        since: datetime | None = kwargs.get("since")

        try:
            self.state = ConnectorState.SYNCING
            items = self._list_and_fetch_objects(prefix=prefix, limit=limit, since=since)
            _logger.info("S3 sync completed: %d items", len(items))
            self.state = ConnectorState.CONNECTED
            return items
        except Exception as exc:
            self._set_error(f"S3 sync error: {exc}")
            return []

    def disconnect(self) -> None:
        """关闭 S3 客户端连接."""
        self._client = None
        self.state = ConnectorState.DISCONNECTED

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_objects(self, prefix: str = "", limit: int = 100) -> list[dict[str, Any]]:
        """列出 S3 bucket 中的对象元数据.

        Parameters
        ----------
        prefix : str
            对象前缀过滤器.
        limit : int
            最大返回数量.

        Returns
        -------
        list[dict]
            对象元数据列表, 每个 dict 包含 Key, Size, LastModified, ETag 等.
        """
        if not self._available or self.state not in (ConnectorState.CONNECTED, ConnectorState.SYNCING) or not self._client:
            _logger.warning("S3Connector not connected")
            return []

        try:
            paginator = self._client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self.bucket,
                Prefix=prefix,
                PaginationConfig={"MaxItems": limit},
            )

            objects: list[dict[str, Any]] = []
            for page in page_iterator:
                for obj in page.get("Contents", []):
                    key: str = obj.get("Key", "")
                    # Skip "folder" placeholders
                    if key.endswith("/"):
                        continue
                    # Suffix filter
                    if self.suffixes and not any(key.endswith(s) for s in self.suffixes):
                        continue
                    objects.append(obj)
                    if len(objects) >= limit:
                        return objects
            return objects
        except Exception as exc:
            _logger.error("S3 list_objects error: %s", exc)
            return []

    def download_object(self, key: str) -> bytes | None:
        """下载 S3 对象内容.

        Parameters
        ----------
        key : str
            对象键 (路径).

        Returns
        -------
        bytes | None
            对象内容字节, 失败返回 None.
        """
        if not self._available or self.state not in (ConnectorState.CONNECTED, ConnectorState.SYNCING) or not self._client:
            _logger.warning("S3Connector not connected")
            return None

        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            body = response["Body"]
            content = body.read()
            body.close()

            if len(content) > self.max_file_size:
                _logger.warning(
                    "Object %s exceeds max size (%d > %d)",
                    key,
                    len(content),
                    self.max_file_size,
                )
                return None
            return content
        except ClientError as exc:
            _logger.error("S3 download_object error: %s", exc)
            return None

    def sync_bucket(self, prefix: str = "", limit: int = 100) -> list[SyncItem]:
        """同步整个 bucket (或指定前缀下) 的对象.

        与 sync() 行为一致, 但显式指定 prefix.
        """
        return self.sync(prefix=prefix, limit=limit)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_and_fetch_objects(
        self, prefix: str, limit: int, since: datetime | None
    ) -> list[SyncItem]:
        objects = self.list_objects(prefix=prefix, limit=limit)
        items: list[SyncItem] = []
        for obj_meta in objects:
            # Time filter
            if since:
                last_modified = obj_meta.get("LastModified")
                if last_modified and last_modified < since:
                    continue
            item = self._object_meta_to_item(obj_meta)
            if item:
                items.append(item)
        return items

    def _object_meta_to_item(self, obj_meta: dict[str, Any]) -> SyncItem | None:
        key = obj_meta.get("Key", "")
        size = obj_meta.get("Size", 0)

        content_bytes = self.download_object(key)
        if content_bytes is None:
            return None

        # Try decode as text; fallback to binary marker
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = f"<binary object: {key}, size={size} bytes>"

        if not content:
            return None

        last_modified = obj_meta.get("LastModified")
        updated_at = (
            last_modified
            if isinstance(last_modified, datetime)
            else datetime.now(timezone.utc)
        )

        return SyncItem(
            id=key,
            content=content,
            source="s3",
            source_url=f"s3://{self.bucket}/{key}",
            title=key.split("/")[-1] or key,
            author=None,
            created_at=updated_at,
            updated_at=updated_at,
            metadata={
                "bucket": self.bucket,
                "key": key,
                "size": size,
                "etag": obj_meta.get("ETag", "").strip('"'),
                "storage_class": obj_meta.get("StorageClass", "STANDARD"),
            },
            tags=["s3", key.split(".")[-1] if "." in key else "unknown"],
        )
