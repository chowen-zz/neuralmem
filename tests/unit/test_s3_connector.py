"""S3Connector 单元测试 — 全部使用 mock, 不依赖真实 AWS API."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.connectors.base import ConnectorState, SyncItem
from neuralmem.connectors import s3 as s3_module


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def connector() -> "s3_module.S3Connector":
    with patch.object(s3_module, "_HAS_BOTO3", True):
        return s3_module.S3Connector(
            name="s3",
            config={
                "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
                "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "region_name": "us-east-1",
                "bucket": "neuralmem-test-bucket",
                "prefix": "docs/",
                "suffixes": [".txt", ".md"],
            },
        )


@pytest.fixture
def mock_client() -> MagicMock:
    """返回一个模拟的 boto3 S3 client."""
    return MagicMock()


# --------------------------------------------------------------------------- #
# Constructor / availability
# --------------------------------------------------------------------------- #

def test_connector_init(connector: "s3_module.S3Connector") -> None:
    assert connector.name == "s3"
    assert connector.bucket == "neuralmem-test-bucket"
    assert connector.prefix == "docs/"
    assert connector.suffixes == [".txt", ".md"]
    assert connector.region_name == "us-east-1"
    assert connector.state == ConnectorState.DISCONNECTED


def test_connector_unavailable_without_boto3() -> None:
    with patch.object(s3_module, "_HAS_BOTO3", False):
        conn = s3_module.S3Connector(config={"bucket": "test"})
        assert not conn.is_available()
        assert conn.authenticate() is False
        assert conn.state == ConnectorState.ERROR


# --------------------------------------------------------------------------- #
# authenticate
# --------------------------------------------------------------------------- #

def test_authenticate_success(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        result = connector.authenticate()

        assert result is True
        assert connector.state == ConnectorState.CONNECTED
        assert connector._client is mock_client
        mock_client.head_bucket.assert_called_once_with(
            Bucket="neuralmem-test-bucket"
        )


def test_authenticate_missing_bucket() -> None:
    with patch.object(s3_module, "_HAS_BOTO3", True):
        conn = s3_module.S3Connector(config={})
    result = conn.authenticate()
    assert result is False
    assert conn.state == ConnectorState.ERROR
    assert "bucket" in (conn.last_error() or "").lower()


def test_authenticate_client_error(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    error_response = {
        "Error": {"Code": "NoSuchBucket", "Message": "The bucket does not exist"}
    }
    client_error_cls = type(
        "ClientError",
        (Exception,),
        {
            "__init__": lambda self, *args, **kw: Exception.__init__(
                self, "NoSuchBucket"
            ),
            "response": error_response,
        },
    )
    mock_client.head_bucket.side_effect = client_error_cls(error_response, "HeadBucket")

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        result = connector.authenticate()

        assert result is False
        assert "NoSuchBucket" in (connector.last_error() or "")


def test_authenticate_generic_error(
    connector: "s3_module.S3Connector"
) -> None:
    with patch.object(
        s3_module, "boto3"
    ) as mock_boto:
        mock_boto.client = MagicMock(side_effect=Exception("network down"))
        result = connector.authenticate()

        assert result is False
        assert "network down" in (connector.last_error() or "")


# --------------------------------------------------------------------------- #
# disconnect
# --------------------------------------------------------------------------- #

def test_disconnect(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()

    connector.disconnect()
    assert connector.state == ConnectorState.DISCONNECTED
    assert connector._client is None


# --------------------------------------------------------------------------- #
# list_objects
# --------------------------------------------------------------------------- #

def test_list_objects_success(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    page = {
        "Contents": [
            {
                "Key": "docs/readme.md",
                "Size": 100,
                "LastModified": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "ETag": '"abc123"',
                "StorageClass": "STANDARD",
            },
            {
                "Key": "docs/notes.txt",
                "Size": 200,
                "LastModified": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "ETag": '"def456"',
                "StorageClass": "STANDARD",
            },
            {
                "Key": "docs/data.json",
                "Size": 50,
                "LastModified": datetime(2024, 1, 3, tzinfo=timezone.utc),
                "ETag": '"ghi789"',
                "StorageClass": "STANDARD",
            },
        ]
    }
    mock_client.get_paginator.return_value.paginate.return_value = [page]

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        objects = connector.list_objects(prefix="docs/", limit=10)

    assert len(objects) == 2  # data.json filtered by suffix
    assert objects[0]["Key"] == "docs/readme.md"
    assert objects[1]["Key"] == "docs/notes.txt"


def test_list_objects_skips_folder_placeholders(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    page = {
        "Contents": [
            {
                "Key": "docs/",
                "Size": 0,
                "LastModified": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
            {
                "Key": "docs/readme.md",
                "Size": 100,
                "LastModified": datetime(2024, 1, 1, tzinfo=timezone.utc),
            },
        ]
    }
    mock_client.get_paginator.return_value.paginate.return_value = [page]

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        objects = connector.list_objects(prefix="docs/", limit=10)

    assert len(objects) == 1
    assert objects[0]["Key"] == "docs/readme.md"


def test_list_objects_not_connected(
    connector: "s3_module.S3Connector"
) -> None:
    assert connector.list_objects() == []


def test_list_objects_client_error(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    error_response = {
        "Error": {"Code": "AccessDenied", "Message": "Access Denied"}
    }
    client_error_cls = type(
        "ClientError",
        (Exception,),
        {
            "__init__": lambda self, *args, **kw: Exception.__init__(
                self, "AccessDenied"
            ),
            "response": error_response,
        },
    )
    mock_client.get_paginator.return_value.paginate.side_effect = client_error_cls(
        error_response, "ListObjectsV2"
    )

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        objects = connector.list_objects()

    assert objects == []


# --------------------------------------------------------------------------- #
# download_object
# --------------------------------------------------------------------------- #

def test_download_object_success(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    body_mock = MagicMock()
    body_mock.read.return_value = b"hello s3 world"
    mock_client.get_object.return_value = {"Body": body_mock}

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        result = connector.download_object("docs/readme.md")

    assert result == b"hello s3 world"
    mock_client.get_object.assert_called_once_with(
        Bucket="neuralmem-test-bucket", Key="docs/readme.md"
    )
    body_mock.close.assert_called_once()


def test_download_object_size_limit(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    body_mock = MagicMock()
    body_mock.read.return_value = b"x" * (11 * 1024 * 1024)  # 11MB > default 10MB
    mock_client.get_object.return_value = {"Body": body_mock}

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        result = connector.download_object("docs/huge.bin")

    assert result is None


def test_download_object_not_connected(
    connector: "s3_module.S3Connector"
) -> None:
    assert connector.download_object("docs/readme.md") is None


def test_download_object_client_error(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    error_response = {
        "Error": {
            "Code": "NoSuchKey",
            "Message": "The specified key does not exist.",
        }
    }
    client_error_cls = type(
        "ClientError",
        (Exception,),
        {
            "__init__": lambda self, *args, **kw: Exception.__init__(
                self, "NoSuchKey"
            ),
            "response": error_response,
        },
    )
    mock_client.get_object.side_effect = client_error_cls(
        error_response, "GetObject"
    )

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        result = connector.download_object("docs/missing.md")

    assert result is None


# --------------------------------------------------------------------------- #
# sync / sync_bucket
# --------------------------------------------------------------------------- #

def test_sync_success(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    page = {
        "Contents": [
            {
                "Key": "docs/readme.md",
                "Size": 14,
                "LastModified": datetime(
                    2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc
                ),
                "ETag": '"abc123"',
                "StorageClass": "STANDARD",
            },
        ]
    }
    mock_client.get_paginator.return_value.paginate.return_value = [page]

    body_mock = MagicMock()
    body_mock.read.return_value = b"hello s3 world"
    mock_client.get_object.return_value = {"Body": body_mock}

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        items = connector.sync(prefix="docs/", limit=10)

    assert len(items) == 1
    assert items[0].id == "docs/readme.md"
    assert items[0].content == "hello s3 world"
    assert items[0].source == "s3"
    assert items[0].title == "readme.md"
    assert items[0].source_url == "s3://neuralmem-test-bucket/docs/readme.md"
    assert items[0].metadata["etag"] == "abc123"
    assert items[0].metadata["storage_class"] == "STANDARD"
    assert connector.state == ConnectorState.CONNECTED


def test_sync_with_since_filter(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    page = {
        "Contents": [
            {
                "Key": "docs/old.md",
                "Size": 10,
                "LastModified": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "ETag": '"old"',
            },
            {
                "Key": "docs/new.md",
                "Size": 10,
                "LastModified": datetime(2024, 6, 1, tzinfo=timezone.utc),
                "ETag": '"new"',
            },
        ]
    }
    mock_client.get_paginator.return_value.paginate.return_value = [page]

    body_mock = MagicMock()
    body_mock.read.return_value = b"content"
    mock_client.get_object.return_value = {"Body": body_mock}

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        since = datetime(2024, 3, 1, tzinfo=timezone.utc)
        items = connector.sync(prefix="docs/", limit=10, since=since)

    assert len(items) == 1
    assert items[0].id == "docs/new.md"


def test_sync_binary_object(
    mock_client: MagicMock
) -> None:
    with patch.object(s3_module, "_HAS_BOTO3", True):
        conn = s3_module.S3Connector(
            name="s3",
            config={
                "bucket": "neuralmem-test-bucket",
                "prefix": "docs/",
            },
        )
    page = {
        "Contents": [
            {
                "Key": "docs/image.png",
                "Size": 1024,
                "LastModified": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "ETag": '"img123"',
            },
        ]
    }
    mock_client.get_paginator.return_value.paginate.return_value = [page]

    body_mock = MagicMock()
    body_mock.read.return_value = b"\x89PNG\r\n\x1a\n"
    mock_client.get_object.return_value = {"Body": body_mock}

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        conn.authenticate()
        items = conn.sync(prefix="docs/", limit=10)

    assert len(items) == 1
    assert "<binary object" in items[0].content


def test_sync_not_connected(
    connector: "s3_module.S3Connector"
) -> None:
    assert connector.sync() == []


def test_sync_bucket(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    mock_client.get_paginator.return_value.paginate.return_value = [
        {"Contents": []}
    ]

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        items = connector.sync_bucket(prefix="archive/", limit=5)

    assert items == []
    mock_client.get_paginator.assert_called_once_with("list_objects_v2")
    paginator = mock_client.get_paginator.return_value
    paginator.paginate.assert_called_once()
    call_kwargs = paginator.paginate.call_args.kwargs
    assert call_kwargs["Bucket"] == "neuralmem-test-bucket"
    assert call_kwargs["Prefix"] == "archive/"


# --------------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------------- #

def test_context_manager() -> None:
    mock_client = MagicMock()
    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        with patch.object(s3_module, "_HAS_BOTO3", True):
            conn = s3_module.S3Connector(
                config={
                    "bucket": "test-bucket",
                    "aws_access_key_id": "key",
                    "aws_secret_access_key": "secret",
                }
            )
        with conn:
            assert conn.state == ConnectorState.CONNECTED
        assert conn.state == ConnectorState.DISCONNECTED


# --------------------------------------------------------------------------- #
# Custom endpoint (MinIO compatibility)
# --------------------------------------------------------------------------- #

def test_authenticate_with_custom_endpoint() -> None:
    with patch.object(s3_module, "_HAS_BOTO3", True):
        conn = s3_module.S3Connector(
            config={
                "bucket": "minio-bucket",
                "endpoint_url": "http://localhost:9000",
                "aws_access_key_id": "minioadmin",
                "aws_secret_access_key": "minioadmin",
            }
        )
    mock_client = MagicMock()
    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        result = conn.authenticate()

    assert result is True
    mock_boto.client.assert_called_once()
    call_kwargs = mock_boto.client.call_args.kwargs
    assert call_kwargs["endpoint_url"] == "http://localhost:9000"


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_sync_empty_bucket(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    mock_client.get_paginator.return_value.paginate.return_value = [
        {"Contents": []}
    ]

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        items = connector.sync(limit=10)

    assert items == []


def test_object_meta_to_item_with_no_last_modified(
    connector: "s3_module.S3Connector", mock_client: MagicMock
) -> None:
    """Test that missing LastModified falls back to current time."""
    page = {
        "Contents": [
            {
                "Key": "docs/readme.md",
                "Size": 14,
                "ETag": '"abc123"',
            },
        ]
    }
    mock_client.get_paginator.return_value.paginate.return_value = [page]

    body_mock = MagicMock()
    body_mock.read.return_value = b"hello s3 world"
    mock_client.get_object.return_value = {"Body": body_mock}

    with patch.object(s3_module, "boto3") as mock_boto:
        mock_boto.client.return_value = mock_client
        connector.authenticate()
        items = connector.sync(limit=10)

    assert len(items) == 1
    assert isinstance(items[0].updated_at, datetime)
