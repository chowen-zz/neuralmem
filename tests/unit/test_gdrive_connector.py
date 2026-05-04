"""GoogleDriveConnector 单元测试 — 全部使用 mock, 不依赖真实 Google API."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from neuralmem.connectors.base import ConnectorState, SyncItem
from neuralmem.connectors import gdrive as gdrive_module


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_credentials() -> dict[str, str]:
    return {
        "token": "test-token",
        "refresh_token": "test-refresh",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
    }


@pytest.fixture
def connector(mock_credentials: dict[str, str]) -> "gdrive_module.GoogleDriveConnector":
    with patch.object(gdrive_module, "_HAS_GOOGLE_API", True):
        return gdrive_module.GoogleDriveConnector(
            name="gdrive",
            config={
                "credentials": mock_credentials,
                "folder_id": "folder123",
                "mime_types": ["text/plain", "application/pdf"],
            },
        )


@pytest.fixture
def mock_service() -> MagicMock:
    """返回一个模拟的 Google Drive API service."""
    service = MagicMock()
    # Ensure .files() returns the same mock so method chaining works
    files_mock = MagicMock()
    service.files.return_value = files_mock
    return service


# --------------------------------------------------------------------------- #
# Constructor / availability
# --------------------------------------------------------------------------- #

def test_connector_init(connector: "gdrive_module.GoogleDriveConnector") -> None:
    assert connector.name == "gdrive"
    assert connector.folder_id == "folder123"
    assert connector.mime_types == ["text/plain", "application/pdf"]
    assert connector.state == ConnectorState.DISCONNECTED


def test_connector_unavailable_without_google_api() -> None:
    with patch.object(gdrive_module, "_HAS_GOOGLE_API", False):
        conn = gdrive_module.GoogleDriveConnector(config={"credentials": {}})
        assert not conn.is_available()
        assert conn.authenticate() is False
        assert conn.state == ConnectorState.ERROR


# --------------------------------------------------------------------------- #
# authenticate
# --------------------------------------------------------------------------- #

def test_authenticate_success(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service):
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance

        result = connector.authenticate()

        assert result is True
        assert connector.state == ConnectorState.CONNECTED
        assert connector._service is mock_service
        MockCreds.assert_called_once()


def test_authenticate_refresh_token(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service), \
         patch.object(gdrive_module, "Request") as MockRequest:
        creds_instance = MagicMock()
        creds_instance.expired = True
        creds_instance.refresh_token = "refresh-me"
        MockCreds.return_value = creds_instance

        result = connector.authenticate()

        assert result is True
        creds_instance.refresh.assert_called_once_with(MockRequest.return_value)


def test_authenticate_missing_credentials() -> None:
    with patch.object(gdrive_module, "_HAS_GOOGLE_API", True):
        conn = gdrive_module.GoogleDriveConnector(config={})
    result = conn.authenticate()
    assert result is False
    assert conn.state == ConnectorState.ERROR
    assert "credentials" in (conn.last_error() or "").lower()


def test_authenticate_build_error(
    connector: "gdrive_module.GoogleDriveConnector"
) -> None:
    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", side_effect=Exception("build failed")):
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance

        result = connector.authenticate()

        assert result is False
        assert "build failed" in (connector.last_error() or "")


# --------------------------------------------------------------------------- #
# disconnect
# --------------------------------------------------------------------------- #

def test_disconnect(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service):
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance
        connector.authenticate()

    connector.disconnect()
    assert connector.state == ConnectorState.DISCONNECTED
    assert connector._service is None
    assert connector._creds is None


# --------------------------------------------------------------------------- #
# list_files
# --------------------------------------------------------------------------- #

def test_list_files_success(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    files_result = {
        "files": [
            {
                "id": "file1",
                "name": "doc1.txt",
                "mimeType": "text/plain",
                "size": "100",
                "modifiedTime": "2024-01-01T00:00:00Z",
                "createdTime": "2024-01-01T00:00:00Z",
                "owners": [{"displayName": "Alice"}],
                "webViewLink": "https://drive.google.com/file/d/file1/view",
            },
            {
                "id": "file2",
                "name": "doc2.pdf",
                "mimeType": "application/pdf",
                "size": "200",
                "modifiedTime": "2024-01-02T00:00:00Z",
                "createdTime": "2024-01-02T00:00:00Z",
                "owners": [{"displayName": "Bob"}],
                "webViewLink": "https://drive.google.com/file/d/file2/view",
            },
        ]
    }
    mock_service.files.return_value.list.return_value.execute.return_value = files_result

    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service):
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance
        connector.authenticate()

    files = connector.list_files(folder_id="folder123", limit=10)

    assert len(files) == 2
    assert files[0]["id"] == "file1"
    assert files[1]["name"] == "doc2.pdf"
    mock_service.files.return_value.list.assert_called_once()
    call_kwargs = mock_service.files.return_value.list.call_args.kwargs
    assert "folder123" in call_kwargs["q"]
    assert "text/plain" in call_kwargs["q"]


def test_list_files_not_connected(
    connector: "gdrive_module.GoogleDriveConnector"
) -> None:
    assert connector.list_files() == []


def test_list_files_api_error(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    mock_service.files.return_value.list.return_value.execute.side_effect = Exception("API error")

    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service):
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance
        connector.authenticate()

    files = connector.list_files()
    assert files == []


# --------------------------------------------------------------------------- #
# download_file
# --------------------------------------------------------------------------- #

def test_download_file_success(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    mock_request = MagicMock()
    mock_service.files().get_media.return_value = mock_request

    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service), \
         patch.object(gdrive_module, "MediaIoBaseDownload") as MockDownloader, \
         patch.object(gdrive_module, "io") as MockIO:
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance
        connector.authenticate()

        # Simulate chunked download
        fh_mock = MagicMock()
        fh_mock.getvalue.return_value = b"hello world"
        MockIO.BytesIO.return_value = fh_mock

        downloader_instance = MockDownloader.return_value
        downloader_instance.next_chunk.side_effect = [
            (MagicMock(progress=MagicMock(return_value=0.5)), False),
            (MagicMock(progress=MagicMock(return_value=1.0)), True),
        ]

        result = connector.download_file("file1")

        assert result == b"hello world"
        mock_service.files().get_media.assert_called_once_with(fileId="file1")


def test_download_file_google_workspace_export(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    mock_request = MagicMock()
    mock_service.files().export_media.return_value = mock_request

    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service), \
         patch.object(gdrive_module, "MediaIoBaseDownload") as MockDownloader, \
         patch.object(gdrive_module, "io") as MockIO:
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance
        connector.authenticate()

        fh_mock = MagicMock()
        fh_mock.getvalue.return_value = b"exported doc content"
        MockIO.BytesIO.return_value = fh_mock

        downloader_instance = MockDownloader.return_value
        downloader_instance.next_chunk.side_effect = [
            (MagicMock(progress=MagicMock(return_value=1.0)), True),
        ]

        result = connector.download_file(
            "doc1", mime_type="application/vnd.google-apps.document"
        )

        mock_service.files().export_media.assert_called_once_with(
            fileId="doc1", mimeType="text/plain"
        )
        assert result == b"exported doc content"


def test_download_file_not_connected(
    connector: "gdrive_module.GoogleDriveConnector"
) -> None:
    assert connector.download_file("file1") is None


# --------------------------------------------------------------------------- #
# sync / sync_folder
# --------------------------------------------------------------------------- #

def test_sync_success(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    files_result = {
        "files": [
            {
                "id": "file1",
                "name": "doc1.txt",
                "mimeType": "text/plain",
                "size": "12",
                "modifiedTime": "2024-01-01T00:00:00Z",
                "createdTime": "2024-01-01T00:00:00Z",
                "owners": [{"displayName": "Alice"}],
                "webViewLink": "https://drive.google.com/file/d/file1/view",
            },
        ]
    }
    mock_service.files.return_value.list.return_value.execute.return_value = files_result
    mock_service.files.return_value.get_media.return_value = MagicMock()

    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service), \
         patch.object(gdrive_module, "MediaIoBaseDownload") as MockDownloader, \
         patch.object(gdrive_module, "io") as MockIO:
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance
        connector.authenticate()

        fh_mock = MagicMock()
        fh_mock.getvalue.return_value = b"hello world"
        MockIO.BytesIO.return_value = fh_mock

        downloader_instance = MockDownloader.return_value
        downloader_instance.next_chunk.side_effect = [
            (MagicMock(progress=MagicMock(return_value=1.0)), True),
        ]

        items = connector.sync(limit=10)

    assert len(items) == 1
    assert items[0].id == "file1"
    assert items[0].content == "hello world"
    assert items[0].source == "gdrive"
    assert items[0].title == "doc1.txt"
    assert items[0].author == "Alice"
    assert connector.state == ConnectorState.CONNECTED


def test_sync_skips_folders(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    files_result = {
        "files": [
            {
                "id": "folder1",
                "name": "MyFolder",
                "mimeType": "application/vnd.google-apps.folder",
                "modifiedTime": "2024-01-01T00:00:00Z",
            },
        ]
    }
    mock_service.files.return_value.list.return_value.execute.return_value = files_result

    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service):
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance
        connector.authenticate()

        items = connector.sync(limit=10)
        assert items == []


def test_sync_binary_file(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    files_result = {
        "files": [
            {
                "id": "file1",
                "name": "image.png",
                "mimeType": "image/png",
                "size": "1024",
                "modifiedTime": "2024-01-01T00:00:00Z",
                "createdTime": "2024-01-01T00:00:00Z",
                "owners": [],
            },
        ]
    }
    mock_service.files.return_value.list.return_value.execute.return_value = files_result
    mock_service.files.return_value.get_media.return_value = MagicMock()

    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service), \
         patch.object(gdrive_module, "MediaIoBaseDownload") as MockDownloader, \
         patch.object(gdrive_module, "io") as MockIO:
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance
        connector.authenticate()

        fh_mock = MagicMock()
        fh_mock.getvalue.return_value = b"\x89PNG\r\n\x1a\n"
        MockIO.BytesIO.return_value = fh_mock

        downloader_instance = MockDownloader.return_value
        downloader_instance.next_chunk.side_effect = [
            (MagicMock(progress=MagicMock(return_value=1.0)), True),
        ]

        items = connector.sync(limit=10)

    assert len(items) == 1
    assert "<binary file" in items[0].content


def test_sync_not_connected(
    connector: "gdrive_module.GoogleDriveConnector"
) -> None:
    assert connector.sync() == []


def test_sync_folder(
    connector: "gdrive_module.GoogleDriveConnector", mock_service: MagicMock
) -> None:
    files_result = {"files": []}
    mock_service.files.return_value.list.return_value.execute.return_value = files_result

    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=mock_service):
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance
        connector.authenticate()

        items = connector.sync_folder(folder_id="other_folder", limit=5)

    assert items == []
    call_kwargs = mock_service.files.return_value.list.call_args.kwargs
    assert "other_folder" in call_kwargs["q"]


# --------------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------------- #

def test_context_manager(mock_credentials: dict[str, str]) -> None:
    with patch.object(gdrive_module, "Credentials") as MockCreds, \
         patch.object(gdrive_module, "build", return_value=MagicMock()):
        creds_instance = MagicMock()
        creds_instance.expired = False
        MockCreds.return_value = creds_instance

        with patch.object(gdrive_module, "_HAS_GOOGLE_API", True):
            conn = gdrive_module.GoogleDriveConnector(
                config={"credentials": mock_credentials}
            )
        with conn:
            assert conn.state == ConnectorState.CONNECTED
        assert conn.state == ConnectorState.DISCONNECTED


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def test_parse_rfc3339() -> None:
    dt = gdrive_module.GoogleDriveConnector._parse_rfc3339("2024-03-15T12:30:00Z")
    assert dt.year == 2024
    assert dt.month == 3
    assert dt.day == 15
    assert dt.hour == 12
    assert dt.minute == 30


def test_parse_rfc3339_empty() -> None:
    dt = gdrive_module.GoogleDriveConnector._parse_rfc3339("")
    assert isinstance(dt, datetime)


def test_get_export_mime() -> None:
    assert (
        gdrive_module.GoogleDriveConnector._get_export_mime(
            "application/vnd.google-apps.document"
        )
        == "text/plain"
    )
    assert (
        gdrive_module.GoogleDriveConnector._get_export_mime(
            "application/vnd.google-apps.spreadsheet"
        )
        == "text/csv"
    )
    assert (
        gdrive_module.GoogleDriveConnector._get_export_mime(
            "application/vnd.google-apps.drawing"
        )
        == "image/png"
    )
    assert gdrive_module.GoogleDriveConnector._get_export_mime("unknown") == "text/plain"
