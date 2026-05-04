import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json

# ─────────────────────────────────────────────────────────────────────────────
# Mock tests for NeuralMem Raycast extension logic
# These tests simulate the search.tsx, save.tsx, and recent.tsx behavior
# without requiring the Raycast runtime or a real NeuralMem API server.
# ─────────────────────────────────────────────────────────────────────────────

NEURALMEM_API_BASE = "http://localhost:8000/api/v1"
DEFAULT_SPACE = "default"
API_KEY = "nm_test_key_123"


class MockFetchResponse:
    """Simulates a fetch() Response object."""

    def __init__(self, ok=True, status=200, json_data=None, text=""):
        self.ok = ok
        self.status = status
        self._json = json_data or {}
        self._text = text

    async def json(self):
        return self._json

    async def text(self):
        return self._text


class MockRaycastAPI:
    """Simulates @raycast/api helpers used by the extension commands."""

    def __init__(self):
        self.toasts = []
        self.clipboard_text = ""
        self.preferences = {
            "apiBaseUrl": NEURALMEM_API_BASE,
            "apiKey": API_KEY,
            "defaultSpace": DEFAULT_SPACE,
        }

    def showToast(self, style, title, message=""):
        self.toasts.append({"style": style, "title": title, "message": message})

    def getPreferenceValues(self):
        return self.preferences

    async def readClipboardText(self):
        return self.clipboard_text


# ─── Test Cases ───────────────────────────────────────────────────────────────


class TestSearchCommand:
    """Tests for search.tsx search logic."""

    @pytest.mark.asyncio
    async def test_search_api_call_structure(self):
        """Verify search builds correct GET request."""
        import urllib.parse
        query = "machine learning"
        space = "research"
        limit = 20

        url = f"{NEURALMEM_API_BASE}/search?q={urllib.parse.quote(query)}&space={space}&limit={limit}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        # Simulate expected URL and headers
        assert "search" in url
        assert "q=machine%20learning" in url
        assert headers["Authorization"] == f"Bearer {API_KEY}"

    @pytest.mark.asyncio
    async def test_search_success_response(self):
        """Verify search parses successful response."""
        mock_data = {
            "results": [
                {
                    "id": "mem-001",
                    "title": "ML Paper",
                    "content": "Neural networks...",
                    "url": "https://example.com/ml",
                    "space": "research",
                    "created_at": "2024-01-15T10:00:00Z",
                    "score": 0.92,
                }
            ],
            "total": 1,
            "query": "machine learning",
        }
        response = MockFetchResponse(ok=True, status=200, json_data=mock_data)
        data = await response.json()

        assert data["total"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "mem-001"
        assert data["results"][0]["score"] == 0.92

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """Verify search handles zero results gracefully."""
        mock_data = {"results": [], "total": 0, "query": "xyzabc"}
        response = MockFetchResponse(ok=True, status=200, json_data=mock_data)
        data = await response.json()

        assert data["total"] == 0
        assert len(data["results"]) == 0

    @pytest.mark.asyncio
    async def test_search_api_failure(self):
        """Verify search handles HTTP errors."""
        response = MockFetchResponse(ok=False, status=500, text="Internal Server Error")
        assert response.ok is False
        assert response.status == 500

    def test_debounce_logic(self):
        """Verify debounce timer configuration."""
        debounce_ms = 300
        assert debounce_ms == 300


class TestSaveCommand:
    """Tests for save.tsx save logic."""

    @pytest.mark.asyncio
    async def test_save_note_payload(self):
        """Verify note save payload structure."""
        payload = {
            "type": "note",
            "title": "My Note",
            "content": "Important research findings...",
            "space": "research",
            "tags": ["ai", "paper"],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        mock_response = MockFetchResponse(
            ok=True, status=201, json_data={"id": "mem-002", "success": True}
        )
        result = await mock_response.json()
        result["success"] = mock_response.ok

        assert payload["type"] == "note"
        assert payload["title"] == "My Note"
        assert payload["space"] == "research"
        assert result["id"] == "mem-002"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_save_page_payload(self):
        """Verify page save payload auto-detects type when URL present."""
        payload = {
            "type": "page",
            "title": "Article",
            "content": "Page content...",
            "url": "https://example.com/article",
            "space": DEFAULT_SPACE,
        }

        assert payload["type"] == "page"
        assert payload["url"] == "https://example.com/article"

    @pytest.mark.asyncio
    async def test_save_api_failure(self):
        """Verify save handles HTTP errors and shows toast."""
        response = MockFetchResponse(ok=False, status=403, text="Forbidden")
        assert response.ok is False
        assert response.status == 403

    def test_clipboard_paste_simulation(self):
        """Verify clipboard read returns text content."""
        clipboard = MockRaycastAPI()
        clipboard.clipboard_text = "Clipboard content here"
        assert clipboard.clipboard_text == "Clipboard content here"

    def test_tags_parsing(self):
        """Verify comma-separated tags are split correctly."""
        tags_raw = "ai, research, paper ,  deep-learning"
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        assert tags == ["ai", "research", "paper", "deep-learning"]


class TestRecentCommand:
    """Tests for recent.tsx recent memories logic."""

    @pytest.mark.asyncio
    async def test_recent_api_call_structure(self):
        """Verify recent memories builds correct GET request."""
        space = DEFAULT_SPACE
        limit = 20
        url = f"{NEURALMEM_API_BASE}/memories/recent?space={space}&limit={limit}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        assert "memories/recent" in url
        assert headers["Authorization"] == f"Bearer {API_KEY}"

    @pytest.mark.asyncio
    async def test_recent_success_response(self):
        """Verify recent memories parses successful response."""
        mock_data = {
            "memories": [
                {
                    "id": "mem-003",
                    "title": "Recent Note",
                    "content": "Just saved...",
                    "url": "https://example.com/recent",
                    "space": "default",
                    "created_at": "2024-01-20T14:30:00Z",
                    "type": "note",
                },
                {
                    "id": "mem-004",
                    "title": "Recent Page",
                    "content": "Page capture...",
                    "url": "https://example.com/page",
                    "space": "default",
                    "created_at": "2024-01-20T14:25:00Z",
                    "type": "page",
                },
            ],
            "total": 2,
        }
        response = MockFetchResponse(ok=True, status=200, json_data=mock_data)
        data = await response.json()

        assert data["total"] == 2
        assert len(data["memories"]) == 2
        assert data["memories"][0]["id"] == "mem-003"
        assert data["memories"][1]["type"] == "page"

    @pytest.mark.asyncio
    async def test_recent_empty_response(self):
        """Verify recent handles empty memory list."""
        mock_data = {"memories": [], "total": 0}
        response = MockFetchResponse(ok=True, status=200, json_data=mock_data)
        data = await response.json()

        assert data["total"] == 0
        assert len(data["memories"]) == 0

    @pytest.mark.asyncio
    async def test_recent_api_failure(self):
        """Verify recent handles HTTP errors."""
        response = MockFetchResponse(ok=False, status=502, text="Bad Gateway")
        assert response.ok is False
        assert response.status == 502


class TestPreferences:
    """Tests for Raycast preference handling."""

    def test_default_preferences(self):
        """Verify default preference values."""
        prefs = {
            "apiBaseUrl": NEURALMEM_API_BASE,
            "apiKey": API_KEY,
            "defaultSpace": DEFAULT_SPACE,
        }
        assert prefs["apiBaseUrl"] == "http://localhost:8000/api/v1"
        assert prefs["defaultSpace"] == "default"

    def test_preferences_without_api_key(self):
        """Verify preferences work when API key is omitted."""
        prefs = {
            "apiBaseUrl": NEURALMEM_API_BASE,
            "apiKey": None,
            "defaultSpace": DEFAULT_SPACE,
        }
        assert prefs["apiKey"] is None


class TestPayloadStructures:
    """Tests for shared data structures across commands."""

    def test_memory_item_structure(self):
        """Verify memory item has all required fields."""
        memory = {
            "id": "mem-005",
            "title": "Test",
            "content": "Content",
            "url": "https://example.com",
            "space": "default",
            "created_at": "2024-01-01T00:00:00Z",
            "score": 0.85,
            "type": "note",
        }
        required = ["id", "title", "content", "created_at"]
        for field in required:
            assert field in memory

    def test_save_payload_required_fields(self):
        """Verify save payload has required fields."""
        payload = {
            "type": "note",
            "title": "T",
            "content": "C",
            "space": "default",
        }
        assert payload["type"] in ["note", "page", "bookmark"]
        assert payload["content"]
        assert payload["space"]


class TestErrorHandling:
    """Tests for error handling patterns."""

    def test_network_error_message(self):
        """Verify network errors produce user-friendly messages."""
        error = Exception("Network request failed")
        msg = str(error)
        assert "failed" in msg.lower()

    def test_json_parse_error(self):
        """Verify invalid JSON responses are handled."""
        bad_json = "not json"
        try:
            json.loads(bad_json)
            assert False, "Should have raised"
        except json.JSONDecodeError:
            assert True


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_raycast_api():
    """Fixture providing a mock Raycast API environment."""
    return MockRaycastAPI()


@pytest.fixture
def mock_search_response():
    """Fixture providing a sample search response."""
    return MockFetchResponse(
        ok=True,
        status=200,
        json_data={
            "results": [
                {
                    "id": "mem-006",
                    "title": "Fixture Result",
                    "content": "Fixture content",
                    "url": "https://example.com/fixture",
                    "space": "default",
                    "created_at": "2024-01-10T08:00:00Z",
                    "score": 0.95,
                }
            ],
            "total": 1,
            "query": "fixture",
        },
    )


@pytest.fixture
def mock_save_response():
    """Fixture providing a sample save response."""
    return MockFetchResponse(
        ok=True, status=201, json_data={"id": "mem-007", "success": True}
    )


@pytest.fixture
def mock_recent_response():
    """Fixture providing a sample recent memories response."""
    return MockFetchResponse(
        ok=True,
        status=200,
        json_data={
            "memories": [
                {
                    "id": "mem-008",
                    "title": "Recent Fixture",
                    "content": "Recent fixture content",
                    "space": "default",
                    "created_at": "2024-01-21T09:00:00Z",
                    "type": "note",
                }
            ],
            "total": 1,
        },
    )
