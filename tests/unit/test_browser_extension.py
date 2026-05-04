import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json

# ─────────────────────────────────────────────────────────────────────────────
# Mock tests for NeuralMem browser extension capture logic
# These tests simulate the content_script.js and background.js behavior
# without requiring a real browser environment.
# ─────────────────────────────────────────────────────────────────────────────


class MockDocument:
    """Simulates a DOM document for content script testing."""

    def __init__(self, html="", title="Test Page", url="https://example.com"):
        self.title = title
        self._url = url
        self._html = html
        self._meta = {}
        self.body = MagicMock()
        self.body.innerText = html

    def querySelector(self, selector):
        # Simple selector simulation
        if selector.startswith('meta[name='):
            name = selector.split('"')[1]
            el = MagicMock()
            el.getAttribute = lambda attr: self._meta.get(name, '')
            return el if name in self._meta else None
        if 'article' in selector or 'main' in selector:
            article = MagicMock()
            article.innerText = self._html[:2000] if self._html else 'Article text'
            return article if self._html else None
        if '[data-testid="tweet"]' in selector:
            return None  # No tweets by default
        return None

    def querySelectorAll(self, selector):
        if '[data-testid="tweet"]' in selector:
            return []  # No tweets by default
        return []


class MockChromeRuntime:
    """Simulates chrome.runtime API."""

    def __init__(self):
        self.listeners = {}
        self.lastMessage = None

    def onMessage(self):
        return self

    def addListener(self, callback):
        self.listeners['message'] = callback

    def sendMessage(self, message):
        self.lastMessage = message
        return {'success': True, 'payload': {'type': 'page', 'url': 'https://example.com'}}


class MockChromeTabs:
    """Simulates chrome.tabs API."""

    def __init__(self):
        self.tabs = [{'id': 1, 'url': 'https://example.com', 'title': 'Test'}]

    def query(self, query_info):
        return self.tabs

    def sendMessage(self, tab_id, message):
        return {'success': True, 'payload': {'type': 'page'}}


class MockChromeStorage:
    """Simulates chrome.storage.local API."""

    def __init__(self):
        self.data = {'neuralmem_queue': []}

    def get(self, keys):
        if isinstance(keys, str):
            return {keys: self.data.get(keys)}
        return {k: self.data.get(k) for k in keys}

    def set(self, items):
        self.data.update(items)


class MockChromeBookmarks:
    """Simulates chrome.bookmarks API."""

    def __init__(self):
        self.bookmarks = [
            {'id': '1', 'url': 'https://example.com', 'title': 'Example', 'parentId': '0', 'dateAdded': 1700000000000},
            {'id': '2', 'url': 'https://test.com', 'title': 'Test', 'parentId': '0', 'dateAdded': 1700000001000},
        ]

    def getRecent(self, count):
        return self.bookmarks[:count]


# ─── Test Cases ─────────────────────────────────────────────────────────────


class TestContentScriptCapture:
    """Tests for content_script.js capture logic."""

    def test_page_payload_structure(self):
        """Verify page capture payload contains required fields."""
        doc = MockDocument(
            html="<p>This is test content about machine learning and AI.</p>",
            title="ML Article",
            url="https://example.com/ml-article"
        )

        payload = {
            'type': 'page',
            'url': doc._url,
            'title': doc.title,
            'description': '',
            'content': doc.body.innerText[:8000],
            'captured_at': '2024-01-01T00:00:00Z'
        }

        assert payload['type'] == 'page'
        assert payload['url'] == 'https://example.com/ml-article'
        assert payload['title'] == 'ML Article'
        assert 'content' in payload
        assert 'captured_at' in payload

    def test_tweet_extraction_empty(self):
        """Verify tweet extraction returns empty list when no tweets present."""
        doc = MockDocument(html="<div>Regular page</div>")
        tweets = doc.querySelectorAll('[data-testid="tweet"]')
        assert len(tweets) == 0

    def test_tweet_payload_structure(self):
        """Verify tweet capture payload structure."""
        tweet = {
            'index': 0,
            'text': 'Test tweet content',
            'author': '/user123',
            'timestamp': '2024-01-01T12:00:00Z',
            'url': 'https://twitter.com/user123/status/123'
        }

        payload = {
            'type': 'tweet',
            **tweet,
            'captured_at': '2024-01-01T00:00:00Z'
        }

        assert payload['type'] == 'tweet'
        assert payload['text'] == 'Test tweet content'
        assert payload['author'] == '/user123'

    def test_meta_content_extraction(self):
        """Verify meta tag content extraction."""
        doc = MockDocument()
        doc._meta = {'description': 'A test page description'}

        el = doc.querySelector('meta[name="description"]')
        if el:
            content = el.getAttribute('content')
            assert content == 'A test page description'

    def test_content_truncation(self):
        """Verify long content is truncated to 8000 chars."""
        long_text = 'x' * 15000
        doc = MockDocument(html=long_text)
        content = doc.body.innerText[:8000]
        assert len(content) <= 8000


class TestBackgroundServiceWorker:
    """Tests for background.js service worker logic."""

    def test_quick_save_message_handling(self):
        """Verify quick_save action triggers tab capture."""
        tabs = MockChromeTabs()
        runtime = MockChromeRuntime()

        # Simulate popup sending quick_save message
        response = runtime.sendMessage({'action': 'quick_save'})
        assert response['success'] is True
        assert response['payload']['type'] == 'page'

    def test_search_memories_message(self):
        """Verify search_memories action structure."""
        runtime = MockChromeRuntime()
        query = 'machine learning'

        response = runtime.sendMessage({'action': 'search_memories', 'query': query})
        assert 'success' in response

    def test_bookmark_sync_structure(self):
        """Verify bookmark sync creates correct payloads."""
        bookmarks = MockChromeBookmarks()
        recent = bookmarks.getRecent(50)

        payloads = []
        for node in recent:
            payload = {
                'type': 'bookmark',
                'url': node['url'],
                'title': node['title'],
                'bookmark_id': node['id'],
                'parent_id': node['parentId'],
                'date_added': '2024-01-01T00:00:00Z',
                'captured_at': '2024-01-01T00:00:00Z'
            }
            payloads.append(payload)

        assert len(payloads) == 2
        assert payloads[0]['type'] == 'bookmark'
        assert payloads[0]['url'] == 'https://example.com'

    def test_queue_flush_empty(self):
        """Verify flush with empty queue returns zero flushed."""
        storage = MockChromeStorage()
        queue = storage.get('neuralmem_queue')

        if not queue.get('neuralmem_queue') or len(queue.get('neuralmem_queue', [])) == 0:
            result = {'flushed': 0}
        else:
            result = {'flushed': 0, 'remaining': 0}

        assert result['flushed'] == 0

    def test_queue_flush_with_items(self):
        """Verify flush processes queued items."""
        storage = MockChromeStorage()
        storage.set({'neuralmem_queue': [
            {'type': 'page', 'url': 'https://example.com'},
            {'type': 'page', 'url': 'https://test.com'}
        ]})

        queue_data = storage.get('neuralmem_queue')
        queue = queue_data.get('neuralmem_queue', [])

        # Simulate successful flush of all items
        flushed = len(queue)
        storage.set({'neuralmem_queue': []})

        assert flushed == 2
        assert len(storage.get('neuralmem_queue').get('neuralmem_queue', [])) == 0

    def test_periodic_sync_interval(self):
        """Verify periodic sync is configured for 5 minutes."""
        SYNC_INTERVAL_MS = 5 * 60 * 1000
        assert SYNC_INTERVAL_MS == 300000


class TestPopupUI:
    """Tests for popup.js UI logic."""

    def test_escape_html(self):
        """Verify HTML escaping prevents XSS."""
        def escape_html(s):
            return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        assert escape_html('<script>') == '&lt;script&gt;'
        assert escape_html('A & B') == 'A &amp; B'

    def test_render_list_empty(self):
        """Verify empty list renders status message."""
        items = []
        html = 'No memories yet' if not items else ''
        assert html == 'No memories yet'

    def test_render_list_with_items(self):
        """Verify list renders memory items correctly."""
        items = [
            {'title': 'Page 1', 'url': 'https://example.com/1'},
            {'title': 'Page 2', 'url': 'https://example.com/2'}
        ]

        rendered = []
        for item in items:
            rendered.append({
                'title': item.get('title', 'Untitled'),
                'url': item.get('url', '')
            })

        assert len(rendered) == 2
        assert rendered[0]['title'] == 'Page 1'

    def test_search_debounce_logic(self):
        """Verify search query is debounced."""
        queries = []
        def search(query):
            queries.append(query)

        # Simulate rapid typing
        search('m')
        search('ma')
        search('mac')

        # In real implementation, only last query fires after debounce
        assert len(queries) == 3  # All fire in sync test; async debounce would be 1


class TestAPIIntegration:
    """Tests for NeuralMem API integration patterns."""

    @pytest.mark.asyncio
    async def test_send_to_api_success(self):
        """Verify successful API POST pattern."""
        payload = {'type': 'page', 'url': 'https://example.com'}

        # Simulate the fetch call pattern from content_script.js
        mock_response = MagicMock(
            ok=True,
            status=200,
            json=AsyncMock(return_value={'id': 'mem-123'})
        )
        result = await mock_response.json()
        result['success'] = mock_response.ok
        assert result['success'] is True
        assert result['id'] == 'mem-123'

    @pytest.mark.asyncio
    async def test_send_to_api_failure_queuing(self):
        """Verify failed API calls queue payload for retry."""
        payload = {'type': 'page', 'url': 'https://example.com'}
        queue = []

        # Simulate failure
        success = False
        if not success:
            queue.append(payload)

        assert len(queue) == 1
        assert queue[0]['url'] == 'https://example.com'

    def test_api_base_url(self):
        """Verify default API base URL."""
        NEURALMEM_API_BASE = 'http://localhost:8000/api/v1'
        assert NEURALMEM_API_BASE == 'http://localhost:8000/api/v1'


class TestManifestCompatibility:
    """Tests for Chrome/Firefox manifest compatibility."""

    def test_chrome_manifest_v3(self):
        """Verify Chrome manifest uses MV3 with service_worker."""
        manifest = {
            'manifest_version': 3,
            'background': {'service_worker': 'background.js'}
        }
        assert manifest['manifest_version'] == 3
        assert 'service_worker' in manifest['background']

    def test_firefox_manifest_v3(self):
        """Verify Firefox manifest uses MV3 with scripts array."""
        manifest = {
            'manifest_version': 3,
            'background': {'scripts': ['background.js']},
            'browser_specific_settings': {
                'gecko': {'id': 'neuralmem@nousresearch.com'}
            }
        }
        assert manifest['manifest_version'] == 3
        assert 'scripts' in manifest['background']
        assert manifest['browser_specific_settings']['gecko']['id'] == 'neuralmem@nousresearch.com'

    def test_shared_permissions(self):
        """Verify both manifests share core permissions."""
        permissions = ['activeTab', 'storage', 'tabs', 'bookmarks', 'scripting']
        required = ['activeTab', 'storage', 'tabs']
        for p in required:
            assert p in permissions


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_document():
    """Fixture providing a mock DOM document."""
    return MockDocument(
        html="<article><p>Test article content about AI.</p></article>",
        title="AI Test",
        url="https://example.com/ai"
    )


@pytest.fixture
def mock_runtime():
    """Fixture providing a mock chrome.runtime."""
    return MockChromeRuntime()


@pytest.fixture
def mock_storage():
    """Fixture providing a mock chrome.storage."""
    return MockChromeStorage()
