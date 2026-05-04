# NeuralMem Browser Extension (V2.3)

Scaffold for the NeuralMem browser extension supporting **Chrome** and **Firefox** (Manifest V3).

## Structure

```
extensions/
├── chrome/
│   ├── manifest.json      # Manifest V3 scaffold
│   ├── content_script.js  # Page/tweet/bookmark capture logic
│   ├── background.js      # Service worker: sync, search, queue flush
│   ├── popup.html         # Popup UI scaffold
│   └── popup.js           # Popup interactions (save, search, recent)
└── firefox/
    ├── manifest.json      # Manifest V3 + gecko ID
    ├── content_script.js  # Same capture logic (browser.* API)
    ├── background.js      # Event page variant (browser.* API)
    ├── popup.html
    └── popup.js
```

## Features

1. **Content Script** — Injected into all pages to:
   - Capture full page content (title, description, body text)
   - Extract tweets from twitter.com / x.com
   - Send payloads to NeuralMem API (`localhost:8000/api/v1`)
   - Queue failed captures in `localStorage` for retry

2. **Background Service Worker** — Handles:
   - `quick_save` message from popup (capture active tab)
   - Periodic bookmark sync (every 5 min via `chrome.alarms`)
   - Queue flush for offline-captured items
   - Search & recent memories proxy to NeuralMem API
   - Context menu: "Save to NeuralMem"

3. **Popup UI** — Quick actions:
   - **Quick Save Page** — one-click capture of current tab
   - **Capture Tweets** — scrape visible tweets on Twitter/X
   - **Search** — live search with debounce
   - **Recent Memories** — list of last 10 saved items

## Setup

### Chrome (unpacked)

1. Open `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select `extensions/chrome/`

### Firefox (temporary)

1. Open `about:debugging`
2. Click **This Firefox** → **Load Temporary Add-on**
3. Select `extensions/firefox/manifest.json`

For permanent install, package with `web-ext` or submit to AMO.

## Configuration

Default API endpoint: `http://localhost:8000/api/v1`

Change in:
- `content_script.js` (`NEURALMEM_API_BASE`)
- `background.js` (`NEURALMEM_API_BASE`)

Or via Options page (`options.html` — scaffold only).

## Permissions

| Permission | Purpose |
|------------|---------|
| `activeTab` | Capture current tab content |
| `storage` | Queue failed captures, settings |
| `tabs` | Query active tab, open URLs |
| `bookmarks` | Sync browser bookmarks |
| `scripting` | Inject content script dynamically |
| `host_permissions` | Access NeuralMem API & all sites |

## Testing

Run unit tests:

```bash
cd /Users/Zhuanz/Desktop/ai-agent-research/neuralmem-main
python -m pytest tests/unit/test_browser_extension.py -v
```

## Notes

- This is a **scaffold/architecture module** — not a full production build.
- Icons (`icons/icon16.png`, etc.) are placeholders; add your own assets.
- `options.html` is referenced in manifest but not included in scaffold.
- Firefox uses `browser.*` API with promise-based calls; Chrome uses `chrome.*` with callbacks (polyfilled in popup via `const api = chrome || browser`).
