# NeuralMem Raycast Extension (V2.3)

Scaffold for the NeuralMem Raycast extension providing quick save, search, and recent-memory access directly from the macOS command palette.

## Structure

```
extensions/raycast/
├── package.json           # Raycast extension manifest
├── src/
│   ├── search.tsx         # Search memories command
│   ├── save.tsx           # Save to memory command
│   └── recent.tsx         # Recent memories command
└── README.md
```

## Features

1. **Search Memories** (`search.tsx`) — Live search with debounce:
   - Semantic / keyword queries against NeuralMem API
   - Displays title, space, relevance score, and creation date
   - Actions: Open in browser, copy content, copy URL

2. **Save to Memory** (`save.tsx`) — Form-based capture:
   - Title, content, URL, space, and tags
   - "Paste from Clipboard" shortcut
   - Auto-detects type (`page` if URL provided, else `note`)

3. **Recent Memories** (`recent.tsx`) — Auto-loaded list:
   - Fetches last 20 memories on open
   - Filter bar for quick narrowing
   - Refresh action

## Setup

### Prerequisites

- macOS with [Raycast](https://raycast.com) installed
- Node.js >= 18
- A running NeuralMem API server (default: `http://localhost:8000/api/v1`)

### Install Dependencies

```bash
cd /Users/Zhuanz/Desktop/ai-agent-research/neuralmem-main/extensions/raycast
npm install
```

### Development

```bash
npm run dev
```

This starts the Raycast development server and hot-reloads changes.

### Build

```bash
npm run build
```

### Publish (internal / private)

```bash
npm run publish
```

## Configuration

Configure via Raycast Preferences after installing the extension:

| Preference      | Default                        | Description                           |
|-----------------|--------------------------------|---------------------------------------|
| API Base URL    | `http://localhost:8000/api/v1` | NeuralMem API server endpoint         |
| API Key         | *(empty)*                      | Optional Bearer token                 |
| Default Space   | `default`                      | Namespace for saves and recent list   |

## API Endpoints Used

| Command | Method | Endpoint                          | Params                         |
|---------|--------|-----------------------------------|--------------------------------|
| Search  | GET    | `{apiBaseUrl}/search`             | `q`, `space`, `limit=20`       |
| Save    | POST   | `{apiBaseUrl}/memories`           | JSON body (SavePayload)        |
| Recent  | GET    | `{apiBaseUrl}/memories/recent`    | `space`, `limit=20`            |

## Testing

Run unit tests (mock-based, no Raycast runtime required):

```bash
cd /Users/Zhuanz/Desktop/ai-agent-research/neuralmem-main
python -m pytest tests/unit/test_raycast_extension.py -v
```

## Notes

- This is a **scaffold/architecture module** — not a full production build.
- Icon (`icon.png`) is referenced in `package.json` but not included in scaffold; add your own 512x512 PNG.
- The extension uses `@raycast/api` and `@raycast/utils` from the Raycast ecosystem.
- TypeScript is configured for strict mode; run `npm run lint` to validate.
