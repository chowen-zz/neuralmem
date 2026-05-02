# neuralmem (npm)

> Local-first, MCP-native agent memory for Node.js

[![npm](https://img.shields.io/npm/v/neuralmem)](https://www.npmjs.com/package/neuralmem)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

This is the official Node.js/TypeScript client for [NeuralMem](https://github.com/chowen-zz/neuralmem) — the local-first agent memory library with 4-strategy parallel retrieval.

## Prerequisites

- **Node.js** ≥ 18
- **Python** ≥ 3.10 with `neuralmem` installed:
  ```bash
  pip install neuralmem
  ```

## Install

```bash
npm install neuralmem
```

## Quick Start

```typescript
import { NeuralMem } from "neuralmem";

const mem = new NeuralMem();
await mem.connect();

// Store memories
await mem.remember("User prefers TypeScript over JavaScript");
await mem.remember("User uses React for frontend", {
  tags: ["tech", "frontend"],
  userId: "user-123",
});

// Search memories (4-strategy parallel: semantic + BM25 + graph + temporal)
const results = await mem.recall("What frontend framework does the user use?");
for (const r of results) {
  console.log(`[${r.score.toFixed(2)}] ${r.memory.content}`);
}

// Cleanup
await mem.disconnect();
```

## API

### `new NeuralMem(config?)`

Create a new client instance.

```typescript
interface NeuralMemConfig {
  transport?: "stdio" | "http";  // default: "stdio"
  command?: string;               // default: "neuralmem"
  args?: string[];                // default: ["mcp"]
  cwd?: string;                   // working directory
  env?: Record<string, string>;   // environment variables
  timeout?: number;               // connection timeout (ms), default: 30000
}
```

### `mem.connect()` / `mem.disconnect()`

Manage the connection to the NeuralMem MCP server. If not called explicitly, `connect()` is called automatically on the first API call.

### `mem.remember(content, options?)`

Store a new memory.

```typescript
await mem.remember("Alice is the team lead", {
  userId: "u1",
  tags: ["team", "leadership"],
  memoryType: "fact",  // fact | preference | instruction | context | event | relationship
});
```

### `mem.recall(query, options?)`

Search memories using 4-strategy parallel retrieval + RRF fusion.

```typescript
const results = await mem.recall("Who leads the team?", {
  userId: "u1",
  limit: 5,
  explain: true,  // include retrieval explanations
});
```

### `mem.recallWithExplanation(query, options?)`

Same as `recall()` but always includes explanations.

### `mem.reflect(memoryId, newContent?, importance?)`

Update or reinforce an existing memory.

```typescript
await mem.reflect("01HXYZ...", "Updated content", 0.9);
```

### `mem.forget(memoryId)`

Delete a memory by ID.

### `mem.consolidate(options?)`

Merge similar memories (Ebbinghaus forgetting curve).

```typescript
const result = await mem.consolidate({ similarityThreshold: 0.9 });
```

### `mem.resolveConflict(memoryId, action?)`

Resolve a superseded memory conflict.

```typescript
await mem.resolveConflict("01HXYZ...", "reactivate");  // or "delete"
```

### `mem.rememberBatch(contents, options?)`

Store multiple memories at once.

```typescript
await mem.rememberBatch(
  ["Fact 1", "Fact 2", "Fact 3"],
  { tags: ["bulk"], userId: "u1" }
);
```

### `mem.exportMemories(options?)`

Export memories in various formats.

```typescript
const json = await mem.exportMemories({ format: "json" });
const md = await mem.exportMemories({ format: "markdown" });
```

### `mem.forgetBatch(options?)`

Batch delete memories by IDs or tags.

```typescript
await mem.forgetBatch({ ids: ["id1", "id2"] });
await mem.forgetBatch({ tags: ["temporary"] });
await mem.forgetBatch({ tags: ["old"], dryRun: true });  // preview only
```

### `mem.listTools()`

List available MCP tools on the server.

## Usage Patterns

### With Express/Fastify API server

```typescript
import express from "express";
import { NeuralMem } from "neuralmem";

const app = express();
const mem = new NeuralMem();
await mem.connect();

app.post("/remember", async (req, res) => {
  const result = await mem.remember(req.body.content, {
    userId: req.body.userId,
  });
  res.json({ result });
});

app.get("/recall", async (req, res) => {
  const results = await mem.recall(req.query.q as string, {
    userId: req.query.userId as string,
  });
  res.json({ results });
});
```

### With AI SDK (Vercel)

```typescript
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { NeuralMem } from "neuralmem";

const mem = new NeuralMem();
await mem.connect();

// Before generation — retrieve relevant memories
const memories = await mem.recall("user preferences");
const context = memories.map((r) => r.memory.content).join("\n");

const { text } = await generateText({
  model: openai("gpt-4o"),
  prompt: `User context:\n${context}\n\nUser: ${userMessage}`,
});

// After generation — store the interaction
await mem.remember(`User asked: ${userMessage}. Assistant replied: ${text}`);
```

## Architecture

```
Node.js App
    ↓
neuralmem (npm) — TypeScript client
    ↓ MCP protocol (stdio)
neuralmem mcp — Python MCP server
    ↓
SQLite + sqlite-vec (local storage)
NetworkX graph (knowledge graph)
4-strategy retrieval engine
```

## License

Apache-2.0
