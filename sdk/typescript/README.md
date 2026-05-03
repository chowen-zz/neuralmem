# @neuralmem/sdk

TypeScript SDK for [NeuralMem](https://github.com/nousresearch/neuralmem) — Memory as Infrastructure.

Zero dependencies. Uses the native `fetch` API (Node 18+ or any modern browser).

## Installation

```bash
npm install @neuralmem/sdk
```

## Quick Start

```typescript
import { NeuralMemClient } from "@neuralmem/sdk";

const client = new NeuralMemClient({
  baseUrl: "http://localhost:8000",
});

// Store a memory
const memories = await client.remember(
  "User prefers TypeScript over JavaScript",
  "user-123"
);
console.log(`Stored ${memories.length} memory item(s)`);

// Search memories
const results = await client.recall(
  "What programming language does the user prefer?",
  "user-123",
  5
);

for (const result of results) {
  console.log(`[${result.score.toFixed(2)}] ${result.memory.content}`);
}

// Health check
const health = await client.health();
console.log(`Status: ${health.status}`);
```

## API Reference

### `NeuralMemClient(options)`

Create a new client instance.

| Parameter     | Type   | Description                              |
|---------------|--------|------------------------------------------|
| `options.baseUrl` | `string` | Base URL of the NeuralMem server     |
| `options.apiKey`  | `string` | Optional API key for authentication  |

### Methods

#### `remember(content, userId?, memoryTypes?) → Promise<Memory[]>`

Store one or more memories extracted from the given content.

```typescript
const memories = await client.remember(
  "The meeting is scheduled for Friday at 3pm",
  "user-1"
);
```

#### `recall(query, userId?, limit?) → Promise<SearchResult[]>`

Search for memories matching the query.

```typescript
const results = await client.recall("When is the meeting?", "user-1", 10);
```

#### `reflect(userId?) → Promise<ReflectResult>`

Reflect on stored memories — retrieve and synthesize related memories.

```typescript
const reflection = await client.reflect("user-1");
console.log(reflection.content);
```

#### `forget(memoryId?) → Promise<void>`

Delete a specific memory by ID.

```typescript
await client.forget("01ABCDEF123456789");
```

#### `health() → Promise<HealthReport>`

Check server health status.

```typescript
const report = await client.health();
console.log(report.status); // "healthy" | "degraded" | "unhealthy"
console.log(report.checks.storage); // "healthy"
```

#### `listMemories(limit?, offset?) → Promise<MemoryListResponse>`

List memories with pagination (dashboard API).

```typescript
const page = await client.listMemories(25, 0);
console.log(`Total: ${page.total}, showing ${page.memories.length}`);
```

## Types

All types are exported from the package:

```typescript
import type {
  Memory,
  MemoryType,
  SearchResult,
  HealthReport,
  NeuralMemClientOptions,
} from "@neuralmem/sdk";
```

## Error Handling

The client throws `NeuralMemError` on API failures:

```typescript
import { NeuralMemClient, NeuralMemError } from "@neuralmem/sdk";

try {
  await client.recall("test query");
} catch (err) {
  if (err instanceof NeuralMemError) {
    console.error(`API error (${err.statusCode}): ${err.message}`);
  }
}
```

## Requirements

- Node.js >= 16.0.0 (native `fetch` in 18+, or use `node-fetch` polyfill)
- A running NeuralMem server instance

## License

MIT
