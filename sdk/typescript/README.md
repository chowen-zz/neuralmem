# @neuralmem/sdk

TypeScript SDK for [NeuralMem](https://github.com/nousresearch/neuralmem) V1.6 — Memory as Infrastructure.

Zero dependencies. Uses the native `fetch` API (Node 18+ or any modern browser). Supports MCP stdio transport for local tool-server integration.

## Installation

```bash
npm install @neuralmem/sdk
```

## Quick Start

```typescript
import { NeuralMemClient, search } from "@neuralmem/sdk";

const client = new NeuralMemClient({
  baseUrl: "http://localhost:8000",
});

// Store a memory
const memories = await client.remember(
  "User prefers TypeScript over JavaScript",
  { userId: "user-123" }
);
console.log(`Stored ${memories.length} memory item(s)`);

// Search memories
const results = await client.recall(
  "What programming language does the user prefer?",
  { userId: "user-123", limit: 5 }
);

for (const result of results) {
  console.log(`[${result.score.toFixed(2)}] ${result.memory.content}`);
}

// Fluent search query builder
const query = search("What does the user like?")
  .forUser("user-123")
  .ofTypes("preference", "fact")
  .withTags("food", "hobby")
  .limit(5)
  .minScore(0.5)
  .build();

// Health check
const health = await client.health();
console.log(`Status: ${health.status}`);
```

## API Reference

### `NeuralMemClient(options)`

Create a new client instance.

| Parameter         | Type   | Description                              |
|-------------------|--------|------------------------------------------|
| `options.baseUrl` | `string` | Base URL of the NeuralMem server       |
| `options.apiKey`  | `string` | Optional API key for authentication    |
| `options.timeout` | `number` | Request timeout in ms (default 30000)  |

### Core Methods

#### `remember(content, options?) → Promise<Memory[]>`

Store one or more memories extracted from the given content.

```typescript
const memories = await client.remember(
  "The meeting is scheduled for Friday at 3pm",
  { userId: "user-1", tags: ["meeting"], importance: 0.8 }
);
```

#### `recall(query, options?) → Promise<SearchResult[]>`

Search for memories matching the query.

```typescript
const results = await client.recall("When is the meeting?", {
  userId: "user-1",
  limit: 10,
  minScore: 0.3,
});
```

#### `reflect(topic, options?) → Promise<ReflectResult>`

Reflect on a topic using memory retrieval and graph traversal.

```typescript
const reflection = await client.reflect("User's work schedule", {
  userId: "user-1",
  depth: 2,
});
console.log(reflection.content);
```

#### `forget(options?) → Promise<number>`

Delete memories matching criteria. Returns number deleted.

```typescript
const deleted = await client.forget({ memoryId: "01ABCDEF123456789" });
```

### Batch Operations

#### `rememberBatch(contents, options?) → Promise<Memory[]>`

Batch remember multiple items with optional progress callback.

```typescript
const all = await client.rememberBatch(
  ["Fact 1", "Fact 2", "Fact 3"],
  {
    userId: "user-1",
    progressCallback: (current, total, preview) => {
      console.log(`${current}/${total}: ${preview}`);
    },
  }
);
```

#### `forgetBatch(memoryIds?, options?) → Promise<ForgetBatchResult>`

Batch delete with optional dry-run preview.

```typescript
const result = await client.forgetBatch(["id1", "id2"], { dryRun: true });
console.log(`Would delete ${result.count} memories`);
```

### Lifecycle & Maintenance

#### `consolidate(userId?) → Promise<ConsolidateResult>`

Run memory consolidation: decay old memories, merge similar ones.

```typescript
const stats = await client.consolidate("user-1");
console.log(`Decayed: ${stats.decayed}, Merged: ${stats.merged}`);
```

#### `cleanupExpired() → Promise<number>`

Remove all expired memories.

```typescript
const deleted = await client.cleanupExpired();
```

### Single Record & History

#### `get(memoryId) → Promise<Memory | null>`

Retrieve a single memory by ID.

#### `update(memoryId, content, metadata?) → Promise<Memory | null>`

Update a memory's content. Records version history automatically.

#### `history(memoryId) → Promise<MemoryHistoryEntry[]>`

Retrieve version history for a memory.

### Stats, Export, Import

#### `getStats() → Promise<MemoryStats>`

Return memory store statistics.

#### `exportMemories(userId?, format?, includeEmbeddings?) → Promise<string>`

Export memories as JSON, markdown, or CSV.

#### `importMemories(data, format?, userId?, skipDuplicates?) → Promise<number>`

Import memories from exported data.

### Dashboard

#### `health() → Promise<HealthReport>`

Check server health status.

```typescript
const report = await client.health();
console.log(report.status); // "healthy" | "degraded" | "unhealthy"
console.log(report.checks.storage); // "healthy"
```

#### `listMemories(limit?, offset?) → Promise<MemoryListResponse>`

List memories with pagination.

```typescript
const page = await client.listMemories(25, 0);
console.log(`Total: ${page.total}, showing ${page.memories.length}`);
```

### Conflict Resolution

#### `resolveConflict(memoryId, action?) → Promise<boolean>`

Resolve a memory conflict manually.

```typescript
const ok = await client.resolveConflict("mem-id", "reactivate");
```

## Search Query Builder

The SDK provides a fluent `SearchQueryBuilder` for constructing search queries:

```typescript
import { search } from "@neuralmem/sdk";

const query = search("What does the user like?")
  .forUser("user-123")
  .ofTypes("preference", "fact")
  .withTags("food", "hobby")
  .between("2024-01-01", "2024-12-31")
  .limit(5)
  .minScore(0.5)
  .build();
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
  SearchQueryBuilder,
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

- Node.js >= 18.0.0 (native `fetch`)
- A running NeuralMem server instance

## License

MIT
