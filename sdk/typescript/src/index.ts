/**
 * @neuralmem/sdk — TypeScript SDK for NeuralMem V1.6.
 *
 * Zero dependencies. Uses the native `fetch` API (Node 18+ or any modern browser).
 * Supports MCP stdio transport for local tool-server integration.
 *
 * @example
 * ```ts
 * import { NeuralMemClient } from "@neuralmem/sdk";
 *
 * const client = new NeuralMemClient({ baseUrl: "http://localhost:8000" });
 *
 * // Store a memory
 * const memories = await client.remember("User prefers dark mode", { userId: "user-1" });
 *
 * // Search memories
 * const results = await client.recall("What UI theme does user prefer?", { userId: "user-1" });
 *
 * // Health check
 * const health = await client.health();
 * console.log(health.status); // "healthy"
 * ```
 */

// Client
export { NeuralMemClient, NeuralMemError } from "./client.js";

// Types
export type {
  // Enums
  MemoryType,
  MemoryScope,
  SessionLayer,
  ExportFormat,
  HealthStatus,
  McpTransportType,
  // Core models
  Entity,
  Relation,
  Memory,
  SearchResult,
  SearchQuery,
  MemoryHistoryEntry,
  // API responses
  RecallResponse,
  RememberResponse,
  ReflectResult,
  HealthCheck,
  HealthReport,
  MemoryListResponse,
  GraphStats,
  ApiError,
  // Client options
  NeuralMemClientOptions,
  McpStdioConfig,
  MemoryStats,
  ConsolidateResult,
  ForgetBatchResult,
  ProgressCallback,
} from "./types.js";

// Memory operations interfaces & helpers
export type {
  RememberOptions,
  RecallOptions,
  ReflectOptions,
  ForgetOptions,
  MemoryOperations,
} from "./memory.js";

// Search query builder
export { SearchQueryBuilder, search } from "./search.js";
