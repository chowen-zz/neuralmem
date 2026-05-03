/**
 * @neuralmem/sdk — TypeScript SDK for NeuralMem V0.7.
 *
 * @example
 * ```ts
 * import { NeuralMemClient } from "@neuralmem/sdk";
 *
 * const client = new NeuralMemClient({ baseUrl: "http://localhost:8000" });
 *
 * // Store a memory
 * const memories = await client.remember("User prefers dark mode", "user-1");
 *
 * // Search memories
 * const results = await client.recall("What UI theme does user prefer?", "user-1");
 *
 * // Health check
 * const health = await client.health();
 * console.log(health.status); // "healthy"
 * ```
 */

export { NeuralMemClient, NeuralMemError } from "./client.js";
export type {
  Memory,
  MemoryType,
  MemoryScope,
  SearchResult,
  RecallResponse,
  RememberResponse,
  ReflectResult,
  HealthStatus,
  HealthCheck,
  HealthReport,
  NeuralMemClientOptions,
  ApiError,
  MemoryListResponse,
  GraphStats,
} from "./types.js";
