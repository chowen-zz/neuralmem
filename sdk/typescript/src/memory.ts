/**
 * NeuralMem V1.6 TypeScript SDK — Memory operations.
 *
 * Provides fluent helpers for remember / recall / reflect / forget
 * that match the Python NeuralMem API surface.
 */
import type {
  Memory,
  MemoryType,
  SearchResult,
  ReflectResult,
  ProgressCallback,
} from "./types.js";

/**
 * Options for the `remember` operation.
 */
export interface RememberOptions {
  userId?: string;
  agentId?: string;
  sessionId?: string;
  memoryType?: MemoryType;
  tags?: string[];
  importance?: number;
  expiresAt?: string;
  expiresInMs?: number;
  infer?: boolean;
  metadata?: Record<string, unknown>;
}

/**
 * Options for the `recall` operation.
 */
export interface RecallOptions {
  userId?: string;
  agentId?: string;
  memoryTypes?: MemoryType[];
  tags?: string[];
  timeRange?: [string, string];
  limit?: number;
  minScore?: number;
}

/**
 * Options for the `reflect` operation.
 */
export interface ReflectOptions {
  userId?: string;
  depth?: number;
}

/**
 * Options for the `forget` operation.
 */
export interface ForgetOptions {
  memoryId?: string;
  userId?: string;
  before?: string;
  tags?: string[];
}

/**
 * High-level memory operations interface.
 *
 * Implementations are provided by `NeuralMemClient` (HTTP) and
 * `McpNeuralMemClient` (MCP stdio). Both expose the same fluent
 * surface so callers can swap transports without changing code.
 */
export interface MemoryOperations {
  /**
   * Store a new memory.
   */
  remember(content: string, options?: RememberOptions): Promise<Memory[]>;

  /**
   * Search for relevant memories.
   */
  recall(query: string, options?: RecallOptions): Promise<SearchResult[]>;

  /**
   * Reflect on a topic using memory retrieval and graph traversal.
   */
  reflect(topic: string, options?: ReflectOptions): Promise<ReflectResult>;

  /**
   * Delete memories matching the given criteria.
   *
   * @returns Number of memories deleted.
   */
  forget(options?: ForgetOptions): Promise<number>;

  /**
   * Batch remember multiple items.
   */
  rememberBatch(
    contents: string[],
    options?: RememberOptions & { progressCallback?: ProgressCallback },
  ): Promise<Memory[]>;
}
