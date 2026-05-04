/**
 * NeuralMem V1.6 TypeScript SDK Client.
 *
 * Provides a zero-dependency HTTP client for interacting with a
 * NeuralMem server instance. Uses the native `fetch` API so it
 * works in both Node.js (18+) and modern browsers.
 *
 * Also supports MCP stdio transport for local tool-server integration.
 *
 * @example
 * ```ts
 * import { NeuralMemClient } from "@neuralmem/sdk";
 *
 * const client = new NeuralMemClient({ baseUrl: "http://localhost:8000" });
 * await client.remember("User prefers TypeScript", { userId: "user-1" });
 * const results = await client.recall("What language does the user prefer?", { userId: "user-1" });
 * ```
 */
import type {
  ConsolidateResult,
  ForgetBatchResult,
  HealthReport,
  Memory,
  MemoryHistoryEntry,
  MemoryListResponse,
  MemoryStats,
  NeuralMemClientOptions,
  ProgressCallback,
  RecallResponse,
  ReflectResult,
  RememberResponse,
  SearchResult,
} from "./types.js";
import type {
  ForgetOptions,
  RecallOptions,
  ReflectOptions,
  RememberOptions,
} from "./memory.js";

export class NeuralMemClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly timeout: number;

  /**
   * Create a new NeuralMem client.
   *
   * @param options - Configuration with `baseUrl` and optional `apiKey`.
   */
  constructor(options: NeuralMemClientOptions) {
    this.baseUrl = (options.baseUrl ?? "http://localhost:8000").replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? 30000;
  }

  // ------------------------------------------------------------------
  // Core API — remember / recall / reflect / forget
  // ------------------------------------------------------------------

  /**
   * Store a new memory.
   *
   * @param content - The text content to remember.
   * @param options - Optional scoping, typing, tags, importance, expiration.
   * @returns The stored memory objects.
   */
  async remember(
    content: string,
    options: RememberOptions = {},
  ): Promise<Memory[]> {
    const body: Record<string, unknown> = { content };
    if (options.userId) body.user_id = options.userId;
    if (options.agentId) body.agent_id = options.agentId;
    if (options.sessionId) body.session_id = options.sessionId;
    if (options.memoryType) body.memory_type = options.memoryType;
    if (options.tags) body.tags = options.tags;
    if (options.importance !== undefined) body.importance = options.importance;
    if (options.expiresAt) body.expires_at = options.expiresAt;
    if (options.expiresInMs) body.expires_in_ms = options.expiresInMs;
    if (options.infer !== undefined) body.infer = options.infer;
    if (options.metadata) body.metadata = options.metadata;

    const data = await this.request<RememberResponse>("/api/remember", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return data.memories ?? [];
  }

  /**
   * Search for relevant memories.
   *
   * @param query - The search query text.
   * @param options - Optional filters, limits, and score thresholds.
   * @returns Array of search results sorted by relevance.
   */
  async recall(
    query: string,
    options: RecallOptions = {},
  ): Promise<SearchResult[]> {
    const body: Record<string, unknown> = { query };
    if (options.userId) body.user_id = options.userId;
    if (options.agentId) body.agent_id = options.agentId;
    if (options.memoryTypes) body.memory_types = options.memoryTypes;
    if (options.tags) body.tags = options.tags;
    if (options.timeRange) body.time_range = options.timeRange;
    if (options.limit !== undefined) body.limit = options.limit;
    if (options.minScore !== undefined) body.min_score = options.minScore;

    const data = await this.request<RecallResponse>("/api/recall", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return data.results ?? [];
  }

  /**
   * Reflect on a topic — retrieve and synthesize related memories.
   *
   * @param topic - The topic to reflect on.
   * @param options - Optional user scoping and graph traversal depth.
   * @returns Reflection results.
   */
  async reflect(
    topic: string,
    options: ReflectOptions = {},
  ): Promise<ReflectResult> {
    const body: Record<string, unknown> = { topic };
    if (options.userId) body.user_id = options.userId;
    if (options.depth !== undefined) body.depth = options.depth;

    return this.request<ReflectResult>("/api/reflect", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  /**
   * Delete memories matching the given criteria.
   *
   * @param options - memoryId, userId, before timestamp, or tags.
   * @returns Number of memories deleted.
   */
  async forget(options: ForgetOptions = {}): Promise<number> {
    const body: Record<string, unknown> = {};
    if (options.memoryId) body.memory_id = options.memoryId;
    if (options.userId) body.user_id = options.userId;
    if (options.before) body.before = options.before;
    if (options.tags) body.tags = options.tags;

    const data = await this.request<{ deleted: number }>("/api/forget", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return data.deleted ?? 0;
  }

  // ------------------------------------------------------------------
  // Batch operations
  // ------------------------------------------------------------------

  /**
   * Batch remember multiple items.
   *
   * @param contents - List of content strings to remember.
   * @param options - Shared options plus optional progress callback.
   * @returns All stored memories across all items.
   */
  async rememberBatch(
    contents: string[],
    options: RememberOptions & { progressCallback?: ProgressCallback } = {},
  ): Promise<Memory[]> {
    const all: Memory[] = [];
    const total = contents.length;
    const { progressCallback, ...rest } = options;

    for (let idx = 0; idx < total; idx++) {
      const content = contents[idx];
      if (progressCallback) {
        const preview = content.length > 60 ? content.slice(0, 60) + "..." : content;
        progressCallback(idx, total, preview);
      }
      const memories = await this.remember(content, rest);
      all.push(...memories);
    }

    if (progressCallback) {
      progressCallback(total, total, "done");
    }
    return all;
  }

  /**
   * Batch delete with optional dry-run preview.
   *
   * @param memoryIds - Specific memory IDs to delete.
   * @param options - userId, tags, dryRun flag.
   * @returns Preview or actual deletion result.
   */
  async forgetBatch(
    memoryIds: string[] | null = null,
    options: {
      userId?: string;
      tags?: string[];
      dryRun?: boolean;
    } = {},
  ): Promise<ForgetBatchResult> {
    const body: Record<string, unknown> = {};
    if (memoryIds) body.memory_ids = memoryIds;
    if (options.userId) body.user_id = options.userId;
    if (options.tags) body.tags = options.tags;
    if (options.dryRun) body.dry_run = options.dryRun;

    return this.request<ForgetBatchResult>("/api/forget/batch", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  // ------------------------------------------------------------------
  // Lifecycle & maintenance
  // ------------------------------------------------------------------

  /**
   * Run memory consolidation: decay old memories, merge similar ones.
   *
   * @param userId - Limit consolidation to a specific user.
   * @returns Counts of decayed, forgotten, and merged memories.
   */
  async consolidate(userId?: string): Promise<ConsolidateResult> {
    const body: Record<string, unknown> = {};
    if (userId) body.user_id = userId;

    return this.request<ConsolidateResult>("/api/consolidate", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  /**
   * Remove all expired memories.
   * @returns Number of expired memories deleted.
   */
  async cleanupExpired(): Promise<number> {
    const data = await this.request<{ deleted: number }>("/api/cleanup-expired", {
      method: "POST",
    });
    return data.deleted ?? 0;
  }

  // ------------------------------------------------------------------
  // Single-record & history
  // ------------------------------------------------------------------

  /**
   * Retrieve a single memory by ID.
   * @returns Memory object if found, null otherwise.
   */
  async get(memoryId: string): Promise<Memory | null> {
    try {
      return await this.request<Memory>(`/api/memories/${encodeURIComponent(memoryId)}`);
    } catch (err) {
      if (err instanceof NeuralMemError && err.statusCode === 404) {
        return null;
      }
      throw err;
    }
  }

  /**
   * Update a memory's content. Records version history automatically.
   * @returns Updated Memory object, or null if not found.
   */
  async update(
    memoryId: string,
    content: string,
    metadata?: Record<string, unknown>,
  ): Promise<Memory | null> {
    const body: Record<string, unknown> = { content };
    if (metadata) body.metadata = metadata;

    try {
      return await this.request<Memory>(`/api/memories/${encodeURIComponent(memoryId)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      });
    } catch (err) {
      if (err instanceof NeuralMemError && err.statusCode === 404) {
        return null;
      }
      throw err;
    }
  }

  /**
   * Retrieve the version history for a memory.
   * @returns List of history entries, ordered chronologically.
   */
  async history(memoryId: string): Promise<MemoryHistoryEntry[]> {
    return this.request<MemoryHistoryEntry[]>(
      `/api/memories/${encodeURIComponent(memoryId)}/history`,
    );
  }

  // ------------------------------------------------------------------
  // Stats, export, import
  // ------------------------------------------------------------------

  /**
   * Return memory store statistics.
   */
  async getStats(): Promise<MemoryStats> {
    return this.request<MemoryStats>("/api/stats");
  }

  /**
   * Export memories as JSON, markdown, or CSV.
   * @param userId - Filter by user. Omit for all users.
   * @param format - Output format.
   * @param includeEmbeddings - Whether to include embedding vectors.
   * @returns Exported data string.
   */
  async exportMemories(
    userId?: string,
    format = "json",
    includeEmbeddings = false,
  ): Promise<string> {
    const params = new URLSearchParams();
    if (userId) params.append("user_id", userId);
    params.append("format", format);
    params.append("include_embeddings", String(includeEmbeddings));

    const data = await this.request<{ data: string }>(`/api/export?${params.toString()}`);
    return data.data ?? "";
  }

  /**
   * Import memories from exported data.
   * @param data - The exported data string.
   * @param format - Data format.
   * @param userId - Override user_id for imported memories.
   * @param skipDuplicates - Skip memories with similar content already stored.
   * @returns Number of memories imported.
   */
  async importMemories(
    data: string,
    format = "json",
    userId?: string,
    skipDuplicates = true,
  ): Promise<number> {
    const body: Record<string, unknown> = { data, format };
    if (userId) body.user_id = userId;
    body.skip_duplicates = skipDuplicates;

    const res = await this.request<{ imported: number }>("/api/import", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return res.imported ?? 0;
  }

  // ------------------------------------------------------------------
  // Health & dashboard
  // ------------------------------------------------------------------

  /**
   * Check the health status of the NeuralMem server.
   * @returns Health report with status and individual checks.
   */
  async health(): Promise<HealthReport> {
    return this.request<HealthReport>("/api/health");
  }

  /**
   * List memories with pagination (dashboard API).
   * @param limit - Max items per page.
   * @param offset - Pagination offset.
   * @returns Paginated memory list.
   */
  async listMemories(
    limit = 50,
    offset = 0,
  ): Promise<MemoryListResponse> {
    return this.request<MemoryListResponse>(
      `/api/memories?limit=${limit}&offset=${offset}`,
    );
  }

  // ------------------------------------------------------------------
  // Conflict resolution
  // ------------------------------------------------------------------

  /**
   * Resolve a memory conflict manually.
   * @param memoryId - Target memory ID.
   * @param action - "reactivate" or "delete".
   * @returns Whether the operation succeeded.
   */
  async resolveConflict(
    memoryId: string,
    action: "reactivate" | "delete" = "reactivate",
  ): Promise<boolean> {
    const body = { memory_id: memoryId, action };
    const res = await this.request<{ success: boolean }>("/api/resolve-conflict", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return res.success ?? false;
  }

  // ------------------------------------------------------------------
  // Internal request helper
  // ------------------------------------------------------------------

  private async request<T>(
    path: string,
    init?: RequestInit,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
      ...(init?.headers as Record<string, string> ?? {}),
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...init,
        headers,
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        let detail = text;
        try {
          const parsed = JSON.parse(text);
          detail = parsed.error || parsed.detail || text;
        } catch {
          // use raw text
        }
        throw new NeuralMemError(
          `Request failed: ${response.status} ${response.statusText} — ${detail}`,
          response.status,
        );
      }

      return (await response.json()) as T;
    } catch (err) {
      if (err instanceof NeuralMemError) {
        throw err;
      }
      if (err instanceof Error && err.name === "AbortError") {
        throw new NeuralMemError(`Request timed out after ${this.timeout}ms`, 408);
      }
      throw new NeuralMemError(`Network error: ${(err as Error).message}`, 0);
    } finally {
      clearTimeout(timeoutId);
    }
  }
}

/** Custom error class for NeuralMem SDK errors. */
export class NeuralMemError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
  ) {
    super(message);
    this.name = "NeuralMemError";
  }
}
