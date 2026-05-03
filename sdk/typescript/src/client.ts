/**
 * NeuralMem TypeScript SDK Client.
 *
 * Provides a zero-dependency HTTP client for interacting with a
 * NeuralMem server instance. Uses the native `fetch` API.
 *
 * @example
 * ```ts
 * import { NeuralMemClient } from "@neuralmem/sdk";
 *
 * const client = new NeuralMemClient({ baseUrl: "http://localhost:8000" });
 * await client.remember("User prefers TypeScript", "user-1");
 * const results = await client.recall("What language does the user prefer?", "user-1");
 * ```
 */
import type {
  HealthReport,
  Memory,
  MemoryListResponse,
  NeuralMemClientOptions,
  RecallResponse,
  ReflectResult,
  SearchResult,
} from "./types.js";

export class NeuralMemClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;

  /**
   * Create a new NeuralMem client.
   *
   * @param options - Configuration with `baseUrl` and optional `apiKey`.
   */
  constructor(options: NeuralMemClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
  }

  // ------------------------------------------------------------------
  // Core API
  // ------------------------------------------------------------------

  /**
   * Store a new memory.
   *
   * @param content - The text content to remember.
   * @param userId - User identifier for scoping.
   * @param memoryTypes - Optional memory type override.
   * @returns The stored memory object(s).
   */
  async remember(
    content: string,
    userId?: string,
    memoryTypes?: string,
  ): Promise<Memory[]> {
    const body: Record<string, unknown> = { content };
    if (userId) body.user_id = userId;
    if (memoryTypes) body.memory_type = memoryTypes;

    const data = await this.request<Record<string, unknown>>(
      "/api/remember",
      {
        method: "POST",
        body: JSON.stringify(body),
      },
    );

    // Server may return { memories: [...] } or a single memory
    if (Array.isArray(data.memories)) {
      return data.memories as Memory[];
    }
    return [data as unknown as Memory];
  }

  /**
   * Search for relevant memories.
   *
   * @param query - The search query text.
   * @param userId - Optional user ID for scoped search.
   * @param limit - Maximum number of results (default 10).
   * @returns Array of search results sorted by relevance.
   */
  async recall(
    query: string,
    userId?: string,
    limit?: number,
  ): Promise<SearchResult[]> {
    const body: Record<string, unknown> = { query };
    if (userId) body.user_id = userId;
    if (limit !== undefined) body.limit = limit;

    const data = await this.request<RecallResponse>("/api/recall", {
      method: "POST",
      body: JSON.stringify(body),
    });

    return data.results ?? [];
  }

  /**
   * Reflect on a topic — retrieve and synthesize related memories.
   *
   * @param userId - Optional user ID for scoped reflection.
   * @returns Reflection results.
   */
  async reflect(userId?: string): Promise<ReflectResult> {
    const body: Record<string, unknown> = {};
    if (userId) body.user_id = userId;

    return this.request<ReflectResult>("/api/reflect", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  /**
   * Delete a memory by ID, or all memories for a user.
   *
   * @param memoryId - Optional specific memory ID to delete.
   */
  async forget(memoryId?: string): Promise<void> {
    const body: Record<string, unknown> = {};
    if (memoryId) body.memory_id = memoryId;

    await this.request("/api/forget", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  /**
   * Check the health status of the NeuralMem server.
   *
   * @returns Health report with status and individual checks.
   */
  async health(): Promise<HealthReport> {
    return this.request<HealthReport>("/api/health");
  }

  // ------------------------------------------------------------------
  // Dashboard API
  // ------------------------------------------------------------------

  /**
   * List memories with pagination (dashboard API).
   *
   * @param limit - Max items per page.
   * @param offset - Pagination offset.
   * @returns Paginated memory list.
   */
  async listMemories(
    limit: number = 50,
    offset: number = 0,
  ): Promise<MemoryListResponse> {
    return this.request<MemoryListResponse>(
      `/api/memories?limit=${limit}&offset=${offset}`,
    );
  }

  // ------------------------------------------------------------------
  // Internal
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

    const response = await fetch(url, { ...init, headers });

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
