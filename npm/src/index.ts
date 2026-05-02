/**
 * NeuralMem — Local-first, MCP-native agent memory for Node.js
 *
 * Usage:
 *   import { NeuralMem } from "neuralmem";
 *
 *   const mem = new NeuralMem();
 *   await mem.connect();
 *
 *   await mem.remember("User prefers TypeScript");
 *   const results = await mem.recall("What language does the user prefer?");
 *   console.log(results);
 *
 *   await mem.disconnect();
 */
import { NeuralMemClient } from "./client.js";
import type {
  ConsolidateOptions,
  ConsolidateResult,
  ExportOptions,
  ForgetBatchOptions,
  Memory,
  NeuralMemConfig,
  RecallOptions,
  RememberBatchOptions,
  RememberOptions,
  SearchResult,
} from "./types.js";

// Re-export types
export type {
  ConsolidateOptions,
  ConsolidateResult,
  ExportOptions,
  ForgetBatchOptions,
  Memory,
  NeuralMemConfig,
  RecallOptions,
  RememberBatchOptions,
  RememberOptions,
  SearchResult,
};

/**
 * NeuralMem Node.js client.
 *
 * Wraps the NeuralMem MCP server with a clean async API.
 * By default, spawns `neuralmem mcp` as a child process (stdio transport).
 */
export class NeuralMem {
  private client: NeuralMemClient;
  private autoConnect: boolean;

  constructor(config: NeuralMemConfig = {}) {
    this.client = new NeuralMemClient(config);
    this.autoConnect = true; // auto-connect on first call
  }

  /** Connect to the NeuralMem MCP server */
  async connect(): Promise<void> {
    await this.client.connect();
    this.autoConnect = false;
  }

  /** Disconnect from the server */
  async disconnect(): Promise<void> {
    await this.client.disconnect();
  }

  /** Whether the client is connected */
  get isConnected(): boolean {
    return this.client.isConnected;
  }

  // ─── Core API ───────────────────────────────────────────────

  /**
   * Store a new memory.
   *
   * @param content - The text content to remember
   * @param options - Optional scoping and metadata
   * @returns Confirmation message with memory ID
   *
   * @example
   *   await mem.remember("User prefers dark mode");
   *   await mem.remember("Alice is the team lead", { tags: ["team"], userId: "u1" });
   */
  async remember(
    content: string,
    options: RememberOptions = {}
  ): Promise<string> {
    await this.ensureConnected();
    const args: Record<string, unknown> = { content };
    if (options.userId) args.user_id = options.userId;
    if (options.tags) args.tags = options.tags.join(",");
    if (options.memoryType) args.memory_type = options.memoryType;
    return this.client.callTool<string>("remember", args);
  }

  /**
   * Search memories using 4-strategy parallel retrieval + RRF fusion.
   *
   * @param query - Natural language search query
   * @param options - Search options
   * @returns Array of search results with scores
   *
   * @example
   *   const results = await mem.recall("What are the user's preferences?");
   *   for (const r of results) {
   *     console.log(`[${r.score}] ${r.memory.content}`);
   *   }
   */
  async recall(
    query: string,
    options: RecallOptions = {}
  ): Promise<SearchResult[]> {
    await this.ensureConnected();
    const args: Record<string, unknown> = { query };
    if (options.userId) args.user_id = options.userId;
    if (options.limit !== undefined) args.limit = options.limit;
    if (options.explain !== undefined) args.explain = options.explain;

    // recall returns formatted text; parse into structured results
    const text = await this.client.callTool<string>("recall", args);
    return this.parseRecallResults(text, options.explain);
  }

  /**
   * Recall with explanations for why each memory was retrieved.
   *
   * @param query - Natural language search query
   * @param options - Search options (explain is always true)
   * @returns Array of search results with explanations
   */
  async recallWithExplanation(
    query: string,
    options: Omit<RecallOptions, "explain"> = {}
  ): Promise<SearchResult[]> {
    await this.ensureConnected();
    const args: Record<string, unknown> = { query };
    if (options.userId) args.user_id = options.userId;
    if (options.limit !== undefined) args.limit = options.limit;
    const text = await this.client.callTool<string>(
      "recall_with_explanation",
      args
    );
    return this.parseRecallResults(text, true);
  }

  /**
   * Update or reinforce a memory.
   *
   * @param memoryId - The ID of the memory to update
   * @param newContent - New content (optional)
   * @param importance - New importance score (optional)
   */
  async reflect(
    memoryId: string,
    newContent?: string,
    importance?: number
  ): Promise<string> {
    await this.ensureConnected();
    const args: Record<string, unknown> = { memory_id: memoryId };
    if (newContent !== undefined) args.new_content = newContent;
    if (importance !== undefined) args.importance = importance;
    return this.client.callTool<string>("reflect", args);
  }

  /**
   * Delete a memory by ID.
   *
   * @param memoryId - The ID of the memory to delete
   */
  async forget(memoryId: string): Promise<string> {
    await this.ensureConnected();
    return this.client.callTool<string>("forget", { memory_id: memoryId });
  }

  /**
   * Merge similar memories to reduce redundancy.
   *
   * @param options - Consolidation options
   * @returns Consolidation result (text or structured)
   */
  async consolidate(
    options: ConsolidateOptions = {}
  ): Promise<string | ConsolidateResult> {
    await this.ensureConnected();
    const args: Record<string, unknown> = {};
    if (options.similarityThreshold !== undefined) {
      args.similarity_threshold = options.similarityThreshold;
    }
    const result = await this.client.callTool<string | ConsolidateResult>(
      "consolidate",
      args
    );
    return result;
  }

  /**
   * Resolve a conflict by reactivating or deleting a superseded memory.
   *
   * @param memoryId - The memory to resolve
   * @param action - "reactivate" or "delete"
   */
  async resolveConflict(
    memoryId: string,
    action: "reactivate" | "delete" = "reactivate"
  ): Promise<string> {
    await this.ensureConnected();
    return this.client.callTool<string>("resolve_conflict", {
      memory_id: memoryId,
      action,
    });
  }

  /**
   * Store multiple memories in batch.
   *
   * @param contents - Array of text contents to remember
   * @param options - Shared options for all memories
   * @returns Confirmation message with memory IDs
   */
  async rememberBatch(
    contents: string[],
    options: RememberBatchOptions = {}
  ): Promise<string> {
    await this.ensureConnected();
    const args: Record<string, unknown> = { contents };
    if (options.userId) args.user_id = options.userId;
    if (options.tags) args.tags = options.tags.join(",");
    if (options.memoryType) args.memory_type = options.memoryType;
    return this.client.callTool<string>("remember_batch", args);
  }

  /**
   * Export memories in various formats.
   *
   * @param options - Export options
   * @returns Exported data as string
   */
  async exportMemories(options: ExportOptions = {}): Promise<string> {
    await this.ensureConnected();
    const args: Record<string, unknown> = {};
    if (options.userId) args.user_id = options.userId;
    if (options.format) args.format = options.format;
    if (options.includeEmbeddings !== undefined) {
      args.include_embeddings = options.includeEmbeddings;
    }
    return this.client.callTool<string>("export_memories", args);
  }

  /**
   * Batch delete memories by IDs or tags.
   *
   * @param options - Batch forget options
   * @returns Result (text or structured)
   */
  async forgetBatch(options: ForgetBatchOptions = {}): Promise<string> {
    await this.ensureConnected();
    const args: Record<string, unknown> = {};
    if (options.ids) args.ids = options.ids.join(",");
    if (options.tags) args.tags = options.tags.join(",");
    if (options.userId) args.user_id = options.userId;
    if (options.dryRun !== undefined) args.dry_run = options.dryRun;
    return this.client.callTool<string>("forget_batch", args);
  }

  /**
   * List available MCP tools on the server.
   */
  async listTools(): Promise<Array<{ name: string; description?: string }>> {
    await this.ensureConnected();
    return this.client.listTools();
  }

  // ─── Internal helpers ───────────────────────────────────────

  private async ensureConnected(): Promise<void> {
    if (!this.client.isConnected && this.autoConnect) {
      await this.client.connect();
    }
  }

  /**
   * Parse the formatted recall text from MCP into structured results.
   *
   * The MCP server returns text like:
   *   [1] (score=0.85) memory content here
   *   [2] (score=0.72) another memory
   */
  private parseRecallResults(
    text: string,
    _explain?: boolean
  ): SearchResult[] {
    if (typeof text !== "string") return [];

    const results: SearchResult[] = [];
    const lines = text.split("\n").filter((l) => l.trim());

    for (const line of lines) {
      // Pattern: [N] (score=X.XX) content
      const match = line.match(
        /^\[(\d+)]\s*\(score=([0-9.]+)\)\s*(.+)$/
      );
      if (match) {
        results.push({
          memory: {
            id: `result-${match[1]}`,
            content: match[3].trim(),
            memoryType: "fact",
            importance: 0.5,
            tags: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            entityIds: [],
          },
          score: parseFloat(match[2]),
        });
      }
    }

    return results;
  }
}
