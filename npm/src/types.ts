/**
 * NeuralMem TypeScript type definitions
 */

/** Memory types supported by NeuralMem */
export type MemoryType =
  | "fact"
  | "preference"
  | "instruction"
  | "context"
  | "event"
  | "relationship";

/** Options for the remember() call */
export interface RememberOptions {
  /** User ID to scope the memory to */
  userId?: string;
  /** Tags for categorization */
  tags?: string[];
  /** Memory type classification */
  memoryType?: MemoryType;
}

/** A stored memory item */
export interface Memory {
  /** Unique memory ID (ULID) */
  id: string;
  /** Memory content text */
  content: string;
  /** Memory type */
  memoryType: MemoryType;
  /** Importance score (0.0 - 1.0) */
  importance: number;
  /** Associated tags */
  tags: string[];
  /** Creation timestamp (ISO 8601) */
  createdAt: string;
  /** Last update timestamp (ISO 8601) */
  updatedAt: string;
  /** User ID */
  userId?: string;
  /** Entity IDs extracted from content */
  entityIds: string[];
}

/** A search result with relevance score */
export interface SearchResult {
  /** The matched memory */
  memory: Memory;
  /** Relevance score (0.0 - 1.0) */
  score: number;
  /** Explanation of why this memory was retrieved (when using explain mode) */
  explanation?: string;
}

/** Options for the recall() call */
export interface RecallOptions {
  /** User ID to scope search to */
  userId?: string;
  /** Maximum number of results */
  limit?: number;
  /** Include explanations for each result */
  explain?: boolean;
}

/** Options for the consolidate() call */
export interface ConsolidateOptions {
  /** Similarity threshold for merging (0.0 - 1.0, default 0.9) */
  similarityThreshold?: number;
}

/** Consolidation result statistics */
export interface ConsolidateResult {
  /** Number of memories merged */
  merged: number;
  /** Number of memories decayed */
  decayed: number;
  /** Number of memories forgotten (below threshold) */
  forgotten: number;
}

/** Options for the remember_batch() call */
export interface RememberBatchOptions {
  userId?: string;
  tags?: string[];
  memoryType?: MemoryType;
}

/** Options for the export_memories() call */
export interface ExportOptions {
  userId?: string;
  format?: "json" | "markdown" | "csv";
  includeEmbeddings?: boolean;
}

/** Options for the forget_batch() call */
export interface ForgetBatchOptions {
  /** Memory IDs to delete */
  ids?: string[];
  /** Delete by tags */
  tags?: string[];
  /** Scope to user ID */
  userId?: string;
  /** Preview without actually deleting */
  dryRun?: boolean;
}

/** NeuralMem client configuration */
export interface NeuralMemConfig {
  /**
   * Transport mode.
   * - "stdio": spawn `neuralmem mcp` as a child process (default)
   * - "http": connect to an existing HTTP MCP server
   */
  transport?: "stdio" | "http";
  /** HTTP endpoint URL (required when transport="http") */
  httpUrl?: string;
  /** Path to the `neuralmem` CLI binary (default: "neuralmem") */
  command?: string;
  /** Additional args to pass to the neuralmem command */
  args?: string[];
  /** Working directory for the spawned process */
  cwd?: string;
  /** Environment variables for the spawned process */
  env?: Record<string, string>;
  /** Connection timeout in milliseconds (default: 30000) */
  timeout?: number;
}
