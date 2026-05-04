/**
 * NeuralMem V1.6 TypeScript SDK — Core type definitions.
 *
 * These interfaces mirror the Python Pydantic models in
 * `neuralmem.core.types` and API response shapes.
 */

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/** Memory type classification. */
export type MemoryType =
  | "fact"
  | "preference"
  | "episodic"
  | "semantic"
  | "procedural"
  | "working";

/** Memory scope — who owns the memory. */
export type MemoryScope = "user" | "agent" | "session" | "shared";

/** Session memory layer architecture. */
export type SessionLayer = "working" | "session" | "long_term";

/** Memory export format. */
export type ExportFormat = "json" | "markdown" | "csv";

/** Health status levels. */
export type HealthStatus = "healthy" | "degraded" | "unhealthy";

// ---------------------------------------------------------------------------
// Core models
// ---------------------------------------------------------------------------

/** Knowledge graph entity. */
export interface Entity {
  id: string;
  name: string;
  entity_type: string;
  aliases: string[];
  attributes: Record<string, unknown>;
  first_seen: string;
  last_seen: string;
}

/** Knowledge graph relation. */
export interface Relation {
  source_id: string;
  target_id: string;
  relation_type: string;
  weight: number;
  timestamp: string;
  metadata: Record<string, unknown>;
}

/** A stored memory record. */
export interface Memory {
  id: string;
  content: string;
  memory_type: MemoryType;
  scope: MemoryScope;

  // Ownership
  user_id: string | null;
  agent_id: string | null;
  session_id: string | null;

  // Metadata
  tags: string[];
  source: string | null;
  importance: number;

  // Linked entities
  entity_ids: string[];

  // Conflict resolution
  is_active: boolean;
  superseded_by: string | null;
  supersedes: string[];

  // Timestamps
  created_at: string;
  updated_at: string;
  last_accessed: string;
  access_count: number;

  // Expiration
  expires_at: string | null;
}

/** A single search result returned by the recall endpoint. */
export interface SearchResult {
  memory: Memory;
  score: number;
  retrieval_method: string;
  explanation: string | null;
}

/** Search request payload. */
export interface SearchQuery {
  query: string;
  user_id: string | null;
  agent_id: string | null;
  memory_types: MemoryType[] | null;
  tags: string[] | null;
  time_range: [string, string] | null;
  limit: number;
  min_score: number;
}

/** A single version history entry for a memory. */
export interface MemoryHistoryEntry {
  id: number;
  memory_id: string;
  old_content: string | null;
  new_content: string;
  event: string; // 'CREATE', 'UPDATE', 'DELETE'
  changed_at: string;
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// API responses
// ---------------------------------------------------------------------------

/** Response from the recall API endpoint. */
export interface RecallResponse {
  results: SearchResult[];
  count: number;
}

/** Response from the remember API endpoint. */
export interface RememberResponse {
  memories: Memory[];
  count: number;
}

/** Result of a reflect operation. */
export interface ReflectResult {
  topic: string;
  content: string;
  memories_found: number;
  entities_found: number;
}

/** Individual health check result. */
export interface HealthCheck {
  status: string;
  detail: string;
}

/** Full health report from the health endpoint. */
export interface HealthReport {
  status: HealthStatus;
  checks: Record<string, string>;
  details: Record<string, string>;
}

/** Memory list response with pagination. */
export interface MemoryListResponse {
  memories: Memory[];
  total: number;
  limit: number;
  offset: number;
}

/** Graph statistics. */
export interface GraphStats {
  node_count: number;
  edge_count: number;
}

/** API error response shape. */
export interface ApiError {
  error: string;
  detail?: string;
}

// ---------------------------------------------------------------------------
// Client options
// ---------------------------------------------------------------------------

/** Transport type for MCP communication. */
export type McpTransportType = "stdio" | "sse" | "websocket";

/** MCP stdio transport configuration. */
export interface McpStdioConfig {
  command: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
}

/** Client configuration options. */
export interface NeuralMemClientOptions {
  /** Base URL of the NeuralMem server (e.g. "http://localhost:8000"). */
  baseUrl?: string;
  /** Optional API key for authentication. */
  apiKey?: string;
  /** MCP stdio transport config (alternative to HTTP). */
  mcpStdio?: McpStdioConfig;
  /** Request timeout in milliseconds (default 30000). */
  timeout?: number;
}

/** Statistics returned by getStats(). */
export interface MemoryStats {
  total_memories: number;
  active_memories: number;
  graph: GraphStats;
}

/** Consolidation result. */
export interface ConsolidateResult {
  decayed: number;
  forgotten: number;
  merged: number;
}

/** Batch forget preview / result. */
export interface ForgetBatchResult {
  count: number;
  memory_ids: string[];
  dry_run: boolean;
}

/** Progress callback signature for batch operations. */
export type ProgressCallback = (
  current: number,
  total: number,
  preview: string,
) => void;
