/**
 * TypeScript type definitions for NeuralMem V0.7.
 *
 * These interfaces mirror the Python Pydantic models in
 * `neuralmem.core.types` and API response shapes.
 */

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

/** A stored memory record. */
export interface Memory {
  id: string;
  content: string;
  memory_type: MemoryType;
  scope: MemoryScope;
  user_id: string | null;
  agent_id: string | null;
  session_id: string | null;
  tags: string[];
  source: string | null;
  importance: number;
  entity_ids: string[];
  is_active: boolean;
  superseded_by: string | null;
  supersedes: string[];
  created_at: string;
  updated_at: string;
  last_accessed: string;
  access_count: number;
  expires_at: string | null;
}

/** A single search result returned by the recall endpoint. */
export interface SearchResult {
  memory: Memory;
  score: number;
  retrieval_method: string;
  explanation: string | null;
}

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

/** Health status levels. */
export type HealthStatus = "healthy" | "degraded" | "unhealthy";

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

/** Client configuration options. */
export interface NeuralMemClientOptions {
  /** Base URL of the NeuralMem server (e.g. "http://localhost:8000"). */
  baseUrl: string;
  /** Optional API key for authentication. */
  apiKey?: string;
}

/** API error response shape. */
export interface ApiError {
  error: string;
  detail?: string;
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
