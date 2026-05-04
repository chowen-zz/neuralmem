/**
 * NeuralMem V1.6 TypeScript SDK — Search query builder.
 *
 * Provides a fluent, chainable API for constructing search queries
 * that mirror the Python `SearchQuery` Pydantic model.
 */
import type { MemoryType, SearchQuery } from "./types.js";

/**
 * Fluent builder for NeuralMem search queries.
 *
 * @example
 * ```ts
 * const query = new SearchQueryBuilder("What does the user like?")
 *   .forUser("user-123")
 *   .ofTypes("preference", "fact")
 *   .withTags("food", "hobby")
 *   .limit(5)
 *   .minScore(0.5)
 *   .build();
 * ```
 */
export class SearchQueryBuilder {
  private _query: string;
  private _userId: string | null = null;
  private _agentId: string | null = null;
  private _memoryTypes: MemoryType[] | null = null;
  private _tags: string[] | null = null;
  private _timeRange: [string, string] | null = null;
  private _limit = 10;
  private _minScore = 0.3;

  constructor(query: string) {
    if (!query || !query.trim()) {
      throw new Error("Search query cannot be empty");
    }
    this._query = query;
  }

  /** Scope the search to a specific user. */
  forUser(userId: string): this {
    this._userId = userId;
    return this;
  }

  /** Scope the search to a specific agent. */
  forAgent(agentId: string): this {
    this._agentId = agentId;
    return this;
  }

  /** Filter by one or more memory types. */
  ofTypes(...types: MemoryType[]): this {
    this._memoryTypes = types;
    return this;
  }

  /** Require all memories to have every listed tag. */
  withTags(...tags: string[]): this {
    this._tags = tags;
    return this;
  }

  /** Restrict results to a time window (ISO-8601 strings). */
  between(start: string, end: string): this {
    this._timeRange = [start, end];
    return this;
  }

  /** Set the maximum number of results (1–100, default 10). */
  limit(n: number): this {
    if (n < 1 || n > 100) {
      throw new Error("limit must be between 1 and 100");
    }
    this._limit = n;
    return this;
  }

  /** Set the minimum relevance score (0.0–1.0, default 0.3). */
  minScore(score: number): this {
    if (score < 0 || score > 1) {
      throw new Error("minScore must be between 0.0 and 1.0");
    }
    this._minScore = score;
    return this;
  }

  /** Build the final `SearchQuery` object. */
  build(): SearchQuery {
    return {
      query: this._query,
      user_id: this._userId,
      agent_id: this._agentId,
      memory_types: this._memoryTypes,
      tags: this._tags,
      time_range: this._timeRange,
      limit: this._limit,
      min_score: this._minScore,
    };
  }

  /** Convenience: build and immediately JSON-serialise. */
  toJSON(): string {
    return JSON.stringify(this.build());
  }
}

/**
 * Create a new `SearchQueryBuilder` for the given query string.
 *
 * Shorthand for `new SearchQueryBuilder(query)`.
 */
export function search(query: string): SearchQueryBuilder {
  return new SearchQueryBuilder(query);
}
