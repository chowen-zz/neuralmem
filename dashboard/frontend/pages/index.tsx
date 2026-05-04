import React, { useState, useEffect } from "react";
import SearchBar from "../components/SearchBar";
import StatsPanel from "../components/StatsPanel";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface MemoryItem {
  id: string;
  content: string;
  memory_type: string;
  importance: number;
  is_active: boolean;
  created_at: string;
  tags: string[];
}

interface SearchResultItem {
  memory: MemoryItem;
  score: number;
  retrieval_method: string;
  explanation: string;
}

interface StatsData {
  memory_count: number;
  node_count: number;
  edge_count: number;
  p99_latency_ms: number;
  p95_latency_ms: number;
  p50_latency_ms: number;
  mean_latency_ms: number;
  recall_calls: number;
  remember_calls: number;
  cache_hit_rate: number;
  active_memories: number;
  superseded_memories: number;
}

export default function IndexPage(): JSX.Element {
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [stats, setStats] = useState<StatsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  // Fetch stats on mount
  useEffect(() => {
    fetchStats();
    fetchMemories();
  }, []);

  async function fetchStats() {
    try {
      const res = await fetch(`${API_BASE}/api/stats`);
      if (!res.ok) throw new Error(`Stats error: ${res.status}`);
      const data: StatsData = await res.json();
      setStats(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load stats");
    }
  }

  async function fetchMemories() {
    try {
      const res = await fetch(`${API_BASE}/api/memories?limit=20`);
      if (!res.ok) throw new Error(`Memories error: ${res.status}`);
      const data = await res.json();
      setMemories(data.memories || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load memories");
    }
  }

  async function handleSearch(q: string) {
    setQuery(q);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, limit: 10 }),
      });
      if (!res.ok) throw new Error(`Search error: ${res.status}`);
      const data = await res.json();
      setSearchResults(data.results || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  function clearSearch() {
    setQuery("");
    setSearchResults([]);
  }

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif", maxWidth: 1200, margin: "0 auto" }}>
      <h1>NeuralMem V1.6 Dashboard</h1>

      <StatsPanel stats={stats} />

      <div style={{ marginTop: "2rem" }}>
        <SearchBar onSearch={handleSearch} loading={loading} />
        {query && (
          <div style={{ marginTop: "0.5rem" }}>
            <span>Results for: <strong>{query}</strong></span>
            <button onClick={clearSearch} style={{ marginLeft: "1rem" }}>
              Clear
            </button>
          </div>
        )}
      </div>

      {error && (
        <div style={{ color: "red", marginTop: "1rem" }}>
          Error: {error}
        </div>
      )}

      <div style={{ marginTop: "2rem" }}>
        <h2>{searchResults.length > 0 ? "Search Results" : "Recent Memories"}</h2>
        <ul style={{ listStyle: "none", padding: 0 }}>
          {(searchResults.length > 0
            ? searchResults.map((r) => ({ ...r.memory, score: r.score }))
            : memories
          ).map((m: any) => (
            <li
              key={m.id}
              style={{
                border: "1px solid #ddd",
                borderRadius: 6,
                padding: "1rem",
                marginBottom: "0.75rem",
                background: m.is_active === false ? "#f5f5f5" : "#fff",
              }}
            >
              <div style={{ fontWeight: 600 }}>{m.content}</div>
              <div style={{ fontSize: "0.85rem", color: "#666", marginTop: "0.25rem" }}>
                Type: {m.memory_type} | Importance: {m.importance} | Active: {String(m.is_active)}
                {m.score !== undefined && ` | Score: ${m.score.toFixed(3)}`}
              </div>
              <div style={{ fontSize: "0.8rem", color: "#999", marginTop: "0.25rem" }}>
                {m.tags?.join(", ")} | {m.created_at}
              </div>
            </li>
          ))}
        </ul>
        {(searchResults.length > 0 ? searchResults.length === 0 : memories.length === 0) && (
          <p>No items to display.</p>
        )}
      </div>
    </div>
  );
}
