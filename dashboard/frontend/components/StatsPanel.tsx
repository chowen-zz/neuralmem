import React from "react";

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

interface StatsPanelProps {
  stats: StatsData | null;
}

export default function StatsPanel({ stats }: StatsPanelProps): JSX.Element {
  if (!stats) {
    return (
      <div style={{ padding: "1rem", color: "#666" }}>
        Loading stats...
      </div>
    );
  }

  const cards = [
    { label: "Memories", value: stats.memory_count },
    { label: "Active", value: stats.active_memories },
    { label: "Superseded", value: stats.superseded_memories },
    { label: "Graph Nodes", value: stats.node_count },
    { label: "Graph Edges", value: stats.edge_count },
    { label: "Recall Calls", value: stats.recall_calls },
    { label: "Remember Calls", value: stats.remember_calls },
    { label: "Cache Hit Rate", value: `${(stats.cache_hit_rate * 100).toFixed(1)}%` },
    { label: "P99 Latency", value: `${stats.p99_latency_ms.toFixed(1)} ms` },
    { label: "P95 Latency", value: `${stats.p95_latency_ms.toFixed(1)} ms` },
    { label: "P50 Latency", value: `${stats.p50_latency_ms.toFixed(1)} ms` },
    { label: "Mean Latency", value: `${stats.mean_latency_ms.toFixed(1)} ms` },
  ];

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
        gap: "1rem",
        marginTop: "1rem",
      }}
    >
      {cards.map((card) => (
        <div
          key={card.label}
          style={{
            border: "1px solid #e0e0e0",
            borderRadius: 6,
            padding: "1rem",
            background: "#fafafa",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: "1.5rem", fontWeight: 700 }}>{card.value}</div>
          <div style={{ fontSize: "0.8rem", color: "#666", marginTop: "0.25rem" }}>
            {card.label}
          </div>
        </div>
      ))}
    </div>
  );
}
