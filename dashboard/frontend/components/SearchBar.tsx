import React, { useState } from "react";

interface SearchBarProps {
  onSearch: (query: string) => void;
  loading?: boolean;
}

export default function SearchBar({ onSearch, loading = false }: SearchBarProps): JSX.Element {
  const [query, setQuery] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{ display: "flex", gap: "0.5rem" }}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search memories..."
        disabled={loading}
        style={{
          flex: 1,
          padding: "0.6rem 0.8rem",
          fontSize: "1rem",
          border: "1px solid #ccc",
          borderRadius: 4,
        }}
      />
      <button
        type="submit"
        disabled={loading || !query.trim()}
        style={{
          padding: "0.6rem 1.2rem",
          fontSize: "1rem",
          cursor: loading ? "not-allowed" : "pointer",
          opacity: loading ? 0.6 : 1,
        }}
      >
        {loading ? "Searching..." : "Search"}
      </button>
    </form>
  );
}
