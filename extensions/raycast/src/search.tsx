import { useState, useEffect, useCallback } from "react";
import {
  List,
  ActionPanel,
  Action,
  Icon,
  showToast,
  Toast,
  getPreferenceValues,
} from "@raycast/api";

// ─────────────────────────────────────────────────────────────────────────────
// NeuralMem Search Command (Raycast)
// Quick semantic / keyword search across saved memories.
// ─────────────────────────────────────────────────────────────────────────────

interface Preferences {
  apiBaseUrl: string;
  apiKey?: string;
  defaultSpace?: string;
}

interface Memory {
  id: string;
  title: string;
  content: string;
  url?: string;
  space?: string;
  created_at: string;
  score?: number;
}

interface SearchResponse {
  results: Memory[];
  total: number;
  query: string;
}

const NEURALMEM_API_BASE = getPreferenceValues<Preferences>().apiBaseUrl;
const API_KEY = getPreferenceValues<Preferences>().apiKey;
const DEFAULT_SPACE = getPreferenceValues<Preferences>().defaultSpace || "default";

async function searchMemories(query: string, space = DEFAULT_SPACE): Promise<SearchResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (API_KEY) {
    headers["Authorization"] = `Bearer ${API_KEY}`;
  }

  const url = new URL(`${NEURALMEM_API_BASE}/search`);
  url.searchParams.set("q", query);
  url.searchParams.set("space", space);
  url.searchParams.set("limit", "20");

  const res = await fetch(url.toString(), { method: "GET", headers });
  if (!res.ok) {
    throw new Error(`Search failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as SearchResponse;
}

export default function SearchMemoriesCommand() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Memory[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const performSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setResults([]);
      setError(null);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const data = await searchMemories(q);
      setResults(data.results || []);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setResults([]);
      showToast({ style: Toast.Style.Failure, title: "Search failed", message: msg });
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Debounce search input (300 ms)
  useEffect(() => {
    const timer = setTimeout(() => performSearch(query), 300);
    return () => clearTimeout(timer);
  }, [query, performSearch]);

  return (
    <List
      isLoading={isLoading}
      searchText={query}
      onSearchTextChange={setQuery}
      searchBarPlaceholder="Search your NeuralMem memories..."
      throttle
    >
      {error ? (
        <List.EmptyView icon={Icon.ExclamationMark} title="Search Error" description={error} />
      ) : query.trim() === "" ? (
        <List.EmptyView icon={Icon.MagnifyingGlass} title="Start typing to search" description="Search across all saved memories by keyword or semantic meaning." />
      ) : results.length === 0 ? (
        <List.EmptyView icon={Icon.Document} title="No memories found" description={`No results for "${query}"`} />
      ) : (
        results.map((mem) => (
          <List.Item
            key={mem.id}
            title={mem.title || "Untitled"}
            subtitle={mem.space || DEFAULT_SPACE}
            accessories={[
              { text: mem.score ? `Score: ${mem.score.toFixed(2)}` : undefined },
              { date: new Date(mem.created_at) },
            ]}
            actions={
              <ActionPanel>
                <Action.OpenInBrowser url={mem.url || "#"} />
                <Action.CopyToClipboard
                  title="Copy Content"
                  content={mem.content}
                  shortcut={{ modifiers: ["cmd"], key: "c" }}
                />
                <Action.CopyToClipboard
                  title="Copy URL"
                  content={mem.url || ""}
                  shortcut={{ modifiers: ["cmd", "shift"], key: "c" }}
                />
                <Action
                  title="Open in NeuralMem"
                  icon={Icon.Globe}
                  onAction={() => {
                    const url = `${NEURALMEM_API_BASE}/memories/${mem.id}`;
                    // Raycast Action.OpenInBrowser handles this; placeholder for custom logic
                  }}
                />
              </ActionPanel>
            }
          />
        ))
      )}
    </List>
  );
}
