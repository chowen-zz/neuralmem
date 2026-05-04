import { useState, useEffect } from "react";
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
// NeuralMem Recent Memories Command (Raycast)
// Browse and open your most recently saved memories.
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
  type?: string;
}

interface RecentResponse {
  memories: Memory[];
  total: number;
}

const NEURALMEM_API_BASE = getPreferenceValues<Preferences>().apiBaseUrl;
const API_KEY = getPreferenceValues<Preferences>().apiKey;
const DEFAULT_SPACE = getPreferenceValues<Preferences>().defaultSpace || "default";

async function fetchRecentMemories(space = DEFAULT_SPACE, limit = 20): Promise<RecentResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (API_KEY) {
    headers["Authorization"] = `Bearer ${API_KEY}`;
  }

  const url = new URL(`${NEURALMEM_API_BASE}/memories/recent`);
  url.searchParams.set("space", space);
  url.searchParams.set("limit", String(limit));

  const res = await fetch(url.toString(), { method: "GET", headers });
  if (!res.ok) {
    throw new Error(`Fetch failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as RecentResponse;
}

export default function RecentMemoriesCommand() {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setIsLoading(true);
        setError(null);
        const data = await fetchRecentMemories();
        if (!cancelled) {
          setMemories(data.memories || []);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (!cancelled) {
          setError(msg);
          setMemories([]);
          showToast({ style: Toast.Style.Failure, title: "Failed to load recent memories", message: msg });
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <List
      isLoading={isLoading}
      searchBarPlaceholder="Filter recent memories..."
      throttle
    >
      {error ? (
        <List.EmptyView icon={Icon.ExclamationMark} title="Error" description={error} />
      ) : memories.length === 0 ? (
        <List.EmptyView
          icon={Icon.Document}
          title="No recent memories"
          description="Save your first memory with the 'Save to Memory' command."
        />
      ) : (
        memories.map((mem) => (
          <List.Item
            key={mem.id}
            title={mem.title || "Untitled"}
            subtitle={mem.space || DEFAULT_SPACE}
            accessories={[
              { text: mem.type || "note" },
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
                  title="Refresh"
                  icon={Icon.ArrowClockwise}
                  shortcut={{ modifiers: ["cmd"], key: "r" }}
                  onAction={async () => {
                    setIsLoading(true);
                    try {
                      const data = await fetchRecentMemories();
                      setMemories(data.memories || []);
                      showToast({ style: Toast.Style.Success, title: "Refreshed" });
                    } catch (err) {
                      const msg = err instanceof Error ? err.message : String(err);
                      showToast({ style: Toast.Style.Failure, title: "Refresh failed", message: msg });
                    } finally {
                      setIsLoading(false);
                    }
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
