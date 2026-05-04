import { useState, useCallback } from "react";
import {
  Form,
  ActionPanel,
  Action,
  Icon,
  showToast,
  Toast,
  getPreferenceValues,
  Clipboard,
} from "@raycast/api";

// ─────────────────────────────────────────────────────────────────────────────
// NeuralMem Save Command (Raycast)
// Quickly save selected text, clipboard content, or typed notes to NeuralMem.
// ─────────────────────────────────────────────────────────────────────────────

interface Preferences {
  apiBaseUrl: string;
  apiKey?: string;
  defaultSpace?: string;
}

interface SavePayload {
  type: "note" | "page" | "bookmark";
  title: string;
  content: string;
  url?: string;
  space: string;
  tags?: string[];
}

interface SaveResponse {
  id: string;
  success: boolean;
}

const NEURALMEM_API_BASE = getPreferenceValues<Preferences>().apiBaseUrl;
const API_KEY = getPreferenceValues<Preferences>().apiKey;
const DEFAULT_SPACE = getPreferenceValues<Preferences>().defaultSpace || "default";

async function saveMemory(payload: SavePayload): Promise<SaveResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (API_KEY) {
    headers["Authorization"] = `Bearer ${API_KEY}`;
  }

  const res = await fetch(`${NEURALMEM_API_BASE}/memories`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`Save failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as SaveResponse;
}

export default function SaveMemoryCommand() {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [url, setUrl] = useState("");
  const [space, setSpace] = useState(DEFAULT_SPACE);
  const [tags, setTags] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handlePasteClipboard = useCallback(async () => {
    const text = await Clipboard.readText();
    if (text) {
      setContent(text);
      showToast({ style: Toast.Style.Success, title: "Pasted from clipboard" });
    } else {
      showToast({ style: Toast.Style.Failure, title: "Clipboard is empty" });
    }
  }, []);

  const handleSubmit = useCallback(async (values: { title: string; content: string; url: string; space: string; tags: string }) => {
    if (!values.content.trim()) {
      showToast({ style: Toast.Style.Failure, title: "Content is required" });
      return;
    }

    setIsLoading(true);
    try {
      const payload: SavePayload = {
        type: values.url?.trim() ? "page" : "note",
        title: values.title.trim() || "Untitled",
        content: values.content.trim(),
        url: values.url.trim() || undefined,
        space: values.space.trim() || DEFAULT_SPACE,
        tags: values.tags
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
      };

      const result = await saveMemory(payload);
      showToast({
        style: Toast.Style.Success,
        title: "Saved to NeuralMem",
        message: `ID: ${result.id}`,
      });

      // Reset form
      setTitle("");
      setContent("");
      setUrl("");
      setSpace(DEFAULT_SPACE);
      setTags("");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      showToast({ style: Toast.Style.Failure, title: "Save failed", message: msg });
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <Form
      isLoading={isLoading}
      actions={
        <ActionPanel>
          <Action.SubmitForm title="Save to NeuralMem" icon={Icon.Plus} onSubmit={handleSubmit} />
          <Action
            title="Paste from Clipboard"
            icon={Icon.Clipboard}
            shortcut={{ modifiers: ["cmd", "shift"], key: "v" }}
            onAction={handlePasteClipboard}
          />
        </ActionPanel>
      }
    >
      <Form.TextField
        id="title"
        title="Title"
        placeholder="e.g. Machine Learning Notes"
        value={title}
        onChange={setTitle}
      />
      <Form.TextArea
        id="content"
        title="Content"
        placeholder="Paste or type the content you want to save..."
        value={content}
        onChange={setContent}
        enableMarkdown
      />
      <Form.Separator />
      <Form.TextField
        id="url"
        title="URL (optional)"
        placeholder="https://example.com/article"
        value={url}
        onChange={setUrl}
      />
      <Form.TextField
        id="space"
        title="Space"
        placeholder={DEFAULT_SPACE}
        value={space}
        onChange={setSpace}
      />
      <Form.TextField
        id="tags"
        title="Tags"
        placeholder="ai, research, paper (comma separated)"
        value={tags}
        onChange={setTags}
      />
    </Form>
  );
}
