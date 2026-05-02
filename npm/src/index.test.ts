/**
 * NeuralMem npm package tests
 *
 * Tests the TypeScript API layer with mocked MCP transport.
 * Does NOT require a running NeuralMem server or Python.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { NeuralMem } from "./index.js";
import type { SearchResult } from "./types.js";

// Mock the MCP SDK
vi.mock("@modelcontextprotocol/sdk/client/index.js", () => {
  const mockCallTool = vi.fn();
  const mockClose = vi.fn();
  const mockListTools = vi.fn();

  return {
    Client: vi.fn().mockImplementation(() => ({
      connect: vi.fn(),
      callTool: mockCallTool,
      close: mockClose,
      listTools: mockListTools,
    })),
    __mockCallTool: mockCallTool,
    __mockClose: mockClose,
    __mockListTools: mockListTools,
  };
});

vi.mock("@modelcontextprotocol/sdk/client/stdio.js", () => ({
  StdioClientTransport: vi.fn().mockImplementation(() => ({})),
}));

// Helper to get the mocked callTool
async function getMocks() {
  const mod = await import("@modelcontextprotocol/sdk/client/index.js");
  const anyMod = mod as unknown as {
    __mockCallTool: ReturnType<typeof vi.fn>;
    __mockClose: ReturnType<typeof vi.fn>;
    __mockListTools: ReturnType<typeof vi.fn>;
  };
  return {
    callTool: anyMod.__mockCallTool,
    close: anyMod.__mockClose,
    listTools: anyMod.__mockListTools,
  };
}

function makeToolResult(text: string) {
  return { content: [{ type: "text", text }] };
}

describe("NeuralMem", () => {
  let mem: NeuralMem;

  beforeEach(async () => {
    vi.clearAllMocks();
    mem = new NeuralMem();
    const mocks = await getMocks();
    mocks.callTool.mockReset();
    mocks.close.mockReset();
    mocks.listTools.mockReset();
  });

  // ─── remember() ─────────────────────────────────────────

  describe("remember()", () => {
    it("should call the remember tool with content", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(
        makeToolResult("Remembered 1 memory item(s). IDs: ['01HXYZ...']")
      );

      const result = await mem.remember("User prefers TypeScript");
      expect(result).toContain("Remembered");
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "remember",
        arguments: { content: "User prefers TypeScript" },
      });
    });

    it("should pass userId when provided", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Remembered 1 memory item(s)."));

      await mem.remember("Test memory", { userId: "user-123" });
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "remember",
        arguments: { content: "Test memory", user_id: "user-123" },
      });
    });

    it("should pass tags as comma-separated string", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Remembered 1 memory item(s)."));

      await mem.remember("Tagged memory", { tags: ["work", "important"] });
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "remember",
        arguments: { content: "Tagged memory", tags: "work,important" },
      });
    });

    it("should pass memoryType", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Remembered 1 memory item(s)."));

      await mem.remember("A preference", { memoryType: "preference" });
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "remember",
        arguments: { content: "A preference", memory_type: "preference" },
      });
    });
  });

  // ─── recall() ───────────────────────────────────────────

  describe("recall()", () => {
    it("should parse recall results from formatted text", async () => {
      const mocks = await getMocks();
      const recallText = [
        "[1] (score=0.85) User prefers TypeScript for frontend",
        "[2] (score=0.72) User dislikes JavaScript",
        "[3] (score=0.65) User uses React with TypeScript",
      ].join("\n");
      mocks.callTool.mockResolvedValue(makeToolResult(recallText));

      const results: SearchResult[] = await mem.recall("TypeScript preferences");
      expect(results).toHaveLength(3);
      expect(results[0].score).toBe(0.85);
      expect(results[0].memory.content).toBe("User prefers TypeScript for frontend");
      expect(results[1].score).toBe(0.72);
    });

    it("should return empty array for no results", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("No memories found."));

      const results = await mem.recall("nonexistent topic");
      expect(results).toHaveLength(0);
    });

    it("should pass limit and userId", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("No memories found."));

      await mem.recall("test query", { userId: "u1", limit: 5 });
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "recall",
        arguments: { query: "test query", user_id: "u1", limit: 5 },
      });
    });
  });

  // ─── reflect() ──────────────────────────────────────────

  describe("reflect()", () => {
    it("should update memory content", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Updated memory 01HXYZ"));

      const result = await mem.reflect("mem-123", "New content");
      expect(result).toContain("Updated");
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "reflect",
        arguments: { memory_id: "mem-123", new_content: "New content" },
      });
    });

    it("should update importance", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Updated memory 01HXYZ"));

      await mem.reflect("mem-123", undefined, 0.9);
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "reflect",
        arguments: { memory_id: "mem-123", importance: 0.9 },
      });
    });
  });

  // ─── forget() ───────────────────────────────────────────

  describe("forget()", () => {
    it("should delete a memory by ID", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Deleted memory 01HXYZ"));

      const result = await mem.forget("mem-123");
      expect(result).toContain("Deleted");
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "forget",
        arguments: { memory_id: "mem-123" },
      });
    });
  });

  // ─── consolidate() ──────────────────────────────────────

  describe("consolidate()", () => {
    it("should call consolidate with default threshold", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(
        makeToolResult("Consolidation complete: {'merged': 2}")
      );

      await mem.consolidate();
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "consolidate",
        arguments: {},
      });
    });

    it("should pass custom threshold", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Consolidation complete."));

      await mem.consolidate({ similarityThreshold: 0.95 });
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "consolidate",
        arguments: { similarity_threshold: 0.95 },
      });
    });
  });

  // ─── resolveConflict() ──────────────────────────────────

  describe("resolveConflict()", () => {
    it("should resolve conflict with reactivate", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Conflict resolved: reactivate"));

      await mem.resolveConflict("mem-123", "reactivate");
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "resolve_conflict",
        arguments: { memory_id: "mem-123", action: "reactivate" },
      });
    });

    it("should resolve conflict with delete", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Conflict resolved: delete"));

      await mem.resolveConflict("mem-123", "delete");
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "resolve_conflict",
        arguments: { memory_id: "mem-123", action: "delete" },
      });
    });
  });

  // ─── rememberBatch() ────────────────────────────────────

  describe("rememberBatch()", () => {
    it("should batch store memories", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(
        makeToolResult("Batch stored 3 memory item(s).")
      );

      const result = await mem.rememberBatch(["fact1", "fact2", "fact3"]);
      expect(result).toContain("Batch stored 3");
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "remember_batch",
        arguments: { contents: ["fact1", "fact2", "fact3"] },
      });
    });

    it("should pass shared options", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Batch stored 2 memory item(s)."));

      await mem.rememberBatch(["a", "b"], { userId: "u1", tags: ["bulk"] });
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "remember_batch",
        arguments: { contents: ["a", "b"], user_id: "u1", tags: "bulk" },
      });
    });
  });

  // ─── exportMemories() ───────────────────────────────────

  describe("exportMemories()", () => {
    it("should export as JSON by default", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult('{"memories": []}'));

      const result = await mem.exportMemories();
      expect(result).toBeDefined();
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "export_memories",
        arguments: {},
      });
    });

    it("should export as markdown", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("# Memories\n\n- fact 1"));

      await mem.exportMemories({ format: "markdown" });
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "export_memories",
        arguments: { format: "markdown" },
      });
    });
  });

  // ─── forgetBatch() ──────────────────────────────────────

  describe("forgetBatch()", () => {
    it("should batch delete by IDs", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Deleted 2 memories"));

      await mem.forgetBatch({ ids: ["id1", "id2"] });
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "forget_batch",
        arguments: { ids: "id1,id2" },
      });
    });

    it("should batch delete by tags", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Deleted 5 memories"));

      await mem.forgetBatch({ tags: ["temp", "ephemeral"] });
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "forget_batch",
        arguments: { tags: "temp,ephemeral" },
      });
    });

    it("should support dry run", async () => {
      const mocks = await getMocks();
      mocks.callTool.mockResolvedValue(makeToolResult("Would delete 3 memories"));

      await mem.forgetBatch({ tags: ["old"], dryRun: true });
      expect(mocks.callTool).toHaveBeenCalledWith({
        name: "forget_batch",
        arguments: { tags: "old", dry_run: true },
      });
    });
  });

  // ─── listTools() ────────────────────────────────────────

  describe("listTools()", () => {
    it("should list available MCP tools", async () => {
      const mocks = await getMocks();
      mocks.listTools.mockResolvedValue({
        tools: [
          { name: "remember", description: "Store a new memory" },
          { name: "recall", description: "Search memories" },
        ],
      });

      await mem.connect();
      const tools = await mem.listTools();
      expect(tools).toHaveLength(2);
      expect(tools[0].name).toBe("remember");
    });
  });

  // ─── connect/disconnect ─────────────────────────────────

  describe("connection lifecycle", () => {
    it("should track connection state", async () => {
      expect(mem.isConnected).toBe(false);
      await mem.connect();
      expect(mem.isConnected).toBe(true);
      await mem.disconnect();
      expect(mem.isConnected).toBe(false);
    });
  });
});
