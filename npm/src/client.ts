/**
 * NeuralMem MCP Client — low-level MCP protocol wrapper
 *
 * Handles connection lifecycle (stdio or HTTP) and raw tool calls.
 * Most users should import from index.ts instead.
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { NeuralMemConfig } from "./types.js";

export class NeuralMemClient {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private config: Required<
    Pick<NeuralMemConfig, "transport" | "command" | "args" | "timeout">
  > &
    NeuralMemConfig;
  private connected = false;

  constructor(config: NeuralMemConfig = {}) {
    this.config = {
      transport: config.transport ?? "stdio",
      command: config.command ?? "neuralmem",
      args: config.args ?? ["mcp"],
      timeout: config.timeout ?? 30_000,
      ...config,
    };
  }

  /** Connect to the NeuralMem MCP server */
  async connect(): Promise<void> {
    if (this.connected) return;

    if (this.config.transport === "http") {
      throw new Error(
        "HTTP transport not yet implemented. Use transport='stdio' (default)."
      );
    }

    // Stdio transport — spawn `neuralmem mcp`
    this.transport = new StdioClientTransport({
      command: this.config.command,
      args: this.config.args,
      cwd: this.config.cwd,
      env: this.config.env as Record<string, string> | undefined,
    });

    this.client = new Client(
      { name: "neuralmem-node", version: "0.1.0" },
      { capabilities: {} }
    );

    await this.client.connect(this.transport);
    this.connected = true;
  }

  /** Disconnect from the server */
  async disconnect(): Promise<void> {
    if (!this.connected) return;
    if (this.client) {
      await this.client.close();
      this.client = null;
    }
    this.transport = null;
    this.connected = false;
  }

  /** Call an MCP tool by name */
  async callTool<T = string>(
    name: string,
    args: Record<string, unknown> = {}
  ): Promise<T> {
    if (!this.connected || !this.client) {
      throw new Error("Not connected. Call connect() first.");
    }

    const result = await this.client.callTool({ name, arguments: args });

    // MCP tools return content array; extract text
    if (result.content && Array.isArray(result.content)) {
      const textPart = result.content.find((c: { type: string }) => c.type === "text");
      if (textPart && "text" in textPart) {
        const text = (textPart as { text: string }).text;
        // Try to parse as JSON for structured responses
        try {
          return JSON.parse(text) as T;
        } catch {
          return text as T;
        }
      }
    }

    return result as T;
  }

  /** List available tools on the server */
  async listTools(): Promise<Array<{ name: string; description?: string }>> {
    if (!this.connected || !this.client) {
      throw new Error("Not connected. Call connect() first.");
    }
    const result = await this.client.listTools();
    return result.tools.map((t) => ({
      name: t.name,
      description: t.description,
    }));
  }

  /** Whether the client is currently connected */
  get isConnected(): boolean {
    return this.connected;
  }
}
