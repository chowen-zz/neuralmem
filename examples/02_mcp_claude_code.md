# 将 NeuralMem 接入 AI 客户端

> 完整的多平台配置指南：[docs/mcp-integrations.md](../docs/mcp-integrations.md)

## 快速配置

所有支持 MCP 的客户端都使用同一配置格式：

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "neuralmem",
      "args": ["mcp"],
      "env": {
        "NEURALMEM_DB_PATH": "~/.neuralmem/memory.db"
      }
    }
  }
}
```

## 按客户端查看配置

| 客户端 | 配置位置 |
|--------|---------|
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Claude Code | `.claude/mcp.json` 或 `claude mcp add neuralmem -- neuralmem mcp` |
| Cursor | `.cursor/mcp.json` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| Cline (VS Code) | `~/.cline/mcp_settings.json` |
| Continue | `~/.continue/config.json` |
| Zed | `~/.config/zed/settings.json` |
| ChatBox / Cherry Studio / Trae | Settings → MCP Servers |

## 可用工具（10 个）

| 工具 | 功能 |
|------|------|
| `remember` | 存储记忆（自动提取实体和关系） |
| `recall` | 检索相关记忆（4策略并行 + RRF 融合） |
| `reflect` | 更新记忆 |
| `forget` | 删除记忆 |
| `consolidate` | 合并相似记忆 |
| `remember_batch` | 批量存储 |
| `forget_batch` | 批量删除 |
| `export_memories` | 导出（JSON/CSV/Markdown） |
| `resolve_conflict` | 解决记忆冲突 |
| `recall_with_explanation` | 带解释的检索 |

## HTTP 模式

```bash
neuralmem mcp --http
# → http://localhost:8000/mcp
```

## CLI 快速测试

```bash
neuralmem add "用户偏好 TypeScript"
neuralmem search "编程语言偏好"
neuralmem stats
```
