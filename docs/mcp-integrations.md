# NeuralMem MCP 接入指南 — 全平台配置

NeuralMem 实现了标准 [Model Context Protocol (MCP)](https://modelcontextprotocol.io)，可接入所有支持 MCP 的 AI 客户端。

## 前置条件

```bash
pip install neuralmem
```

验证安装：
```bash
neuralmem --help
neuralmem mcp  # 测试 stdio 模式（Ctrl+C 退出）
```

---

## 1. Claude Desktop

**macOS** 编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`：
**Windows** 编辑 `%APPDATA%\Claude\claude_desktop_config.json`：

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

---

## 2. Claude Code (CLI)

在项目根目录创建 `.claude/mcp.json`：

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "neuralmem",
      "args": ["mcp"]
    }
  }
}
```

或通过命令行添加：
```bash
claude mcp add neuralmem -- neuralmem mcp
```

---

## 3. Cursor

**方式 A：项目级配置**（推荐）

在项目根目录创建 `.cursor/mcp.json`：

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "neuralmem",
      "args": ["mcp"],
      "env": {
        "NEURALMEM_DB_PATH": ".neuralmem/memory.db"
      }
    }
  }
}
```

**方式 B：全局配置**

编辑 `~/.cursor/mcp.json`，格式同上。

配置后在 Cursor Settings → MCP 中确认状态为绿色 ✅。

---

## 4. Windsurf (Codeium)

编辑 `~/.codeium/windsurf/mcp_config.json`：

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

重启 Windsurf 后在 Cascade 面板中可用。

---

## 5. Cline (VS Code 扩展)

在 VS Code 中打开 Cline 面板 → ⚙️ MCP Servers → Configure：

编辑 `~/.cline/mcp_settings.json`（全局）或项目 `.cline/mcp_settings.json`：

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "neuralmem",
      "args": ["mcp"],
      "env": {
        "NEURALMEM_DB_PATH": "~/.neuralmem/memory.db"
      },
      "disabled": false,
      "autoApprove": ["remember", "recall"]
    }
  }
}
```

> `autoApprove` 可减少高频操作的确认弹窗。建议至少对 `recall` 自动批准。

---

## 6. Continue (VS Code / JetBrains)

编辑 Continue 配置文件 `~/.continue/config.json`：

```json
{
  "mcpServers": [
    {
      "name": "neuralmem",
      "command": "neuralmem",
      "args": ["mcp"],
      "env": {
        "NEURALMEM_DB_PATH": "~/.neuralmem/memory.db"
      }
    }
  ]
}
```

---

## 7. Zed Editor

编辑 `~/.config/zed/settings.json`，在 `mcp` 字段中添加：

```json
{
  "mcp": {
    "servers": {
      "neuralmem": {
        "command": "neuralmem",
        "args": ["mcp"],
        "env": {
          "NEURALMEM_DB_PATH": "~/.neuralmem/memory.db"
        }
      }
    }
  }
}
```

Zed 的 Agent Panel 会自动发现 NeuralMem 工具。

---

## 8. ChatBox (全平台桌面客户端)

ChatBox 支持自定义 MCP Provider。在 Settings → MCP Servers 中添加：

- **Name**: `neuralmem`
- **Type**: `stdio`
- **Command**: `neuralmem mcp`
- **Environment**: `NEURALMEM_DB_PATH=~/.neuralmem/memory.db`

或在配置文件中（位置因平台而异）：

```json
{
  "mcpServers": {
    "neuralmem": {
      "type": "stdio",
      "command": "neuralmem",
      "args": ["mcp"]
    }
  }
}
```

---

## 9. Cherry Studio

在 Cherry Studio 设置 → MCP 服务器中添加：

- **类型**: `本地进程 (stdio)`
- **命令**: `neuralmem`
- **参数**: `mcp`
- **环境变量**: `NEURALMEM_DB_PATH=~/.neuralmem/memory.db`

配置文件方式（`~/.cherrystudio/mcp.json`）：

```json
{
  "servers": {
    "neuralmem": {
      "transport": "stdio",
      "command": "neuralmem",
      "args": ["mcp"]
    }
  }
}
```

---

## 10. Trae (字节跳动)

编辑 MCP 配置（Settings → MCP Servers）：

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

---

## 11. Augment Code

在 Augment 设置中添加 MCP Server：

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "neuralmem",
      "args": ["mcp"]
    }
  }
}
```

---

## 12. HTTP 模式（远程/共享场景）

对于不支持 stdio 或需要远程访问的场景，启动 HTTP 模式：

```bash
neuralmem mcp --http
# 默认监听 http://localhost:8000/mcp
```

客户端配置（以支持 HTTP 的客户端为例）：

```json
{
  "mcpServers": {
    "neuralmem": {
      "url": "http://localhost:8000/mcp",
      "transport": "streamable-http"
    }
  }
}
```

> HTTP 模式适合：多客户端共享同一记忆库、远程服务器部署、Docker 容器化场景。

---

## 13. Node.js / npm 接入

```bash
npm install neuralmem
```

```typescript
import { NeuralMem } from "neuralmem";

const mem = new NeuralMem();
await mem.connect();

await mem.remember("用户偏好 TypeScript", { tags: ["偏好"] });
const results = await mem.recall("TypeScript");
for (const r of results) {
  console.log(`[${r.score.toFixed(2)}] ${r.memory.content}`);
}

await mem.disconnect();
```

---

## 通用配置项

所有平台共享以下环境变量：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `NEURALMEM_DB_PATH` | `~/.neuralmem/memory.db` | SQLite 数据库路径 |
| `NEURALMEM_EMBEDDING_PROVIDER` | `local` | 嵌入模型：`local` / `openai` / `cohere` / `gemini` |
| `NEURALMEM_LLM_EXTRACTOR` | `none` | LLM 实体提取：`none` / `ollama` / `openai` / `anthropic` |
| `NEURALMEM_ENABLE_RERANKER` | `false` | 启用 Cross-Encoder 重排序 |

示例 — 使用 OpenAI 嵌入 + Ollama 实体提取：

```json
{
  "env": {
    "NEURALMEM_EMBEDDING_PROVIDER": "openai",
    "NEURALMEM_LLM_EXTRACTOR": "ollama",
    "OPENAI_API_KEY": "sk-..."
  }
}
```

---

## 接入后可用工具

| 工具 | 功能 | 示例触发 |
|------|------|----------|
| `remember` | 存储记忆 | "记住这个项目用 PostgreSQL" |
| `recall` | 检索记忆 | "我们之前用什么数据库？" |
| `reflect` | 更新记忆 | "更新那条记忆，改成 MySQL" |
| `forget` | 删除记忆 | "忘掉刚才那条" |
| `consolidate` | 合并重复 | "整理一下记忆库" |
| `remember_batch` | 批量存储 | 从文件导入 |
| `forget_batch` | 批量删除 | 按标签清理 |
| `export_memories` | 导出记忆 | JSON/CSV/Markdown |
| `resolve_conflict` | 解决冲突 | 重新激活被覆盖的记忆 |
| `recall_with_explanation` | 带解释检索 | 显示每条结果的匹配原因 |

---

## 故障排查

**1. 工具未出现**
- 确认 `neuralmem` 在 PATH 中：`which neuralmem`
- 尝试直接运行：`neuralmem mcp` 看是否有报错

**2. 连接失败**
- 检查 Python 版本：`python3 --version`（需要 3.10+）
- 检查依赖：`pip show neuralmem mcp`

**3. 数据库锁定**
- 多客户端同时写入时可能出现。使用 HTTP 模式避免冲突。

**4. macOS 安全提示**
- 首次运行可能弹出安全提示，在 系统设置 → 隐私与安全 中允许。
