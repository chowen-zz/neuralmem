# Claude Code 集成

## 配置

在项目目录创建 `.claude/mcp.json`：

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "neuralmem",
      "args": ["mcp"],
      "env": {
        "NEURALMEM_DB_PATH": "./.neuralmem/project.db"
      }
    }
  }
}
```

## 使用方式

配置后，Claude Code 会自动在合适时机调用记忆工具：

- **存储**：`remember("这个项目使用 PostgreSQL 和 React")`
- **检索**：`recall("数据库是什么？")`
- **反思**：`reflect("项目技术栈")`

## 项目级记忆

建议为每个项目使用独立的数据库文件（通过 `NEURALMEM_DB_PATH` 设置），避免不同项目的记忆混淆。
