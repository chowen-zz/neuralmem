# Cursor 集成

## 配置

在项目根目录创建 `.cursor/mcp.json`：

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

配置完成后重启 Cursor，Composer 中即可使用记忆功能。
