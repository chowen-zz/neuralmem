# 将 NeuralMem 接入 Claude Desktop / Claude Code

## 安装

```bash
pip install neuralmem
```

## Claude Desktop 配置

编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`（macOS）：

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

## Claude Code 配置

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

## 可用工具

接入后，Claude 可以使用以下工具：

| 工具 | 功能 |
|------|------|
| `remember` | 存储记忆（自动提取实体和关系） |
| `recall` | 检索相关记忆（4策略并行 + RRF 融合） |
| `reflect` | 对主题进行多跳推理总结 |
| `forget` | 删除指定记忆（支持 GDPR 合规） |
| `consolidate` | 后台整理：衰减旧记忆、合并重复 |

## 使用示例

在 Claude 对话中，Claude 会自动调用这些工具：

- "记住这个项目使用 PostgreSQL 和 React"
  → Claude 调用 `remember` 存储技术栈信息

- "我们之前用什么数据库？"
  → Claude 调用 `recall` 检索相关记忆

## CLI 快速测试

```bash
# 添加记忆
neuralmem add "用户偏好 TypeScript"

# 搜索记忆
neuralmem search "编程语言偏好"

# 查看统计
neuralmem stats

# 启动 MCP Server（stdio 模式）
neuralmem mcp
```
