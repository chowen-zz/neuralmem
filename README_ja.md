# NeuralMem

[简体中文](README.md) | [English](README_en.md) | **日本語** | [한국어](README_ko.md) | [Tiếng Việt](README_vi.md) | [繁體中文](README_zh-TW.md)

> **メモリ・アズ・インフラ** — ローカルファースト、MCPネイティブ、エンタープライズ対応

SQLiteのようなエージェントメモリ — ゼロ依存インストール、ローカルファースト、エンタープライズ対応。

[![Tests](https://img.shields.io/badge/tests-160%20passing-brightgreen)](https://github.com/chowen-zz/neuralmem)
[![Coverage](https://img.shields.io/badge/coverage-83%25-green)](https://github.com/chowen-zz/neuralmem)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/neuralmem/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## なぜ NeuralMem なのか？

200K トークンのコンテキストウィンドウがあっても、本番環境のリクエストに全ての会話履歴を詰め込むのは現実的ではありません — コストとレイテンシの両面で許容できないからです。NeuralMem はより優れた解決策を提供します：

- **永続メモリ**：エージェントがセッションをまたいでユーザーの好み、プロジェクトの背景、過去の意思決定を記憶します
- **スマート検索**：4つの並列戦略（セマンティック + BM25 + グラフ + 時系列）と RRF フュージョンにより、最も関連性の高いメモリを抽出します
- **ゼロ依存**：`pip install neuralmem` だけで準備完了 — Docker も API キーも不要
- **MCPネイティブ**：Model Context Protocol を一級市民としてサポート。30秒で Claude Desktop に接続できます

## クイックスタート

```bash
pip install neuralmem
```

```python
from neuralmem import NeuralMem

mem = NeuralMem()

# メモリを保存する
mem.remember("ユーザーはフロントエンドに TypeScript を好み、JavaScript を嫌っている")

# メモリを検索する（4戦略並列検索 + RRF フュージョン）
results = mem.recall("ユーザーの技術的な好みは何ですか？")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

## Claude Desktop への MCP 統合

`~/Library/Application Support/Claude/claude_desktop_config.json` に以下を追加します：

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

接続後、Claude は5つのツールを利用できるようになります：`remember`、`recall`、`reflect`、`forget`、`consolidate`。

## コア機能

### 4戦略並列検索

```
セマンティック検索   → 意味的に類似したメモリを取得
BM25 キーワード      → 完全一致する用語を取得
グラフ走査           → エンティティの関係を通じて関連メモリを検索
時系列加重           → 最近のメモリほど高い重みを付与
                   ↓
          RRF フュージョン（Reciprocal Rank Fusion）
                   ↓
     Cross-Encoder 再ランキング（オプション）
```

### Ebbinghaus 忘却曲線

```python
# 忘却曲線を適用するために定期的に consolidate を呼び出す
stats = mem.consolidate()
# {"decayed": 12, "forgotten": 3, "merged": 0}
```

### ナレッジグラフ

エンティティと関係を自動的に抽出して NetworkX グラフに保存し、多段階推論を可能にします：

```python
report = mem.reflect("Alice の技術スタック")
# グラフを自動走査：Alice → Python → 機械学習 → 関連メモリ
```

### エンティティ曖昧性解消

「Alice」「同僚の Alice」「彼女」といった表現は自動的に同一エンティティとして解決され、グラフに重複ノードが生成されません。

## CLI

```bash
neuralmem add "ユーザーは TypeScript を好む"   # メモリを追加
neuralmem search "プログラミング言語"           # メモリを検索
neuralmem stats                               # 統計情報を表示
neuralmem mcp                                 # MCP サーバーを起動
```

## オプション機能の拡張

```bash
# Cross-Encoder 再ランキングを有効化（約 67MB のモデルダウンロード）
pip install neuralmem[reranker]
```

```python
from neuralmem import NeuralMem
from neuralmem.core.config import NeuralMemConfig

mem = NeuralMem(config=NeuralMemConfig(enable_reranker=True))
```

## 競合製品との比較

| 機能 | NeuralMem | Mem0 | Zep |
|------|-----------|------|-----|
| ローカル実行 | ✅ ゼロ依存 | ❌ Docker 必要 | ❌ Neo4j 必要 |
| グラフ機能 | ✅ 無料 | ❌ $249/月 | ✅ Neo4j 必要 |
| MCPネイティブ | ✅ | ✅ | ✅ |
| 忘却曲線 | ✅ | ❌ | ❌ |
| エンティティ曖昧性解消 | ✅ | ❌ | ✅ |
| ライセンス | Apache-2.0 | Apache-2.0 | Apache-2.0 |

## ライセンス

Apache-2.0。商用プロジェクトでの使用は無料です。
