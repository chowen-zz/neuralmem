# 检索策略

NeuralMem 使用四策略并行检索 + RRF 融合，确保找到最相关的记忆。

## 四策略

### 1. 语义搜索（Semantic）
基于向量相似度，使用 FastEmbed（all-MiniLM-L6-v2）生成嵌入向量，通过余弦相似度找到语义相近的记忆。

### 2. BM25 关键词（Keyword）
基于 SQLite FTS5，精确匹配关键词，适合捕获技术术语等精确匹配场景。

### 3. 图谱遍历（Graph）
通过知识图谱中的实体关系，发现间接关联的记忆。例如搜索"Alice"可以找到包含"她的项目"的记忆。

### 4. 时序加权（Temporal）
语义搜索结果 + 时间衰减权重，近期记忆获得更高分数。

## RRF 融合

四策略结果通过倒数排名融合（Reciprocal Rank Fusion）合并：

```
RRF(d) = Σ 1/(k + rank_i(d)),  k=60
```

## 可选：Cross-Encoder 重排序

```bash
pip install neuralmem[reranker]
```

```python
mem = NeuralMem(config=NeuralMemConfig(enable_reranker=True))
```
