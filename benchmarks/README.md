# NeuralMem 基准测试

## LongMemEval

LongMemEval 是一个评估长期记忆系统的公开基准，测试记忆准确性、召回率和精确率。

### 运行

```bash
# 安装依赖
pip install neuralmem datasets

# 运行（使用示例数据）
python benchmarks/longmemeval/run_benchmark.py --sample 100

# 完整评测
python benchmarks/longmemeval/run_benchmark.py
```

### 评测维度

| 指标 | 说明 |
|------|------|
| Recall@K | 前 K 个结果中包含正确答案的比例 |
| Precision@K | 前 K 个结果中正确答案的占比 |
| MRR | Mean Reciprocal Rank，越高越好 |
| F1 | Recall 和 Precision 的调和平均 |
