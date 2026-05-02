# NeuralMem

[简体中文](README.md) | [English](README_en.md) | [日本語](README_ja.md) | **한국어** | [Tiếng Việt](README_vi.md) | [繁體中文](README_zh-TW.md)

> **인프라로서의 메모리** — 로컬 우선, MCP 네이티브, 엔터프라이즈 지원

SQLite 같은 에이전트 메모리 — 제로 의존성 설치, 로컬 우선, 엔터프라이즈 확장 가능.

[![Tests](https://img.shields.io/badge/tests-160%20passing-brightgreen)](https://github.com/chowen-zz/neuralmem)
[![Coverage](https://img.shields.io/badge/coverage-83%25-green)](https://github.com/chowen-zz/neuralmem)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/neuralmem/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## 왜 NeuralMem인가?

200K 토큰 컨텍스트 창이 있더라도, 전체 대화 기록을 프로덕션 요청마다 집어넣는 것은 현실적으로 불가능합니다 — 비용과 지연 시간이 너무 크기 때문입니다. NeuralMem은 더 나은 솔루션을 제공합니다:

- **영속 메모리**: 에이전트가 사용자 선호 설정, 프로젝트 컨텍스트, 과거 결정 사항을 세션을 넘어 기억합니다
- **스마트 검색**: 시맨틱, BM25, 그래프, 시간적 가중치의 4가지 병렬 전략과 RRF 융합을 통해 가장 관련성 높은 메모리를 반환합니다
- **제로 의존성**: `pip install neuralmem` 한 줄로 바로 시작 — Docker도, API 키도 필요 없습니다
- **MCP 네이티브**: Model Context Protocol 네이티브 지원으로 30초 안에 Claude Desktop에 연결할 수 있습니다

## 빠른 시작

```bash
pip install neuralmem
```

```python
from neuralmem import NeuralMem

mem = NeuralMem()

# 메모리 저장
mem.remember("User prefers TypeScript for frontend, dislikes JavaScript")

# 메모리 검색 (4가지 전략 병렬 검색 + RRF 융합)
results = mem.recall("What are the user's tech preferences?")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

## Claude Desktop MCP 통합

`~/Library/Application Support/Claude/claude_desktop_config.json` 파일에 다음 내용을 추가하세요:

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

연결이 완료되면 Claude는 `remember`, `recall`, `reflect`, `forget`, `consolidate` 5가지 도구를 사용할 수 있게 됩니다.

## 핵심 기능

### 4가지 병렬 검색 전략

```
시맨틱 검색        → 의미적으로 유사한 메모리 검색
BM25 키워드        → 정확한 키워드 매칭
그래프 탐색        → 개체 관계를 통한 연관 메모리 검색
시간적 가중치      → 최근 메모리에 더 높은 가중치 부여
                  ↓
         RRF 융합 (Reciprocal Rank Fusion)
                  ↓
   Cross-Encoder 재순위화 (선택적)
```

### 에빙하우스 망각 곡선

```python
# consolidate()를 주기적으로 호출하여 망각 곡선 적용
stats = mem.consolidate()
# {"decayed": 12, "forgotten": 3, "merged": 0}
```

### 지식 그래프

개체와 관계가 자동으로 추출되어 NetworkX 그래프에 저장되며, 다중 홉 추론이 가능합니다:

```python
report = mem.reflect("Alice's tech stack")
# 그래프 자동 탐색: Alice → Python → Machine Learning → 연관 메모리
```

### 개체 명확화

"Alice", "동료 Alice", "그녀" 같은 표현이 동일한 개체로 자동 연결되어 그래프 내 중복 노드가 생기지 않습니다.

## CLI

```bash
neuralmem add "User prefers TypeScript"   # 메모리 추가
neuralmem search "programming language"   # 메모리 검색
neuralmem stats                           # 통계 보기
neuralmem mcp                             # MCP 서버 시작
```

## 선택적 기능 강화

```bash
# Cross-Encoder 재순위화 활성화 (약 67MB 모델 다운로드)
pip install neuralmem[reranker]
```

```python
from neuralmem import NeuralMem
from neuralmem.core.config import NeuralMemConfig

mem = NeuralMem(config=NeuralMemConfig(enable_reranker=True))
```

## 비교

| 기능 | NeuralMem | Mem0 | Zep |
|------|-----------|------|-----|
| 로컬 실행 | ✅ 제로 의존성 | ❌ Docker 필요 | ❌ Neo4j 필요 |
| 그래프 기능 | ✅ 무료 | ❌ $249/월 | ✅ Neo4j 필요 |
| MCP 네이티브 | ✅ | ✅ | ✅ |
| 망각 곡선 | ✅ | ❌ | ❌ |
| 개체 명확화 | ✅ | ❌ | ✅ |
| 라이선스 | Apache-2.0 | Apache-2.0 | Apache-2.0 |

## 라이선스

Apache-2.0. 상업 프로젝트에서 무료로 사용 가능합니다.
