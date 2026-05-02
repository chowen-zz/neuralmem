# NeuralMem

[简体中文](README_zh.md) | [English](README.md) | [日本語](README_ja.md) | [한국어](README_ko.md) | **Tiếng Việt** | [繁體中文](README_zh-TW.md)

> **Bộ nhớ như Hạ tầng — Ưu tiên cục bộ, tích hợp MCP nguyên bản, sẵn sàng cho doanh nghiệp**

Bộ nhớ cho Agent AI như SQLite — Cài đặt không phụ thuộc bên ngoài, ưu tiên cục bộ, có thể mở rộng cho doanh nghiệp.

[![Tests](https://img.shields.io/badge/tests-160%20passing-brightgreen)](https://github.com/chowen-zz/neuralmem)
[![Coverage](https://img.shields.io/badge/coverage-83%25-green)](https://github.com/chowen-zz/neuralmem)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/neuralmem/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

---

## Tại sao NeuralMem?

Dù cửa sổ ngữ cảnh hiện đại có thể lên tới 200K token, việc nhét toàn bộ lịch sử hội thoại vào môi trường production vẫn là điều không khả thi — cả về chi phí lẫn độ trễ. NeuralMem cung cấp một giải pháp tốt hơn:

- **Bộ nhớ lâu dài** — Agent ghi nhớ thông tin xuyên suốt các phiên làm việc, không bị mất khi kết thúc hội thoại
- **Tìm kiếm thông minh** — 4 chiến lược kết hợp: semantic + BM25 + đồ thị tri thức + trọng số thời gian, hợp nhất qua RRF fusion
- **Không phụ thuộc bên ngoài** — Chỉ cần `pip install neuralmem`, không cần Docker, không cần API key
- **Hỗ trợ MCP nguyên bản** — Kết nối với Claude Desktop trong vòng 30 giây

---

## Bắt đầu nhanh

```bash
pip install neuralmem
```

```python
from neuralmem import MemoryEngine

engine = MemoryEngine()

# Lưu một ký ức
engine.remember("Alice thích dùng Python và đang xây dựng một ứng dụng RAG.")

# Truy xuất thông tin liên quan
results = engine.recall("Alice đang làm dự án gì?")
for r in results:
    print(r.content, r.score)
```

---

## Tích hợp MCP với Claude Desktop

Chỉnh sửa file `claude_desktop_config.json` của bạn:

```json
{
  "mcpServers": {
    "neuralmem": {
      "command": "python",
      "args": ["-m", "neuralmem.mcp_server"]
    }
  }
}
```

Sau khi khởi động lại Claude Desktop, bạn sẽ có ngay **5 công cụ bộ nhớ**:

| Công cụ | Mô tả |
|---------|-------|
| `remember` | Lưu thông tin mới vào bộ nhớ |
| `recall` | Truy xuất ký ức liên quan theo ngữ nghĩa |
| `reflect` | Suy luận trên đồ thị tri thức |
| `forget` | Xóa một ký ức cụ thể |
| `consolidate` | Áp dụng đường cong quên lãng, gộp ký ức trùng lặp |

---

## Tính năng cốt lõi

### Tìm kiếm song song 4 chiến lược

NeuralMem chạy đồng thời 4 chiến lược tìm kiếm, sau đó hợp nhất kết quả bằng RRF (Reciprocal Rank Fusion). Tùy chọn có thể bật Cross-Encoder để xếp hạng lại:

```
Semantic Search  ──┐
BM25 Keyword     ──┤──► RRF Fusion ──► [Cross-Encoder] ──► Kết quả cuối
Graph Traversal  ──┤
Time Weighting   ──┘
```

### Đường cong quên lãng Ebbinghaus

Bộ nhớ không dùng đến sẽ dần mờ đi — giống như trí nhớ con người. Gọi `consolidate()` để kích hoạt quá trình này:

```python
stats = engine.consolidate()
print(stats)
# {"decayed": 12, "forgotten": 3, "merged": 5}
```

- `decayed`: số ký ức bị giảm điểm quan trọng
- `forgotten`: số ký ức bị xóa vì quá lâu không dùng
- `merged`: số ký ức trùng lặp được gộp lại

### Đồ thị tri thức

NeuralMem xây dựng đồ thị quan hệ giữa các thực thể bằng NetworkX, hỗ trợ suy luận đa bước:

```python
engine.remember("Alice là kỹ sư tại Acme Corp.")
engine.remember("Alice đang dùng FastAPI và PostgreSQL cho dự án mới.")
engine.remember("Acme Corp sử dụng kiến trúc microservices.")

# Suy luận: tech stack của Alice liên quan đến kiến trúc công ty như thế nào?
insights = engine.reflect("tech stack của Alice")
print(insights)
# ["Alice → dùng → FastAPI, PostgreSQL",
#  "Alice → làm tại → Acme Corp",
#  "Acme Corp → áp dụng → microservices"]
```

### Phân giải thực thể

"Alice", "đồng nghiệp Alice", "cô ấy" đều được nhận diện là cùng một thực thể — không có nút trùng lặp trong đồ thị tri thức:

```python
engine.remember("Alice thích uống cà phê buổi sáng.")
engine.remember("Đồng nghiệp Alice hay đến muộn.")
engine.remember("Cô ấy vừa hoàn thành sprint đầu tiên.")

# Ba câu trên đều được liên kết với cùng một thực thể "Alice"
```

---

## CLI

```bash
# Thêm ký ức từ dòng lệnh
neuralmem add "Cuộc họp với khách hàng vào thứ Sáu lúc 14:00"

# Tìm kiếm ký ức
neuralmem search "lịch họp tuần này"

# Xem thống kê bộ nhớ
neuralmem stats

# Khởi động MCP server
neuralmem mcp
```

---

## Cải tiến tùy chọn

Cài thêm Cross-Encoder để xếp hạng lại kết quả tìm kiếm với độ chính xác cao hơn:

```bash
pip install neuralmem[reranker]
```

```python
engine = MemoryEngine(enable_reranker=True)
```

Cross-Encoder sẽ chạy sau bước RRF fusion để đánh giá lại mức độ liên quan của từng kết quả, đặc biệt hữu ích khi cần độ chính xác cao trong các tác vụ phức tạp.

---

## So sánh

| Tính năng | NeuralMem | Mem0 | Zep |
|-----------|-----------|------|-----|
| Chạy cục bộ | ✅ Không phụ thuộc | ❌ Cần Docker | ❌ Cần Neo4j |
| Tính năng đồ thị | ✅ Miễn phí | ❌ $249/tháng | ✅ Cần Neo4j |
| Hỗ trợ MCP nguyên bản | ✅ | ✅ | ✅ |
| Đường cong quên lãng | ✅ | ❌ | ❌ |
| Phân giải thực thể | ✅ | ❌ | ✅ |
| Giấy phép | Apache-2.0 | Apache-2.0 | Apache-2.0 |

---

## Giấy phép

Dự án được phát hành theo giấy phép [Apache-2.0](LICENSE).

Miễn phí sử dụng trong các dự án thương mại.
