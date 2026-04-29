"""NeuralMem 配置管理"""
from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class NeuralMemConfig(BaseModel):
    """NeuralMem 主配置类。所有字段均可通过环境变量覆盖。"""

    # 存储配置
    db_path: str = Field(
        default="~/.neuralmem/memory.db",
        description="SQLite 数据库路径",
    )

    # Embedding 配置
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="FastEmbed 模型名称",
    )
    embedding_provider: str = Field(
        default="local",
        description="Embedding 提供商：local | openai | ollama",
    )
    embedding_dim: int = Field(default=384, description="向量维度")

    # 检索配置
    enable_reranker: bool = Field(default=False, description="是否启用 Cross-Encoder 重排序")
    recency_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="时序检索中近期权重"
    )
    default_search_limit: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)

    # LLM 提取配置（Ollama 可选）
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama 服务地址",
    )
    ollama_model: str = Field(
        default="llama3.2:3b",
        description="Ollama 提取模型",
    )
    enable_llm_extraction: bool = Field(
        default=False,
        description="是否启用 LLM 辅助提取（需要 Ollama 运行中）",
    )

    # OpenAI（可选）
    openai_api_key: str | None = Field(default=None)
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    @field_validator("db_path")
    @classmethod
    def expand_db_path(cls, v: str) -> str:
        return str(Path(v).expanduser())

    @classmethod
    def from_env(cls) -> "NeuralMemConfig":
        """从环境变量构建配置（NEURALMEM_ 前缀）"""
        return cls(
            db_path=os.getenv("NEURALMEM_DB_PATH", "~/.neuralmem/memory.db"),
            embedding_model=os.getenv("NEURALMEM_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_provider=os.getenv("NEURALMEM_EMBEDDING_PROVIDER", "local"),
            enable_reranker=os.getenv("NEURALMEM_ENABLE_RERANKER", "false").lower() == "true",
            ollama_url=os.getenv("NEURALMEM_OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("NEURALMEM_OLLAMA_MODEL", "llama3.2:3b"),
            enable_llm_extraction=os.getenv("NEURALMEM_LLM_EXTRACTION", "false").lower() == "true",
            openai_api_key=os.getenv("NEURALMEM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )

    def get_db_path(self) -> Path:
        p = Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
