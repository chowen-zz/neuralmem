"""NeuralMem 配置管理"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class NeuralMemConfig(BaseModel):
    """NeuralMem 主配置类。所有字段均可通过环境变量覆盖。"""

    # 存储配置
    db_path: str = Field(
        default="~/.neuralmem/memory.db",
        description="SQLite 数据库路径",
    )
    pg_dsn: str = Field(
        default="postgresql://localhost:5432/neuralmem",
        description="PostgreSQL DSN（用于 PgVectorStorage 后端）",
    )

    # Embedding 配置
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed 模型名称 (all-MiniLM-L6-v2 已不兼容 fastembed>=0.8)",
    )
    embedding_provider: str = Field(
        default="local",
        description="Embedding 提供商：local | openai | ollama",
    )
    embedding_dim: int = Field(default=384, description="向量维度")

    # 冲突解决配置
    conflict_threshold_low: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="冲突检测下界：低于此值视为无关记忆",
    )
    conflict_threshold_high: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="冲突检测上界：高于此值视为完全重复",
    )

    # 重要性强化配置
    enable_importance_reinforcement: bool = Field(
        default=True, description="recall()命中的记忆自动提升 importance",
    )
    reinforcement_boost: float = Field(
        default=0.05, ge=0.0, le=0.5,
        description="每次 recall 命中的 importance 增量",
    )

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

    # Ollama Embedding
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model",
    )

    # OpenAI（可选）
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # Cohere
    cohere_api_key: str | None = Field(default=None, repr=False)
    cohere_embedding_model: str = Field(default="embed-multilingual-v3.0")

    # Gemini
    gemini_api_key: str | None = Field(default=None, repr=False)
    gemini_embedding_model: str = Field(default="text-embedding-004")

    # HuggingFace Inference API
    hf_api_key: str | None = Field(default=None, repr=False)
    hf_model: str = Field(default="BAAI/bge-m3")
    hf_inference_url: str = Field(default="https://api-inference.huggingface.co")

    # Azure OpenAI
    azure_endpoint: str | None = Field(default=None)
    azure_api_key: str | None = Field(default=None, repr=False)
    azure_deployment: str = Field(default="text-embedding-3-small")
    azure_api_version: str = Field(default="2024-02-01")

    # Metrics / Observability
    enable_metrics: bool = Field(
        default=False,
        description="是否启用结构化指标收集（MetricsCollector）",
    )

    # LLM extractor selection (replaces enable_llm_extraction bool)
    llm_extractor: str = Field(
        default="none",
        description="LLM 提取器：none | ollama | openai | anthropic",
    )
    openai_extractor_model: str = Field(default="gpt-4o-mini")
    anthropic_api_key: str | None = Field(default=None, repr=False)
    anthropic_model: str = Field(default="claude-haiku-4-5-20251001")

    @model_validator(mode="after")
    def _backcompat_llm_extraction(self) -> NeuralMemConfig:
        if self.enable_llm_extraction and self.llm_extractor == "none":
            self.llm_extractor = "ollama"
        return self

    @field_validator("db_path")
    @classmethod
    def expand_db_path(cls, v: str) -> str:
        return str(Path(v).expanduser())

    @classmethod
    def from_env(cls) -> NeuralMemConfig:
        """从环境变量构建配置（NEURALMEM_ 前缀）"""
        return cls(
            db_path=os.getenv("NEURALMEM_DB_PATH", "~/.neuralmem/memory.db"),
            pg_dsn=os.getenv("NEURALMEM_PG_DSN", "postgresql://localhost:5432/neuralmem"),
            embedding_model=os.getenv("NEURALMEM_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_provider=os.getenv("NEURALMEM_EMBEDDING_PROVIDER", "local"),
            conflict_threshold_low=float(os.getenv("NEURALMEM_CONFLICT_LOW", "0.75")),
            conflict_threshold_high=float(os.getenv("NEURALMEM_CONFLICT_HIGH", "0.95")),
            enable_importance_reinforcement=(
                os.getenv("NEURALMEM_REINFORCE", "true").lower() == "true"
            ),
            reinforcement_boost=float(os.getenv("NEURALMEM_REINFORCE_BOOST", "0.05")),
            enable_reranker=os.getenv("NEURALMEM_ENABLE_RERANKER", "false").lower() == "true",
            ollama_url=os.getenv("NEURALMEM_OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.getenv("NEURALMEM_OLLAMA_MODEL", "llama3.2:3b"),
            ollama_embedding_model=os.getenv(
                "NEURALMEM_OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
            ),
            enable_llm_extraction=os.getenv("NEURALMEM_LLM_EXTRACTION", "false").lower() == "true",
            openai_api_key=os.getenv("NEURALMEM_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            cohere_api_key=os.getenv("NEURALMEM_COHERE_API_KEY"),
            cohere_embedding_model=os.getenv(
                "NEURALMEM_COHERE_EMBEDDING_MODEL", "embed-multilingual-v3.0"
            ),
            gemini_api_key=os.getenv("NEURALMEM_GEMINI_API_KEY"),
            gemini_embedding_model=os.getenv(
                "NEURALMEM_GEMINI_EMBEDDING_MODEL", "text-embedding-004"
            ),
            hf_api_key=os.getenv("NEURALMEM_HF_API_KEY"),
            hf_model=os.getenv("NEURALMEM_HF_MODEL", "BAAI/bge-m3"),
            hf_inference_url=os.getenv(
                "NEURALMEM_HF_INFERENCE_URL", "https://api-inference.huggingface.co"
            ),
            azure_endpoint=os.getenv("NEURALMEM_AZURE_ENDPOINT"),
            azure_api_key=os.getenv("NEURALMEM_AZURE_API_KEY"),
            azure_deployment=os.getenv("NEURALMEM_AZURE_DEPLOYMENT", "text-embedding-3-small"),
            azure_api_version=os.getenv("NEURALMEM_AZURE_API_VERSION", "2024-02-01"),
            llm_extractor=os.getenv("NEURALMEM_LLM_EXTRACTOR", "none"),
            openai_extractor_model=os.getenv("NEURALMEM_OPENAI_EXTRACTOR_MODEL", "gpt-4o-mini"),
            anthropic_api_key=os.getenv("NEURALMEM_ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv(
                "NEURALMEM_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
            ),
        )

    def get_db_path(self) -> Path:
        p = Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
