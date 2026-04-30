"""Tests for NeuralMemRetriever — synchronous path."""
from __future__ import annotations

import pytest
from neuralmem.core.exceptions import NeuralMemError

from neuralmem_langchain import NeuralMemRetriever


def test_returns_documents(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    docs = retriever.invoke("user preferences")
    assert len(docs) == 1
    assert docs[0].page_content == "User prefers Python for backend development"


def test_document_metadata_has_all_fields(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    docs = retriever.invoke("user preferences")
    meta = docs[0].metadata
    assert meta["memory_id"] == "abc123def456"
    assert meta["score"] == pytest.approx(0.85)
    assert meta["retrieval_method"] == "semantic"
    assert meta["memory_type"] == "semantic"
    assert meta["tags"] == ["preference", "technology"]
    assert meta["created_at"] == "2026-04-30T00:00:00+00:00"
    assert meta["user_id"] == "test-user"


def test_empty_results_returns_empty_list(mock_mem):
    mock_mem.recall.return_value = []
    retriever = NeuralMemRetriever(mem=mock_mem)
    docs = retriever.invoke("nothing here")
    assert docs == []


def test_k_param_forwarded_to_recall(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, k=3)
    retriever.invoke("query")
    assert mock_mem.recall.call_args.kwargs["limit"] == 3


def test_user_id_forwarded_to_recall(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, user_id="alice")
    retriever.invoke("query")
    assert mock_mem.recall.call_args.kwargs["user_id"] == "alice"


def test_min_score_forwarded_to_recall(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, min_score=0.7)
    retriever.invoke("query")
    assert mock_mem.recall.call_args.kwargs["min_score"] == pytest.approx(0.7)


def test_default_user_id_is_default(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    retriever.invoke("query")
    assert mock_mem.recall.call_args.kwargs["user_id"] == "default"


def test_neuralmem_error_propagates(mock_mem):
    mock_mem.recall.side_effect = NeuralMemError("recall failed")
    retriever = NeuralMemRetriever(mem=mock_mem)
    with pytest.raises(NeuralMemError, match="recall failed"):
        retriever.invoke("query")
