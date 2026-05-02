"""Tests for NeuralMemRetriever."""
from __future__ import annotations

import pytest
from neuralmem.core.exceptions import NeuralMemError

from neuralmem_llamaindex import NeuralMemRetriever


def test_returns_nodes(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    nodes = retriever.retrieve("user preferences")
    assert len(nodes) == 1


def test_node_text_is_memory_content(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    nodes = retriever.retrieve("user preferences")
    assert nodes[0].node.text == "User prefers Python for backend development"


def test_node_score(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    nodes = retriever.retrieve("user preferences")
    assert nodes[0].score == pytest.approx(0.85)


def test_node_metadata_fields(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    nodes = retriever.retrieve("user preferences")
    meta = nodes[0].node.metadata
    assert meta["memory_id"] == "abc123def456"
    assert meta["memory_type"] == "semantic"
    assert meta["tags"] == ["preference", "technology"]
    assert meta["created_at"] == "2026-04-30T00:00:00+00:00"
    assert meta["user_id"] == "test-user"


def test_empty_results(mock_mem):
    mock_mem.recall.return_value = []
    retriever = NeuralMemRetriever(mem=mock_mem)
    nodes = retriever.retrieve("nothing here")
    assert nodes == []


def test_k_forwarded_to_recall(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, k=3)
    retriever.retrieve("query")
    assert mock_mem.recall.call_args.kwargs["limit"] == 3


def test_user_id_forwarded_to_recall(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, user_id="alice")
    retriever.retrieve("query")
    assert mock_mem.recall.call_args.kwargs["user_id"] == "alice"


def test_error_propagates(mock_mem):
    mock_mem.recall.side_effect = NeuralMemError("recall failed")
    retriever = NeuralMemRetriever(mem=mock_mem)
    with pytest.raises(NeuralMemError, match="recall failed"):
        retriever.retrieve("query")
