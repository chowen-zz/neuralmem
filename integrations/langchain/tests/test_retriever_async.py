"""Tests for NeuralMemRetriever — asynchronous path."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from neuralmem_langchain import NeuralMemRetriever


@pytest.mark.asyncio
async def test_async_returns_documents(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem)
    docs = await retriever.ainvoke("user preferences")
    assert len(docs) == 1
    assert docs[0].page_content == "User prefers Python for backend development"


@pytest.mark.asyncio
async def test_async_uses_to_thread(mock_mem):
    """Verify asyncio.to_thread() is used so recall() doesn't block the event loop."""
    retriever = NeuralMemRetriever(mem=mock_mem)
    with patch(
        "neuralmem_langchain.retriever.asyncio.to_thread",
        wraps=asyncio.to_thread,
    ) as mock_thread:
        await retriever.ainvoke("query")
        mock_thread.assert_called_once()


@pytest.mark.asyncio
async def test_async_empty_results(mock_mem):
    mock_mem.recall.return_value = []
    retriever = NeuralMemRetriever(mem=mock_mem)
    docs = await retriever.ainvoke("nothing")
    assert docs == []


@pytest.mark.asyncio
async def test_async_user_id_forwarded(mock_mem):
    retriever = NeuralMemRetriever(mem=mock_mem, user_id="bob")
    await retriever.ainvoke("query")
    assert mock_mem.recall.call_args.kwargs["user_id"] == "bob"
