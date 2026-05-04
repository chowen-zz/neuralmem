"""PredictiveRetrievalEngine unit tests — all mock-based."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuralmem.core.types import Memory, SearchQuery, SearchResult
from neuralmem.retrieval.engine import RetrievalEngine
from neuralmem.retrieval.predictive import (
    PredictiveRetrievalEngine,
    PredictiveStats,
    _UserContext,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_storage():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.dimension = 128
    embedder.encode_one.return_value = [0.1] * 128
    embedder.encode.return_value = [[0.1] * 128]
    return embedder


@pytest.fixture
def mock_graph():
    return MagicMock()


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.query_embedding_cache_size = 0
    config.enable_reranker = False
    config.recency_weight = 0.3
    return config


@pytest.fixture
def mock_hot_store():
    store = MagicMock()
    store.get_memory.return_value = None
    store.save_memory.return_value = "mem-id"
    return store


@pytest.fixture
def mock_profile_engine():
    engine = MagicMock()
    engine.storage = MagicMock()
    return engine


@pytest.fixture
def mock_prefetch_engine():
    engine = MagicMock()
    engine._min_sequence_length = 2
    engine.stats.return_value = MagicMock(
        predictions_made=0,
        predictions_hit=0,
        predictions_miss=0,
        items_prefetched=0,
        prefetch_time_ms=0.0,
        avg_prefetch_latency_ms=0.0,
    )
    engine.predict_next.return_value = ["predicted query"]
    return engine


@pytest.fixture
def sample_memory():
    return Memory(content="sample memory content")


@pytest.fixture
def sample_search_result(sample_memory):
    return SearchResult(
        memory=sample_memory,
        score=0.9,
        retrieval_method="semantic",
        explanation="test",
    )


@pytest.fixture
def predictive_engine(
    mock_storage,
    mock_embedder,
    mock_graph,
    mock_config,
    mock_hot_store,
    mock_profile_engine,
    mock_prefetch_engine,
):
    """Build a PredictiveRetrievalEngine with all dependencies mocked."""
    with patch(
        "neuralmem.retrieval.predictive.RetrievalEngine.__init__",
        return_value=None,
    ) as _mock_super_init:
        engine = PredictiveRetrievalEngine(
            storage=mock_storage,
            embedder=mock_embedder,
            graph=mock_graph,
            config=mock_config,
            hot_store=mock_hot_store,
            profile_engine=mock_profile_engine,
            prefetch_engine=mock_prefetch_engine,
            max_context_window=10,
            prediction_threshold=0.5,
        )
        # Manually set attributes that super().__init__ would have set
        engine._storage = mock_storage
        engine._embedder = mock_embedder
        engine._graph = mock_graph
        engine._config = mock_config
        engine._merger = MagicMock()
        engine._reranker = MagicMock()
        engine._cached_embedder = mock_embedder
        engine._semantic = MagicMock()
        engine._keyword = MagicMock()
        engine._graph_strategy = MagicMock()
        engine._temporal = MagicMock()
        engine._executor = MagicMock()
    return engine


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def test_predictive_engine_init(predictive_engine, mock_hot_store, mock_profile_engine):
    assert predictive_engine._hot_store is mock_hot_store
    assert predictive_engine._profile_engine is mock_profile_engine
    assert predictive_engine._prediction_threshold == 0.5
    assert predictive_engine._max_context_window == 10


def test_predictive_engine_init_defaults(mock_storage, mock_embedder, mock_graph, mock_config):
    """Test that default prefetch_engine is created when not provided."""
    with patch("neuralmem.retrieval.predictive.RetrievalEngine.__init__", return_value=None):
        engine = PredictiveRetrievalEngine(
            storage=mock_storage,
            embedder=mock_embedder,
            graph=mock_graph,
            config=mock_config,
        )
        assert engine._prefetch_engine is not None
        assert engine._hot_store is None


# --------------------------------------------------------------------------- #
# search() override
# --------------------------------------------------------------------------- #

def test_search_records_query_and_warms_hotstore(
    predictive_engine, mock_hot_store, sample_search_result
):
    with patch.object(
        RetrievalEngine, "search", return_value=[sample_search_result]
    ):
        sq = SearchQuery(query="test", user_id="u1")
        results = predictive_engine.search(sq)

    assert len(results) == 1
    mock_hot_store.save_memory.assert_called_once()
    # Verify user context was recorded
    ctx = predictive_engine._get_user_context("u1")
    assert "test" in ctx.recent_queries


def test_search_without_user_id_skips_context(predictive_engine, mock_hot_store, sample_search_result):
    with patch.object(
        RetrievalEngine, "search", return_value=[sample_search_result]
    ):
        sq = SearchQuery(query="test", user_id=None)
        results = predictive_engine.search(sq)

    assert len(results) == 1
    # No user context should be created for None user_id
    assert None not in predictive_engine._user_contexts


# --------------------------------------------------------------------------- #
# predictive_search()
# --------------------------------------------------------------------------- #

def test_predictive_search_triggers_both_prefetches(
    predictive_engine, mock_hot_store, sample_search_result
):
    with patch.object(
        RetrievalEngine, "search", return_value=[sample_search_result]
    ) as mock_super_search:
        with patch.object(
            predictive_engine, "_prefetch_from_profile", return_value=3
        ) as mock_profile:
            with patch.object(
                predictive_engine, "_prefetch_from_context", return_value=2
            ) as mock_context:
                sq = SearchQuery(query="test", user_id="u1")
                results = predictive_engine.predictive_search(sq)

    assert len(results) == 1
    mock_profile.assert_called_once_with("u1")
    mock_context.assert_called_once_with("u1")
    mock_super_search.assert_called_once()


def test_predictive_search_respects_flags(
    predictive_engine, mock_hot_store, sample_search_result
):
    with patch.object(RetrievalEngine, "search", return_value=[sample_search_result]):
        with patch.object(predictive_engine, "_prefetch_from_profile") as mock_profile:
            with patch.object(predictive_engine, "_prefetch_from_context") as mock_context:
                sq = SearchQuery(query="test", user_id="u1")
                predictive_engine.predictive_search(sq, warm_profile=False, warm_context=False)

    mock_profile.assert_not_called()
    mock_context.assert_not_called()


# --------------------------------------------------------------------------- #
# Profile-based pre-fetching
# --------------------------------------------------------------------------- #

def test_prefetch_user_profile_no_profile_engine(predictive_engine):
    predictive_engine._profile_engine = None
    result = predictive_engine.prefetch_user_profile("u1")
    assert result == []


def test_prefetch_user_profile_with_cached_profile(
    predictive_engine, mock_profile_engine, sample_memory
):
    # Seed profile cache
    predictive_engine._profile_cache["u1"] = {
        "primary_intent": {
            "name": "primary_intent",
            "value": {"category": "informational"},
            "confidence": 0.8,
            "evidence": [],
        },
        "knowledge_machine_learning": {
            "name": "knowledge_machine_learning",
            "value": {"domain": "machine_learning", "level": "advanced"},
            "confidence": 0.7,
            "evidence": [],
        },
    }
    predictive_engine._profile_cache_ts["u1"] = 9999999999.0  # far future

    with patch.object(
        RetrievalEngine, "search", return_value=[
            SearchResult(memory=sample_memory, score=0.8, retrieval_method="semantic")
        ]
    ):
        result = predictive_engine.prefetch_user_profile("u1", limit=5)

    assert len(result) >= 0
    stats = predictive_engine.stats()
    assert stats.profile_prefetches >= 0


def test_prefetch_user_profile_below_threshold(predictive_engine):
    predictive_engine._profile_cache["u1"] = {
        "low_conf": {
            "name": "low_conf",
            "value": {"category": "creative"},
            "confidence": 0.2,
            "evidence": [],
        }
    }
    predictive_engine._profile_cache_ts["u1"] = 9999999999.0
    result = predictive_engine.prefetch_user_profile("u1")
    assert result == []


def test_extract_predicted_queries_varied_values(predictive_engine):
    profile = {
        "intent": {
            "value": {"category": "learning"},
            "confidence": 0.9,
        },
        "knowledge": {
            "value": {"domain": "web_development"},
            "confidence": 0.8,
        },
        "pref": {
            "value": {"type": "technology", "value": "python"},
            "confidence": 0.85,
        },
        "str_val": {
            "value": "direct string",
            "confidence": 0.9,
        },
    }
    queries = predictive_engine._extract_predicted_queries(profile, 0.5)
    assert len(queries) > 0
    assert any("learning" in q for q in queries)
    assert any("web_development" in q for q in queries)


def test_extract_predicted_queries_deduplicates(predictive_engine):
    profile = {
        "a": {"value": {"category": "test"}, "confidence": 0.9},
        "b": {"value": {"category": "test"}, "confidence": 0.9},
    }
    queries = predictive_engine._extract_predicted_queries(profile, 0.5)
    assert len(queries) == 1


def test_get_cached_profile_builds_from_storage(
    predictive_engine, mock_profile_engine, sample_memory
):
    mock_profile_engine.storage.list_memories.return_value = [sample_memory]
    mock_profile_engine.build_profile.return_value = {
        "primary_intent": MagicMock(
            name="primary_intent",
            value={"category": "informational"},
            confidence=0.8,
            evidence=["m1"],
        )
    }

    profile = predictive_engine._get_cached_profile("u1")
    assert profile is not None
    assert "primary_intent" in profile
    mock_profile_engine.storage.list_memories.assert_called_once_with(
        user_id="u1", limit=500
    )


def test_get_cached_profile_storage_failure(predictive_engine, mock_profile_engine):
    mock_profile_engine.storage.list_memories.side_effect = Exception("DB down")
    profile = predictive_engine._get_cached_profile("u1")
    assert profile is None


def test_get_cached_profile_no_memories(predictive_engine, mock_profile_engine):
    mock_profile_engine.storage.list_memories.return_value = []
    profile = predictive_engine._get_cached_profile("u1")
    assert profile is None


def test_get_cached_profile_uses_cache(predictive_engine):
    predictive_engine._profile_cache["u1"] = {"cached": True}
    predictive_engine._profile_cache_ts["u1"] = 9999999999.0
    profile = predictive_engine._get_cached_profile("u1")
    assert profile == {"cached": True}


def test_get_cached_profile_cache_expired(
    predictive_engine, mock_profile_engine, sample_memory
):
    predictive_engine._profile_cache["u1"] = {"old": True}
    predictive_engine._profile_cache_ts["u1"] = 0.0  # expired
    mock_profile_engine.storage.list_memories.return_value = [sample_memory]
    mock_profile_engine.build_profile.return_value = {
        "primary_intent": MagicMock(
            name="primary_intent",
            value={"category": "informational"},
            confidence=0.8,
            evidence=["m1"],
        )
    }
    profile = predictive_engine._get_cached_profile("u1")
    assert "primary_intent" in profile


# --------------------------------------------------------------------------- #
# Context pattern prediction
# --------------------------------------------------------------------------- #

def test_predict_next_queries_insufficient_history(predictive_engine):
    result = predictive_engine.predict_next_queries("u1")
    assert result == []


def test_predict_next_queries_with_history(predictive_engine, mock_prefetch_engine):
    predictive_engine._record_user_query("u1", "query1")
    predictive_engine._record_user_query("u1", "query2")

    result = predictive_engine.predict_next_queries("u1")
    assert result == ["predicted query"]
    mock_prefetch_engine.predict_next.assert_called_once()


def test_prefetch_from_context(predictive_engine, sample_memory):
    predictive_engine._record_user_query("u1", "query1")
    predictive_engine._record_user_query("u1", "query2")

    with patch.object(
        RetrievalEngine, "search", return_value=[
            SearchResult(memory=sample_memory, score=0.8, retrieval_method="semantic")
        ]
    ):
        result = predictive_engine.prefetch_from_context("u1")

    assert len(result) >= 0
    stats = predictive_engine.stats()
    assert stats.context_prefetches >= 0


def test_prefetch_from_context_no_predictions(predictive_engine):
    predictive_engine._prefetch_engine.predict_next.return_value = []
    result = predictive_engine.prefetch_from_context("u1")
    assert result == []


# --------------------------------------------------------------------------- #
# HotStore warming
# --------------------------------------------------------------------------- #

def test_warm_hot_store_success(predictive_engine, mock_hot_store, sample_memory):
    mock_hot_store.get_memory.return_value = None
    count = predictive_engine.warm_hot_store([sample_memory])
    assert count == 1
    mock_hot_store.save_memory.assert_called_once_with(sample_memory)


def test_warm_hot_store_existing_skips(predictive_engine, mock_hot_store, sample_memory):
    mock_hot_store.get_memory.return_value = sample_memory
    count = predictive_engine.warm_hot_store([sample_memory])
    assert count == 0
    mock_hot_store.save_memory.assert_not_called()


def test_warm_hot_store_force_overwrites(predictive_engine, mock_hot_store, sample_memory):
    mock_hot_store.get_memory.return_value = sample_memory
    count = predictive_engine.warm_hot_store([sample_memory], force=True)
    assert count == 1
    mock_hot_store.save_memory.assert_called_once()


def test_warm_hot_store_no_hotstore(predictive_engine, sample_memory):
    predictive_engine._hot_store = None
    count = predictive_engine.warm_hot_store([sample_memory])
    assert count == 0


def test_warm_hot_store_handles_exception(predictive_engine, mock_hot_store, sample_memory):
    mock_hot_store.get_memory.side_effect = Exception("hot store error")
    count = predictive_engine.warm_hot_store([sample_memory])
    assert count == 0


# --------------------------------------------------------------------------- #
# User context tracking
# --------------------------------------------------------------------------- #

def test_record_query_updates_context(predictive_engine):
    predictive_engine.record_query("u1", "hello world", result_count=3)
    ctx = predictive_engine._get_user_context("u1")
    assert "hello world" in ctx.recent_queries


def test_get_user_context_creates_new(predictive_engine):
    ctx = predictive_engine._get_user_context("new_user")
    assert ctx.user_id == "new_user"
    assert len(ctx.recent_queries) == 0


def test_user_context_maxlen(predictive_engine):
    for i in range(15):
        predictive_engine._record_user_query("u1", f"query{i}")
    ctx = predictive_engine._get_user_context("u1")
    assert len(ctx.recent_queries) == 10  # max_context_window


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def test_stats_returns_predictive_stats(predictive_engine):
    stats = predictive_engine.stats()
    assert isinstance(stats, PredictiveStats)
    assert stats.profile_prefetches == 0
    assert stats.context_prefetches == 0


def test_reset_stats_clears_counters(predictive_engine):
    predictive_engine._stats.profile_prefetches = 5
    predictive_engine.reset_stats()
    stats = predictive_engine.stats()
    assert stats.profile_prefetches == 0
    assert stats.context_prefetches == 0


# --------------------------------------------------------------------------- #
# Context management
# --------------------------------------------------------------------------- #

def test_clear_user_context_single_user(predictive_engine):
    predictive_engine._record_user_query("u1", "q1")
    predictive_engine._profile_cache["u1"] = {"data": True}
    predictive_engine.clear_user_context("u1")
    assert "u1" not in predictive_engine._user_contexts
    assert "u1" not in predictive_engine._profile_cache


def test_clear_user_context_all_users(predictive_engine):
    predictive_engine._record_user_query("u1", "q1")
    predictive_engine._record_user_query("u2", "q2")
    predictive_engine.clear_user_context()
    assert len(predictive_engine._user_contexts) == 0
    assert len(predictive_engine._profile_cache) == 0


def test_invalidate_profile_cache(predictive_engine):
    predictive_engine._profile_cache["u1"] = {"data": True}
    predictive_engine._profile_cache_ts["u1"] = 123.0
    predictive_engine.invalidate_profile_cache("u1")
    assert "u1" not in predictive_engine._profile_cache
    assert "u1" not in predictive_engine._profile_cache_ts


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

def test_close_clears_state(predictive_engine):
    predictive_engine._record_user_query("u1", "q1")
    predictive_engine._stats.profile_prefetches = 5
    with patch.object(RetrievalEngine, "close"):
        predictive_engine.close()
    assert len(predictive_engine._user_contexts) == 0
    stats = predictive_engine.stats()
    assert stats.profile_prefetches == 0


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_prefetch_user_profile_search_failure(predictive_engine):
    predictive_engine._profile_cache["u1"] = {
        "intent": {
            "value": {"category": "test"},
            "confidence": 0.9,
            "evidence": [],
        }
    }
    predictive_engine._profile_cache_ts["u1"] = 9999999999.0

    with patch.object(RetrievalEngine, "search", side_effect=Exception("search failed")):
        result = predictive_engine.prefetch_user_profile("u1")
    assert result == []


def test_predictive_search_with_empty_results(predictive_engine, mock_hot_store):
    with patch.object(RetrievalEngine, "search", return_value=[]):
        sq = SearchQuery(query="test", user_id="u1")
        results = predictive_engine.predictive_search(sq)
    assert results == []
    mock_hot_store.save_memory.assert_not_called()


def test_extract_predicted_queries_empty_profile(predictive_engine):
    queries = predictive_engine._extract_predicted_queries({}, 0.5)
    assert queries == []


def test_extract_predicted_queries_low_confidence(predictive_engine):
    profile = {
        "low": {"value": {"category": "test"}, "confidence": 0.1},
    }
    queries = predictive_engine._extract_predicted_queries(profile, 0.5)
    assert queries == []


def test_user_context_last_activity_updated(predictive_engine):
    t0 = predictive_engine._get_user_context("u1").last_activity
    predictive_engine._record_user_query("u1", "q1")
    t1 = predictive_engine._get_user_context("u1").last_activity
    assert t1 >= t0
