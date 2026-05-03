"""Unit tests for the LoCoMo benchmark utilities."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure benchmark module is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from locomo_benchmark import (
    EvalResult,
    LoCoMoBenchmarker,
    LoCoMoConversation,
    LoCoMoMessage,
    LoCoMoQuestion,
    mean_reciprocal_rank,
    percentile,
    precision_at_k,
    recall_at_k,
    simple_answer_match,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATASET_PATH = Path(__file__).parent.parent.parent / "benchmarks" / "locomo_sample.json"


@pytest.fixture
def sample_dataset_path():
    """Path to the sample LoCoMo dataset."""
    return str(SAMPLE_DATASET_PATH)


@pytest.fixture
def sample_conversations():
    """Pre-parsed sample conversations for testing."""
    return [
        LoCoMoConversation(
            conversation_id="test_001",
            messages=[
                LoCoMoMessage(role="user", content="I love Python programming"),
                LoCoMoMessage(role="assistant", content="Python is great!"),
                LoCoMoMessage(role="user", content="I use PyTorch for deep learning"),
            ],
            questions=[
                LoCoMoQuestion(
                    question="What programming language does the user love?",
                    answer="Python",
                    evidence_ids=["msg_001_0"],
                ),
                LoCoMoQuestion(
                    question="What framework does the user use for deep learning?",
                    answer="PyTorch",
                    evidence_ids=["msg_001_2"],
                ),
            ],
        ),
        LoCoMoConversation(
            conversation_id="test_002",
            messages=[
                LoCoMoMessage(role="user", content="My cat's name is Whiskers"),
                LoCoMoMessage(role="assistant", content="What a cute name!"),
                LoCoMoMessage(role="user", content="He is 5 years old"),
            ],
            questions=[
                LoCoMoQuestion(
                    question="What is the user's cat's name?",
                    answer="Whiskers",
                    evidence_ids=["msg_002_0"],
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Tests: Dataset loading
# ---------------------------------------------------------------------------

class TestDatasetLoading:
    """Test loading and parsing of LoCoMo JSON datasets."""

    def test_load_sample_dataset(self, sample_dataset_path):
        """The sample dataset should load successfully."""
        conversations = LoCoMoBenchmarker.load_locomo_dataset(sample_dataset_path)
        assert len(conversations) == 5

    def test_conversation_ids_unique(self, sample_dataset_path):
        """Each conversation should have a unique ID."""
        conversations = LoCoMoBenchmarker.load_locomo_dataset(sample_dataset_path)
        ids = [c.conversation_id for c in conversations]
        assert len(ids) == len(set(ids))

    def test_messages_have_content(self, sample_dataset_path):
        """All messages should have non-empty content."""
        conversations = LoCoMoBenchmarker.load_locomo_dataset(sample_dataset_path)
        for conv in conversations:
            for msg in conv.messages:
                assert msg.content, f"Empty message in {conv.conversation_id}"

    def test_questions_have_answers(self, sample_dataset_path):
        """All questions should have non-empty answers."""
        conversations = LoCoMoBenchmarker.load_locomo_dataset(sample_dataset_path)
        for conv in conversations:
            for qa in conv.questions:
                assert qa.question, f"Empty question in {conv.conversation_id}"
                assert qa.answer, f"Empty answer in {conv.conversation_id}"

    def test_min_messages_per_conversation(self, sample_dataset_path):
        """Each conversation should have at least 3 messages."""
        conversations = LoCoMoBenchmarker.load_locomo_dataset(sample_dataset_path)
        for conv in conversations:
            assert len(conv.messages) >= 3, (
                f"{conv.conversation_id} has only {len(conv.messages)} messages"
            )

    def test_min_questions_per_conversation(self, sample_dataset_path):
        """Each conversation should have at least 2 questions."""
        conversations = LoCoMoBenchmarker.load_locomo_dataset(sample_dataset_path)
        for conv in conversations:
            assert len(conv.questions) >= 2, (
                f"{conv.conversation_id} has only {len(conv.questions)} questions"
            )

    def test_load_nonexistent_file(self, tmp_path):
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            LoCoMoBenchmarker.load_locomo_dataset(str(tmp_path / "missing.json"))

    def test_load_malformed_json(self, tmp_path):
        """Loading malformed JSON should raise an error."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            LoCoMoBenchmarker.load_locomo_dataset(str(bad_file))

    def test_load_empty_conversations(self, tmp_path):
        """Loading a dataset with empty conversations list should return empty list."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text('{"conversations": []}')
        conversations = LoCoMoBenchmarker.load_locomo_dataset(str(empty_file))
        assert conversations == []


# ---------------------------------------------------------------------------
# Tests: Dataclass parsing
# ---------------------------------------------------------------------------

class TestConversationDataclass:
    """Test the LoCoMoConversation and related dataclasses."""

    def test_message_defaults(self):
        """LoCoMoMessage should have sensible defaults."""
        msg = LoCoMoMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp == ""

    def test_question_defaults(self):
        """LoCoMoQuestion should have sensible defaults."""
        q = LoCoMoQuestion(question="What?", answer="This")
        assert q.evidence_ids == []

    def test_conversation_creation(self):
        """LoCoMoConversation should be creatable with all fields."""
        conv = LoCoMoConversation(
            conversation_id="c1",
            messages=[LoCoMoMessage(role="user", content="hi")],
            questions=[LoCoMoQuestion(question="q", answer="a", evidence_ids=["e1"])],
        )
        assert conv.conversation_id == "c1"
        assert len(conv.messages) == 1
        assert len(conv.questions) == 1
        assert conv.questions[0].evidence_ids == ["e1"]

    def test_conversation_from_json_dict(self):
        """LoCoMoConversation should parse correctly from a dict (as from JSON)."""
        data = {
            "conversation_id": "conv_xyz",
            "messages": [
                {"role": "user", "content": "hello", "timestamp": "2025-01-01T00:00:00Z"},
                {"role": "assistant", "content": "hi there"},
            ],
            "questions": [
                {"question": "What did the user say?", "answer": "hello"},
            ],
        }
        # Simulate the parsing logic from load_locomo_dataset
        messages = [LoCoMoMessage(**m) for m in data["messages"]]
        questions = [LoCoMoQuestion(**q) for q in data["questions"]]
        conv = LoCoMoConversation(
            conversation_id=data["conversation_id"],
            messages=messages,
            questions=questions,
        )
        assert conv.conversation_id == "conv_xyz"
        assert conv.messages[0].role == "user"
        assert conv.messages[1].timestamp == ""  # missing field default


# ---------------------------------------------------------------------------
# Tests: Metric calculation helpers
# ---------------------------------------------------------------------------

class TestMetricHelpers:
    """Test pure metric calculation functions."""

    def test_recall_at_k_perfect(self):
        """All relevant items in top-K should give recall=1.0."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_recall_at_k_partial(self):
        """Only one of two relevant items in top-K should give recall=0.5."""
        retrieved = ["a", "x", "y", "b", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.5

    def test_recall_at_k_none(self):
        """No relevant items in results should give recall=0.0."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_at_k_empty_relevant(self):
        """Empty relevant set should give recall=0.0."""
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0

    def test_recall_at_k_k_larger_than_results(self):
        """K larger than retrieved list should still work."""
        retrieved = ["a"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=5) == 0.5

    def test_precision_at_k_perfect(self):
        """All top-K items are relevant -> precision=1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_precision_at_k_partial(self):
        """One of three top-3 items relevant -> precision=1/3."""
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        assert abs(precision_at_k(retrieved, relevant, k=3) - 1 / 3) < 1e-6

    def test_precision_at_k_zero_k(self):
        """K=0 should give precision=0.0."""
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0

    def test_mrr_first_result(self):
        """First result is relevant -> MRR=1.0."""
        assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_mrr_second_result(self):
        """Second result is relevant -> MRR=0.5."""
        assert mean_reciprocal_rank(["x", "a", "c"], {"a"}) == 0.5

    def test_mrr_third_result(self):
        """Third result is relevant -> MRR=1/3."""
        result = mean_reciprocal_rank(["x", "y", "a"], {"a"})
        assert abs(result - 1 / 3) < 1e-6

    def test_mrr_no_relevant(self):
        """No relevant results -> MRR=0.0."""
        assert mean_reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_mrr_multiple_relevant(self):
        """Multiple relevant -> uses the first (highest rank)."""
        assert mean_reciprocal_rank(["b", "a", "c"], {"a", "b"}) == 1.0

    def test_percentile_empty(self):
        """Empty list -> 0.0."""
        assert percentile([], 50) == 0.0

    def test_percentile_single(self):
        """Single element -> that element for any percentile."""
        assert percentile([5.0], 50) == 5.0
        assert percentile([5.0], 99) == 5.0

    def test_percentile_sorted(self):
        """Percentile from a sorted list."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(data, 50) == 3.0  # index 2
        assert percentile(data, 0) == 1.0   # index 0

    def test_percentile_95th(self):
        """95th percentile from 100 items."""
        data = list(range(100))
        assert percentile(data, 95) == 95  # index 95


# ---------------------------------------------------------------------------
# Tests: Answer matching
# ---------------------------------------------------------------------------

class TestAnswerMatching:
    """Test the simple_answer_match heuristic."""

    def test_exact_match(self):
        """Exact substring match should return True."""
        assert simple_answer_match("What language?", "Python", ["User loves Python"])

    def test_term_match(self):
        """Key terms from answer appearing in text should match."""
        assert simple_answer_match(
            "What restaurant?",
            "Bella Notte on Main Street",
            ["Went to Bella Notte restaurant last night"],
        )

    def test_no_match(self):
        """Unrelated text should not match."""
        assert not simple_answer_match(
            "What language?",
            "Python",
            ["User went to the store and bought apples"],
        )

    def test_short_answer_fallback(self):
        """Short answers (< 4 chars) use full substring match."""
        assert simple_answer_match("?", "Rust", ["switching to Rust from Python"])

    def test_partial_terms_match(self):
        """Matching half the key terms should be sufficient."""
        assert simple_answer_match(
            "What?",
            "Carrot cake from Sweet Delights bakery",
            ["Ordered carrot cake at the bakery on Elm Street"],
        )


# ---------------------------------------------------------------------------
# Tests: EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:
    """Test EvalResult dataclass."""

    def test_defaults(self):
        """EvalResult should have sensible defaults."""
        r = EvalResult()
        assert r.recall_at_k == {}
        assert r.precision_at_k == {}
        assert r.mrr == 0.0
        assert r.total_questions == 0

    def test_with_values(self):
        """EvalResult should store values correctly."""
        r = EvalResult(
            recall_at_k={1: 0.5, 3: 0.8, 5: 0.9},
            precision_at_k={1: 0.5, 3: 0.3, 5: 0.2},
            mrr=0.75,
            avg_latency_ms=12.5,
            p50_latency_ms=10.0,
            p95_latency_ms=25.0,
            total_questions=20,
        )
        assert r.recall_at_k[5] == 0.9
        assert r.mrr == 0.75
        assert r.total_questions == 20


# ---------------------------------------------------------------------------
# Tests: Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    """Test generate_report static method."""

    def test_report_structure(self):
        """Report should contain expected top-level keys."""
        result = EvalResult(
            recall_at_k={1: 0.5, 3: 0.7},
            precision_at_k={1: 0.5, 3: 0.3},
            mrr=0.6,
            avg_latency_ms=10.0,
            p50_latency_ms=8.0,
            p95_latency_ms=20.0,
            total_questions=10,
            total_conversations=2,
        )
        report = LoCoMoBenchmarker.generate_report(result)
        assert report["benchmark"] == "LoCoMo"
        assert "summary" in report
        assert "recall_at_k" in report
        assert "precision_at_k" in report
        assert "latency" in report

    def test_report_saves_to_file(self, tmp_path):
        """Report should be saved to disk when output_path is given."""
        result = EvalResult(total_questions=5, total_conversations=1, mrr=0.4)
        output = str(tmp_path / "report.json")
        LoCoMoBenchmarker.generate_report(result, output_path=output)

        with open(output) as f:
            data = json.load(f)
        assert data["benchmark"] == "LoCoMo"
        assert data["summary"]["mrr"] == 0.4

    def test_report_returns_dict(self):
        """generate_report should return a dict even without output_path."""
        result = EvalResult()
        report = LoCoMoBenchmarker.generate_report(result)
        assert isinstance(report, dict)
