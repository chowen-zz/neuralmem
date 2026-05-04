"""Unit tests for NeuralMem V1.7 CommunityAnalytics — all mock-based.

Covers:
  • RepoMetrics / Contributor / GrowthFunnel / TrendReport dataclasses
  • GitHubAPIClient (mock HTTP)
  • CommunityAnalytics (metrics tracking, funnel, trend reports, health score)
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from neuralmem.community.analytics import (
    CommunityAnalytics,
    Contributor,
    GitHubAPIClient,
    GrowthFunnel,
    RepoMetrics,
    TrendReport,
)


# =============================================================================
# Dataclass tests
# =============================================================================

class TestRepoMetrics:
    def test_to_dict(self):
        m = RepoMetrics(repo="o/r", stars=10, forks=2, open_issues=3, closed_issues=1, watchers=5, subscribers=4)
        d = m.to_dict()
        assert d["repo"] == "o/r"
        assert d["stars"] == 10
        assert "created_at" in d


class TestContributor:
    def test_to_dict_with_dates(self):
        c = Contributor(
            username="alice",
            contributions=42,
            first_contribution=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_contribution=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        d = c.to_dict()
        assert d["username"] == "alice"
        assert d["first_contribution"] == "2024-01-01T00:00:00+00:00"

    def test_to_dict_without_dates(self):
        c = Contributor(username="bob", contributions=5)
        d = c.to_dict()
        assert d["first_contribution"] is None


class TestGrowthFunnel:
    def test_conversion_rate_basic(self):
        f = GrowthFunnel(date=datetime.now(timezone.utc), visitors=100, stars_gained=10)
        assert f.conversion_rate("visitors", "stars_gained") == 10.0

    def test_conversion_rate_zero_from(self):
        f = GrowthFunnel(date=datetime.now(timezone.utc), visitors=0, stars_gained=5)
        assert f.conversion_rate("visitors", "stars_gained") == 0.0

    def test_to_dict(self):
        f = GrowthFunnel(date=datetime(2024, 1, 1, tzinfo=timezone.utc), visitors=100)
        d = f.to_dict()
        assert d["visitors"] == 100
        assert d["date"] == "2024-01-01T00:00:00+00:00"


class TestTrendReport:
    def test_star_velocity(self):
        r = TrendReport(
            repo="o/r",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 11, tzinfo=timezone.utc),
            metrics_snapshots=[
                RepoMetrics(repo="o/r", stars=100, forks=0, open_issues=0, closed_issues=0, watchers=0, subscribers=0),
            ],
        )
        assert r.star_velocity() == 10.0

    def test_issue_resolution_rate(self):
        r = TrendReport(
            repo="o/r",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            metrics_snapshots=[
                RepoMetrics(repo="o/r", stars=0, forks=0, open_issues=2, closed_issues=8, watchers=0, subscribers=0),
            ],
        )
        assert r.issue_resolution_rate() == 80.0

    def test_issue_resolution_rate_zero_total(self):
        r = TrendReport(
            repo="o/r",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            metrics_snapshots=[
                RepoMetrics(repo="o/r", stars=0, forks=0, open_issues=0, closed_issues=0, watchers=0, subscribers=0),
            ],
        )
        assert r.issue_resolution_rate() == 0.0

    def test_top_contributors(self):
        r = TrendReport(
            repo="o/r",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            contributors=[
                Contributor(username="a", contributions=10),
                Contributor(username="b", contributions=50),
                Contributor(username="c", contributions=30),
            ],
        )
        top = r.top_contributors(limit=2)
        assert len(top) == 2
        assert top[0].username == "b"
        assert top[1].username == "c"

    def test_to_dict(self):
        r = TrendReport(
            repo="o/r",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        d = r.to_dict()
        assert d["repo"] == "o/r"
        assert d["star_velocity"] == 0.0
        assert d["issue_resolution_rate"] == 0.0


# =============================================================================
# GitHubAPIClient
# =============================================================================

class TestGitHubAPIClientMockHTTP:
    def test_get_repo_metrics(self):
        mock_http = MagicMock(return_value={
            "stargazers_count": 150,
            "forks_count": 30,
            "open_issues_count": 12,
            "watchers_count": 45,
            "subscribers_count": 20,
        })
        client = GitHubAPIClient(token="tok", http_client=mock_http)
        metrics = client.get_repo_metrics("neuralmem", "neuralmem")
        assert metrics.repo == "neuralmem/neuralmem"
        assert metrics.stars == 150
        assert metrics.forks == 30
        assert metrics.open_issues == 12
        assert metrics.watchers == 45
        assert metrics.subscribers == 20
        mock_http.assert_called_once()

    def test_get_contributors(self):
        mock_http = MagicMock(return_value=[
            {"login": "alice", "contributions": 42, "avatar_url": "http://a", "html_url": "http://h"},
            {"login": "bob", "contributions": 7},
        ])
        client = GitHubAPIClient(http_client=mock_http)
        contributors = client.get_contributors("o", "r")
        assert len(contributors) == 2
        assert contributors[0].username == "alice"
        assert contributors[0].contributions == 42
        assert contributors[1].username == "bob"

    def test_get_contributors_empty(self):
        mock_http = MagicMock(return_value={})
        client = GitHubAPIClient(http_client=mock_http)
        contributors = client.get_contributors("o", "r")
        assert contributors == []

    def test_get_closed_issues_count(self):
        mock_http = MagicMock(return_value={"total_count": 88})
        client = GitHubAPIClient(http_client=mock_http)
        count = client.get_closed_issues_count("o", "r")
        assert count == 88

    def test_call_log(self):
        mock_http = MagicMock(return_value={})
        client = GitHubAPIClient(http_client=mock_http)
        client.get_repo_metrics("o", "r")
        log = client.get_call_log()
        assert len(log) == 1
        assert log[0]["method"] == "GET"
        assert "repos/o/r" in log[0]["url"]

    def test_no_http_client_raises(self):
        client = GitHubAPIClient()
        with pytest.raises(RuntimeError, match="No http_client configured"):
            client.get_repo_metrics("o", "r")


# =============================================================================
# CommunityAnalytics
# =============================================================================

class TestCommunityAnalyticsTrack:
    def test_track_current_metrics(self):
        mock_metrics = RepoMetrics(repo="o/r", stars=10, forks=2, open_issues=1, closed_issues=0, watchers=5, subscribers=3)
        mock_client = MagicMock(spec=GitHubAPIClient)
        mock_client.get_repo_metrics.return_value = mock_metrics
        mock_client.get_closed_issues_count.return_value = 5

        analytics = CommunityAnalytics(repo="o/r", github_client=mock_client)
        metrics = analytics.track_current_metrics()

        assert metrics.stars == 10
        assert metrics.closed_issues == 5
        mock_client.get_repo_metrics.assert_called_once_with("o", "r")
        mock_client.get_closed_issues_count.assert_called_once_with("o", "r")
        assert len(analytics.get_metrics_history()) == 1

    def test_track_current_metrics_closed_issues_fallback(self):
        mock_metrics = RepoMetrics(repo="o/r", stars=10, forks=2, open_issues=1, closed_issues=0, watchers=5, subscribers=3)
        mock_client = MagicMock(spec=GitHubAPIClient)
        mock_client.get_repo_metrics.return_value = mock_metrics
        mock_client.get_closed_issues_count.side_effect = Exception("API error")

        analytics = CommunityAnalytics(repo="o/r", github_client=mock_client)
        metrics = analytics.track_current_metrics()
        assert metrics.closed_issues == 0

    def test_track_contributors(self):
        mock_client = MagicMock(spec=GitHubAPIClient)
        mock_client.get_contributors.return_value = [
            Contributor(username="alice", contributions=10),
            Contributor(username="bob", contributions=5),
        ]
        analytics = CommunityAnalytics(repo="o/r", github_client=mock_client)
        contributors = analytics.track_contributors()
        assert len(contributors) == 2
        assert analytics.get_contributor_leaderboard()[0].username == "alice"


class TestCommunityAnalyticsFunnel:
    def test_record_funnel(self):
        analytics = CommunityAnalytics(repo="o/r")
        funnel = GrowthFunnel(date=datetime.now(timezone.utc), visitors=1000, stars_gained=50)
        recorded = analytics.record_funnel(funnel)
        assert recorded.visitors == 1000
        assert len(analytics.get_funnel_history()) == 1

    def test_generate_funnel(self):
        analytics = CommunityAnalytics(repo="o/r")
        funnel = analytics.generate_funnel(visitors=500, stars_gained=25, forks_gained=5)
        assert funnel.visitors == 500
        assert funnel.stars_gained == 25
        assert len(analytics.get_funnel_history()) == 1


class TestCommunityAnalyticsTrendReport:
    def test_build_trend_report_filters_by_date(self):
        analytics = CommunityAnalytics(repo="o/r")
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)

        analytics._metrics_history.append(
            RepoMetrics(repo="o/r", stars=10, forks=0, open_issues=0, closed_issues=0, watchers=0, subscribers=0, created_at=old)
        )
        analytics._metrics_history.append(
            RepoMetrics(repo="o/r", stars=20, forks=0, open_issues=0, closed_issues=0, watchers=0, subscribers=0, created_at=now)
        )
        analytics._funnel_history.append(
            GrowthFunnel(date=old, visitors=100)
        )
        analytics._funnel_history.append(
            GrowthFunnel(date=now, visitors=200)
        )

        report = analytics.build_trend_report(start_date=now - timedelta(days=30), end_date=now)
        assert len(report.metrics_snapshots) == 1
        assert len(report.funnels) == 1
        assert report.metrics_snapshots[0].stars == 20
        assert report.funnels[0].visitors == 200

    def test_build_trend_report_default_range(self):
        analytics = CommunityAnalytics(repo="o/r")
        report = analytics.build_trend_report()
        assert report.repo == "o/r"
        assert (report.end_date - report.start_date).days == 30


class TestCommunityAnalyticsStarGrowth:
    def test_get_star_growth_filters_by_days(self):
        analytics = CommunityAnalytics(repo="o/r")
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)
        analytics._metrics_history.append(
            RepoMetrics(repo="o/r", stars=5, forks=0, open_issues=0, closed_issues=0, watchers=0, subscribers=0, created_at=old)
        )
        analytics._metrics_history.append(
            RepoMetrics(repo="o/r", stars=15, forks=0, open_issues=0, closed_issues=0, watchers=0, subscribers=0, created_at=now)
        )
        growth = analytics.get_star_growth(days=30)
        assert len(growth) == 1
        assert growth[0][1] == 15


class TestCommunityAnalyticsHealthScore:
    def test_health_score_no_metrics(self):
        analytics = CommunityAnalytics(repo="o/r")
        score = analytics.get_health_score()
        assert score["score"] == 0
        assert "No metrics" in score["reason"]

    def test_health_score_with_metrics(self):
        analytics = CommunityAnalytics(repo="o/r")
        analytics._metrics_history.append(
            RepoMetrics(repo="o/r", stars=100, forks=20, open_issues=5, closed_issues=15, watchers=10, subscribers=5)
        )
        analytics._contributor_history = [
            Contributor(username="a", contributions=10),
            Contributor(username="b", contributions=5),
        ]
        score = analytics.get_health_score()
        assert score["score"] > 0
        assert score["score"] <= 100
        assert score["resolution_rate"] == 75.0
        assert score["contributor_count"] == 2


class TestCommunityAnalyticsExport:
    def test_export_report(self, tmp_path: Path):
        analytics = CommunityAnalytics(repo="o/r")
        report = TrendReport(
            repo="o/r",
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            metrics_snapshots=[
                RepoMetrics(repo="o/r", stars=10, forks=0, open_issues=0, closed_issues=0, watchers=0, subscribers=0),
            ],
            funnels=[
                GrowthFunnel(date=datetime(2024, 1, 1, tzinfo=timezone.utc), visitors=100),
            ],
            contributors=[
                Contributor(username="alice", contributions=5),
            ],
        )
        path = analytics.export_report(report, str(tmp_path / "report.json"))
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["repo"] == "o/r"
        assert data["star_velocity"] == 10.0
        assert len(data["top_contributors"]) == 1
        assert len(data["metrics_snapshots"]) == 1
        assert len(data["funnels"]) == 1


class TestCommunityAnalyticsHistory:
    def test_get_metrics_history_returns_copy(self):
        analytics = CommunityAnalytics(repo="o/r")
        analytics._metrics_history.append(
            RepoMetrics(repo="o/r", stars=1, forks=0, open_issues=0, closed_issues=0, watchers=0, subscribers=0)
        )
        hist = analytics.get_metrics_history()
        assert len(hist) == 1
        hist.clear()
        assert len(analytics.get_metrics_history()) == 1

    def test_get_funnel_history_returns_copy(self):
        analytics = CommunityAnalytics(repo="o/r")
        analytics._funnel_history.append(
            GrowthFunnel(date=datetime.now(timezone.utc), visitors=1)
        )
        hist = analytics.get_funnel_history()
        assert len(hist) == 1
        hist.clear()
        assert len(analytics.get_funnel_history()) == 1
