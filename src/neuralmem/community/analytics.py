"""NeuralMem V1.7 community growth engine — analytics.

CommunityAnalytics: GitHub stars/forks/issues tracking, growth funnel,
and contributor activity tracking. All real API calls are mockable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable


@dataclass
class RepoMetrics:
    """GitHub repository metrics snapshot."""

    repo: str
    stars: int
    forks: int
    open_issues: int
    closed_issues: int
    watchers: int
    subscribers: int
    latest_release: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "stars": self.stars,
            "forks": self.forks,
            "open_issues": self.open_issues,
            "closed_issues": self.closed_issues,
            "watchers": self.watchers,
            "subscribers": self.subscribers,
            "latest_release": self.latest_release,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Contributor:
    """A repository contributor."""

    username: str
    contributions: int
    avatar_url: str = ""
    profile_url: str = ""
    first_contribution: datetime | None = None
    last_contribution: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "contributions": self.contributions,
            "avatar_url": self.avatar_url,
            "profile_url": self.profile_url,
            "first_contribution": self.first_contribution.isoformat() if self.first_contribution else None,
            "last_contribution": self.last_contribution.isoformat() if self.last_contribution else None,
        }


@dataclass
class GrowthFunnel:
    """User growth funnel snapshot."""

    date: datetime
    visitors: int = 0
    page_views: int = 0
    stars_gained: int = 0
    forks_gained: int = 0
    issues_opened: int = 0
    issues_closed: int = 0
    new_contributors: int = 0
    downloads: int = 0

    def conversion_rate(self, from_stage: str, to_stage: str) -> float:
        """Calculate conversion rate between two funnel stages."""
        stage_map = {
            "visitors": self.visitors,
            "page_views": self.page_views,
            "stars_gained": self.stars_gained,
            "forks_gained": self.forks_gained,
            "issues_opened": self.issues_opened,
            "new_contributors": self.new_contributors,
            "downloads": self.downloads,
        }
        from_val = stage_map.get(from_stage, 0)
        to_val = stage_map.get(to_stage, 0)
        if from_val == 0:
            return 0.0
        return round((to_val / from_val) * 100, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "visitors": self.visitors,
            "page_views": self.page_views,
            "stars_gained": self.stars_gained,
            "forks_gained": self.forks_gained,
            "issues_opened": self.issues_opened,
            "issues_closed": self.issues_closed,
            "new_contributors": self.new_contributors,
            "downloads": self.downloads,
        }


@dataclass
class TrendReport:
    """Aggregated trend report over a date range."""

    repo: str
    start_date: datetime
    end_date: datetime
    metrics_snapshots: list[RepoMetrics] = field(default_factory=list)
    funnels: list[GrowthFunnel] = field(default_factory=list)
    contributors: list[Contributor] = field(default_factory=list)

    def star_velocity(self) -> float:
        """Average stars gained per day."""
        days = max(1, (self.end_date - self.start_date).days)
        total = sum(m.stars for m in self.metrics_snapshots)
        return round(total / days, 2)

    def issue_resolution_rate(self) -> float:
        """Percentage of issues that are closed."""
        total_open = sum(m.open_issues for m in self.metrics_snapshots)
        total_closed = sum(m.closed_issues for m in self.metrics_snapshots)
        total = total_open + total_closed
        if total == 0:
            return 0.0
        return round((total_closed / total) * 100, 2)

    def top_contributors(self, limit: int = 5) -> list[Contributor]:
        """Return top contributors by contribution count."""
        return sorted(self.contributors, key=lambda c: c.contributions, reverse=True)[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "star_velocity": self.star_velocity(),
            "issue_resolution_rate": self.issue_resolution_rate(),
            "metrics_count": len(self.metrics_snapshots),
            "funnel_count": len(self.funnels),
            "contributor_count": len(self.contributors),
        }


class GitHubAPIClient:
    """GitHub API client — mockable for testing."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None, http_client: Callable | None = None) -> None:
        self._token = token
        self._http = http_client
        self._call_log: list[dict[str, Any]] = []

    def get_repo_metrics(self, owner: str, repo: str) -> RepoMetrics:
        """Fetch repository metrics from GitHub API."""
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"
        data = self._request("GET", url)
        return RepoMetrics(
            repo=f"{owner}/{repo}",
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            open_issues=data.get("open_issues_count", 0),
            closed_issues=0,  # Requires separate issues search
            watchers=data.get("watchers_count", 0),
            subscribers=data.get("subscribers_count", 0),
            latest_release="",
        )

    def get_contributors(self, owner: str, repo: str, limit: int = 100) -> list[Contributor]:
        """Fetch contributor list from GitHub API."""
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/contributors?per_page={limit}"
        data = self._request("GET", url)
        contributors: list[Contributor] = []
        for item in data if isinstance(data, list) else []:
            contributors.append(
                Contributor(
                    username=item.get("login", ""),
                    contributions=item.get("contributions", 0),
                    avatar_url=item.get("avatar_url", ""),
                    profile_url=item.get("html_url", ""),
                )
            )
        return contributors

    def get_closed_issues_count(self, owner: str, repo: str) -> int:
        """Fetch closed issues count from GitHub API."""
        url = f"{self.BASE_URL}/search/issues?q=repo:{owner}/{repo}+type:issue+state:closed"
        data = self._request("GET", url)
        return data.get("total_count", 0)

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Make an HTTP request — override or inject http_client for mocking."""
        self._call_log.append({"method": method, "url": url, "kwargs": kwargs})
        if self._http:
            return self._http(method, url, **kwargs)
        raise RuntimeError("No http_client configured — provide one or mock this method")

    def get_call_log(self) -> list[dict[str, Any]]:
        """Return log of all API calls made."""
        return self._call_log.copy()


class CommunityAnalytics:
    """Community growth analytics engine for NeuralMem."""

    def __init__(
        self,
        repo: str,
        github_client: GitHubAPIClient | None = None,
    ) -> None:
        self._repo = repo
        self._github = github_client or GitHubAPIClient()
        self._metrics_history: list[RepoMetrics] = []
        self._funnel_history: list[GrowthFunnel] = []
        self._contributor_history: list[Contributor] = []

    def track_current_metrics(self) -> RepoMetrics:
        """Fetch and store current repository metrics."""
        owner, repo_name = self._repo.split("/")
        metrics = self._github.get_repo_metrics(owner, repo_name)
        try:
            metrics.closed_issues = self._github.get_closed_issues_count(owner, repo_name)
        except Exception:
            metrics.closed_issues = 0
        self._metrics_history.append(metrics)
        return metrics

    def track_contributors(self) -> list[Contributor]:
        """Fetch and store current contributor list."""
        owner, repo_name = self._repo.split("/")
        contributors = self._github.get_contributors(owner, repo_name)
        self._contributor_history = contributors
        return contributors

    def record_funnel(self, funnel: GrowthFunnel) -> GrowthFunnel:
        """Record a growth funnel snapshot."""
        self._funnel_history.append(funnel)
        return funnel

    def generate_funnel(
        self,
        visitors: int = 0,
        page_views: int = 0,
        stars_gained: int = 0,
        forks_gained: int = 0,
        issues_opened: int = 0,
        issues_closed: int = 0,
        new_contributors: int = 0,
        downloads: int = 0,
    ) -> GrowthFunnel:
        """Create and record a growth funnel snapshot."""
        funnel = GrowthFunnel(
            date=datetime.now(timezone.utc),
            visitors=visitors,
            page_views=page_views,
            stars_gained=stars_gained,
            forks_gained=forks_gained,
            issues_opened=issues_opened,
            issues_closed=issues_closed,
            new_contributors=new_contributors,
            downloads=downloads,
        )
        return self.record_funnel(funnel)

    def build_trend_report(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> TrendReport:
        """Build a trend report for a date range."""
        start = start_date or datetime.now(timezone.utc) - timedelta(days=30)
        end = end_date or datetime.now(timezone.utc)

        metrics = [
            m for m in self._metrics_history
            if start <= m.created_at <= end
        ]
        funnels = [
            f for f in self._funnel_history
            if start <= f.date <= end
        ]

        return TrendReport(
            repo=self._repo,
            start_date=start,
            end_date=end,
            metrics_snapshots=metrics,
            funnels=funnels,
            contributors=self._contributor_history.copy(),
        )

    def get_star_growth(self, days: int = 30) -> list[tuple[datetime, int]]:
        """Return star count over the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            (m.created_at, m.stars)
            for m in self._metrics_history
            if m.created_at >= cutoff
        ]

    def get_contributor_leaderboard(self, limit: int = 10) -> list[Contributor]:
        """Return top contributors sorted by contribution count."""
        return sorted(
            self._contributor_history,
            key=lambda c: c.contributions,
            reverse=True,
        )[:limit]

    def get_health_score(self) -> dict[str, Any]:
        """Calculate a repository health score (0-100)."""
        if not self._metrics_history:
            return {"score": 0, "reason": "No metrics tracked yet"}

        latest = self._metrics_history[-1]
        total_issues = latest.open_issues + latest.closed_issues
        resolution_rate = (
            (latest.closed_issues / total_issues * 100) if total_issues > 0 else 100
        )

        # Simple scoring formula
        score = min(100, max(0, round(
            30 +  # base
            min(latest.stars / 10, 30) +  # stars up to 30
            min(latest.forks / 5, 15) +   # forks up to 15
            (resolution_rate / 100 * 20) +  # issue resolution up to 20
            min(len(self._contributor_history), 5)  # contributors up to 5
        )))

        return {
            "score": score,
            "stars": latest.stars,
            "forks": latest.forks,
            "open_issues": latest.open_issues,
            "resolution_rate": round(resolution_rate, 2),
            "contributor_count": len(self._contributor_history),
        }

    def export_report(self, report: TrendReport, output_path: str) -> str:
        """Export a trend report to JSON."""
        data = {
            "repo": report.repo,
            "start_date": report.start_date.isoformat(),
            "end_date": report.end_date.isoformat(),
            "star_velocity": report.star_velocity(),
            "issue_resolution_rate": report.issue_resolution_rate(),
            "top_contributors": [c.to_dict() for c in report.top_contributors()],
            "metrics_snapshots": [m.to_dict() for m in report.metrics_snapshots],
            "funnels": [f.to_dict() for f in report.funnels],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path

    def get_metrics_history(self) -> list[RepoMetrics]:
        """Return all tracked metrics snapshots."""
        return self._metrics_history.copy()

    def get_funnel_history(self) -> list[GrowthFunnel]:
        """Return all recorded funnel snapshots."""
        return self._funnel_history.copy()
