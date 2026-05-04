"""NeuralMem community package exports (V1.7)."""
from __future__ import annotations

from neuralmem.community.sharing import MemorySharing, ShareRecord
from neuralmem.community.collaboration import (
    CollaborationSpace,
    MemberRole,
    SpaceManager,
    SpaceMember,
)
from neuralmem.community.feedback import FeedbackEntry, FeedbackLoop
from neuralmem.community.blog_generator import (
    BlogPost,
    PublishResult,
    SEOOptimizer,
    PlatformPublisher,
    TechBlogGenerator,
)
from neuralmem.community.example_generator import (
    ExampleProjectGenerator,
    FileNode,
    ProjectTemplate,
)
from neuralmem.community.analytics import (
    CommunityAnalytics,
    Contributor,
    GitHubAPIClient,
    GrowthFunnel,
    RepoMetrics,
    TrendReport,
)

__all__ = [
    "MemorySharing",
    "ShareRecord",
    "CollaborationSpace",
    "MemberRole",
    "SpaceManager",
    "SpaceMember",
    "FeedbackEntry",
    "FeedbackLoop",
    # V1.7 growth engine
    "BlogPost",
    "PublishResult",
    "SEOOptimizer",
    "PlatformPublisher",
    "TechBlogGenerator",
    "ExampleProjectGenerator",
    "FileNode",
    "ProjectTemplate",
    "CommunityAnalytics",
    "Contributor",
    "GitHubAPIClient",
    "GrowthFunnel",
    "RepoMetrics",
    "TrendReport",
]
