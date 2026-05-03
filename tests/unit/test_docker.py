"""Tests for Docker deployment files (Dockerfile, docker-compose, .dockerignore)."""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDockerfile:
    """Verify Dockerfile structure and best practices."""

    @pytest.fixture()
    def dockerfile(self) -> str:
        path = PROJECT_ROOT / "Dockerfile"
        assert path.exists(), "Dockerfile not found"
        return path.read_text()

    def test_uses_multi_stage_build(self, dockerfile: str) -> None:
        """Dockerfile should define at least two stages."""
        from_s = [
            line for line in dockerfile.splitlines()
            if line.strip().upper().startswith("FROM ")
        ]
        assert len(from_s) >= 2, (
            f"Expected >=2 FROM stages, found {len(from_s)}"
        )

    def test_uses_python_slim_base(self, dockerfile: str) -> None:
        """Runtime stage should use python:3.11-slim."""
        assert "python:3.11-slim" in dockerfile

    def test_non_root_user(self, dockerfile: str) -> None:
        """Dockerfile should create and switch to a non-root user."""
        assert "useradd" in dockerfile or "adduser" in dockerfile
        assert "USER neuralmem" in dockerfile

    def test_exposes_port_8080(self, dockerfile: str) -> None:
        """Dockerfile should EXPOSE 8080 for MCP HTTP server."""
        lines = dockerfile.splitlines()
        expose_lines = [
            ln for ln in lines if ln.strip().upper().startswith("EXPOSE")
        ]
        assert any("8080" in ln for ln in expose_lines), (
            "Port 8080 should be exposed"
        )

    def test_healthcheck_configured(self, dockerfile: str) -> None:
        """Dockerfile should have a HEALTHCHECK instruction."""
        assert "HEALTHCHECK" in dockerfile

    def test_healthcheck_uses_curl(self, dockerfile: str) -> None:
        """Healthcheck should use curl to /health endpoint."""
        assert "curl" in dockerfile
        assert "/health" in dockerfile

    def test_entrypoint_runs_neuralmem(self, dockerfile: str) -> None:
        """ENTRYPOINT should invoke neuralmem mcp."""
        assert "neuralmem" in dockerfile
        assert "mcp" in dockerfile

    def test_sets_db_path_env(self, dockerfile: str) -> None:
        """Dockerfile should set NEURALMEM_DB_PATH env var."""
        assert "NEURALMEM_DB_PATH" in dockerfile

    def test_installs_neuralmem_package(self, dockerfile: str) -> None:
        """Builder stage should pip install the neuralmem package."""
        assert "pip install" in dockerfile

    def test_creates_data_directory(self, dockerfile: str) -> None:
        """Dockerfile should create /data directory for persistence."""
        assert "/data" in dockerfile


class TestDockerCompose:
    """Verify docker-compose.yml structure."""

    @pytest.fixture()
    def compose(self) -> str:
        path = PROJECT_ROOT / "docker-compose.yml"
        assert path.exists(), "docker-compose.yml not found"
        return path.read_text()

    def test_has_neuralmem_service(self, compose: str) -> None:
        """Should define a neuralmem-server service."""
        assert "neuralmem-server" in compose

    def test_exposes_port_8080(self, compose: str) -> None:
        """Should map host port 8080 to container port 8080."""
        assert "8080:8080" in compose

    def test_volume_for_data_persistence(self, compose: str) -> None:
        """Should mount a volume for /data (SQLite DB persistence)."""
        assert "neuralmem-data" in compose or "/data" in compose

    def test_environment_variables(self, compose: str) -> None:
        """Should pass NEURALMEM_* environment variables."""
        assert "NEURALMEM_DB_PATH" in compose

    def test_healthcheck_in_compose(self, compose: str) -> None:
        """Should define a healthcheck for the service."""
        assert "healthcheck" in compose.lower()

    def test_restart_policy(self, compose: str) -> None:
        """Should set a restart policy."""
        assert "restart:" in compose

    def test_optional_chromadb_service(self, compose: str) -> None:
        """Should reference optional chromadb service (commented or active)."""
        assert "chromadb" in compose.lower() or "chroma" in compose.lower()

    def test_builds_from_dockerfile(self, compose: str) -> None:
        """Service should build from the Dockerfile."""
        assert "Dockerfile" in compose or "build:" in compose


class TestDockerIgnore:
    """Verify .dockerignore excludes unnecessary files."""

    @pytest.fixture()
    def dockerignore(self) -> str:
        path = PROJECT_ROOT / ".dockerignore"
        assert path.exists(), ".dockerignore not found"
        return path.read_text()

    def test_excludes_tests(self, dockerignore: str) -> None:
        assert "tests" in dockerignore

    def test_excludes_git(self, dockerignore: str) -> None:
        assert ".git" in dockerignore

    def test_excludes_pycache(self, dockerignore: str) -> None:
        assert "__pycache__" in dockerignore

    def test_excludes_venv(self, dockerignore: str) -> None:
        assert ".venv" in dockerignore or "venv" in dockerignore
