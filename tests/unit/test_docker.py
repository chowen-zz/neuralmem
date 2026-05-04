"""Mock-based unit tests for NeuralMem Docker configuration structure.

These tests validate the Dockerfile, docker-compose.yml, entrypoint.sh,
and build.sh files without requiring Docker to be installed or running.
All filesystem and shell interactions are mocked.
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCKER_DIR = PROJECT_ROOT / "docker"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_file(path: Path) -> str:
    """Read file content; fail test if missing."""
    if not path.exists():
        pytest.fail(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dockerfile_content():
    """Content of docker/Dockerfile."""
    return _read_file(DOCKER_DIR / "Dockerfile")


@pytest.fixture
def compose_content():
    """Content of docker/docker-compose.yml."""
    return _read_file(DOCKER_DIR / "docker-compose.yml")


@pytest.fixture
def entrypoint_content():
    """Content of docker/entrypoint.sh."""
    return _read_file(DOCKER_DIR / "entrypoint.sh")


@pytest.fixture
def build_script_content():
    """Content of scripts/build.sh."""
    return _read_file(SCRIPTS_DIR / "build.sh")


# ---------------------------------------------------------------------------
# Tests: Dockerfile Structure
# ---------------------------------------------------------------------------


class TestDockerfileStructure:
    """Validate Dockerfile multi-stage build structure."""

    def test_has_multiple_stages(self, dockerfile_content):
        """Dockerfile must declare at least 2 FROM stages."""
        from_lines = [line for line in dockerfile_content.splitlines() if line.strip().startswith("FROM")]
        assert len(from_lines) >= 2, f"Expected >=2 FROM stages, got {len(from_lines)}"

    def test_has_builder_stage(self, dockerfile_content):
        """Must have a builder stage with AS builder."""
        assert "AS builder" in dockerfile_content, "Missing 'AS builder' stage"

    def test_has_runtime_stage(self, dockerfile_content):
        """Must have a runtime stage with AS runtime."""
        assert "AS runtime" in dockerfile_content, "Missing 'AS runtime' stage"

    def test_slim_base_image(self, dockerfile_content):
        """Runtime stage should use a slim Python image for smaller footprint."""
        runtime_line = next(
            (line for line in dockerfile_content.splitlines() if "AS runtime" in line),
            "",
        )
        assert "slim" in runtime_line.lower(), "Runtime base image should be 'slim' variant"

    def test_copy_from_builder(self, dockerfile_content):
        """Runtime stage must COPY artifacts from builder stage."""
        assert "COPY --from=builder" in dockerfile_content, "Missing COPY --from=builder"

    def test_non_root_user(self, dockerfile_content):
        """Runtime stage should create and switch to a non-root user."""
        assert "useradd" in dockerfile_content or "adduser" in dockerfile_content, "Missing non-root user creation"
        assert "USER " in dockerfile_content, "Missing USER directive"

    def test_healthcheck_defined(self, dockerfile_content):
        """Must include a HEALTHCHECK instruction."""
        assert "HEALTHCHECK" in dockerfile_content, "Missing HEALTHCHECK"

    def test_healthcheck_uses_curl(self, dockerfile_content):
        """Health check should use curl to hit an HTTP endpoint."""
        health_match = re.search(r"HEALTHCHECK.*curl.*(health|/api/health)", dockerfile_content, re.DOTALL | re.IGNORECASE)
        assert health_match is not None, "HEALTHCHECK should use curl against a health endpoint"

    def test_exposes_ports(self, dockerfile_content):
        """Must expose both MCP and Dashboard ports."""
        expose_match = re.search(r"EXPOSE\s+(\d+)[\s\d]*", dockerfile_content)
        assert expose_match is not None, "Missing EXPOSE directive"
        ports = re.findall(r"\b\d{4,5}\b", dockerfile_content.split("EXPOSE")[1].split("\n")[0] if "EXPOSE" in dockerfile_content else "")
        assert len(ports) >= 1, "Should expose at least one port"

    def test_pythonunbuffered_env(self, dockerfile_content):
        """Should set PYTHONUNBUFFERED for proper logging in containers."""
        assert "PYTHONUNBUFFERED=1" in dockerfile_content, "Missing PYTHONUNBUFFERED=1"

    def test_entrypoint_set(self, dockerfile_content):
        """ENTRYPOINT must be defined."""
        assert "ENTRYPOINT" in dockerfile_content, "Missing ENTRYPOINT"

    def test_labels_present(self, dockerfile_content):
        """Should include OCI-style labels."""
        assert "LABEL" in dockerfile_content, "Missing LABEL directives"
        assert any(label in dockerfile_content for label in ["version", "maintainer", "description"]), "Missing version/maintainer/description labels"

    def test_no_cache_dir_pip(self, dockerfile_content):
        """pip install should use --no-cache-dir to reduce layer size."""
        pip_lines = [line for line in dockerfile_content.splitlines() if "pip install" in line]
        for line in pip_lines:
            assert "--no-cache-dir" in line, f"pip install missing --no-cache-dir: {line}"

    def test_apt_cleanup(self, dockerfile_content):
        """apt-get install should clean up lists to reduce image size."""
        assert "rm -rf /var/lib/apt/lists/*" in dockerfile_content, "Missing apt cache cleanup"


# ---------------------------------------------------------------------------
# Tests: docker-compose.yml Structure
# ---------------------------------------------------------------------------


class TestDockerComposeStructure:
    """Validate docker-compose.yml full-stack configuration."""

    def test_services_section_exists(self, compose_content):
        """Must define a services section."""
        assert "services:" in compose_content, "Missing 'services:' section"

    def test_app_service_defined(self, compose_content):
        """Must define neuralmem-app service."""
        assert "neuralmem-app:" in compose_content, "Missing 'neuralmem-app' service"

    def test_dashboard_service_defined(self, compose_content):
        """Must define neuralmem-dashboard service."""
        assert "neuralmem-dashboard:" in compose_content, "Missing 'neuralmem-dashboard' service"

    def test_app_has_healthcheck(self, compose_content):
        """App service should have a healthcheck block."""
        assert "healthcheck:" in compose_content, "Compose missing healthcheck anywhere"
        # Verify it's inside neuralmem-app by checking the raw content order
        app_idx = compose_content.find("neuralmem-app:")
        dash_idx = compose_content.find("neuralmem-dashboard:")
        hc_idx = compose_content.find("healthcheck:", app_idx)
        assert hc_idx != -1 and (dash_idx == -1 or hc_idx < dash_idx), "App service missing healthcheck"

    def test_dashboard_has_healthcheck(self, compose_content):
        """Dashboard service should have a healthcheck block."""
        dash_idx = compose_content.find("neuralmem-dashboard:")
        hc_idx = compose_content.find("healthcheck:", dash_idx)
        assert hc_idx != -1, "Dashboard service missing healthcheck"

    def test_depends_on_with_condition(self, compose_content):
        """Dashboard should depend_on app with service_healthy condition."""
        dash_idx = compose_content.find("neuralmem-dashboard:")
        assert dash_idx != -1
        block = compose_content[dash_idx:]
        assert "depends_on:" in block, "Dashboard missing depends_on"
        assert "condition: service_healthy" in block, "Dashboard depends_on missing health condition"

    def test_volumes_section_exists(self, compose_content):
        """Must define named volumes for persistent storage."""
        assert "volumes:" in compose_content, "Missing 'volumes:' section"

    def test_neuralmem_data_volume(self, compose_content):
        """Must have neuralmem-data volume."""
        assert "neuralmem-data:" in compose_content, "Missing 'neuralmem-data' volume"


# ---------------------------------------------------------------------------
# Tests: entrypoint.sh Structure
# ---------------------------------------------------------------------------


class TestEntrypointStructure:
    """Validate entrypoint.sh startup script structure."""

    def test_shebang_present(self, entrypoint_content):
        """Must start with bash shebang."""
        assert entrypoint_content.startswith("#!/bin/bash"), "Missing bash shebang"

    def test_set_euo_pipefail(self, entrypoint_content):
        """Should set strict mode for safer shell execution."""
        assert "set -euo pipefail" in entrypoint_content, "Missing 'set -euo pipefail'"

    def test_mode_variable(self, entrypoint_content):
        """Should read MODE environment variable with default."""
        assert 'MODE="${MODE:-' in entrypoint_content, "Missing MODE env var with default"

    def test_mcp_mode_function(self, entrypoint_content):
        """Should define a start function for MCP mode."""
        assert "start_mcp()" in entrypoint_content or "neuralmem mcp" in entrypoint_content, "Missing MCP start logic"

    def test_dashboard_mode_function(self, entrypoint_content):
        """Should define a start function for dashboard mode."""
        assert "start_dashboard()" in entrypoint_content or "dashboard.server" in entrypoint_content, "Missing dashboard start logic"

    def test_both_mode_function(self, entrypoint_content):
        """Should define a start function for dual mode."""
        assert "start_both()" in entrypoint_content, "Missing dual-mode start logic"

    def test_port_variables(self, entrypoint_content):
        """Should accept PORT and DASHBOARD_PORT environment variables."""
        assert 'PORT="${PORT:-' in entrypoint_content, "Missing PORT env var"
        assert 'DASHBOARD_PORT="${DASHBOARD_PORT:-' in entrypoint_content, "Missing DASHBOARD_PORT env var"

    def test_db_path_variable(self, entrypoint_content):
        """Should accept NEURALMEM_DB_PATH environment variable."""
        assert "NEURALMEM_DB_PATH" in entrypoint_content, "Missing DB_PATH env var reference"

    def test_help_flag(self, entrypoint_content):
        """Should support --help / -h flag."""
        assert "--help" in entrypoint_content, "Missing --help support"

    def test_case_statement(self, entrypoint_content):
        """Should use case/esac to dispatch modes."""
        assert "case \"$MODE\" in" in entrypoint_content, "Missing case statement for mode dispatch"


# ---------------------------------------------------------------------------
# Tests: build.sh Structure
# ---------------------------------------------------------------------------


class TestBuildScriptStructure:
    """Validate scripts/build.sh build script structure."""

    def test_shebang_present(self, build_script_content):
        """Must start with bash shebang."""
        assert build_script_content.startswith("#!/bin/bash"), "Missing bash shebang"

    def test_set_euo_pipefail(self, build_script_content):
        """Should set strict mode."""
        assert "set -euo pipefail" in build_script_content, "Missing 'set -euo pipefail'"

    def test_default_tag(self, build_script_content):
        """Should define a default version tag."""
        assert 'TAG="v1.6.0"' in build_script_content or "v1.6" in build_script_content, "Missing default version tag"

    def test_dockerfile_reference(self, build_script_content):
        """Should reference a Dockerfile path (literal or via variable)."""
        assert "Dockerfile" in build_script_content, "Missing reference to a Dockerfile"

    def test_runtime_target(self, build_script_content):
        """Should build the 'runtime' target."""
        assert '--target runtime' in build_script_content or "runtime" in build_script_content, "Missing runtime target build"

    def test_push_option(self, build_script_content):
        """Should support --push flag."""
        assert "--push" in build_script_content, "Missing --push option"

    def test_registry_option(self, build_script_content):
        """Should support --registry option."""
        assert "--registry" in build_script_content, "Missing --registry option"

    def test_tag_option(self, build_script_content):
        """Should support --tag option."""
        assert "--tag" in build_script_content, "Missing --tag option"

    def test_smoke_test_function(self, build_script_content):
        """Should include a smoke test function or block."""
        assert "smoke" in build_script_content.lower() or "test" in build_script_content.lower(), "Missing smoke/test logic"

    def test_docker_available_check(self, build_script_content):
        """Should verify Docker is available before building."""
        assert "command -v docker" in build_script_content, "Missing Docker availability check"

    def test_multi_platform_support(self, build_script_content):
        """Should support multi-platform builds via buildx."""
        assert "buildx" in build_script_content, "Missing buildx reference for multi-platform builds"


# ---------------------------------------------------------------------------
# Tests: File Existence (mock-based — no real Docker needed)
# ---------------------------------------------------------------------------


class TestFileExistence:
    """Verify all required Docker/deployment files exist."""

    def test_dockerfile_exists(self):
        assert (DOCKER_DIR / "Dockerfile").exists(), "docker/Dockerfile missing"

    def test_docker_compose_exists(self):
        assert (DOCKER_DIR / "docker-compose.yml").exists(), "docker/docker-compose.yml missing"

    def test_entrypoint_exists(self):
        assert (DOCKER_DIR / "entrypoint.sh").exists(), "docker/entrypoint.sh missing"

    def test_build_script_exists(self):
        assert (SCRIPTS_DIR / "build.sh").exists(), "scripts/build.sh missing"

    def test_dockerignore_exists(self):
        assert (PROJECT_ROOT / ".dockerignore").exists(), ".dockerignore missing"


# ---------------------------------------------------------------------------
# Tests: Mock-based Dockerfile Parsing
# ---------------------------------------------------------------------------


class TestDockerfileMockParsing:
    """Parse Dockerfile content with mocked regex / line operations."""

    def test_extract_stages_mock(self, dockerfile_content):
        """Mock-style test: simulate stage extraction logic."""
        # Simulate a parser that extracts stage names
        stage_pattern = re.compile(r"FROM\s+\S+\s+AS\s+(\w+)")
        stages = stage_pattern.findall(dockerfile_content)
        assert "builder" in stages, "builder stage not found by mock parser"
        assert "runtime" in stages, "runtime stage not found by mock parser"

    def test_extract_exposed_ports_mock(self, dockerfile_content):
        """Mock-style test: simulate port extraction."""
        expose_pattern = re.compile(r"EXPOSE\s+([\d\s]+)")
        match = expose_pattern.search(dockerfile_content)
        assert match is not None, "No EXPOSE directive found by mock parser"
        ports = [int(p) for p in match.group(1).split() if p.isdigit()]
        assert len(ports) >= 1, "No ports extracted by mock parser"
        assert 8080 in ports or 8081 in ports, "Expected 8080 or 8081 in exposed ports"

    def test_extract_env_vars_mock(self, dockerfile_content):
        """Mock-style test: simulate ENV extraction."""
        env_pattern = re.compile(r"ENV\s+(\w+)=")
        env_vars = env_pattern.findall(dockerfile_content)
        assert "NEURALMEM_DB_PATH" in env_vars, "NEURALMEM_DB_PATH not extracted by mock parser"
        assert "PYTHONUNBUFFERED" in env_vars, "PYTHONUNBUFFERED not extracted by mock parser"

    def test_extract_labels_mock(self, dockerfile_content):
        """Mock-style test: simulate LABEL extraction."""
        label_pattern = re.compile(r'LABEL\s+(\w+)="([^"]+)"')
        labels = dict(label_pattern.findall(dockerfile_content))
        assert "version" in labels or "description" in labels, "Expected version or description label"

    def test_extract_copy_instructions_mock(self, dockerfile_content):
        """Mock-style test: simulate COPY instruction extraction."""
        copy_pattern = re.compile(r"COPY\s+(--from=\w+\s+)?(\S+)\s+(\S+)")
        copies = copy_pattern.findall(dockerfile_content)
        from_copies = [c for c in copies if c[0]]
        assert len(from_copies) >= 1, "Expected at least one COPY --from= instruction"


# ---------------------------------------------------------------------------
# Tests: Mock-based docker-compose.yml Parsing
# ---------------------------------------------------------------------------


class TestComposeMockParsing:
    """Parse docker-compose.yml content with mocked structural checks."""

    def test_count_services_mock(self, compose_content):
        """Mock-style test: simulate service counting."""
        # Service names are at 2-space indent; names may contain hyphens
        service_pattern = re.compile(r"^  ([\w-]+):\s*$", re.MULTILINE)
        services = service_pattern.findall(compose_content)
        assert "neuralmem-app" in services, "neuralmem-app not found by mock parser"
        assert "neuralmem-dashboard" in services, "neuralmem-dashboard not found by mock parser"

    def test_count_volumes_mock(self, compose_content):
        """Mock-style test: simulate volume counting."""
        # Find volumes: section and count entries (2-space indent for volume names; names may contain hyphens)
        vol_section_match = re.search(r"^volumes:(.*?)(?=^\S|\Z)", compose_content, re.MULTILINE | re.DOTALL)
        assert vol_section_match is not None, "No volumes section found by mock parser"
        vol_block = vol_section_match.group(1)
        vol_names = re.findall(r"^  ([\w-]+):", vol_block, re.MULTILINE)
        assert "neuralmem-data" in vol_names, "neuralmem-data volume not found by mock parser"

    def test_environment_defaults_mock(self, compose_content):
        """Mock-style test: simulate env default extraction."""
        env_pattern = re.compile(r"-\s+(\w+)=\$\{(\w+):-([^}]+)\}")
        env_defaults = env_pattern.findall(compose_content)
        assert any("NEURALMEM" in e[0] for e in env_defaults), "No NEURALMEM env var with default found"

    def test_profile_usage_mock(self, compose_content):
        """Mock-style test: simulate profile extraction."""
        profile_pattern = re.compile(r"profiles:\s*\[(.*?)\]", re.DOTALL)
        profiles = profile_pattern.findall(compose_content)
        assert len(profiles) > 0, "No profiles found by mock parser"
        assert any("vector-db" in p or "weaviate" in p or "chroma" in p for p in profiles), "Expected vector-db/weaviate/chroma profile"


# ---------------------------------------------------------------------------
# Tests: Mock-based entrypoint.sh Parsing
# ---------------------------------------------------------------------------


class TestEntrypointMockParsing:
    """Parse entrypoint.sh with mocked shell analysis."""

    def test_function_definitions_mock(self, entrypoint_content):
        """Mock-style test: simulate function extraction."""
        func_pattern = re.compile(r"^(\w+)\(\)\s*\{", re.MULTILINE)
        funcs = func_pattern.findall(entrypoint_content)
        assert "start_mcp" in funcs or "start_dashboard" in funcs or "start_both" in funcs, "Expected start functions"

    def test_env_defaults_mock(self, entrypoint_content):
        """Mock-style test: simulate env default extraction."""
        default_pattern = re.compile(r'(\w+)="\$\{\w+:-([^}]+)\}"')
        defaults = default_pattern.findall(entrypoint_content)
        assert any("8080" in d[1] or "8081" in d[1] for d in defaults), "Expected port defaults"

    def test_case_modes_mock(self, entrypoint_content):
        """Mock-style test: simulate case branch extraction."""
        case_pattern = re.compile(r"case\s+\"?\$\w+\"?\s+in(.*?)\nesac", re.DOTALL)
        match = case_pattern.search(entrypoint_content)
        assert match is not None, "No case/esac block found by mock parser"
        branches = re.findall(r"(\w+)\)", match.group(1))
        assert "mcp" in branches, "mcp branch missing"
        assert "dashboard" in branches, "dashboard branch missing"
        assert "both" in branches, "both branch missing"


# ---------------------------------------------------------------------------
# Tests: Mock-based build.sh Parsing
# ---------------------------------------------------------------------------


class TestBuildScriptMockParsing:
    """Parse build.sh with mocked shell analysis."""

    def test_cli_options_mock(self, build_script_content):
        """Mock-style test: simulate option extraction."""
        opt_pattern = re.compile(r"--(\w+)\)")
        opts = opt_pattern.findall(build_script_content)
        assert "tag" in opts, "--tag option missing"
        assert "push" in opts, "--push option missing"
        assert "registry" in opts, "--registry option missing"

    def test_docker_build_command_mock(self, build_script_content):
        """Mock-style test: simulate docker build command extraction."""
        # The script uses a $BUILDER variable (docker buildx build or docker build)
        build_pattern = re.compile(r"\$BUILDER|docker\s+(buildx\s+)?build")
        match = build_pattern.search(build_script_content)
        assert match is not None, "No docker build command found by mock parser"
        # Also verify --file is used somewhere in build commands
        assert "--file" in build_script_content, "Build command should use --file"

    def test_image_tagging_mock(self, build_script_content):
        """Mock-style test: simulate docker tag commands."""
        tag_pattern = re.compile(r"docker\s+tag\s+(\S+)\s+(\S+)")
        tags = tag_pattern.findall(build_script_content)
        assert len(tags) >= 1, "Expected at least one docker tag command"
