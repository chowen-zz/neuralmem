"""Tests for sdk/publish.py — Python SDK publish tooling."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sdk.publish as publish_mod
from sdk.publish import (
    bump_version,
    check_changelog_has_version,
    check_clean_git,
    check_versions_in_sync,
    read_version_file,
    read_version_pyproject,
    write_version_file,
    write_pyproject_version,
)

# ── Helpers ──────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent


# ── Version reading ──────────────────────────────────────────────


class TestVersionReading:
    """Tests for version reading functions."""

    def test_read_version_pyproject(self):
        version = read_version_pyproject()
        assert isinstance(version, str)
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_read_version_file(self):
        # Ensure _version.py exists first
        write_version_file("0.9.0")
        version = read_version_file()
        assert version == "0.9.0"

    def test_read_versions_consistent(self):
        write_version_file("0.9.0")
        write_pyproject_version("0.9.0")
        assert read_version_pyproject() == read_version_file()


# ── Version bumping ─────────────────────────────────────────────


class TestVersionBumping:
    """Tests for version bump helpers."""

    def test_bump_patch(self):
        assert bump_version("0.9.0", "patch") == "0.9.1"

    def test_bump_minor(self):
        assert bump_version("0.9.0", "minor") == "0.10.0"

    def test_bump_major(self):
        assert bump_version("0.9.0", "major") == "1.0.0"

    def test_bump_patch_from_zero(self):
        assert bump_version("0.0.0", "patch") == "0.0.1"

    def test_bump_invalid_part_raises(self):
        with pytest.raises(ValueError, match="Unknown bump part"):
            bump_version("0.9.0", "revision")

    def test_bump_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Invalid semver"):
            bump_version("not-a-version", "patch")


# ── Write version ────────────────────────────────────────────────


class TestWriteVersion:
    """Tests for version writing functions."""

    def test_write_version_file_roundtrip(self, tmp_path):
        vfile = tmp_path / "_version.py"
        with patch.object(publish_mod, "VERSION_FILE", vfile):
            write_version_file("1.2.3")
            content = vfile.read_text()
            assert '__version__ = "1.2.3"' in content
            assert "from __future__ import annotations" in content

    def test_write_pyproject_version_roundtrip(self, tmp_path):
        pfile = tmp_path / "pyproject.toml"
        pfile.write_text('[project]\nname = "test"\nversion = "0.1.0"\n')
        with patch.object(publish_mod, "PYPROJECT", pfile):
            write_pyproject_version("2.0.0")
            text = pfile.read_text()
            assert 'version = "2.0.0"' in text


# ── Pre-flight checks ───────────────────────────────────────────


class TestPreflightChecks:
    """Tests for pre-flight check functions."""

    def test_check_clean_git_clean(self):
        with patch("sdk.publish.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(stdout="")
            assert check_clean_git() is True

    def test_check_clean_git_dirty(self):
        with patch("sdk.publish.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(
                stdout=" M src/foo.py\n"
            )
            assert check_clean_git() is False

    def test_check_changelog_has_version_true(self, tmp_path):
        cl = tmp_path / "CHANGELOG.md"
        cl.write_text("# Changelog\n## 0.9.0\n- stuff\n")
        with patch.object(publish_mod, "CHANGELOG", cl):
            assert check_changelog_has_version("0.9.0") is True

    def test_check_changelog_has_version_false(self, tmp_path):
        cl = tmp_path / "CHANGELOG.md"
        cl.write_text("# Changelog\n## 0.8.0\n- stuff\n")
        with patch.object(publish_mod, "CHANGELOG", cl):
            assert check_changelog_has_version("0.9.0") is False

    def test_check_versions_in_sync_true(self, tmp_path):
        vf = tmp_path / "_version.py"
        vf.write_text('__version__ = "0.9.0"\n')
        pf = tmp_path / "pyproject.toml"
        pf.write_text('version = "0.9.0"\n')
        with (
            patch.object(publish_mod, "VERSION_FILE", vf),
            patch.object(publish_mod, "PYPROJECT", pf),
        ):
            assert check_versions_in_sync() is True

    def test_check_versions_in_sync_false(self, tmp_path):
        vf = tmp_path / "_version.py"
        vf.write_text('__version__ = "0.8.0"\n')
        pf = tmp_path / "pyproject.toml"
        pf.write_text('version = "0.9.0"\n')
        with (
            patch.object(publish_mod, "VERSION_FILE", vf),
            patch.object(publish_mod, "PYPROJECT", pf),
        ):
            assert check_versions_in_sync() is False


# ── Build / upload ───────────────────────────────────────────────


class TestBuildUpload:
    """Tests for build and upload functions."""

    @patch("sdk.publish.subprocess")
    def test_build_package_success(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=0)
        from sdk.publish import build_package

        assert build_package() is True
        mock_sub.run.assert_called_once()

    @patch("sdk.publish.subprocess")
    def test_build_package_failure(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=1)
        from sdk.publish import build_package

        assert build_package() is False

    @patch("sdk.publish.subprocess")
    def test_upload_package_dry_run(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=0)
        from sdk.publish import upload_package

        assert upload_package(dry_run=True) is True

    @patch("sdk.publish.subprocess")
    def test_upload_package_failure(self, mock_sub):
        mock_sub.run.return_value = MagicMock(returncode=1)
        from sdk.publish import upload_package

        assert upload_package(dry_run=False) is False


# ── Full pipeline ────────────────────────────────────────────────


class TestPublishPipeline:
    """Tests for the full publish pipeline."""

    @patch("sdk.publish.upload_package", return_value=True)
    @patch("sdk.publish.build_package", return_value=True)
    @patch("sdk.publish.run_lint", return_value=True)
    @patch("sdk.publish.run_tests", return_value=True)
    @patch("sdk.publish.check_changelog_has_version", return_value=True)
    @patch("sdk.publish.check_clean_git", return_value=True)
    @patch("sdk.publish.check_versions_in_sync", return_value=True)
    @patch("sdk.publish.read_version_pyproject", return_value="0.9.0")
    def test_publish_dry_run_success(
        self,
        mock_ver,
        mock_sync,
        mock_git,
        mock_cl,
        mock_tests,
        mock_lint,
        mock_build,
        mock_upload,
    ):
        from sdk.publish import publish

        assert publish(dry_run=True, skip_tests=True) == 0

    @patch("sdk.publish.check_versions_in_sync", return_value=False)
    @patch("sdk.publish.read_version_pyproject", return_value="0.9.0")
    def test_publish_fails_on_version_mismatch(self, mock_ver, mock_sync):
        from sdk.publish import publish

        assert publish(dry_run=True, skip_tests=True) == 1
