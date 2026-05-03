"""SDK publish script for neuralmem Python package (PyPI).

Usage:
    python sdk/publish.py                  # normal publish
    python sdk/publish.py --dry-run        # dry-run (no upload)
    python sdk/publish.py --bump patch     # bump patch, then publish
    python sdk/publish.py --bump minor     # bump minor, then publish
    python sdk/publish.py --bump major     # bump major, then publish
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
VERSION_FILE = ROOT / "src" / "neuralmem" / "_version.py"
TS_PACKAGE = ROOT / "sdk" / "typescript" / "package.json"
CHANGELOG = ROOT / "CHANGELOG.md"


# ── Version helpers ─────────────────────────────────────────────────


def read_version_pyproject() -> str:
    """Read version from pyproject.toml."""
    text = PYPROJECT.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("Cannot find version in pyproject.toml")
    return match.group(1)


def read_version_file() -> str:
    """Read version from src/neuralmem/_version.py."""
    text = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError("Cannot find __version__ in _version.py")
    return match.group(1)


def write_version_file(version: str) -> None:
    """Write version to src/neuralmem/_version.py."""
    VERSION_FILE.write_text(
        f'from __future__ import annotations\n\n__version__ = "{version}"\n'
    )


def write_pyproject_version(version: str) -> None:
    """Write version to pyproject.toml."""
    text = PYPROJECT.read_text()
    new_text = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        f'\\1"{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    PYPROJECT.write_text(new_text)


def _parse_semver(version: str) -> list[int]:
    """Parse a semver string into [major, minor, patch]."""
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid semver: {version}")
    return [int(p) for p in parts]


def bump_version(version: str, part: str) -> str:
    """Bump a semver version by the given part (major, minor, patch)."""
    major, minor, patch = _parse_semver(version)
    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    elif part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Unknown bump part: {part}")


# ── Pre-flight checks ──────────────────────────────────────────────


def check_clean_git() -> bool:
    """Check that git working tree is clean."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.stdout.strip() == ""


def check_changelog_has_version(version: str) -> bool:
    """Check that CHANGELOG.md mentions the current version."""
    if not CHANGELOG.exists():
        return False
    text = CHANGELOG.read_text()
    return version in text


def check_versions_in_sync() -> bool:
    """Check that pyproject.toml and _version.py versions match."""
    return read_version_pyproject() == read_version_file()


def run_tests() -> bool:
    """Run pytest and return True if all tests pass."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.returncode == 0


def run_lint() -> bool:
    """Run ruff and return True if no issues."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.returncode == 0


def preflight_checks(skip_tests: bool = False) -> list[str]:
    """Run all pre-flight checks, return list of failures."""
    failures: list[str] = []

    if not check_versions_in_sync():
        failures.append(
            "Version mismatch: pyproject.toml and _version.py disagree"
        )

    if not check_clean_git():
        failures.append("Git working tree is not clean")

    version = read_version_pyproject()
    if not check_changelog_has_version(version):
        failures.append(
            f"CHANGELOG.md does not mention version {version}"
        )

    if not skip_tests and not run_tests():
        failures.append("Tests failed")

    return failures


# ── Build & upload ──────────────────────────────────────────────────


def build_package() -> bool:
    """Build sdist + wheel."""
    result = subprocess.run(
        [sys.executable, "-m", "build"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.returncode == 0


def upload_package(dry_run: bool = False) -> bool:
    """Upload dist/* to PyPI via twine."""
    cmd = [sys.executable, "-m", "twine", "upload", "dist/*"]
    if dry_run:
        cmd = [sys.executable, "-m", "twine", "check", "dist/*"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ROOT,
        shell=True,
    )
    return result.returncode == 0


# ── Main ────────────────────────────────────────────────────────────


def publish(dry_run: bool = False, skip_tests: bool = False) -> int:
    """Run the full publish pipeline. Returns exit code."""
    print("=== NeuralMem Python SDK Publish ===")
    print(f"Version: {read_version_pyproject()}")
    print(f"Dry run: {dry_run}")
    print()

    # Pre-flight
    print("Running pre-flight checks...")
    failures = preflight_checks(skip_tests=skip_tests)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("  All pre-flight checks passed.")
    print()

    # Build
    print("Building package...")
    if not build_package():
        print("  FAIL: Build failed")
        return 1
    print("  Build succeeded.")
    print()

    # Upload
    action = "Checking" if dry_run else "Uploading"
    print(f"{action} package...")
    if not upload_package(dry_run=dry_run):
        print(f"  FAIL: {action} failed")
        return 1
    print(f"  {action} succeeded.")
    print()

    print("=== Done ===")
    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Publish neuralmem Python SDK to PyPI"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and check but do not upload",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests during pre-flight",
    )
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Bump version before publishing",
    )
    args = parser.parse_args()

    if args.bump:
        current = read_version_pyproject()
        new = bump_version(current, args.bump)
        print(f"Bumping version: {current} -> {new}")
        write_pyproject_version(new)
        write_version_file(new)

    raise SystemExit(publish(dry_run=args.dry_run, skip_tests=args.skip_tests))


if __name__ == "__main__":
    main()
