"""Cross-SDK version synchronization.

Canonical version source: pyproject.toml

Usage:
    python sdk/version_sync.py --check    # verify all versions match
    python sdk/version_sync.py --sync     # sync canonical version everywhere
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
VERSION_FILE = ROOT / "src" / "neuralmem" / "_version.py"
TS_PACKAGE = ROOT / "sdk" / "typescript" / "package.json"
INIT_FILE = ROOT / "src" / "neuralmem" / "__init__.py"


# ── Read versions ───────────────────────────────────────────────


def read_pyproject_version() -> str:
    """Read version from pyproject.toml (canonical source)."""
    text = PYPROJECT.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("Cannot find version in pyproject.toml")
    return match.group(1)


def read_version_file_version() -> str | None:
    """Read version from src/neuralmem/_version.py."""
    if not VERSION_FILE.exists():
        return None
    text = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError("Cannot find __version__ in _version.py")
    return match.group(1)


def read_ts_version() -> str | None:
    """Read version from sdk/typescript/package.json."""
    if not TS_PACKAGE.exists():
        return None
    pkg = json.loads(TS_PACKAGE.read_text())
    return pkg.get("version")


def read_init_version() -> str | None:
    """Read version from src/neuralmem/__init__.py."""
    if not INIT_FILE.exists():
        return None
    text = INIT_FILE.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        return None
    return match.group(1)


# ── Write versions ──────────────────────────────────────────────


def write_version_file(version: str) -> None:
    """Write version to src/neuralmem/_version.py."""
    VERSION_FILE.write_text(
        f'from __future__ import annotations\n\n'
        f'__version__ = "{version}"\n'
    )


def write_ts_version(version: str) -> None:
    """Write version to sdk/typescript/package.json."""
    pkg = json.loads(TS_PACKAGE.read_text())
    pkg["version"] = version
    TS_PACKAGE.write_text(json.dumps(pkg, indent=2) + "\n")


def write_init_version(version: str) -> None:
    """Write version to src/neuralmem/__init__.py."""
    text = INIT_FILE.read_text()
    new_text = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{version}"',
        text,
    )
    INIT_FILE.write_text(new_text)


# ── Commands ────────────────────────────────────────────────────


def check_versions() -> bool:
    """Check all version files match the canonical version. Returns True if ok."""
    canonical = read_pyproject_version()
    print(f"Canonical version (pyproject.toml): {canonical}")
    ok = True

    checks: list[tuple[str, str | None]] = [
        ("src/neuralmem/_version.py", read_version_file_version()),
        ("sdk/typescript/package.json", read_ts_version()),
        ("src/neuralmem/__init__.py", read_init_version()),
    ]

    for path, version in checks:
        if version is None:
            print(f"  SKIP: {path} (not found)")
        elif version == canonical:
            print(f"  OK:   {path} = {version}")
        else:
            print(f"  DIFF: {path} = {version} (expected {canonical})")
            ok = False

    return ok


def sync_versions() -> None:
    """Sync canonical version to all version files."""
    canonical = read_pyproject_version()
    print(f"Syncing all versions to: {canonical}")

    write_version_file(canonical)
    print(f"  Wrote src/neuralmem/_version.py")

    if TS_PACKAGE.exists():
        write_ts_version(canonical)
        print(f"  Wrote sdk/typescript/package.json")

    if INIT_FILE.exists():
        write_init_version(canonical)
        print(f"  Wrote src/neuralmem/__init__.py")


# ── CLI ─────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-SDK version synchronization"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify all versions match the canonical version",
    )
    group.add_argument(
        "--sync",
        action="store_true",
        help="Sync canonical version to all version files",
    )
    args = parser.parse_args()

    if args.check:
        ok = check_versions()
        raise SystemExit(0 if ok else 1)
    else:
        sync_versions()
        raise SystemExit(0)


if __name__ == "__main__":
    main()
