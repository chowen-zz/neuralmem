/**
 * SDK publish script for neuralmem TypeScript package (npm).
 *
 * Usage:
 *   npx tsx sdk/typescript/publish.ts               # normal publish
 *   npx tsx sdk/typescript/publish.ts --dry-run      # dry-run (no publish)
 *   npx tsx sdk/typescript/publish.ts --sync          # sync version from Python first
 */
import { execSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dirname ?? __dirname, "..", "..");
const TS_PACKAGE = resolve(ROOT, "sdk", "typescript", "package.json");
const PYPROJECT = resolve(ROOT, "pyproject.toml");
const README = resolve(ROOT, "sdk", "typescript", "README.md");
const CHANGELOG = resolve(ROOT, "CHANGELOG.md");

// ── Version helpers ─────────────────────────────────────────────

function readTsVersion(): string {
  const pkg = JSON.parse(readFileSync(TS_PACKAGE, "utf-8"));
  return pkg.version;
}

function writeTsVersion(version: string): void {
  const pkg = JSON.parse(readFileSync(TS_PACKAGE, "utf-8"));
  pkg.version = version;
  writeFileSync(TS_PACKAGE, JSON.stringify(pkg, null, 2) + "\n");
}

function readPythonVersion(): string {
  const text = readFileSync(PYPROJECT, "utf-8");
  const match = text.match(/^version\s*=\s*"([^"]+)"/m);
  if (!match) throw new Error("Cannot find version in pyproject.toml");
  return match[1];
}

// ── Pre-flight checks ──────────────────────────────────────────

function checkReadmeExists(): boolean {
  return existsSync(README);
}

function checkTestsPass(): boolean {
  try {
    execSync("npm test", {
      cwd: resolve(ROOT, "sdk", "typescript"),
      stdio: "pipe",
    });
    return true;
  } catch {
    return false;
  }
}

function checkVersionInChangelog(version: string): boolean {
  if (!existsSync(CHANGELOG)) return false;
  const text = readFileSync(CHANGELOG, "utf-8");
  return text.includes(version);
}

interface PreflightResult {
  passed: boolean;
  failures: string[];
}

function preflightChecks(skipTests = false): PreflightResult {
  const failures: string[] = [];
  const version = readTsVersion();

  if (!checkVersionInChangelog(version)) {
    failures.push(`CHANGELOG.md does not mention version ${version}`);
  }

  if (!checkReadmeExists()) {
    failures.push("sdk/typescript/README.md does not exist");
  }

  if (!skipTests) {
    try {
      checkTestsPass();
    } catch {
      failures.push("TypeScript tests failed");
    }
  }

  return { passed: failures.length === 0, failures };
}

// ── Build & publish ────────────────────────────────────────────

function buildPackage(): boolean {
  try {
    execSync("npm run build", {
      cwd: resolve(ROOT, "sdk", "typescript"),
      stdio: "pipe",
    });
    return true;
  } catch {
    return false;
  }
}

function publishPackage(dryRun: boolean): boolean {
  try {
    const cmd = dryRun ? "npm pack --dry-run" : "npm publish";
    execSync(cmd, {
      cwd: resolve(ROOT, "sdk", "typescript"),
      stdio: "pipe",
    });
    return true;
  } catch {
    return false;
  }
}

// ── Main ───────────────────────────────────────────────────────

function main(): void {
  const args = process.argv.slice(2);
  const dryRun = args.includes("--dry-run");
  const sync = args.includes("--sync");
  const skipTests = args.includes("--skip-tests");

  console.log("=== NeuralMem TypeScript SDK Publish ===");

  if (sync) {
    const pyVersion = readPythonVersion();
    const tsVersion = readTsVersion();
    console.log(`Syncing version: ${tsVersion} -> ${pyVersion}`);
    writeTsVersion(pyVersion);
  }

  console.log(`Version: ${readTsVersion()}`);
  console.log(`Dry run: ${dryRun}`);
  console.log();

  // Pre-flight
  console.log("Running pre-flight checks...");
  const result = preflightChecks(skipTests);
  if (!result.passed) {
    for (const f of result.failures) {
      console.log(`  FAIL: ${f}`);
    }
    process.exit(1);
  }
  console.log("  All pre-flight checks passed.");
  console.log();

  // Build
  console.log("Building package...");
  if (!buildPackage()) {
    console.log("  FAIL: Build failed");
    process.exit(1);
  }
  console.log("  Build succeeded.");
  console.log();

  // Publish
  const action = dryRun ? "Checking" : "Publishing";
  console.log(`${action} package...`);
  if (!publishPackage(dryRun)) {
    console.log(`  FAIL: ${action} failed`);
    process.exit(1);
  }
  console.log(`  ${action} succeeded.`);
  console.log();

  console.log("=== Done ===");
}

main();
