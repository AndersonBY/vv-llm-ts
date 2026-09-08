#!/usr/bin/env node

import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const scriptPath = fileURLToPath(import.meta.url);
const root = resolve(dirname(scriptPath), "..");
const vendorRoot = join(root, "contract");
const vendorLockPath = join(vendorRoot, "consumer-lock.v1.json");
const generatedCatalogPath = join(root, "src", "generated", "contract-catalog.ts");
const EXPECTED_CONSUMER_LOCK_SHA256 = "88040f2f41e84c45bad0e7ff70239df4ba33424246db2305dcf411babdcb7396";
const EXPECTED_CONTRACT_VERSION = "1.1.0";
const EXPECTED_SCHEMA_VERSION = 2;
const EXPECTED_FIXTURE_VERSION = 2;
const EXPECTED_CATALOG_REVISION = 3;
const args = process.argv.slice(2);
const checkOnly = args.includes("--check");
const sourceFlagIndex = args.indexOf("--source");
const sourceArgument = sourceFlagIndex >= 0 ? args[sourceFlagIndex + 1] : undefined;
const sourceValue = sourceArgument ?? process.env.VV_LLM_CONTRACT_SOURCE;
const sourceRoot = sourceValue ? resolve(process.cwd(), sourceValue) : undefined;

if (sourceFlagIndex >= 0 && !sourceArgument) {
  fail("--source requires a contract source tree path");
}
if (args.some((argument, index) => argument !== "--check" && argument !== "--source" && index !== sourceFlagIndex + 1)) {
  fail("Usage: node scripts/contract-sync.mjs [--check] [--source <contract-source>]");
}

function fail(message) {
  throw new Error(`[contract] ${message}`);
}

function readJson(path) {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    fail(`cannot read JSON ${relative(root, path)}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function readConsumerLock(path, sourceLabel) {
  let bytes;
  try {
    bytes = readFileSync(path);
  } catch (error) {
    fail(`cannot read ${sourceLabel} consumer lock: ${error instanceof Error ? error.message : String(error)}`);
  }
  const actual = createHash("sha256").update(bytes).digest("hex");
  if (actual !== EXPECTED_CONSUMER_LOCK_SHA256) {
    fail(`${sourceLabel} consumer lock SHA-256 mismatch: expected ${EXPECTED_CONSUMER_LOCK_SHA256}, got ${actual}`);
  }
  try {
    return JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    fail(`cannot parse ${sourceLabel} consumer lock: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function assertObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${label} must be an object`);
}

function assertRelativeArtifact(path) {
  if (typeof path !== "string" || path.length === 0 || path.includes("\\") || path.startsWith("/") || path.startsWith("..")) {
    fail(`invalid artifact path: ${String(path)}`);
  }
  const resolved = resolve(vendorRoot, path);
  const relativePath = relative(vendorRoot, resolved);
  if (!relativePath || relativePath.startsWith(`..${sep}`) || relativePath === "..") {
    fail(`artifact escapes vendor root: ${path}`);
  }
}

function artifactPaths(lock) {
  assertObject(lock.artifacts, "consumer lock artifacts");
  return Object.keys(lock.artifacts).sort();
}

function bundlePaths(lock) {
  return ["manifest.json", "checksums.sha256", "consumer-lock.v1.json", ...artifactPaths(lock)];
}

function bundleFiles(base) {
  const files = [];
  const visit = (directory, prefix = "") => {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
      const path = join(directory, entry.name);
      if (entry.isDirectory()) visit(path, relativePath);
      else files.push(relativePath);
    }
  };
  visit(base);
  return files.sort();
}

function validateLockShape(lock, sourceLabel) {
  assertObject(lock, `${sourceLabel} lock`);
  if (lock.format !== "vv-llm-contract-consumer-lock.v1") fail(`${sourceLabel} lock format is invalid`);
  if (lock.contract_version !== EXPECTED_CONTRACT_VERSION) {
    fail(`${sourceLabel} contract version must be ${EXPECTED_CONTRACT_VERSION}`);
  }
  const expectedRevisions = {
    schema_version: EXPECTED_SCHEMA_VERSION,
    fixture_version: EXPECTED_FIXTURE_VERSION,
    catalog_revision: EXPECTED_CATALOG_REVISION,
  };
  for (const [key, expected] of Object.entries(expectedRevisions)) {
    if (lock[key] !== expected) fail(`${sourceLabel} ${key} must be ${expected}`);
  }
  if (typeof lock.manifest_sha256 !== "string" || typeof lock.checksums_sha256 !== "string") {
    fail(`${sourceLabel} lock is missing manifest/checksums hashes`);
  }
  for (const artifact of artifactPaths(lock)) assertRelativeArtifact(artifact);
}

function validateBundle(base, lock, sourceLabel, { strictFiles = false } = {}) {
  validateLockShape(lock, sourceLabel);
  for (const relativePath of bundlePaths(lock)) {
    const path = join(base, relativePath);
    if (!existsSync(path)) fail(`${sourceLabel} is missing ${relativePath}`);
  }
  const expectedFiles = new Set(bundlePaths(lock));
  const artifactDirectories = ["catalog/", "fixtures/", "schemas/"];
  const extras = bundleFiles(base).filter((path) => !expectedFiles.has(path)
    && (strictFiles || artifactDirectories.some((prefix) => path.startsWith(prefix))));
  if (extras.length > 0) fail(`${sourceLabel} has unlocked artifact(s): ${extras.join(", ")}`);
  const manifestPath = join(base, "manifest.json");
  const checksumsPath = join(base, "checksums.sha256");
  if (sha256(manifestPath) !== lock.manifest_sha256) fail(`${sourceLabel} manifest SHA-256 does not match lock`);
  if (sha256(checksumsPath) !== lock.checksums_sha256) fail(`${sourceLabel} checksums SHA-256 does not match lock`);

  const checksums = new Map();
  for (const line of readFileSync(checksumsPath, "utf8").split(/\r?\n/)) {
    if (!line.trim()) continue;
    const match = /^(\w{64})  (.+)$/.exec(line);
    if (!match) fail(`${sourceLabel} checksums has malformed line: ${line}`);
    if (checksums.has(match[2])) fail(`${sourceLabel} checksums has duplicate entry: ${match[2]}`);
    checksums.set(match[2], match[1]);
  }
  const expectedChecksumPaths = new Set(artifactPaths(lock));
  const checksumExtras = [...checksums.keys()].filter((path) => !expectedChecksumPaths.has(path));
  if (checksumExtras.length > 0) fail(`${sourceLabel} checksums has unlocked artifact(s): ${checksumExtras.join(", ")}`);
  for (const artifact of artifactPaths(lock)) {
    const expected = lock.artifacts[artifact];
    const actual = sha256(join(base, artifact));
    if (actual !== expected) fail(`${sourceLabel} artifact SHA-256 mismatch: ${artifact}`);
    if (checksums.get(artifact) !== expected) fail(`${sourceLabel} checksums entry mismatch: ${artifact}`);
  }
  const bundledLock = readConsumerLock(join(base, "consumer-lock.v1.json"), `${sourceLabel} bundled`);
  if (JSON.stringify(bundledLock) !== JSON.stringify(lock)) fail(`${sourceLabel} bundled consumer lock differs from root lock`);
}

function copyBundle(sourceRoot, lock) {
  const targetRoot = join(vendorRoot, `v${lock.contract_version}`);
  mkdirSync(targetRoot, { recursive: true });
  for (const relativePath of bundlePaths(lock)) {
    const sourcePath = join(sourceRoot, relativePath);
    const targetPath = join(targetRoot, relativePath);
    mkdirSync(dirname(targetPath), { recursive: true });
    cpSync(sourcePath, targetPath);
  }
  mkdirSync(vendorRoot, { recursive: true });
  cpSync(join(sourceRoot, "consumer-lock.v1.json"), vendorLockPath);
}

function generateCatalog(lock, catalog) {
  assertObject(catalog, "catalog");
  assertObject(catalog.default_models, "catalog.default_models");
  assertObject(catalog.backends, "catalog.backends");
  const models = [];
  for (const [backend, backendConfig] of Object.entries(catalog.backends)) {
    if (!backendConfig || typeof backendConfig !== "object" || Array.isArray(backendConfig)) fail(`catalog backend ${backend} is invalid`);
    const backendModels = backendConfig.models;
    if (!backendModels || typeof backendModels !== "object" || Array.isArray(backendModels)) fail(`catalog backend ${backend}.models is invalid`);
    for (const [name, model] of Object.entries(backendModels)) {
      if (!model || typeof model !== "object" || Array.isArray(model)) fail(`catalog model ${backend}/${name} is invalid`);
      models.push({ backend, ...model });
    }
  }
  const artifactMap = Object.fromEntries(Object.entries(lock.artifacts).sort(([left], [right]) => left.localeCompare(right)));
  const output = `/**\n * Generated by scripts/contract-sync.mjs. Do not edit by hand.\n * Source: vv-llm-contract consumer-lock.v1.json and catalog/default-chat-catalog.json\n */\nimport type { ModelConfig } from "../types.js";\n\nexport const CONTRACT_VERSION = ${JSON.stringify(lock.contract_version)} as const;\nexport const CONTRACT_SCHEMA_VERSION = ${lock.schema_version} as const;\nexport const CONTRACT_FIXTURE_VERSION = ${lock.fixture_version} as const;\nexport const CONTRACT_CATALOG_REVISION = ${lock.catalog_revision} as const;\nexport const CONTRACT_CONSUMER_LOCK_SHA256 = ${JSON.stringify(EXPECTED_CONSUMER_LOCK_SHA256)} as const;\nexport const CONTRACT_MANIFEST_SHA256 = ${JSON.stringify(lock.manifest_sha256)} as const;\nexport const CONTRACT_CHECKSUMS_SHA256 = ${JSON.stringify(lock.checksums_sha256)} as const;\nexport const CONTRACT_ARTIFACTS = ${JSON.stringify(artifactMap, null, 2)} as const;\nexport const CONTRACT_DEFAULT_MODELS = ${JSON.stringify(catalog.default_models, null, 2)} as const;\n\nexport interface ContractCatalogModelConfig extends ModelConfig {\n  backend: string;\n}\n\nexport const DEFAULT_MODEL_CONFIGS: readonly ContractCatalogModelConfig[] = ${JSON.stringify(models, null, 2)};\n`;
  return output;
}

function ensureGeneratedCatalog(lock, catalog, mode) {
  const expected = generateCatalog(lock, catalog);
  if (mode === "write") {
    mkdirSync(dirname(generatedCatalogPath), { recursive: true });
    writeFileSync(generatedCatalogPath, expected, "utf8");
    return;
  }
  if (!existsSync(generatedCatalogPath)) fail(`generated catalog is missing: ${relative(root, generatedCatalogPath)}`);
  const actual = readFileSync(generatedCatalogPath, "utf8");
  if (actual !== expected) fail(`generated catalog is stale: ${relative(root, generatedCatalogPath)}; run npm run contract:sync`);
}

function compareSourceLock(sourceLock, vendorLock) {
  if (JSON.stringify(sourceLock) !== JSON.stringify(vendorLock)) {
    fail("vendor consumer lock differs from explicit contract source lock; run npm run contract:sync -- --source <contract-source>");
  }
}

function main() {
  if (!checkOnly && !sourceRoot) {
    fail("contract sync requires an explicit --source path or VV_LLM_CONTRACT_SOURCE");
  }
  if (!checkOnly) {
    if (!existsSync(sourceRoot)) fail(`contract source tree is not available: ${sourceRoot}`);
    const sourceLock = readConsumerLock(join(sourceRoot, "consumer-lock.v1.json"), "contract source");
    validateBundle(sourceRoot, sourceLock, "contract source");
    copyBundle(sourceRoot, sourceLock);
  }

  if (!existsSync(vendorLockPath)) {
    fail(`vendor consumer lock is missing: ${relative(root, vendorLockPath)}; run npm run contract:sync`);
  }
  const vendorLock = readConsumerLock(vendorLockPath, "vendored");
  const versionRoot = join(vendorRoot, `v${vendorLock.contract_version}`);
  validateBundle(versionRoot, vendorLock, "vendored contract", { strictFiles: true });
  if (sourceRoot) {
    if (!existsSync(sourceRoot)) fail(`contract source tree is not available: ${sourceRoot}`);
    const sourceLock = readConsumerLock(join(sourceRoot, "consumer-lock.v1.json"), "contract source");
    validateBundle(sourceRoot, sourceLock, "contract source");
    compareSourceLock(sourceLock, vendorLock);
  }
  const catalog = readJson(join(versionRoot, "catalog", "default-chat-catalog.json"));
  ensureGeneratedCatalog(vendorLock, catalog, checkOnly ? "check" : "write");
  console.log(`${checkOnly ? "checked" : "synced"} vv-llm-contract ${vendorLock.contract_version} catalog_revision=${vendorLock.catalog_revision}`);
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
