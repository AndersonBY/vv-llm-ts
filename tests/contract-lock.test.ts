import assert from "node:assert/strict";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

test("contract check rejects a tampered consumer lock before JSON trust", () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "vv-llm-ts-contract-lock-"));
  try {
    const scriptSource = fileURLToPath(new URL("../../scripts/contract-sync.mjs", import.meta.url));
    const vendorSource = fileURLToPath(new URL("../../contract/", import.meta.url));
    const generatedSource = fileURLToPath(new URL("../../src/generated/", import.meta.url));
    const scriptTarget = join(tempRoot, "scripts", "contract-sync.mjs");
    const vendorTarget = join(tempRoot, "contract");
    const generatedTarget = join(tempRoot, "src", "generated");
    mkdirSync(join(tempRoot, "scripts"), { recursive: true });
    mkdirSync(generatedTarget, { recursive: true });
    cpSync(scriptSource, scriptTarget);
    cpSync(vendorSource, vendorTarget, { recursive: true });
    cpSync(generatedSource, generatedTarget, { recursive: true });

    const lockPath = join(vendorTarget, "consumer-lock.v1.json");
    const original = readFileSync(lockPath, "utf8");
    const tampered = original.replace('"catalog_revision": 3', '"catalog_revision": 1');
    assert.notEqual(tampered, original);
    writeFileSync(lockPath, tampered, "utf8");

    const result = spawnSync(process.execPath, [scriptTarget, "--check"], {
      cwd: tempRoot,
      encoding: "utf8",
    });
    assert.notEqual(result.status, 0);
    assert.match(`${result.stdout}\n${result.stderr}`, /consumer lock SHA-256 mismatch/);
  } finally {
    rmSync(tempRoot, { recursive: true, force: true });
  }
});

test("contract check rejects an extra unlocked artifact", () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "vv-llm-ts-contract-extra-"));
  try {
    const scriptSource = fileURLToPath(new URL("../../scripts/contract-sync.mjs", import.meta.url));
    const vendorSource = fileURLToPath(new URL("../../contract/", import.meta.url));
    const generatedSource = fileURLToPath(new URL("../../src/generated/", import.meta.url));
    const scriptTarget = join(tempRoot, "scripts", "contract-sync.mjs");
    const vendorTarget = join(tempRoot, "contract");
    const generatedTarget = join(tempRoot, "src", "generated");
    mkdirSync(join(tempRoot, "scripts"), { recursive: true });
    mkdirSync(generatedTarget, { recursive: true });
    cpSync(scriptSource, scriptTarget);
    cpSync(vendorSource, vendorTarget, { recursive: true });
    cpSync(generatedSource, generatedTarget, { recursive: true });
    writeFileSync(join(vendorTarget, "v1.0.1", "fixtures", "unlocked.json"), "{}", "utf8");

    const result = spawnSync(process.execPath, [scriptTarget, "--check"], {
      cwd: tempRoot,
      encoding: "utf8",
    });
    assert.notEqual(result.status, 0);
    assert.match(`${result.stdout}\n${result.stderr}`, /unlocked artifact/);
  } finally {
    rmSync(tempRoot, { recursive: true, force: true });
  }
});

test("contract check is vendor-only by default and compares an explicit source", () => {
  const tempRoot = mkdtempSync(join(tmpdir(), "vv-llm-ts-contract-source-"));
  try {
    const scriptSource = fileURLToPath(new URL("../../scripts/contract-sync.mjs", import.meta.url));
    const vendorSource = fileURLToPath(new URL("../../contract/", import.meta.url));
    const generatedSource = fileURLToPath(new URL("../../src/generated/", import.meta.url));
    const scriptTarget = join(tempRoot, "scripts", "contract-sync.mjs");
    const vendorTarget = join(tempRoot, "contract");
    const generatedTarget = join(tempRoot, "src", "generated");
    const sourceTarget = join(tempRoot, "contract-source");
    const releaseTarget = join(vendorTarget, "v1.0.1");
    mkdirSync(join(tempRoot, "scripts"), { recursive: true });
    mkdirSync(generatedTarget, { recursive: true });
    mkdirSync(sourceTarget, { recursive: true });
    cpSync(scriptSource, scriptTarget);
    cpSync(vendorSource, vendorTarget, { recursive: true });
    cpSync(generatedSource, generatedTarget, { recursive: true });
    for (const name of ["manifest.json", "checksums.sha256", "consumer-lock.v1.json"]) {
      cpSync(join(releaseTarget, name), join(sourceTarget, name));
    }
    for (const name of ["catalog", "fixtures", "schemas"]) {
      cpSync(join(releaseTarget, name), join(sourceTarget, name), { recursive: true });
    }
    // A source checkout may contain repository documentation/scripts outside
    // the locked contract artifact directories.
    writeFileSync(join(sourceTarget, "README.md"), "source checkout", "utf8");

    const environment = { ...process.env };
    delete environment.VV_LLM_CONTRACT_SOURCE;
    const vendorOnly = spawnSync(process.execPath, [scriptTarget, "--check"], {
      cwd: tempRoot,
      env: environment,
      encoding: "utf8",
    });
    assert.equal(vendorOnly.status, 0, `${vendorOnly.stdout}\n${vendorOnly.stderr}`);
    const withSource = spawnSync(process.execPath, [scriptTarget, "--check", "--source", sourceTarget], {
      cwd: tempRoot,
      env: environment,
      encoding: "utf8",
    });
    assert.equal(withSource.status, 0, `${withSource.stdout}\n${withSource.stderr}`);
  } finally {
    rmSync(tempRoot, { recursive: true, force: true });
  }
});
