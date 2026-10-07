import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export function readContractFixture<T>(relativePath: string): T {
  const path = fileURLToPath(new URL(relativePath, new URL("../../contract/v1.2.2/", import.meta.url)));
  return JSON.parse(readFileSync(path, "utf8")) as T;
}
