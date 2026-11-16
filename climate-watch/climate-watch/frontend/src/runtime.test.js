import test from "node:test";
import assert from "node:assert/strict";
import { localScanningEnabled } from "./runtime.js";
test("hosted and explicit demo builds never enable local scan services", () => {
  for (const host of [
    "climatewatch.vercel.app",
    "climatewatch.example",
    "localhost.example",
  ])
    assert.equal(localScanningEnabled(host, "production"), false);
  for (const host of ["localhost", "127.0.0.1", "[::1]"]) {
    assert.equal(localScanningEnabled(host, "demo"), false);
    assert.equal(localScanningEnabled(host, "production"), true);
  }
});
