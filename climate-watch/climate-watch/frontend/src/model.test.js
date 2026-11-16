import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { normalizeSource } from "./data.js";
import {
  filterCameras,
  validBox,
  modelDetections,
  frameAtTime,
  formatTime,
} from "./model.js";

const rawLive = JSON.parse(
  readFileSync(new URL("../public/live-cameras/cameras.json", import.meta.url)),
);
const live = rawLive.map(normalizeSource);
const cases = JSON.parse(
  readFileSync(new URL("../public/evidence/sources.json", import.meta.url)),
).map(normalizeSource);

test("continuous feeds and refreshed clips remain distinct, with official coordinates and verified playback", () => {
  assert.equal(live.filter((source) => source.transport === "hls").length, 10);
  assert.equal(
    live.filter((source) => source.transport === "mp4-refresh").length,
    5,
  );
  for (const source of live) {
    assert.equal(source.mediaType, "video");
    assert.equal(new URL(source.streamUrl).protocol, "https:");
    assert.equal(source.coordinates.length, 2);
    assert.ok(source.verification.sampleDecoded);
    if (source.transport === "hls")
      assert.ok(source.verification.sequenceAdvanced);
  }
  assert.equal(filterCameras(live, " london ", "all").length, 5);
});

test("recorded evidence has actual media and unknown camera positions stay unknown", () => {
  assert.equal(new Set(cases.map((source) => source.id)).size, cases.length);
  assert.ok(
    cases.some(
      (source) => source.region === "Europe" && source.topic === "smoke",
    ),
  );
  assert.equal(cases.length, 3);
  assert.equal(
    cases.find((source) => source.id === "bali-river-waste").coordinates,
    null,
  );
  for (const source of cases) {
    assert.ok(source.mediaPath.startsWith("/evidence/"));
    assert.ok(source.poster);
    assert.equal(new URL(source.sourceUrl).protocol, "https:");
    assert.ok(source.license);
  }
});

test("only normalized valid detections render and overlays do not drift to unsampled timestamps", () => {
  const valid = { box: [0.1, 0.2, 0.5, 0.6] };
  assert.equal(validBox(valid.box), true);
  for (const box of [[0, 0, 2, 1], [1, 0, 0, 1], [0, NaN, 1, 1], null])
    assert.equal(validBox(box), false);
  assert.equal(
    modelDetections({ detections: [valid, { box: [0, 0, 2, 1] }] }).length,
    1,
  );
  const frames = [{ time_s: 0 }, { time_s: 4 }, { time_s: 8 }];
  assert.equal(frameAtTime(frames, 4.1), frames[1]);
  assert.equal(frameAtTime(frames, 2), null);
  assert.equal(formatTime(65.5), "1:05");
});

test("shipped evidence carries genuine model provenance and valid masks", () => {
  for (const source of cases) {
    const result = JSON.parse(
      readFileSync(
        new URL(
          "../public/evidence/results/" + source.id + ".json",
          import.meta.url,
        ),
      ),
    );
    assert.equal(result.provenance.mode, "model_inference");
    assert.match(result.provenance.input_sha256, /^[0-9a-f]{64}$/);
    assert.ok(result.models.length >= 2);
    for (const frame of result.frames) {
      assert.ok(frame.thumbnail.startsWith("data:image/jpeg;base64,"));
      for (const detection of frame.detections) {
        assert.ok(validBox(detection.box));
        assert.ok(detection.model_id);
        assert.ok(["smoke", "fire", "waste"].includes(detection.hazard));
        if (detection.mask)
          assert.ok(
            detection.mask.every(
              (point) =>
                point.length === 2 &&
                point.every(
                  (value) => Number.isFinite(value) && value >= 0 && value <= 1,
                ),
            ),
          );
      }
    }
  }
});
