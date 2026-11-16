# Test report (Nov 2026) — `pytest -q tests` → 66 passed, 1 xfailed

**Read this caveat first:** all CV tests use a *synthetic* depth-haze scene (`tests/synth.py`, Koschmieder model with known sky ground truth).
They prove internal consistency and robustness to camera factors. They do **not** prove real-world accuracy: that needs real CCTV frames +
station PM2.5 (`scripts/spike.py`, `scripts/calibrate.py`). Real YOLO26-sem was **not** run (no torch in the sandbox) → `scripts/verify_sem.py`.

| Factor | Result | Status |
|---|---|---|
| Haze monotonic in extinction (7 levels), with/without sky | strictly monotonic; ~25 raw pts clear→dense | PASS |
| Camera gain / auto-exposure (0.5–1.3×) | drift ≤3 pts (was up to 5 before contrast normalisation) | PASS |
| Over-exposure / clipping | flagged `saturated_clipping` (was silent) | PASS (fixed) |
| Sensor noise σ=5–20 | drift ≤4 pts (was −12 at σ=20 before denoise) | PASS (fixed) |
| JPEG q=90→20 | drift ≤3 pts | PASS |
| Resolution 320–1920 px | drift ≤2 pts | PASS |
| Night / dark | `night_or_dark`, confidence <0.5, no alert | PASS |
| Blur / lens fog | `blurry_or_lens_fog` flagged | PASS |
| No sky visible | `no_usable_sky`, confidence −0.2, still monotonic | PASS |
| Smoke (yellow) vs fog (grey) | `sky_yellowness` separates them (≥0.5 gap) | PASS |
| Camera white balance ±15% | raw drift ≤9 pts → **cancelled by per-camera baseline** (≤8 rel pts). Colour removed from the index (it caused the drift) and kept as a separate cue | PASS (design changed) |
| Overcast sky | index still monotonic; cloud vs haze not separable from one frame | KNOWN LIMIT |
| Degenerate input (black, white, flat, 8×8, gray, BGRA, float, uint16) | no NaN, index in 0-100 | PASS |
| Temporal stability (30 noisy frames) | std <1 pt | PASS |
| Speed (CPU, 640 px) | ~30 ms haze + ~4 ms sky heuristic | PASS |
| Heuristic sky IoU (blue / overcast / yellow smoke) | ≥0.8 up to moderate haze | PASS |
| Heuristic sky at extreme haze (β=3e-3) | IoU ≈0.48 | XFAIL (known; expect sem model to do better — verify) |
| Privacy blur (inside box, edges, degenerate boxes, no mutation) | correct | PASS |
| Playbook covers every detectable hazard; no invented advice | enforced by test | PASS |
| Flood-risk monotonic in rain & hazards | correct | PASS |
| Worker: licence guard, 60 s ring buffer, interval, baseline warm-up→relative, alert, offline/recovery, purge, atomic JSON, no private data in public | correct | PASS |
| Worker: global lighting change ≠ event; new object = event | gain ≤1.5× ignored; object caught | PASS (fixed) |
| Worker: heartbeat re-checks persistent hazards | correct | PASS (fixed design) |
| API contract: real Gradio Space ↔ real `gradio_client` ↔ worker tick | works after `handle_file()` fix | PASS (bug found) |

## Bugs / design flaws found by testing (all fixed)
1. Index range compressed + camera-dependent → added per-camera clear-day baseline (`climatewatch/baseline.py`, `relative_haze`); alerts use the relative index.
2. Sky-colour term made the index WB-sensitive → colour is now a separate cue (`sky_yellowness`, flag `yellow_sky_cue`).
3. Noise biased the dark channel low → denoise before DCP. 4. Contrast term depended on gain → normalised by mean brightness.
5. Colour-based sky heuristic failed in yellow smoke (IoU 0.40) → texture/connectivity-based heuristic.
6. "60 s buffer" actually spanned 30 min → 1 fps buffer decoupled from analysis interval.
7. Change-only trigger would never see persistent garbage → detector heartbeat every 10 min.
8. Global lighting changes triggered false events → gain-normalised, blob-based change detection.
9. Worker passed a plain path to the Space → `handle_file()` (only the real-client integration test caught it).
10. Private context frames grew without bound → auto-purge after `PRIVATE_RETENTION_H` (24 h).

## Still NOT tested (needs real data / hardware)
Real CCTV frames; rain/lens droplets; sun glare; PTZ movement; IR night mode; real yolo26-sem output + domain shift; waste/drain/flood detectors (no weights);
haze-vs-station correlation; Space cold-start/ZeroGPU quota behaviour; real RTSP/HLS streams; load with 10+ cameras; frontend.
