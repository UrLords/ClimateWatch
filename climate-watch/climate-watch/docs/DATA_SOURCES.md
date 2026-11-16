# Data sources, limits & validation plan (researched 12 Nov 2026)
Items marked **(verify)** were not confirmed on the primary page. Re-check licences/ToS before relying on anything.

## 1. Legal camera feeds
| Source | What you get | Licence / limits | Verdict |
|---|---|---|---|
| **Singapore LTA Traffic Images** (data.gov.sg dataset `d_6cdb6b405b25aaaacbaf7689bcc6fae0`) | Snapshots with camera id + lat/lon, island-wide | Open Data Licence, free for commercial use. data.gov.sg rate limits per 10 s: lowest without key, higher with free dev/prod key (guide.data.gov.sg/developer-guide/api-overview/api-rate-limits) | **Best first site**: legal, mapped, and has real air-quality ground truth (below). Snapshot refresh ~20 s **(verify)** => no 1 fps context; use `poll_sec: 20` |
| **Denpasar ATCS** (atcs.denpasarkota.go.id) + **Bali Satu Data peta-cctv** | Public HTTPS HLS streams; ~105-164 Denpasar cams (counts differ by source), ~91 provincial cams; Badung = locations only | Government portals. Third-party viewers call the ATCS API with client id/secret headers copied from the site => **not an official open API**. Ask Dishub/Diskominfo for written permission (or an MoU) before scraping | Great local demo story (you are in Bali) **only with permission**; fall back to replaying archived frames |
| **Windy Webcams API v3** | Stills, 5-15 min cadence, ~65k cams | Free tier = link/embed, small images, image URLs expire in 10 min, attribution required; Pro 9,990 EUR/yr | **Not suitable** for CV analysis on free tier (check ToS) |
| Exposed/unsecured RTSP (Insecam-style) | - | Unauthorised access risk | **Never** |

## 2. Ground truth for haze vs station
- **Singapore NEA via data.gov.sg**: `/v2/real-time/api/pm25` (1-hr PM2.5) and `/psi`, hourly, **5 regions** (coarse: expect a weaker correlation than a co-located sensor); historical 1-hr PM2.5 CSV Apr 2014-Jul 2026 (dataset `d_6c0d5fc34838b12472475fdb73c0af29`). Good for real transboundary-haze episodes.
- **Bali is data-poor** (third-party brief baliair.pages.dev **(verify)**): official Denpasar KLHK station offline since Aug 2025 (archive on AQICN, 548 days); one official station in Sempidi (Badung); Nafas has 6 Bali stations; PurpleAir/IQAir/AirGradient community sensors. Typical PM2.5 15-22 ug/m3 => **low dynamic range** => poor correlation likely unless a fire/haze episode occurs.
- **OpenAQ v3**: free key, 60 req/min, 2,000/h; radius search by lat/lon.
- Literature reality check: camera->PM2.5 studies report R2 ~0.76 (daytime, deep model, fitted per site) and r ~0.82 (hand-crafted DCP + fit); the HVAQ open dataset (arXiv 2102.09332) has only 106 images over 3 days, **no night**, and notes rain/snow are mistaken for pollution. Do not promise more than a per-camera, daytime, clear-weather Spearman with CI.

## 3. Datasets for hard conditions (to test/fine-tune)
| Need | Dataset | Notes |
|---|---|---|
| fog / night / rain / snow segmentation | **ACDC** (4,006 adverse + 1,503 normal imgs; Cityscapes label set) acdc.vision.ee.ethz.ch | Directly tests yolo26-sem sky/road. Car-view, not CCTV. Check licence (likely non-commercial) **(verify)** |
| rain, glare, puddle reflections, lens drops, night, thermal | **AAU RainSnow Traffic Surveillance** (22 x 5 min videos, 7 intersections, RGB+thermal) vbn.aau.dk | Real surveillance footage; best match to CCTV |
| rain in surveillance | **RIS set** (2,048 imgs from 154 surveillance cams, in the deraining benchmark arXiv 1903.08558) | |
| raindrops on lens (day/night) | **Raindrop Clarity** (15,186 pairs, 9,744 night; arXiv 2407.16957), Raindrops-on-Windshield | For a lens-droplet detector -> mark low confidence |
| night glare | Flare7K | Mostly synthetic flare |
| **night IR** | none found with haze/PM labels | Treat `ir_mono_mode` as low confidence; collect own |
| waste (street) | TACO (1.5k imgs), DetectWaste (aggregate), MDWD (3,697 street-level, arXiv 2608.00257), GVP (fixed-camera binary waste) | UAVVaste/dumpsite sets are drone views (domain gap) |
| illegal dumping (video, fixed cams) | **MIVIA-IWDD** (WACV 2026 WasteVision workshop paper) | Closest to the "behaviour" feature |
| flood on streets | FloodNet = UAV aerial, CC BY-SA 4.0, only 37 "road flooded" imgs => poor fit; **AlleyFloodNet** (ground-level classification, preprint) better | Expect to label your own water-on-road masks |
| blocked drains | **no public dataset found** | Label a few hundred images yourself |

## 4. ZeroGPU facts and what they mean for this design
- Hosting: free accounts in good standing (verified email, >30 days) can host up to 2 ZeroGPU Spaces; PRO up to 10. (An older doc revision said PRO was required **(verify on your account)**.)
- Daily GPU quota (current docs): unauthenticated 2 min, free 5 min, PRO 40 min (+ pre-paid credits $1 per 10 min). Older revision: 3.5 / 25 min. Quota is **per calling identity**: worker calls with `HF_TOKEN` burn *your* quota. Default function window 60 s; shorter `duration` = better queue priority.
- **Budget math:** 10 cameras x heartbeat every 10 min = 1,440 detector calls/day; at ~1 s GPU each = ~24 min/day > free 5 min. => **Run YOLO26n detectors on CPU inside the worker**; use ZeroGPU only for heavy/occasional models (VLM captions, open-vocab labelling) or the public demo.
- Cold start: free CPU Spaces sleep after 48 h idle and restart on the next visitor; storage is ephemeral; community-reported cold boot ~2 min, worse with GPU **(verify)**. Mitigate: restart/warm the Space 20-30 min before a demo, ping it daily, give the worker a 120 s timeout + retry.

## 5. Load (measured with the real worker code, 1 vCPU sandbox, 1080p JPEG, heuristic sky, no network)
| Cameras | buffer tick | analysis round | avg CPU |
|---|---|---|---|
| 10 | 128 ms/s | 0.45 s | 14% of one core |
| 25 | 322 ms/s | 1.22 s | 36% |
| 50 | 629 ms/s | 2.45 s | 71% |
Network latency and HLS decoding are NOT included and the loop is sequential: use a thread pool for fetches and `poll_sec` per camera (20-60 s for snapshot feeds, 1 s only for HLS). yolo26-sem on CPU is unmeasured.

## 6. Mapping
- **MapLibre GL JS + OpenFreeMap** (vector tiles from OSM; no key, no registration, no stated view limits; OSM attribution required; donate or self-host for production). Fallback: **Protomaps PMTiles** self-hosted on Cloudflare R2/Pages. Leaflet works via maplibre-gl-leaflet.
- Feed the map straight from `public/nodes.json` (GeoJSON-ify client-side); cluster only above ~2,000 markers. Marker colour: green clear / red alert / grey offline / amber low_confidence or warming_up. Show `haze_relative`, `confidence`, flags, `advice[]`, and the blurred frame in the popup.
- Weather overlay: Open-Meteo (free for non-commercial use) already used in `risk.py`. Prior art for Bali camera maps: github.com/prshendra/bali-cctv, github.com/hendraptaksu/cctv-denpasar (the latter uses the ATCS site's credentials: do not copy that part).

## 7. Recommended validation order
1. Collect 5-7 days of LTA snapshots from ~10 cameras + NEA hourly PM2.5 -> `scripts/calibrate.py` (daytime, clear-weather subset; report Spearman + CI).
2. Run `scripts/verify_sem.py` and `scripts/spike.py` on ACDC (fog/night/rain) + AAU RainSnow frames; record sky IoU / flag behaviour per condition.
3. Ask Dishub Denpasar for permission; collect a Bali demo set; keep a replay fallback.
4. Fine-tune one detector (waste or flood-on-road) from MDWD/TACO/AlleyFloodNet + ~300 own labels; report precision/recall on held-out CCTV frames.
