# ClimateWatch

<p align="center">
  <strong>Observe environmental conditions. Inspect the evidence. Understand what to do next.</strong>
</p>

ClimateWatch turns existing public CCTV and webcam feeds into an interface for environmental awareness. Select a camera on an interactive map, view conditions at that location, and explore visual evidence of smoke, fire, or waste alongside practical response and prevention guidance.

The prototype demonstrates how computer vision can support environmental monitoring without installing new cameras.

## Highlights

- Interactive worldwide camera map with 2D and 3D views.
- Large camera viewer for live video, periodically updated clips, and camera snapshots.
- A separate evidence collection with recorded footage, detection boxes, and segmentation masks.
- Frame-by-frame inspection of saved model results, with source dates and attribution.
- Response and prevention guidance linked to the observed condition.
- Downloadable evidence records with model provenance.
- Local computer vision processing for smoke, fire, and waste candidates.

## Demo and Local Application

The Vercel demo presents camera feeds and previously generated evidence. New scans and automatic monitoring run only in the local application.

| Capability                          | Web demo                              | Local application     |
| ----------------------------------- | ------------------------------------- | --------------------- |
| Interactive camera map              | Available                             | Available             |
| Public camera playback              | Available                             | Available             |
| Saved evidence, masks, and guidance | Available                             | Available             |
| Scan an uploaded image or clip      | Local only (for now due to prototype) | Available with models |
| Automatic camera analysis           | Local only (for now due to prototype) | Available with models |
| Save newly detected camera evidence | Local only (for now due to prototype) | Stored locally        |

The current collection contains 10 live-video sources, five periodically updated London clips, and three Finland camera snapshots. The three recorded evidence examples include one smoke video and two waste photographs. Historical evidence does not describe current conditions at a live camera.

## How It Works

**Camera viewing:** select a location, open its camera panel, and view the provider's footage. Camera feeds and recorded evidence remain separate throughout the interface.

**Local monitoring:** the Python service samples camera frames, runs detection and segmentation, and compares observations across checks. Repeated, spatially matching smoke, fire, or waste candidates can produce an evidence record. When no supported target passes the detection policy, no incident or action card is created.

**Evidence review:** inspect the analyzed frames and model overlays, read the associated guidance, and check the source and limitations before interpreting a finding.

## Technology

| Layer                      | Tools                                            |
| -------------------------- | ------------------------------------------------ |
| Interface                  | React, Vite, Lucide                              |
| Map                        | MapLibre GL JS, OpenFreeMap                      |
| Video playback             | HLS.js, browser video                            |
| Local inference            | Python, PyTorch, Ultralytics, Transformers       |
| Detection and segmentation | YOLOE, a fine-tuned smoke/fire detector, CLIPSeg |
| Image processing           | OpenCV, Pillow, NumPy                            |
| Local evidence storage     | SQLite                                           |
| Web deployment             | Vercel static hosting                            |

## Run the Web Demo

Use Node.js 22.12 or later with npm. Internet access is required for the map and public camera feeds. Python, model weights, and API keys are not required for this version.

From the project root:

```bash
cd frontend
npm ci
npm run build:demo
npm run preview
```

Open the local URL printed by Vite, normally [http://127.0.0.1:4173](http://127.0.0.1:4173). This build uses the same local-only scanning labels as the hosted demo.

## Deploy to Vercel

Import the repository into Vercel and select the `frontend` directory as the project root. The included [configuration](frontend/vercel.json) provides these settings:

| Setting               | Value                |
| --------------------- | -------------------- |
| Framework preset      | Vite                 |
| Install command       | `npm ci`             |
| Build command         | `npm run build:demo` |
| Output directory      | `dist`               |
| Environment variables | None required        |

Deploy the frontend only. The Python service and model files are not part of the Vercel deployment, and the hosted interface does not call the local scanning API.

## Run Local Scanning

The local application was tested with Python 3.12. It also requires the frontend dependencies and the prepared model assets described below.

From the project root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS and Linux:

```bash
source .venv/bin/activate
```

Install the Python dependencies and build the local frontend:

```bash
python -m pip install -r requirements-prototype.txt
cd frontend
npm ci
npm run build
cd ..
python scripts/serve_prototype.py
```

Open [http://127.0.0.1:8766](http://127.0.0.1:8766). Keep the server running for automatic camera checks. On Windows, `Start-ClimateWatch.cmd` starts the same service after setup.

### Model Assets

Local inference expects these assets under `models/`:

| Path                           | Purpose                                                              |
| ------------------------------ | -------------------------------------------------------------------- |
| `environment-yoloe-26s-seg.pt` | YOLOE checkpoint prepared with the application's environment prompts |
| `fire-smoke-yolov8n.pt`        | Smoke and flame detection                                            |
| `clipseg/`                     | CLIPSeg weights, model configuration, and tokenizer files            |

Installing Python packages does not prepare these assets. The environment checkpoint must contain the expected prompt configuration; renaming an unconfigured checkpoint is not sufficient. The prepared files are required to reproduce the local prototype. They are unnecessary for viewing the static demo and its saved results.

The experimental waste checkpoint is not enabled by default: it did not pass the prototype's false-positive and example-regression checks.

## Development and Checks

For frontend development, run the Python service and `npm run dev` from `frontend` in separate terminals. Vite forwards local API requests to port 8766.

Frontend checks:

```bash
cd frontend
npm test
npm run build:demo
```

Local service checks, from the project root:

```bash
python -m unittest discover -s scripts -p "test_*.py"
```

Additional backend tests are maintained in `tests/`. Automated checks verify application behavior; they are not a measurement of real-world detection accuracy.

## Project Structure

```text
frontend/src/               Camera map, viewer, and evidence interface
frontend/public/            Camera catalogs, licensed media, and saved results
frontend/vercel.json         Static demo deployment configuration
climatewatch/                Inference and environmental analysis modules
scripts/                    Local server, monitoring, research, and training tools
worker/                     Separate ingestion-worker implementation
models/                     Local model assets and experiment results
data/                       Local runtime evidence and other generated data
tests/                      Additional backend tests
docs/                       Research sources and previous test reports
```

## Scope and Limitations

- Model outputs are candidate observations that require review. Scores are not calibrated probabilities.
- Visible smoke does not establish wildfire cause or a PM2.5 measurement. Waste near a drain does not establish complete blockage or predict flooding.
- Local monitoring samples frames; it does not analyze every frame from every camera.
- Camera availability, update frequency, and browser playback depend on the provider.
- Heat stress, drought, and broader environmental scenario exploration are future applications, not implemented features of this demo.

## Sources and Attribution

Camera and media records retain their provider, source URL, license information, and recording dates. Saved analysis retains its model provenance. Public viewing access does not automatically grant redistribution rights; consult the provider's terms before reusing a feed or recording.

The model stack includes AGPL-3.0 and Apache-2.0 components. Their licenses and the individual media licenses remain applicable. This repository does not currently include a project-wide license file.

Interface design reference: Damir Sokolovsky (@heydamir), CC BY 4.0.
