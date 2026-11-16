import React, { useEffect, useRef, useState } from "react";

import {
  Aperture,
  Camera,
  ChevronRight,
  ExternalLink,
  Focus,
  Globe2,
  Layers3,
  LoaderCircle,
  ScanLine,
  Search,
  Upload,
  X,
  Download,
} from "lucide-react";

import CameraMap from "./CameraMap.jsx";

import { loadSources } from "./data.js";

import {
  filterCameras,
  displayDate,
  validBox,
  modelDetections,
  formatTime,
} from "./model.js";

import Hls from "hls.js";
import { localScanningEnabled } from "./runtime.js";
const LOCAL_SCANNING = localScanningEnabled(
  location.hostname,
  import.meta.env.MODE,
);

const External = ({ href, children }) =>
  href ? (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
      <ExternalLink size={13} />
    </a>
  ) : null;

const textOf = (item) =>
  typeof item === "string" ? item : item?.text || item?.action || "";

const colorFor = (label) =>
  /smoke|fire|flame/i.test(label)
    ? "#ffb55b"
    : /water|flood/i.test(label)
      ? "#72d5f4"
      : /drain/i.test(label)
        ? "#f4df66"
        : "#a9ee8f";

const monitoringLabel = (state) =>
  ({
    waiting: "Queued for automatic check",
    scanning: "Checking camera automatically",
    waiting_for_new_frame: "Waiting for a new camera frame",
    no_target_detected: "No target detected",
    candidate_pending: "Candidate awaiting another check",
    evidence_captured: "Evidence captured automatically",
    offline: "Camera or analysis unavailable",
  })[state?.status] || "Waiting for monitor";
const isLive = (source) =>
  Boolean(source.liveSource || source.streamUrl || source.liveImageUrl);

const scoreLabel = (item) =>
  item.score_type === "mean_mask_activation"
    ? item.confidence.toFixed(2) + " mask score"
    : Math.round(item.confidence * 100) + "% confidence";

const transportLabel = (source) =>
  source.transport === "hls"
    ? "Live video"
    : source.transport === "mp4-refresh"
      ? "Updating video clip"
      : isLive(source)
        ? "Updating camera image"
        : source.mediaType === "video"
          ? "Recorded clip"
          : "Evidence photo";

function VideoSource({ url, hls, onError }) {
  const ref = useRef(null);

  useEffect(() => {
    const video = ref.current;

    let player;

    if (hls && Hls.isSupported()) {
      player = new Hls({
        maxBufferLength: 20,
        liveSyncDurationCount: 3,
        backBufferLength: 15,
      });

      player.loadSource(url);
      player.attachMedia(video);

      player.on(Hls.Events.ERROR, (_event, data) => {
        if (data.fatal)
          onError("The camera stream is unavailable. Try another camera.");
      });
    } else video.src = url;

    return () => {
      player?.destroy();
      video.removeAttribute("src");
      video.load();
    };
  }, [url, hls]);

  return (
    <video
      ref={ref}
      controls
      muted
      autoPlay={hls}
      playsInline
      preload="metadata"
      onError={() =>
        onError("The video could not be loaded. Try another camera.")
      }
      aria-label={hls ? "Live camera video" : "Recorded source video"}
    />
  );
}

function DetectionOverlay({ frame, width, height, active, roi }) {
  return (
    <svg
      className="detection-overlay"
      viewBox={"0 0 " + width + " " + height}
      preserveAspectRatio="xMidYMid meet"
      aria-label="Model detections and segmentation masks"
    >
      {modelDetections(frame).map((detection, index) => {
        const [x1, y1, x2, y2] = detection.box,
          color = colorFor(detection.label);

        const mask = detection.mask || [],
          label =
            detection.label +
            " " +
            (detection.score_type === "mean_mask_activation"
              ? detection.confidence.toFixed(2)
              : Math.round(detection.confidence * 100) + "%");

        return (
          <g
            key={index}
            opacity={active === null || active === index ? 1 : 0.25}
          >
            {mask.length >= 3 && Array.isArray(mask[0]) && (
              <polygon
                points={mask
                  .map(([x, y]) => x * width + "," + y * height)
                  .join(" ")}
                fill={color}
                fillOpacity=".3"
                stroke={color}
                strokeWidth="1.4"
                vectorEffect="non-scaling-stroke"
              />
            )}

            <rect
              x={x1 * width}
              y={y1 * height}
              width={(x2 - x1) * width}
              height={(y2 - y1) * height}
              fill="none"
              stroke={color}
              strokeWidth={active === index ? 3 : 1.5}
              vectorEffect="non-scaling-stroke"
            />

            <rect
              x={x1 * width}
              y={Math.max(0, y1 * height - height * 0.044)}
              width={Math.min(
                label.length * width * 0.0085 + 10,
                width - x1 * width,
              )}
              height={height * 0.044}
              fill={color}
            />

            <text
              x={x1 * width + 4}
              y={Math.max(height * 0.032, y1 * height - height * 0.011)}
              fill="#142019"
              fontSize={height * 0.028}
            >
              {label}
            </text>
          </g>
        );
      })}

      {validBox(roi) && (
        <g>
          <rect
            x={roi[0] * width}
            y={roi[1] * height}
            width={(roi[2] - roi[0]) * width}
            height={(roi[3] - roi[1]) * height}
            fill="#f4df6612"
            stroke="#f4df66"
            strokeDasharray="5 5"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />
          <text
            x={roi[0] * width + 4}
            y={Math.min(height - 5, roi[3] * height - 5)}
            fill="#fff"
            stroke="#142019"
            strokeWidth=".3"
            fontSize={height * 0.025}
          >
            Manual drain reference
          </text>
        </g>
      )}
    </svg>
  );
}

function EvidenceResult({ result, source, stored }) {
  const [index, setIndex] = useState(0),
    [playing, setPlaying] = useState(false),
    [overlay, setOverlay] = useState(true),
    [active, setActive] = useState(null);

  const frames = result.frames || [],
    frame = frames[index] || frames[0];

  const detections = modelDetections(frame),
    summary = result.summary || {};

  const width = result.media?.width || frame?.width || 960,
    height = result.media?.height || frame?.height || 540;

  const roi = frame?.drain_assessment?.drain_roi;

  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const timer = setInterval(
      () => setIndex((current) => (current + 1) % frames.length),
      650,
    );
    return () => clearInterval(timer);
  }, [playing, frames.length]);

  function download() {
    const blob = new Blob(
      [
        JSON.stringify(
          { source: { ...source, file: undefined }, analysis: result },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );

    const url = URL.createObjectURL(blob),
      link = document.createElement("a");

    link.href = url;
    link.download = "climatewatch-" + source.id + "-evidence.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <article className="evidence-result">
      <div className="section-top">
        <div>
          <span className="eyebrow">
            {stored ? "Saved model analysis" : "Captured by this scan"} ·{" "}
            {frames.length} sampled {frames.length === 1 ? "frame" : "frames"}
          </span>
          <h3>
            {summary.hazards?.length
              ? summary.hazards
                  .map((value) => value[0].toUpperCase() + value.slice(1))
                  .join(" / ") + " candidates"
              : "No target passed the detection threshold"}
          </h3>
        </div>
        <button
          className="icon-button"
          onClick={download}
          aria-label="Download model evidence"
        >
          <Download size={19} />
        </button>
      </div>

      <p className="capture-clock">
        {result.capture?.retrieved_at
          ? "Retrieved " +
            result.capture.retrieved_at.replace("T", " ").slice(0, 19) +
            " UTC · exact camera capture time unavailable"
          : source.recordedAt
            ? "Recorded " + displayDate(source.recordedAt)
            : "Upload capture time not provided"}
      </p>
      <div className="evidence-image">
        <img
          src={frame?.thumbnail || source.mediaPath}
          alt={
            "Analyzed evidence from " +
            source.name +
            ", sampled at " +
            formatTime(frame?.time_s)
          }
        />

        {overlay && (
          <DetectionOverlay
            frame={frame}
            width={width}
            height={height}
            active={active}
            roi={roi}
          />
        )}

        <span className="frame-label">
          Analyzed frame · {formatTime(frame?.time_s)}
        </span>
      </div>

      <div className="viewer-controls">
        <label>
          <input
            type="checkbox"
            checked={overlay}
            onChange={(event) => setOverlay(event.target.checked)}
          />
          Show masks & boxes
        </label>
        <span>Model scores · review required</span>
        {frames.length > 1 && (
          <button className="text-button" onClick={() => setPlaying(!playing)}>
            {playing ? "Pause samples" : "Play analyzed samples"}
          </button>
        )}
      </div>

      {frames.length > 1 && (
        <div className="frame-strip" aria-label="Sampled evidence frames">
          {frames.map((item, i) => (
            <button
              key={i}
              className={index === i ? "selected" : ""}
              onClick={() => {
                setIndex(i);
                setActive(null);
                setPlaying(false);
              }}
              aria-label={"Inspect sampled frame at " + formatTime(item.time_s)}
            >
              <img src={item.thumbnail} alt="" />
              <span>
                {formatTime(item.time_s)}
                <small>{modelDetections(item).length} detections</small>
              </span>
            </button>
          ))}
        </div>
      )}

      <div className="finding-list">
        {detections.map((item, i) => (
          <button
            key={i}
            onClick={() => setActive(active === i ? null : i)}
            aria-pressed={active === i}
          >
            <span
              className="finding-color"
              style={{ background: colorFor(item.label) }}
            />
            <strong>{item.label}</strong>
            <span>
              {item.mask?.length ? "Segmentation mask" : "Detection box"}
            </span>
            <b>{scoreLabel(item)}</b>
          </button>
        ))}
        {!detections.length && (
          <p>
            No supported target was detected in this sampled frame. This is not
            a safety assessment.
          </p>
        )}
      </div>

      {frame?.drain_assessment && (
        <p className="drain-finding">
          {Math.round(frame.drain_assessment.waste_mask_coverage * 100)}% of the
          manually marked drain region overlaps a waste mask.{" "}
          {frame.drain_assessment.status === "potential_obstruction"
            ? "Possible obstruction: request an on-site inspection."
            : "This scan does not show substantial waste overlap."}
        </p>
      )}

      <p className="time-context">{summary.time_context}</p>

      <div className="response-grid">
        <section>
          <h4>What to do</h4>
          <ol>
            {(summary.actions || []).map((item, i) => (
              <li key={i}>{textOf(item)}</li>
            ))}
          </ol>
        </section>
        <section>
          <h4>Prevent a repeat</h4>
          <ol>
            {(summary.prevention || []).map((item, i) => (
              <li key={i}>{textOf(item)}</li>
            ))}
          </ol>
        </section>
      </div>

      <div className="guidance-links">
        {(summary.sources || []).map((item) => (
          <External key={item.url} href={item.url}>
            {item.title}
          </External>
        ))}
      </div>

      <details className="technical-details">
        <summary>Models, provenance, and limitations</summary>
        <p>
          Mask scores are mean model activation inside the proposed region, not
          probabilities. Detection confidence is also uncalibrated.
        </p>
        <p>
          {result.models?.map((model) => model.id).join(" + ") ||
            result.model?.id}
        </p>
        <ul>
          {(summary.limitations || result.limitations || []).map((item, i) => (
            <li key={i}>{textOf(item)}</li>
          ))}
        </ul>
        <p>
          Masks describe visible appearance. They do not establish blocked water
          flow, PM2.5, or wildfire cause.
        </p>
        <pre>{JSON.stringify(result.provenance || {}, null, 2)}</pre>
      </details>
    </article>
  );
}

function CameraPanel({
  source,
  onClose,
  onResult,
  health,
  initialResult,
  cameraState,
  cameraEvent,
}) {
  const dialog = useRef(null),
    request = useRef(null);

  const [result, setResult] = useState(initialResult || null),
    [stored, setStored] = useState(false),
    [busy, setBusy] = useState(false);

  const [error, setError] = useState(""),
    [elapsed, setElapsed] = useState(0),
    [stamp, setStamp] = useState(source.recordedAt);

  const [tick, setTick] = useState(Date.now()),
    [loadingSaved, setLoadingSaved] = useState(true);

  const live = isLive(source),
    transport = source.transport || "snapshot";

  const evidenceRef = useRef(null);

  const sourceUrl = live
    ? source.streamUrl || source.liveImageUrl || source.liveSource?.imageUrl
    : source.mediaPath;

  const refreshMs = Math.max(
    30000,
    (source.liveSource?.collectionIntervalSeconds ||
      source.refreshSeconds ||
      600) * 1000,
  );

  useEffect(() => {
    dialog.current.showModal();
    return () => {
      request.current?.abort();
      dialog.current?.close();
    };
  }, []);

  useEffect(() => {
    const abort = new AbortController();

    if (live || source.file || initialResult) {
      setLoadingSaved(false);
      return () => abort.abort();
    }

    fetch(source.analysisPath || "/evidence/results/" + source.id + ".json", {
      signal: abort.signal,
    })
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (data?.frames) {
          setResult(data);
          setStored(true);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!abort.signal.aborted) setLoadingSaved(false);
      });

    return () => abort.abort();
  }, [source.id]);

  useEffect(() => {
    if (!live || transport === "hls") return;

    const timer = setInterval(() => setTick(Date.now()), refreshMs);

    return () => clearInterval(timer);
  }, [source.id, live, transport, refreshMs]);

  useEffect(() => {
    if (!busy) return;

    const started = Date.now();
    setElapsed(0);

    const timer = setInterval(
      () => setElapsed(Math.floor((Date.now() - started) / 1000)),
      1000,
    );

    return () => clearInterval(timer);
  }, [busy]);

  async function scan() {
    if (!LOCAL_SCANNING || busy) return;

    setBusy(true);
    setError("");

    const abort = new AbortController();
    request.current = abort;

    const timeout = setTimeout(() => abort.abort(), 240000);

    try {
      const isVideo = source.mediaType === "video";

      const endpoint = source.file
        ? isVideo
          ? "/api/analyze-clip"
          : "/api/analyze-image"
        : live
          ? "/api/analyze-camera"
          : "/api/analyze-evidence";

      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": source.file ? source.file.type : "application/json",
        },
        body:
          source.file ||
          JSON.stringify({
            id: source.id,
            ...(source.drain_roi ? { drain_roi: source.drain_roi } : {}),
          }),
        signal: abort.signal,
      });

      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "The scan failed.");

      if (!Array.isArray(data.frames))
        throw new Error("The scan returned no analyzed frames.");

      setResult(data);
      setStored(false);
      if (data.source?.recordedAt) setStamp(data.source.recordedAt);

      onResult(source, data);

      setTimeout(
        () =>
          evidenceRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          }),
        50,
      );
    } catch (err) {
      setError(
        err.name === "AbortError"
          ? "Scan stopped or timed out. Try again with a shorter clip."
          : err.message,
      );
    } finally {
      clearTimeout(timeout);
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!LOCAL_SCANNING || !live || !cameraEvent?.created_at) return;
    const abort = new AbortController();
    fetch("/api/camera-evidence?id=" + encodeURIComponent(source.id), {
      signal: abort.signal,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.frames?.length) {
          setResult(data);
          setStored(false);
        }
      })
      .catch(() => {});
    return () => abort.abort();
  }, [source.id, cameraEvent?.created_at]);
  const playbackUrl =
    transport === "mp4-refresh"
      ? sourceUrl + (sourceUrl.includes("?") ? "&" : "?") + "cw=" + tick
      : sourceUrl;

  const imageUrl = live
    ? LOCAL_SCANNING
      ? "/api/camera-frame?id=" + encodeURIComponent(source.id) + "&t=" + tick
      : sourceUrl + (sourceUrl.includes("?") ? "&" : "?") + "cw=" + tick
    : sourceUrl;

  return (
    <dialog
      ref={dialog}
      className="camera-dialog"
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="dialog-header">
        <div>
          <span className="source-kind">
            <span className={"status-dot " + (live ? "live" : "")} />
            {transportLabel(source)} · {source.city}, {source.country}
          </span>
          <h2>{source.name}</h2>
        </div>
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Close camera"
        >
          <X size={23} />
        </button>
      </div>

      <div className="dialog-scroll">
        <div className="source-section">
          <div className="section-top">
            <h3>{live ? "Camera view" : "Original recording"}</h3>
            <span className="muted">
              {transport === "hls"
                ? "Continuous video"
                : live
                  ? "Refreshes approximately every " +
                    Math.round(refreshMs / 60000) +
                    " min"
                  : source.recordedAt
                    ? displayDate(source.recordedAt)
                    : "Historical evidence"}
            </span>
          </div>

          <div className="source-viewer">
            {transport === "hls" || source.mediaType === "video" ? (
              <VideoSource
                key={source.id}
                url={playbackUrl}
                hls={transport === "hls"}
                onError={setError}
              />
            ) : (
              <img
                key={imageUrl}
                src={imageUrl}
                alt={source.name + " camera view"}
                onError={() =>
                  setError(
                    "The current camera image is unavailable. The operator may be offline.",
                  )
                }
              />
            )}
            <span className="frame-label">{transportLabel(source)}</span>
          </div>

          <div className="source-controls">
            <div>
              <External href={source.sourceUrl}>
                {source.sourceName || source.credit || "Camera operator"}
              </External>
              <small>
                {live && transport !== "hls"
                  ? "The camera view refreshes separately from AI analysis."
                  : source.footageKind}
                {!live && stamp && " · " + displayDate(stamp)}
              </small>
            </div>
            {!LOCAL_SCANNING ? (
              <div className="monitor-status" role="status">
                <strong>Scanning runs locally for now</strong>
                <small>
                  This demo shows camera feeds and saved model evidence. Live AI
                  monitoring is not running here.
                </small>
              </div>
            ) : live ? (
              <div className="monitor-status" role="status">
                <strong>{monitoringLabel(cameraState)}</strong>
                <small>
                  {cameraState?.checked_at
                    ? "Last checked " +
                      new Date(cameraState.checked_at).toLocaleTimeString()
                    : "Checks start with the local server"}{" "}
                  · Automatic CPU sampling
                </small>
              </div>
            ) : (
              <button
                className="primary-button"
                onClick={scan}
                disabled={busy || health === "offline"}
              >
                {busy ? (
                  <LoaderCircle size={18} className="spinner" />
                ) : (
                  <ScanLine size={18} />
                )}{" "}
                {busy
                  ? "Scanning · " + elapsed + "s"
                  : live
                    ? "Capture & scan"
                    : "Scan this evidence"}
              </button>
            )}
          </div>

          {busy && (
            <div className="scan-progress" role="status">
              <span>
                Running smoke detection and object segmentation on actual
                frames.
              </span>
              <button
                className="text-button"
                onClick={() => request.current?.abort()}
              >
                Stop waiting
              </button>
            </div>
          )}

          {error && (
            <p className="error-text" role="alert">
              {error}
            </p>
          )}

          {LOCAL_SCANNING && health === "offline" && (
            <p className="error-text">
              The local scan service is offline. Launch Start-ClimateWatch.cmd
              to use the models.
            </p>
          )}
        </div>

        <section className="captured-evidence" ref={evidenceRef}>
          <div className="evidence-title">
            <div>
              <span className="eyebrow">Separate from the source feed</span>
              <h2>Captured evidence</h2>
            </div>
            <span className="muted">Frame → mask / box → response</span>
          </div>

          {result?.frames?.length ? (
            <EvidenceResult
              key={result.analysis_id || String(stored)}
              result={result}
              source={source}
              stored={stored}
            />
          ) : (
            <div className="evidence-empty">
              <ScanLine size={26} />
              <div>
                <h3>
                  {loadingSaved
                    ? "Loading saved model evidence…"
                    : live && !LOCAL_SCANNING
                      ? "Live scanning is local only"
                      : live
                        ? "No threat evidence captured"
                        : "No target detected"}
                </h3>
                <p>
                  {live && !LOCAL_SCANNING
                    ? "This camera is not being analyzed on the hosted demo. Open the Evidence collection to explore saved detections and segmentation."
                    : live
                      ? "Evidence appears automatically after a target persists across two checks. Ordinary water, roads, people, and vehicles are ignored."
                      : "Only qualifying smoke, fire, or waste is shown. No target means no boxes, incident, or action card."}
                </p>
              </div>
            </div>
          )}
        </section>

        <footer className="source-details">
          <strong>Source and recording</strong>
          <p>
            {source.license} · {source.credit}
            {source.licenseUrl && (
              <>
                {" "}
                · <External href={source.licenseUrl}>License</External>
              </>
            )}
          </p>
          <p>{source.locationAccuracy}</p>
          <p>{source.observation || source.observations}</p>
          <External href={source.sourceUrl}>View original source</External>
        </footer>
      </div>
    </dialog>
  );
}

export default function App() {
  const [sources, setSources] = useState([]),
    [selectedId, setSelectedId] = useState(null),
    [openId, setOpenId] = useState(null);

  const [query, setQuery] = useState(""),
    [region, setRegion] = useState("all"),
    [mode, setMode] = useState("cameras");

  const [dimension, setDimension] = useState("3d"),
    [regionRequest, setRegionRequest] = useState(0),
    [focusRequest, setFocusRequest] = useState(0);

  const [health, setHealth] = useState("checking"),
    [error, setError] = useState(""),
    [uploaded, setUploaded] = useState(null),
    [captures, setCaptures] = useState({});

  const upload = useRef(null);
  const [monitor, setMonitor] = useState({
    enabled: false,
    cameras: [],
    events: [],
  });
  useEffect(() => {
    if (!LOCAL_SCANNING) return;
    const abort = new AbortController();
    let active = true;
    const poll = async () => {
      try {
        const response = await fetch("/api/monitor", { signal: abort.signal });
        if (response.ok && active) setMonitor(await response.json());
      } catch {
        if (active) setMonitor((previous) => ({ ...previous, enabled: false }));
      }
    };
    poll();
    const timer = setInterval(poll, 5000);
    return () => {
      active = false;
      abort.abort();
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const abort = new AbortController();

    loadSources(abort.signal)
      .then((items) => {
        setSources(items);
        const requested = new URLSearchParams(location.search).get("camera");
        setSelectedId(
          items.find((item) => item.id === requested)?.id ||
            items.find(isLive)?.id ||
            items[0].id,
        );
      })
      .catch((err) => {
        if (err.name !== "AbortError") setError(err.message);
      });

    if (LOCAL_SCANNING)
      fetch("/api/health", { signal: abort.signal })
        .then((response) => {
          if (!response.ok) throw new Error();
          return response.json();
        })
        .then(() => setHealth("ready"))
        .catch(() => setHealth("offline"));

    return () => abort.abort();
  }, []);

  useEffect(
    () => () => {
      if (uploaded?.mediaPath) URL.revokeObjectURL(uploaded.mediaPath);
    },
    [uploaded],
  );

  const cameras = sources.filter(isLive),
    archives = sources.filter((source) => !isLive(source));

  const capturedSources = monitor.events
    .map((event) => {
      const source = sources.find((item) => item.id === event.camera_id);
      return source
        ? {
            ...source,
            poster: event.preview,
            captured: true,
            recordedAt: event.created_at,
          }
        : null;
    })
    .filter(Boolean);

  const evidence = [...capturedSources, ...archives];

  const visible = filterCameras(
    mode === "cameras" ? cameras : evidence,
    query,
    "all",
  ).filter((source) => region === "all" || source.region === region);

  const selected = sources.find((source) => source.id === selectedId);

  const openSource =
    openId === "upload"
      ? uploaded
      : sources.find((source) => source.id === openId);

  function open(source) {
    setSelectedId(source.id);
    setOpenId(source.id);
    const url = new URL(location.href);
    url.searchParams.set("camera", source.id);
    history.replaceState(null, "", url);
  }

  function selectMap(id) {
    const source = sources.find((item) => item.id === id);
    if (source) open(source);
  }

  function uploadFile(file) {
    if (
      !["image/jpeg", "image/png", "video/mp4", "video/webm"].includes(
        file.type,
      )
    ) {
      setError("Choose a JPEG, PNG, MP4, or WebM file.");
      return;
    }

    const video = file.type.startsWith("video/");

    if (file.size > (video ? 40 : 8) * 1024 * 1024) {
      setError(
        video ? "Choose a clip under 40 MB." : "Choose an image under 8 MB.",
      );
      return;
    }

    const source = {
      id: "upload",
      name: file.name,
      city: "Your upload",
      country: "Location not assigned",
      mediaType: video ? "video" : "image",
      mediaPath: URL.createObjectURL(file),
      file,
      footageKind: "Private local upload",
      license: "Processed locally; not published",
    };

    setUploaded(source);
    setOpenId("upload");
    setError("");
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/">
          <Aperture size={27} />
          <span>ClimateWatch</span>
        </a>
        <nav className="primary-nav" aria-label="Collection">
          <button
            aria-pressed={mode === "cameras"}
            onClick={() => {
              setMode("cameras");
              setQuery("");
            }}
          >
            Cameras <span>{cameras.length}</span>
          </button>
          <button
            aria-pressed={mode === "evidence"}
            onClick={() => {
              setMode("evidence");
              setQuery("");
            }}
          >
            Evidence <span>{evidence.length}</span>
          </button>
        </nav>
        <span className="monitor-indicator">
          {!LOCAL_SCANNING
            ? "Demo · scanning is local only"
            : monitor.enabled
              ? "Monitoring " + monitor.cameras.length + " cameras"
              : "Monitor offline"}
        </span>
        <button
          className="secondary-button"
          disabled={!LOCAL_SCANNING}
          title={
            !LOCAL_SCANNING
              ? "Run ClimateWatch locally to scan your own clip"
              : undefined
          }
          onClick={() => upload.current.click()}
        >
          <Upload size={17} />
          {LOCAL_SCANNING ? "Scan a clip" : "Scan · local only"}
        </button>
        <input
          ref={upload}
          hidden
          type="file"
          accept="image/jpeg,image/png,video/mp4,video/webm"
          onChange={(event) => {
            if (event.target.files?.[0]) uploadFile(event.target.files[0]);
            event.target.value = "";
          }}
        />
      </header>

      {error && (
        <div className="global-error" role="alert">
          {error}
          <button onClick={() => setError("")} aria-label="Dismiss error">
            <X size={16} />
          </button>
        </div>
      )}

      <main className={"workspace mode-" + mode}>
        <aside className="sources-panel">
          <div className="sources-heading">
            <h1>
              {mode === "cameras" ? "Watch a place" : "Inspect the evidence"}
            </h1>
            <p>
              {mode === "cameras"
                ? LOCAL_SCANNING
                  ? "Cameras are checked automatically in the background."
                  : "Explore public CCTV. AI scanning runs only in the local version for now."
                : "Saved analysis of real recordings, with masks, boxes, and practical response guidance."}
            </p>
          </div>
          <label className="search-box">
            <Search size={17} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Find a place"
              aria-label="Search locations"
            />
          </label>
          <select
            className="region-select"
            value={region}
            onChange={(event) => setRegion(event.target.value)}
            aria-label="Filter by region"
          >
            <option value="all">Worldwide</option>
            <option value="Europe">Europe</option>
            <option value="Other regions">Other regions</option>
          </select>

          <div className="source-list">
            {visible.map((source) => (
              <button
                className={
                  "source-row " + (selectedId === source.id ? "selected" : "")
                }
                key={source.id}
                onClick={() => open(source)}
              >
                <div className="source-thumb">
                  {source.poster || source.mediaType !== "video" ? (
                    <img
                      src={source.poster || source.mediaPath}
                      alt=""
                      loading="lazy"
                    />
                  ) : (
                    <Camera size={23} />
                  )}
                </div>
                <div>
                  <strong>{source.city}</strong>
                  <span>{source.name}</span>
                  <small>
                    {source.country} · {transportLabel(source)}
                  </small>
                  {captures[source.id] && (
                    <em>
                      {captures[source.id].summary?.hazards?.length
                        ? "Findings captured"
                        : "Scanned · no target detected"}
                    </em>
                  )}
                </div>
                <ChevronRight size={14} />
              </button>
            ))}
            {!visible.length && (
              <p className="empty">
                {sources.length
                  ? "No matching sources."
                  : "Loading cameras and evidence…"}
              </p>
            )}
          </div>

          <div className="source-list-footer">
            Live source and captured evidence stay separate. Historical cases
            are labeled as recordings.
          </div>
        </aside>

        {mode === "cameras" ? (
          <section
            className="map-workspace"
            aria-label="Worldwide public cameras"
          >
            <CameraMap
              cameras={visible}
              selectedId={selectedId}
              onSelect={selectMap}
              dimension={dimension}
              regionRequest={regionRequest}
              focusRequest={focusRequest}
            />
            <div className="map-toolbar">
              <div className="map-toggle">
                <button
                  aria-pressed={dimension === "2d"}
                  onClick={() => setDimension("2d")}
                >
                  2D
                </button>
                <button
                  aria-pressed={dimension === "3d"}
                  onClick={() => setDimension("3d")}
                >
                  <Layers3 size={15} />
                  3D
                </button>
              </div>
              <button
                className="map-control"
                onClick={() => {
                  setDimension("2d");
                  setRegionRequest((n) => n + 1);
                }}
              >
                <Globe2 size={17} />
                World view
              </button>
            </div>
            <div className="map-caption">
              <div>
                <strong>{selected?.city || "Public camera network"}</strong>
                <span>Select a pin to open the camera</span>
              </div>
              <button
                className="icon-button"
                onClick={() => setFocusRequest((n) => n + 1)}
                aria-label="Focus selected camera"
              >
                <Focus size={20} />
              </button>
            </div>
          </section>
        ) : (
          <section
            className="evidence-gallery"
            aria-label="Recorded environmental evidence"
          >
            <div className="gallery-heading">
              <h2>See what the models found</h2>
              <p>
                Saved model analysis of dated recordings. These examples are
                separate from current camera conditions.
              </p>
            </div>
            <div className="gallery-grid">
              {visible.map((source) => (
                <button
                  key={source.id}
                  className="evidence-card"
                  onClick={() => open(source)}
                >
                  <img
                    src={source.poster || source.mediaPath}
                    alt={source.name}
                  />
                  <div>
                    <span>
                      {source.country} ·{" "}
                      {source.captured
                        ? "Automatic camera evidence"
                        : displayDate(source.recordedAt)}
                    </span>
                    <h3>{source.name}</h3>
                    <p>
                      {source.captured
                        ? "Open captured detections"
                        : source.mediaType === "video"
                          ? "Open clip and scanned frames"
                          : "Open image and segmentation"}
                      <ChevronRight size={16} />
                    </p>
                  </div>
                </button>
              ))}
            </div>
          </section>
        )}
      </main>

      {openSource && (
        <CameraPanel
          key={openSource.id + (openSource.file?.name || "")}
          source={openSource}
          cameraState={monitor.cameras.find(
            (item) => item.id === openSource.id,
          )}
          cameraEvent={monitor.events.find(
            (item) => item.camera_id === openSource.id,
          )}
          initialResult={captures[openSource.id]}
          health={health}
          onClose={() => setOpenId(null)}
          onResult={(source, result) => {
            if (result.frames?.length)
              setCaptures((previous) => ({ ...previous, [source.id]: result }));
          }}
        />
      )}
    </div>
  );
}
