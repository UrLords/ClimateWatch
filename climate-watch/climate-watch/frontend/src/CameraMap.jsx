import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";

maplibregl.setWorkerUrl(workerUrl);

const BUILDINGS = "climatewatch-buildings";
const MARKER_ICON =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="3" y="6" width="13" height="12" rx="3"/><path d="m16 10 5-3v10l-5-3z"/><circle cx="9.5" cy="12" r="2.5"/></svg>';
const validLocation = (camera) =>
  Array.isArray(camera.coordinates) &&
  camera.coordinates.length === 2 &&
  camera.coordinates.every(Number.isFinite) &&
  Math.abs(camera.coordinates[0]) <= 180 &&
  Math.abs(camera.coordinates[1]) <= 85;

export default function CameraMap({
  cameras,
  selectedId,
  onSelect,
  dimension = "3d",
  regionRequest = 0,
  focusRequest = 0,
}) {
  const container = useRef(null);
  const mapRef = useRef(null);
  const markers = useRef(new Map());
  const camerasRef = useRef(cameras);
  const selectRef = useRef(onSelect);
  const dimensionRef = useRef(dimension);
  const [attempt, setAttempt] = useState(0);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  camerasRef.current = cameras;
  selectRef.current = onSelect;
  dimensionRef.current = dimension;
  const selectedCamera = cameras.find(
    (camera) => camera.id === selectedId && validLocation(camera),
  );
  const longitude = selectedCamera?.coordinates[0];
  const latitude = selectedCamera?.coordinates[1];

  useEffect(() => {
    setReady(false);
    setError("");
    let map;
    let loaded = false;
    let resizeObserver;
    let timeout;
    const initial =
      camerasRef.current.find(
        (camera) => camera.id === selectedId && validLocation(camera),
      ) || camerasRef.current.find(validLocation);
    try {
      map = new maplibregl.Map({
        container: container.current,
        style: "https://tiles.openfreemap.org/styles/positron",
        center: initial?.coordinates || [12, 35],
        zoom: initial ? 16 : 2,
        pitch: dimensionRef.current === "3d" ? 58 : 0,
        bearing: dimensionRef.current === "3d" ? -22 : 0,
        attributionControl: false,
        canvasContextAttributes: { antialias: true },
        maxPitch: 65,
        respectPrefersReducedMotion: true,
      });
      mapRef.current = map;
      map.addControl(
        new maplibregl.NavigationControl({ visualizePitch: true }),
        "bottom-right",
      );
      map.addControl(
        new maplibregl.AttributionControl({ compact: true }),
        "bottom-left",
      );
      map
        .getCanvas()
        .setAttribute(
          "aria-label",
          "Camera map. Use arrow keys to pan and plus or minus to zoom.",
        );
      map.on("load", () => {
        try {
          const style = map.getStyle();
          const existingSource = Object.entries(style.sources).find(
            ([, source]) =>
              source.type === "vector" &&
              source.url?.includes("tiles.openfreemap.org/planet"),
          )?.[0];
          const source = existingSource || "climatewatch-openfreemap";
          if (!existingSource)
            map.addSource(source, {
              type: "vector",
              url: "https://tiles.openfreemap.org/planet",
            });
          const labels = style.layers.find(
            (layer) => layer.type === "symbol" && layer.layout?.["text-field"],
          );
          map.addLayer(
            {
              id: BUILDINGS,
              source,
              "source-layer": "building",
              type: "fill-extrusion",
              minzoom: 14,
              filter: ["!=", ["get", "hide_3d"], true],
              layout: {
                visibility: dimensionRef.current === "3d" ? "visible" : "none",
              },
              paint: {
                "fill-extrusion-color": "#aab7ad",
                "fill-extrusion-height": [
                  "interpolate",
                  ["linear"],
                  ["zoom"],
                  14,
                  0,
                  15,
                  ["to-number", ["get", "render_height"], 0],
                ],
                "fill-extrusion-base": [
                  "to-number",
                  ["get", "render_min_height"],
                  0,
                ],
                "fill-extrusion-opacity": 0.82,
              },
            },
            labels?.id,
          );
          loaded = true;
          clearTimeout(timeout);
          setError("");
          setReady(true);
        } catch {
          setError(
            "The 3D map could not finish loading. Camera sources remain available in the list.",
          );
        }
      });
      map.on("error", () => {
        if (!loaded)
          setError(
            "Map data is unavailable. Check your connection or explore the camera list.",
          );
      });
      map.on("webglcontextlost", () => {
        setReady(false);
        setError(
          "Your browser paused the 3D map. Retry the map or use the camera list.",
        );
      });
      timeout = window.setTimeout(() => {
        if (!loaded)
          setError(
            "The map is taking longer to load. Camera sources remain available in the list.",
          );
      }, 20000);
      resizeObserver = new ResizeObserver(() => map.resize());
      resizeObserver.observe(container.current);
    } catch {
      setError(
        "This browser could not start the 3D map. Try another browser or use the camera list.",
      );
    }
    return () => {
      clearTimeout(timeout);
      resizeObserver?.disconnect();
      markers.current.forEach((marker) => marker.remove());
      markers.current.clear();
      map?.remove();
      mapRef.current = null;
    };
  }, [attempt]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const visible = new Set(
      cameras.filter(validLocation).map((camera) => camera.id),
    );
    markers.current.forEach((marker, id) => {
      if (!visible.has(id)) {
        marker.remove();
        markers.current.delete(id);
      }
    });
    for (const camera of cameras.filter(validLocation)) {
      let marker = markers.current.get(camera.id);
      if (!marker) {
        const anchor = document.createElement("div");
        const button = document.createElement("button");
        button.type = "button";
        button.innerHTML = MARKER_ICON;
        button.addEventListener("click", () => selectRef.current(camera.id));
        anchor.appendChild(button);
        marker = new maplibregl.Marker({
          element: anchor,
          anchor: "bottom",
          offset: [0, -6],
          pitchAlignment: "viewport",
          rotationAlignment: "viewport",
          subpixelPositioning: true,
        })
          .setLngLat(camera.coordinates)
          .addTo(map);
        markers.current.set(camera.id, marker);
      }
      marker.setLngLat(camera.coordinates);
      const button = marker.getElement().firstElementChild;
      const selected = camera.id === selectedId;
      button.className = `map-marker map-marker--${camera.topic} marker-${camera.topic}${selected ? " map-marker--selected" : ""}`;
      button.title = `${camera.name} · ${camera.city}`;
      button.setAttribute(
        "aria-label",
        `Select ${camera.name}, ${camera.city}`,
      );
      button.setAttribute("aria-pressed", String(selected));
    }
  }, [cameras, selectedId, attempt]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    if (map.getLayer(BUILDINGS))
      map.setLayoutProperty(
        BUILDINGS,
        "visibility",
        dimension === "3d" ? "visible" : "none",
      );
    map.easeTo({
      pitch: dimension === "3d" ? 58 : 0,
      bearing: dimension === "3d" ? -22 : 0,
      duration: 600,
    });
  }, [dimension, ready]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || longitude === undefined || latitude === undefined)
      return;
    map.flyTo({
      center: [longitude, latitude],
      zoom: 16,
      pitch: dimensionRef.current === "3d" ? 58 : 0,
      bearing: dimensionRef.current === "3d" ? -22 : 0,
      duration: 1400,
    });
  }, [selectedId, longitude, latitude, focusRequest, ready]);

  useEffect(() => {
    const map = mapRef.current;
    const locations = camerasRef.current.filter(validLocation);
    if (!regionRequest || !ready || !map || !locations.length) return;
    const bounds = new maplibregl.LngLatBounds();
    locations.forEach((camera) => bounds.extend(camera.coordinates));
    map.fitBounds(bounds, {
      padding: 80,
      maxZoom: 8,
      pitch: 0,
      bearing: 0,
      duration: 1200,
    });
  }, [regionRequest, ready]);

  return (
    <section
      className="map-shell"
      aria-label="Camera locations"
      aria-busy={!ready && !error}
    >
      <div ref={container} className="map-canvas" />
      {!ready && !error && (
        <div className="map-loading" role="status">
          Loading the camera map…
        </div>
      )}
      {error && (
        <div className="map-failure" role="status">
          <strong>Map unavailable</strong>
          <p>{error}</p>
          <button
            type="button"
            onClick={() => setAttempt((value) => value + 1)}
          >
            Retry map
          </button>
        </div>
      )}
    </section>
  );
}
