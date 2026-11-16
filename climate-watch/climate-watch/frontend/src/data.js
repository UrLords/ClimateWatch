const europeanCountries = new Set([
  "United Kingdom",
  "UK",
  "Germany",
  "Greece",
  "Spain",
  "Portugal",
  "France",
  "Italy",
  "Finland",
  "Norway",
  "Sweden",
  "Netherlands",
  "Belgium",
  "Switzerland",
  "Austria",
  "Czechia",
]);
export function normalizeSource(source) {
  return {
    ...source,
    drain_roi: source.drain_roi?.xyxy || source.drain_roi,
    name: source.title || source.name,
    region:
      source.region ||
      (europeanCountries.has(source.country) ? "Europe" : "Other regions"),
    topic: source.topic === "visibility" ? "smoke" : source.topic,
    mediaType:
      source.mediaType === "hls" ? "video" : source.mediaType || "image",
    transport:
      source.mediaType === "hls"
        ? "hls"
        : source.mode === "rolling-clip"
          ? "mp4-refresh"
          : "snapshot",
    mediaPath: source.mediaPath || source.imagePath,
    poster: source.posterPath || source.poster || source.imageUrl,
    liveImageUrl: source.liveImageUrl || source.liveSource?.imageUrl,
    footageKind: source.footageKind || "Recorded evidence",
  };
}
export async function loadSources(signal) {
  const paths = [
    "/live-cameras/cameras.json",
    "/europe-cameras/cameras.json",
    "/evidence/sources.json",
  ];
  const results = await Promise.all(
    paths.map(async (path) => {
      const response = await fetch(path, { signal });
      if (!response.ok) return [];
      const value = await response.json();
      return (
        Array.isArray(value) ? value : value.sources || value.cameras || []
      ).map(normalizeSource);
    }),
  );
  const sources = results
    .flat()
    .filter((source) => source.id && (source.mediaPath || source.streamUrl));
  if (!sources.length)
    throw new Error(
      "The camera and evidence collection could not be loaded. Refresh the page or try again later.",
    );
  return sources;
}
