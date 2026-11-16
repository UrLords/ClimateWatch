export const topics = {
  smoke: "Smoke & fire",
  waste: "Waste & drains",
  water: "Water",
};
export function filterCameras(cameras, query, topic) {
  const search = query.trim().toLocaleLowerCase();
  return cameras.filter(
    (camera) =>
      (topic === "all" || camera.topic === topic) &&
      (camera.name + " " + camera.city + " " + camera.country)
        .toLocaleLowerCase()
        .includes(search),
  );
}
export function frameAtTime(frames, time, tolerance = 0.35) {
  if (!Number.isFinite(time)) return null;
  const nearby = frames.filter(
    (frame) =>
      Number.isFinite(frame.time_s) &&
      Math.abs(frame.time_s - time) <= tolerance,
  );
  return (
    nearby.sort(
      (a, b) => Math.abs(a.time_s - time) - Math.abs(b.time_s - time),
    )[0] || null
  );
}
export function validBox(box) {
  return (
    Array.isArray(box) &&
    box.length === 4 &&
    box.every((n) => Number.isFinite(n) && n >= 0 && n <= 1) &&
    box[2] > box[0] &&
    box[3] > box[1]
  );
}
export function modelDetections(frame) {
  return (frame?.detections || []).filter((detection) =>
    validBox(detection.box),
  );
}
export function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  return (
    Math.floor(seconds / 60) +
    ":" +
    String(Math.floor(seconds % 60)).padStart(2, "0")
  );
}
export function displayDate(value) {
  if (!value || !Number.isFinite(new Date(value).getTime()))
    return "Date not provided";
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(value));
}
