export function localScanningEnabled(hostname, mode) {
  return (
    mode !== "demo" && ["localhost", "127.0.0.1", "[::1]"].includes(hostname)
  );
}
