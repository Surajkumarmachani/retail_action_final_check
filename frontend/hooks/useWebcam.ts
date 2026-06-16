// hooks/useWebcam.ts
// Manages webcam stream access and frame capture to Base64 JPEG.
// The hidden <canvas> is used only for frame capture; the <video> renders the live feed.

import { useCallback, useEffect, useRef, useState } from "react";

// Frames are sent at this fraction of native resolution.
// Box coords from the server are in scaled space — multiply by 1/FRAME_SEND_SCALE to display.
export const FRAME_SEND_SCALE = 0.5;

export function useWebcam() {
  const videoRef      = useRef<HTMLVideoElement>(null);
  const captureCanvas = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    // React Strict Mode mounts→unmounts→remounts in dev.
    // `active` prevents the stale first-mount's async callbacks from firing.
    let active = true;

    const toMsg = (e: unknown): string =>
      e instanceof Error ? e.message : String(e);

    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false })
      .then((s) => {
        if (!active) {
          // Cleanup already ran before getUserMedia resolved — discard the stream.
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        const video = videoRef.current;
        if (!video) return;
        video.srcObject = s;
        video.play()
          .then(() => { if (active) setReady(true); })
          .catch((e) => {
            if (!active) return;
            // AbortError fires when Strict Mode's first-mount cleanup kills the
            // stream before play() resolves — it's harmless; don't show it.
            if (e instanceof DOMException && e.name === "AbortError") return;
            setError(toMsg(e));
          });
      })
      .catch((e) => {
        if (active) setError(toMsg(e));
      });

    return () => {
      active = false;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Send frames at 50% linear resolution — 4× smaller payload, ~50% faster backend.
  // Box coordinates returned by the server are in this scaled space; callers must
  // multiply by 1/FRAME_SEND_SCALE to convert back to display (video) coordinates.
  const captureFrame = useCallback((): string | null => {
    const video  = videoRef.current;
    const canvas = captureCanvas.current;
    if (!video || !canvas || video.readyState < 2) return null;

    canvas.width  = Math.round(video.videoWidth  * FRAME_SEND_SCALE);
    canvas.height = Math.round(video.videoHeight * FRAME_SEND_SCALE);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.45).split(",")[1];
  }, []);

  return { videoRef, captureCanvas, ready, error, captureFrame };
}
