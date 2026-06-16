"use client";

// Runs face detection + expression recognition at ~5fps in the browser.
// Models are served from /public/models (copied from @vladmandic/face-api).
// Loaded once globally — subsequent hook instances share the same weights.

import { useEffect, useRef, useState, RefObject } from "react";

export type ExpressionName =
  | "neutral" | "happy" | "sad" | "angry" | "fearful" | "disgusted" | "surprised";

export interface FaceResult {
  /** Face centre in raw (un-mirrored) video pixel coords */
  cx: number;
  cy: number;
  dominant: ExpressionName;
  confidence: number;
  scores: Partial<Record<ExpressionName, number>>;
}

// Module-level singletons so models load exactly once across re-mounts
let fa: typeof import("@vladmandic/face-api") | null = null;
let ready  = false;
let loading: Promise<void> | null = null;

async function ensureLoaded() {
  if (ready) return;
  if (loading) { await loading; return; }
  loading = (async () => {
    fa = await import("@vladmandic/face-api");
    await fa.nets.tinyFaceDetector.loadFromUri("/models");
    await fa.nets.faceExpressionNet.loadFromUri("/models");
    ready = true;
  })();
  await loading;
}

export function useFaceExpression(
  videoRef: RefObject<HTMLVideoElement>,
  active: boolean,
): FaceResult[] {
  const [faces, setFaces] = useState<FaceResult[]>([]);
  const aliveRef = useRef(false);

  useEffect(() => {
    if (!active) { setFaces([]); return; }
    aliveRef.current = true;

    (async () => {
      try {
        await ensureLoaded();
      } catch {
        return; // model load failed silently
      }

      const opts = new fa!.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.45 });

      while (aliveRef.current) {
        const video = videoRef.current;
        if (video && video.readyState >= 2) {
          try {
            const detected = await fa!
              .detectAllFaces(video, opts)
              .withFaceExpressions();

            if (aliveRef.current) {
              if (detected.length === 0) {
                setFaces([]);
              } else {
                setFaces(detected.map(d => {
                  const b = d.detection.box;
                  const entries = Object.entries(d.expressions) as [ExpressionName, number][];
                  entries.sort((a, b) => b[1] - a[1]);
                  return {
                    cx: b.x + b.width  / 2,
                    cy: b.y + b.height / 2,
                    dominant:   entries[0][0],
                    confidence: entries[0][1],
                    scores: d.expressions as Partial<Record<ExpressionName, number>>,
                  };
                }));
              }
            }
          } catch { /* inference error — skip frame */ }
        }
        // 5fps is plenty for expression; keeps main thread free for rendering
        await new Promise(r => setTimeout(r, 200));
      }
    })();

    return () => { aliveRef.current = false; };
  }, [videoRef, active]);

  return faces;
}
