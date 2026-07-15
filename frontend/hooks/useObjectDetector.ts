"use client";

// Client-side product detection (MediaPipe ObjectDetector, EfficientDet-Lite0, on GPU).
// Runs entirely in the browser on the live <video> — NO server round-trip — so product
// boxes appear/disappear the instant an object enters/leaves the frame ("millisecond" feel).
// Draws green boxes + bold labels straight onto the overlay canvas each detected frame.

import { useEffect, useRef, useState, RefObject } from "react";

const WASM_PATH = "/mediapipe/wasm";
const MODEL     = "/mediapipe/efficientdet_lite0.tflite";

export interface DetObject {
  label: string;
  score: number;
  box:   [number, number, number, number];   // normalised (0-1) raw-frame coords
}

let filesetPromise: Promise<any> | null = null;
async function getFileset() {
  const { FilesetResolver } = await import("@mediapipe/tasks-vision");
  if (!filesetPromise) {
    filesetPromise = FilesetResolver.forVisionTasks(WASM_PATH).catch(e => { filesetPromise = null; throw e; });
  }
  return filesetPromise;
}

async function createDetector() {
  const { ObjectDetector } = await import("@mediapipe/tasks-vision");
  const fileset = await getFileset();
  const opts: any = {
    baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
    runningMode: "VIDEO", scoreThreshold: 0.35, maxResults: 12,
  };
  try { return await ObjectDetector.createFromOptions(fileset, opts); }
  catch {
    return await ObjectDetector.createFromOptions(fileset, {
      ...opts, baseOptions: { modelAssetPath: MODEL, delegate: "CPU" },
    });
  }
}

function drawBoxes(canvas: HTMLCanvasElement, video: HTMLVideoElement, objs: DetObject[]) {
  const w = video.videoWidth, h = video.videoHeight;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, w, h);
  for (const o of objs) {
    const [nx1, ny1, nx2, ny2] = o.box;
    const x1 = (1 - nx2) * w, x2 = (1 - nx1) * w;   // mirror to match scaleX(-1) video
    const y1 = ny1 * h, y2 = ny2 * h;
    const bw = x2 - x1, bh = y2 - y1;
    const cs = Math.min(bw, bh, 22);

    ctx.save();
    ctx.fillStyle = "rgba(22,163,74,0.08)";
    ctx.fillRect(x1, y1, bw, bh);
    ctx.strokeStyle = "#16a34a"; ctx.lineWidth = 2;
    ctx.shadowColor = "rgba(22,163,74,0.55)"; ctx.shadowBlur = 12;
    ctx.strokeRect(x1, y1, bw, bh);
    // corner brackets
    ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 3; ctx.lineCap = "round";
    for (const [bx, by, dx, dy] of [
      [x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1],
    ] as [number, number, number, number][]) {
      ctx.beginPath();
      ctx.moveTo(bx, by + dy * cs); ctx.lineTo(bx, by); ctx.lineTo(bx + dx * cs, by);
      ctx.stroke();
    }
    ctx.restore();

    const label = `${o.label.toUpperCase()}  ${o.score.toFixed(2)}`;
    ctx.font = "800 14px ui-monospace, monospace";
    const tw = ctx.measureText(label).width;
    const ly = y1 > 30 ? y1 - 26 : y2 + 2;
    ctx.save();
    ctx.fillStyle = "rgba(6,30,15,0.95)";
    ctx.strokeStyle = "rgba(22,163,74,0.75)"; ctx.lineWidth = 1;
    ctx.fillRect(x1, ly, tw + 16, 22);
    ctx.strokeRect(x1, ly, tw + 16, 22);
    ctx.fillStyle = "#dcfce7";
    ctx.fillText(label, x1 + 8, ly + 15);
    ctx.restore();
  }
}

export function useObjectDetector(
  videoRef:  RefObject<HTMLVideoElement>,
  canvasRef: RefObject<HTMLCanvasElement>,
  active:    boolean,
) {
  const [objects, setObjects] = useState<DetObject[]>([]);
  const detRef     = useRef<any>(null);
  const rafRef     = useRef<number | null>(null);
  const runningRef = useRef(false);
  const busyRef    = useRef(false);
  const lastVideoT = useRef(-1);
  const lastRunRef = useRef(0);
  const lastPush   = useRef(0);
  const tsRef      = useRef(0);

  const DETECT_FPS = 25, MIN_DT = 1000 / DETECT_FPS, PUSH_MS = 120;

  useEffect(() => {
    if (!active) { setObjects([]); return; }
    runningRef.current = true;
    let cancelled = false;

    (async () => {
      let d: any;
      try { d = await createDetector(); }
      catch (e) { console.error("[MACHANI] ObjectDetector init failed:", e); return; }
      if (cancelled) { d?.close?.(); return; }
      detRef.current = d;
      loop();
    })();

    function loop() {
      if (!runningRef.current) return;
      rafRef.current = requestAnimationFrame(loop);
      const video = videoRef.current, det = detRef.current, canvas = canvasRef.current;
      if (!video || !det || !canvas || video.readyState < 2 || video.videoWidth === 0) return;

      const now = performance.now();
      if (now - lastRunRef.current < MIN_DT) return;
      if (video.currentTime === lastVideoT.current) return;
      if (busyRef.current) return;
      busyRef.current = true;
      lastRunRef.current = now;
      lastVideoT.current = video.currentTime;

      try {
        tsRef.current = Math.max(tsRef.current + 1, now);
        const r = det.detectForVideo(video, tsRef.current);
        const vw = video.videoWidth, vh = video.videoHeight;
        const objs: DetObject[] = [];
        for (const d of r.detections ?? []) {
          const bb = d.boundingBox; const cat = d.categories?.[0];
          if (!bb || !cat || cat.categoryName === "person") continue;
          objs.push({
            label: cat.categoryName,
            score: cat.score,
            box: [bb.originX / vw, bb.originY / vh, (bb.originX + bb.width) / vw, (bb.originY + bb.height) / vh],
          });
        }
        drawBoxes(canvas, video, objs);          // instant visual, every detected frame
        if (now - lastPush.current > PUSH_MS) {  // low-freq state for panel/events
          lastPush.current = now;
          setObjects(objs);
        }
      } catch { /* skip frame */ }
      finally { busyRef.current = false; }
    }

    return () => {
      cancelled = true;
      runningRef.current = false;
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      detRef.current?.close?.();
      detRef.current = null;
      const c = canvasRef.current;
      if (c) { const ctx = c.getContext("2d"); ctx?.clearRect(0, 0, c.width, c.height); }
      setObjects([]);
    };
  }, [videoRef, canvasRef, active]);

  return { objects };
}
