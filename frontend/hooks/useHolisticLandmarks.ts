"use client";

// MediaPipe Holistic — ENHANCEMENT LAYER only.
// Draws the dense face-mesh grid (478-pt tessellation + eyes/iris/gaze) and the hand
// finger-mesh onto an overlay canvas that sits ON TOP of the reliable server-rendered
// green skeleton + product boxes. The body skeleton and objects come from the backend;
// this only adds the face + hand mesh. If Holistic fails to load, the server layer still
// renders everything — so there is never a "nothing visible" state.
//
// Self-hosted wasm + model under /public/mediapipe (required by COEP: require-corp).

import { useEffect, useRef, useState, RefObject } from "react";
import { drawSkeleton, type HolisticParts, type LM, type Conn } from "@/lib/skeleton";

const WASM_PATH  = "/mediapipe/wasm";
const MODEL_PATH = "/mediapipe/holistic_landmarker.task";

// Cache the fileset across mounts. On failure, clear so the next attempt retries.
let filesetPromise: Promise<any> | null = null;
async function getFileset() {
  const { FilesetResolver } = await import("@mediapipe/tasks-vision");
  if (!filesetPromise) {
    filesetPromise = FilesetResolver.forVisionTasks(WASM_PATH).catch(e => {
      filesetPromise = null;
      throw e;
    });
  }
  return filesetPromise;
}

async function createHolistic() {
  const { HolisticLandmarker } = await import("@mediapipe/tasks-vision");
  const fileset = await getFileset();
  const opts: any = {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
    runningMode: "VIDEO",
    outputFaceBlendshapes: false,
    outputPoseSegmentationMasks: true,   // for the body boundary outline
  };
  try { return await HolisticLandmarker.createFromOptions(fileset, opts); }
  catch {
    return await HolisticLandmarker.createFromOptions(fileset, {
      ...opts, baseOptions: { modelAssetPath: MODEL_PATH, delegate: "CPU" },
    });
  }
}

// Face-mesh tessellation edges (the dense grid), captured once.
let faceTessCache: Conn[] | null = null;
async function getFaceTess(): Promise<Conn[]> {
  if (faceTessCache) return faceTessCache;
  const { FaceLandmarker } = await import("@mediapipe/tasks-vision");
  faceTessCache = (FaceLandmarker as any).FACE_LANDMARKS_TESSELATION as Conn[];
  return faceTessCache;
}

export function useHolisticLandmarks(
  videoRef:  RefObject<HTMLVideoElement>,
  canvasRef: RefObject<HTMLCanvasElement>,
  active:    boolean,
) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const holRef     = useRef<any>(null);
  const rafRef     = useRef<number | null>(null);
  const runningRef = useRef(false);
  const busyRef    = useRef(false);
  const lastVideoT = useRef(-1);
  const lastRunRef = useRef(0);
  const tsRef      = useRef(0);
  const tessRef    = useRef<Conn[] | null>(null);

  // Throttle the face/hand mesh to 15fps — it runs on the main thread, so a lower rate
  // frees the thread for video rendering + the prediction WebSocket loop (higher overall fps).
  const DETECT_FPS = 15, MIN_DT = 1000 / DETECT_FPS;

  useEffect(() => {
    if (!active) return;
    runningRef.current = true;
    let cancelled = false;
    setError(null);

    (async () => {
      let hol: any;
      try { hol = await createHolistic(); }
      catch (e: any) {
        console.error("[MACHANI] Holistic init failed:", e);
        if (!cancelled) setError(e?.message ? String(e.message) : "MediaPipe init failed");
        return;
      }
      if (cancelled) { hol?.close?.(); return; }
      holRef.current = hol;
      getFaceTess().then(t => { if (!cancelled) tessRef.current = t; }).catch(() => {});
      setReady(true);
      loop();
    })();

    function loop() {
      if (!runningRef.current) return;
      rafRef.current = requestAnimationFrame(loop);

      const video = videoRef.current, hol = holRef.current, canvas = canvasRef.current;
      if (!video || !hol || !canvas ||
          video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) return;

      const now = performance.now();
      if (now - lastRunRef.current < MIN_DT) return;
      if (video.currentTime === lastVideoT.current) return;
      if (busyRef.current) return;
      busyRef.current = true;
      lastRunRef.current = now;
      lastVideoT.current = video.currentTime;

      try {
        tsRef.current = Math.max(tsRef.current + 1, now);
        const r = hol.detectForVideo(video, tsRef.current);

        const face  = r.faceLandmarks?.[0];
        const pose  = r.poseLandmarks?.[0] as LM[] | undefined;
        const left  = r.leftHandLandmarks?.[0];
        const right = r.rightHandLandmarks?.[0];
        const hands: LM[][] = [left, right].filter(Boolean) as LM[][];
        const mask  = r.poseSegmentationMasks?.[0];

        const parts: HolisticParts = {
          face, pose, hands, poseMask: mask, faceTess: tessRef.current ?? undefined,
        };
        // Rich enhancement on the primary person: face mesh + hands + eyes/gaze + body boundary.
        // The body SKELETON is server-rendered (YOLOv8-pose) so it covers ALL people + shoulders;
        // Holistic is single-person, so drawing its pose here would only cover one + double up.
        drawSkeleton(canvas, video, parts, {
          drawPose: false, drawObjects: false, drawBoundary: true,
        });
        mask?.close?.();   // free the mask each frame
      } catch { /* frame error — skip */ }
      finally { busyRef.current = false; }
    }

    return () => {
      cancelled = true;
      runningRef.current = false;
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      holRef.current?.close?.();
      holRef.current = null;
      const c = canvasRef.current;
      if (c) { const ctx = c.getContext("2d"); ctx?.clearRect(0, 0, c.width, c.height); }
      setReady(false);
    };
  }, [videoRef, canvasRef, active]);

  return { ready, error };
}
