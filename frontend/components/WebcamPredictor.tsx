"use client";

import { useCallback, useRef, useState, useEffect } from "react";
import { signOut } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { Prediction } from "@/lib/types";
import { useWebcam } from "@/hooks/useWebcam";
import { usePredictorSocket } from "@/hooks/useWebSocket";
import { PredictionOverlay } from "./PredictionOverlay";
import { PersonCard, MiniCard } from "./PersonCard";

const STREAM_FPS  = 15;
const STALE_MS    = 1500; // remove person from UI 1.5 s after last seen

export function WebcamPredictor() {
  // Use a Map for O(1) stale-entry removal
  const predMap  = useRef<Map<number, { pred: Prediction; ts: number }>>(new Map());
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [sessionSec, setSessionSec]   = useState(0);
  const [fps, setFps]                 = useState(0);
  const fpsRef   = useRef(0);
  const startRef = useRef(Date.now());

  const { videoRef, captureCanvas, ready, error, captureFrame } = useWebcam();

  // Merge incoming predictions; stale entries age out after STALE_MS
  const handlePredictions = useCallback((preds: Prediction[]) => {
    fpsRef.current += 1;
    const now = Date.now();
    const map = predMap.current;
    for (const p of preds) map.set(p.id, { pred: p, ts: now });
    for (const [id, { ts }] of map) if (now - ts > STALE_MS) map.delete(id);
    setPredictions(Array.from(map.values()).map(v => v.pred));
  }, []);

  // Periodic cleanup — fires even when no frames arrive (camera blocked, tab hidden)
  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      const map = predMap.current;
      let changed = false;
      for (const [id, { ts }] of map) {
        if (now - ts > STALE_MS) { map.delete(id); changed = true; }
      }
      if (changed) setPredictions(Array.from(map.values()).map(v => v.pred));
    }, 500);
    return () => clearInterval(id);
  }, []);

  const { connected, authError } = usePredictorSocket({
    onPredictions: handlePredictions,
    captureFrame,
    webcamReady: ready,
    fps: STREAM_FPS,
  });

  // Session timer + FPS
  useEffect(() => {
    const id = setInterval(() => {
      setSessionSec(Math.floor((Date.now() - startRef.current) / 1000));
      setFps(fpsRef.current);
      fpsRef.current = 0;
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const frameW    = videoRef.current?.videoWidth  ?? 640;
  const frameH    = videoRef.current?.videoHeight ?? 480;
  const buyCount  = predictions.filter(p => p.verdict === "LIKELY BUY").length;
  const avgProb   = predictions.length > 0
    ? predictions.reduce((s, p) => s + p.prob, 0) / predictions.length : 0;

  // Primary = highest probability; rest = secondary
  const sorted    = [...predictions].sort((a, b) => {
    if (a.verdict !== b.verdict) return a.verdict === "LIKELY BUY" ? -1 : 1;
    return b.prob - a.prob;
  });
  const primary   = sorted[0] ?? null;
  const secondary = sorted.slice(1);

  const sessionStr = sessionSec >= 60
    ? `${Math.floor(sessionSec / 60)}m ${sessionSec % 60}s`
    : `${sessionSec}s`;

  if (error || authError) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-6 max-w-md text-center">
          <p className="text-red-400 text-sm">{String(error ?? authError)}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#030712]">

      {/* ── Header ─────────────────────────────────────────── */}
      <header className="flex flex-shrink-0 items-center justify-between border-b border-white/5 bg-black/60 px-5 py-2.5 backdrop-blur-sm z-20">

        {/* Brand */}
        <div className="flex items-center gap-2.5">
          <div className="relative flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-green-400 to-cyan-500">
            <svg className="h-4 w-4 text-black" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
            </svg>
            <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-green-400 to-cyan-500 opacity-40 animate-ping" style={{ animationDuration: "4s" }} />
          </div>
          <div className="leading-tight">
            <span className="block text-[13px] font-black tracking-tight text-white">MACHANI</span>
            <span className="block text-[9px] font-medium uppercase tracking-[0.15em] text-gray-500">Retail Intelligence</span>
          </div>
        </div>

        {/* Live stats pills */}
        <div className="flex items-center gap-4">
          {[
            { label: "Customers", val: predictions.length, accent: false },
            { label: "Buy intent", val: buyCount, accent: buyCount > 0 },
            { label: "Avg confidence", val: `${(avgProb * 100).toFixed(0)}%`, accent: false },
            { label: "Session", val: sessionStr, accent: false },
          ].map(({ label, val, accent }, i) => (
            <div key={i} className="flex items-center gap-1.5">
              {i > 0 && <div className="h-3 w-px bg-white/10" />}
              <span className="text-[11px] text-gray-500">{label}</span>
              <span className={`text-[11px] font-bold font-mono ${accent ? "text-green-400" : "text-white"}`}>{val}</span>
            </div>
          ))}
        </div>

        {/* Connection status + sign out */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="relative">
              <div className={`h-2 w-2 rounded-full ${connected ? "bg-green-400" : "bg-amber-400"}`} />
              {connected && <div className="absolute inset-0 rounded-full bg-green-400 animate-ping opacity-60" />}
            </div>
            <span className="text-[11px] text-gray-400">{connected ? "Live" : "Connecting…"}</span>
            {connected && fps > 0 && <span className="text-[10px] font-mono text-gray-600">{fps} fps</span>}
          </div>
          <button
            onClick={() => signOut(auth)}
            className="rounded-lg border border-white/10 px-2.5 py-1 text-[11px] text-gray-400 hover:border-white/20 hover:text-white transition-colors"
          >
            Sign out
          </button>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0 overflow-hidden">

        {/* ── Camera feed ──────────────────────────────────── */}
        <div className="relative flex flex-1 items-center justify-center bg-black/50 p-4 min-w-0">
          <div
            className="relative w-full max-w-3xl overflow-hidden rounded-2xl shadow-2xl"
            style={{ aspectRatio: "4/3" }}
          >
            {/* Loading overlay */}
            {(!ready || !connected) && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center rounded-2xl bg-[#030712]">
                <div className="relative mb-5">
                  <div className="h-12 w-12 rounded-full border-2 border-gray-800 border-t-cyan-400 animate-spin" />
                  <div className="absolute inset-1 rounded-full border border-cyan-500/20 animate-ping" />
                </div>
                <p className="text-sm text-gray-500 tracking-wide">
                  {!ready ? "Requesting camera access…" : "Connecting to AI engine…"}
                </p>
              </div>
            )}

            {/* Video (CSS-mirrored for selfie view) */}
            <video
              ref={videoRef}
              autoPlay muted playsInline
              className="h-full w-full rounded-2xl object-cover"
              style={{ transform: "scaleX(-1)" }}
            />

            {/* Prediction canvas (x-coords flipped in code, not CSS) */}
            <div className="absolute inset-0 rounded-2xl overflow-hidden">
              <PredictionOverlay predictions={predictions} frameWidth={frameW} frameHeight={frameH} />
            </div>

            {/* Scanning line */}
            {ready && connected && (
              <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-2xl">
                <div className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent animate-scan" />
              </div>
            )}

            {/* Corner brackets */}
            <div className="pointer-events-none absolute inset-0 rounded-2xl">
              <div className="absolute left-3 top-3 h-5 w-5 border-l-2 border-t-2 border-cyan-500/50 rounded-tl-sm" />
              <div className="absolute right-3 top-3 h-5 w-5 border-r-2 border-t-2 border-cyan-500/50 rounded-tr-sm" />
              <div className="absolute bottom-3 left-3 h-5 w-5 border-b-2 border-l-2 border-cyan-500/50 rounded-bl-sm" />
              <div className="absolute bottom-3 right-3 h-5 w-5 border-b-2 border-r-2 border-cyan-500/50 rounded-br-sm" />
            </div>

            {/* Bottom status badges */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 z-10">
              {connected && (
                <div className="flex items-center gap-1.5 rounded-full bg-black/75 px-3 py-1.5 backdrop-blur-sm border border-white/10">
                  <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-cyan-400">AI Active</span>
                </div>
              )}
              {buyCount > 0 && (
                <div className="flex items-center gap-1.5 rounded-full bg-green-900/90 px-3 py-1.5 backdrop-blur-sm border border-green-500/30">
                  <div className="h-1.5 w-1.5 rounded-full bg-green-400 animate-ping" style={{ animationDuration: "1.5s" }} />
                  <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-green-400">
                    {buyCount} Likely {buyCount === 1 ? "Buyer" : "Buyers"}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── Right panel ──────────────────────────────────── */}
        <aside className="flex w-[340px] flex-shrink-0 flex-col border-l border-white/5 bg-[#030712] overflow-hidden">

          {/* Panel header */}
          <div className="flex flex-shrink-0 items-center justify-between border-b border-white/5 px-4 py-3 bg-black/30">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-gray-500">Customer Intelligence</p>
              <p className="text-[11px] text-gray-400 mt-0.5">
                {predictions.length > 0
                  ? `${predictions.length} person${predictions.length > 1 ? "s" : ""} detected`
                  : "Waiting for customers…"}
              </p>
            </div>
            {buyCount > 0 && (
              <div className="flex items-center gap-1.5 rounded-full bg-green-500 px-2.5 py-1">
                <div className="h-1.5 w-1.5 rounded-full bg-black animate-pulse" />
                <span className="text-[9px] font-extrabold uppercase tracking-wide text-black">
                  {buyCount} Buying
                </span>
              </div>
            )}
          </div>

          {/* Scrollable content */}
          <div className="flex-1 overflow-y-auto">
            {predictions.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full py-16 text-center px-6">
                <div className="mb-4 text-5xl opacity-10 select-none">👥</div>
                <p className="text-sm text-gray-700 leading-relaxed">
                  {connected && ready
                    ? "No one in frame yet.\nStep in front of the camera."
                    : "Connecting to AI…"}
                </p>
              </div>
            ) : (
              <div className="p-3 space-y-3">
                {/* Primary customer — full analysis card */}
                {primary && <PersonCard key={primary.id} p={primary} isPrimary />}

                {/* Secondary customers — compact */}
                {secondary.length > 0 && (
                  <>
                    {secondary.length > 0 && (
                      <p className="px-1 text-[9px] font-bold uppercase tracking-[0.18em] text-gray-700">
                        Other customers
                      </p>
                    )}
                    {secondary.map(p => <MiniCard key={p.id} p={p} />)}
                  </>
                )}
              </div>
            )}
          </div>

          {/* Legend */}
          <div className="flex-shrink-0 border-t border-white/5 px-4 py-3 bg-black/20">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
              {[
                { col: "bg-green-500", label: "Likely buyer" },
                { col: "bg-blue-500",  label: "Browsing" },
                { col: "bg-amber-400", label: "Hand near item" },
                { col: "bg-cyan-400",  label: "Gaze on product" },
              ].map(({ col, label }) => (
                <div key={label} className="flex items-center gap-1.5">
                  <div className={`h-2 w-2 rounded-sm ${col}`} />
                  <span className="text-[10px] text-gray-500">{label}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>

      {/* Hidden capture canvas */}
      <canvas ref={captureCanvas} className="hidden" />
    </div>
  );
}
