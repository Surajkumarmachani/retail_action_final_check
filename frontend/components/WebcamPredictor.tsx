"use client";

import { useCallback, useEffect, useRef, useState, ReactNode } from "react";
import { signOut } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { Prediction, ProductDetection } from "@/lib/types";
import { useWebcam, FRAME_SEND_SCALE } from "@/hooks/useWebcam";
import { usePredictorSocket } from "@/hooks/useWebSocket";
import { useFaceExpression, type FaceResult } from "@/hooks/useFaceExpression";
import { PredictionOverlay } from "./PredictionOverlay";
import { PersonCard, MiniCard } from "./PersonCard";

const STREAM_FPS = 15;
const STALE_MS   = 600;   // tighter — removes stale boxes faster

// ── Intel feed ────────────────────────────────────────────────────────────────

type EvType = "ENTRY" | "EXIT" | "BUY" | "BROWSE" | "OBJ" | "DWELL";
interface IntelEvent { id: string; ts: number; type: EvType; body: string; }

const EV_STYLE: Record<EvType, string> = {
  ENTRY:  "text-cyan-400  border-cyan-500/40",
  EXIT:   "text-gray-500  border-gray-600/30",
  BUY:    "text-green-400 border-green-500/40",
  BROWSE: "text-blue-400  border-blue-500/30",
  OBJ:    "text-amber-400 border-amber-500/40",
  DWELL:  "text-purple-400 border-purple-500/30",
};
const EV_TAG: Record<EvType, string> = {
  ENTRY: "ENTR", EXIT: "EXIT", BUY: "BUY!", BROWSE: "BRWS", OBJ: "OBJ·", DWELL: "DWLL",
};

function IntelFeed({ events }: { events: IntelEvent[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [events]);

  const fmt = (ts: number) => {
    const d = new Date(ts);
    return `${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}:${String(d.getSeconds()).padStart(2,"0")}`;
  };

  return (
    <div className="flex-1 overflow-y-auto px-3 py-1.5 space-y-0.5 min-h-0">
      {events.length === 0 && (
        <p className="text-[9px] font-mono text-gray-700 py-3 text-center tracking-widest uppercase">Awaiting activity…</p>
      )}
      {events.map(ev => (
        <div key={ev.id} className="flex items-center gap-2 py-[3px] border-b border-white/[0.03]">
          <span className="flex-shrink-0 font-mono text-[8.5px] text-gray-600 w-[50px]">{fmt(ev.ts)}</span>
          <span className={`flex-shrink-0 font-mono text-[7.5px] font-bold border rounded px-1 py-px leading-tight ${EV_STYLE[ev.type]}`}>
            {EV_TAG[ev.type]}
          </span>
          <span className="font-mono text-[9px] text-gray-400 leading-tight truncate">{ev.body}</span>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}

// ── Panel frame ───────────────────────────────────────────────────────────────

function Panel({
  title, badge, className = "", children,
}: { title: string; badge?: ReactNode; className?: string; children: ReactNode }) {
  return (
    <div
      className={`relative rounded-xl border border-cyan-500/15 bg-black/60 overflow-hidden flex flex-col ${className}`}
      style={{ boxShadow: "0 0 20px rgba(6,182,212,0.04), inset 0 1px 0 rgba(6,182,212,0.05)" }}
    >
      {/* Corner brackets */}
      <span className="pointer-events-none absolute left-0 top-0 z-10 h-3 w-3 border-l border-t border-cyan-400/40" />
      <span className="pointer-events-none absolute right-0 top-0 z-10 h-3 w-3 border-r border-t border-cyan-400/40" />
      <span className="pointer-events-none absolute left-0 bottom-0 z-10 h-3 w-3 border-l border-b border-cyan-400/40" />
      <span className="pointer-events-none absolute right-0 bottom-0 z-10 h-3 w-3 border-r border-b border-cyan-400/40" />
      {/* Title bar */}
      <div className="flex-shrink-0 flex items-center justify-between px-3 py-1.5 border-b border-cyan-500/10 bg-cyan-950/10">
        <div className="flex items-center gap-2">
          <span className="h-[5px] w-[5px] rounded-full bg-cyan-400/70 animate-pulse" />
          <span className="text-[8px] font-mono font-bold uppercase tracking-[0.28em] text-cyan-500/75">{title}</span>
        </div>
        {badge && <div className="text-[8px] font-mono text-gray-600">{badge}</div>}
      </div>
      {children}
    </div>
  );
}

// ── Subject registry row ──────────────────────────────────────────────────────

function SubjectRow({ p }: { p: Prediction }) {
  const isBuy = p.verdict === "LIKELY BUY";
  const dwell = p.dwell_sec ?? 0;
  const m = Math.floor(dwell / 60), s = Math.floor(dwell % 60);
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 border-b border-white/[0.04] ${isBuy ? "bg-green-950/20" : ""}`}>
      <div className={`flex-shrink-0 h-1.5 w-1.5 rounded-full ${isBuy ? "bg-green-400 animate-pulse" : "bg-blue-400/60"}`} />
      <span className="flex-shrink-0 w-7 font-mono text-[10px] font-bold text-gray-300">#{p.id}</span>
      <div className="flex-1 h-[3px] rounded-full bg-gray-800/80 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${isBuy ? "bg-green-500" : "bg-blue-500"}`}
          style={{ width: `${p.prob * 100}%` }}
        />
      </div>
      <span className={`flex-shrink-0 w-8 text-right font-mono text-[10px] font-bold ${isBuy ? "text-green-400" : "text-blue-400"}`}>
        {Math.round(p.prob * 100)}%
      </span>
      <span className={`flex-shrink-0 w-[68px] text-right font-mono text-[7.5px] font-bold ${isBuy ? "text-green-600" : "text-blue-600"}`}>
        {isBuy ? "LIKELY BUY" : "BROWSING"}
      </span>
      <span className="flex-shrink-0 w-8 text-right font-mono text-[8.5px] text-gray-600">
        {m > 0 ? `${m}m${s}s` : `${s}s`}
      </span>
    </div>
  );
}

// ── IoU deduplication (removes ghost duplicate tracks from ByteTrack) ─────────

function boxIou(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]), iy2 = Math.min(a[3], b[3]);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  const inter = (ix2 - ix1) * (iy2 - iy1);
  const aA = (a[2] - a[0]) * (a[3] - a[1]);
  const aB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (aA + aB - inter);
}

function dedupeByIou(preds: Prediction[], thresh = 0.35): Prediction[] {
  // Sort by descending probability so the "best" track wins overlaps
  const sorted = [...preds].sort((a, b) => b.prob - a.prob);
  const kept: Prediction[] = [];
  for (const p of sorted) {
    if (!kept.some(k => boxIou(p.box, k.box) > thresh)) kept.push(p);
  }
  return kept;
}

// ── Object-proximity detection ────────────────────────────────────────────────
// Determines the closest product that overlaps or is adjacent (within 110px)
// to a person box. Both coordinate systems are in display (640×480) un-mirrored space.

interface NearbyProduct { label: string; conf: number; inHand: boolean; }

function getNearbyProduct(pred: Prediction, prods: ProductDetection[]): NearbyProduct | null {
  if (prods.length === 0) return null;
  const [px1, py1, px2, py2] = pred.box;
  const ph = py2 - py1;

  let best: (NearbyProduct & { score: number }) | null = null;
  const MARGIN = 110; // px, expands person box to catch adjacent products

  for (const prod of prods) {
    if (prod.conf < 0.4) continue;
    const [ox1, oy1, ox2, oy2] = prod.box;

    const overlapX = Math.max(0, Math.min(px2 + MARGIN, ox2) - Math.max(px1 - MARGIN, ox1));
    const overlapY = Math.max(0, Math.min(py2 + MARGIN, oy2) - Math.max(py1 - MARGIN, oy1));
    if (overlapX <= 0 || overlapY <= 0) continue;

    const ocy    = (oy1 + oy2) / 2;
    const inHand = ocy > py1 + ph * 0.35; // below top 35% = hand / torso zone
    const score  = prod.conf * (inHand ? 1.5 : 1.0);

    if (!best || score > best.score) {
      best = { label: prod.label, conf: prod.conf, inHand, score };
    }
  }
  return best ? { label: best.label, conf: best.conf, inHand: best.inHand } : null;
}

// ── Main component ────────────────────────────────────────────────────────────

export function WebcamPredictor() {
  const predMap       = useRef<Map<number, { pred: Prediction; ts: number }>>(new Map());
  const prevVerdicts  = useRef<Map<number, string>>(new Map());
  const prevIds       = useRef<Set<number>>(new Set());
  const prevProds     = useRef<ProductDetection[]>([]);
  // Per-subject event cooldowns
  const entrCooldown  = useRef<Map<number, number>>(new Map()); // id  → last ENTR ts
  const buyCooldown   = useRef<Map<number, number>>(new Map()); // id  → last BUY  ts
  const objCooldown   = useRef<Map<string,  number>>(new Map()); // label → last OBJ ts
  // Delayed EXIT: only fire after subject is gone for EXIT_DELAY ms
  const disappearedAt = useRef<Map<number, number>>(new Map()); // id → disappeared ts
  const eventIdRef    = useRef(0);
  const fpsRef        = useRef(0);
  const startRef      = useRef(Date.now());

  const ENTR_COOLDOWN = 8_000;   // same subject won't spam ENTR within 8 s
  const BUY_COOLDOWN  = 20_000;  // same subject won't spam BUY  within 20 s
  const OBJ_COOLDOWN  = 12_000;  // same object label won't spam within 12 s
  const EXIT_DELAY    = 2_500;   // subject must be gone for 2.5 s before EXIT fires

  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [products, setProducts]       = useState<ProductDetection[]>([]);
  const [sessionSec, setSessionSec]   = useState(0);
  const [fps, setFps]                 = useState(0);
  const [events, setEvents]           = useState<IntelEvent[]>([]);

  const { videoRef, captureCanvas, ready, error, captureFrame } = useWebcam();

  const pushEvent = useCallback((ev: Omit<IntelEvent, "id">) => {
    const id = String(++eventIdRef.current);
    setEvents(prev => [...prev.slice(-99), { ...ev, id }]);
  }, []);

  const handlePredictions = useCallback((preds: Prediction[]) => {
    fpsRef.current += 1;
    const now = Date.now();
    const map = predMap.current;

    // Scale box coords from send-resolution back to display-resolution
    const upscaled = preds.map(p => ({
      ...p,
      box: p.box.map(v => v / FRAME_SEND_SCALE) as Prediction["box"],
    }));

    // Remove ghost duplicate tracks before any processing
    const deduped = dedupeByIou(upscaled);
    const curIds  = new Set(deduped.map(p => p.id));

    for (const p of deduped) {
      // Cancel any pending EXIT for this subject (it reappeared)
      disappearedAt.current.delete(p.id);

      // ENTR — only if subject is genuinely new OR has been gone a long time
      if (!prevIds.current.has(p.id)) {
        const lastEntr = entrCooldown.current.get(p.id) ?? 0;
        if (now - lastEntr > ENTR_COOLDOWN) {
          pushEvent({ ts: now, type: "ENTRY", body: `Subject #${p.id} entered surveillance zone` });
          entrCooldown.current.set(p.id, now);
        }
      }

      // Verdict change events
      const prev = prevVerdicts.current.get(p.id);
      if (prev !== p.verdict) {
        if (p.verdict === "LIKELY BUY") {
          const lastBuy = buyCooldown.current.get(p.id) ?? 0;
          if (now - lastBuy > BUY_COOLDOWN) {
            pushEvent({ ts: now, type: "BUY", body: `Subject #${p.id} — HIGH PURCHASE INTENT ${Math.round(p.prob * 100)}%` });
            buyCooldown.current.set(p.id, now);
          }
        } else if (prev === "LIKELY BUY") {
          pushEvent({ ts: now, type: "BROWSE", body: `Subject #${p.id} intent dropped → browsing` });
        }
        prevVerdicts.current.set(p.id, p.verdict);
      }
      map.set(p.id, { pred: p, ts: now });
    }

    // Schedule delayed EXIT for subjects that disappeared this frame
    for (const id of prevIds.current) {
      if (!curIds.has(id) && !disappearedAt.current.has(id)) {
        disappearedAt.current.set(id, now);
      }
    }
    prevIds.current = curIds;

    for (const [id, { ts }] of map) if (now - ts > STALE_MS) map.delete(id);
    setPredictions(Array.from(map.values()).map(v => v.pred));
  }, [pushEvent, ENTR_COOLDOWN, BUY_COOLDOWN]);

  // Periodic cleanup — also fires delayed EXIT events
  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      const map = predMap.current;
      let changed = false;

      // Stale prediction cleanup
      for (const [id, { ts }] of map) if (now - ts > STALE_MS) { map.delete(id); changed = true; }
      if (changed) setPredictions(Array.from(map.values()).map(v => v.pred));

      // Fire EXIT events for subjects gone longer than EXIT_DELAY
      for (const [id, ts] of disappearedAt.current) {
        if (now - ts >= EXIT_DELAY) {
          pushEvent({ ts: now, type: "EXIT", body: `Subject #${id} exited surveillance zone` });
          prevVerdicts.current.delete(id);
          disappearedAt.current.delete(id);
        }
      }
    }, 500);
    return () => clearInterval(id);
  }, [pushEvent, EXIT_DELAY]);

  const handleProducts = useCallback((prods: ProductDetection[]) => {
    const now = Date.now();
    // Scale product box coords from send-resolution (320×240) → display-resolution (640×480)
    const upscaled = prods.map(p => ({
      ...p,
      box: p.box.map(v => v / FRAME_SEND_SCALE) as ProductDetection["box"],
    }));
    const prevLabels = new Set(prevProds.current.map(p => p.label));
    for (const p of upscaled) {
      const lastT = objCooldown.current.get(p.label) ?? 0;
      if (!prevLabels.has(p.label) && now - lastT > OBJ_COOLDOWN) {
        pushEvent({ ts: now, type: "OBJ", body: `Object lock: ${p.label} (${Math.round(p.conf * 100)}% conf)` });
        objCooldown.current.set(p.label, now);
      }
    }
    prevProds.current = upscaled;
    setProducts(upscaled);
  }, [pushEvent, OBJ_COOLDOWN]);

  const { connected, authError } = usePredictorSocket({
    onPredictions: handlePredictions,
    onProducts:    handleProducts,
    captureFrame,
    webcamReady: ready,
    fps: STREAM_FPS,
  });

  // Face expressions run at 5fps in the browser — no API key needed
  const faceResults = useFaceExpression(videoRef, connected && ready);

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

  // Only show confirmed subjects (dwell >= 2s) in registry/analysis panels.
  // Ghost ByteTrack tracks typically die within 1-2 seconds; this hides them.
  const confirmed  = predictions.filter(p => (p.dwell_sec ?? 0) >= 2);
  const buyCount   = confirmed.filter(p => p.verdict === "LIKELY BUY").length;
  const avgProb    = confirmed.length > 0
    ? confirmed.reduce((s, p) => s + p.prob, 0) / confirmed.length : 0;

  const sorted    = [...confirmed].sort((a, b) =>
    a.verdict !== b.verdict ? (a.verdict === "LIKELY BUY" ? -1 : 1) : b.prob - a.prob
  );
  const primary   = sorted[0] ?? null;
  const secondary = sorted.slice(1);

  // Match face-api expression to the nearest prediction by face-centre within body box
  function getExpression(pred: Prediction): FaceResult | null {
    // Face is in the top 55% of the body box
    const topY = pred.box[1];
    const midY = pred.box[1] + (pred.box[3] - pred.box[1]) * 0.55;
    let best: FaceResult | null = null, bestDist = Infinity;
    for (const f of faceResults) {
      if (f.cx >= pred.box[0] && f.cx <= pred.box[2] && f.cy >= topY && f.cy <= midY) {
        const dx = f.cx - (pred.box[0] + pred.box[2]) / 2;
        const dy = f.cy - topY;
        const d  = Math.hypot(dx, dy);
        if (d < bestDist) { bestDist = d; best = f; }
      }
    }
    return best;
  }

  const mm = Math.floor(sessionSec / 60), ss = sessionSec % 60;
  const sessionStr = `${String(mm).padStart(2,"0")}:${String(ss).padStart(2,"0")}`;

  const intentLevel = buyCount >= 3 ? "HIGH"
    : buyCount >= 1 ? "MODERATE"
    : predictions.length > 0 ? "ACTIVE"
    : "STANDBY";
  const intentCol = buyCount >= 3 ? "text-red-400"
    : buyCount >= 1 ? "text-amber-400"
    : predictions.length > 0 ? "text-cyan-400"
    : "text-gray-600";

  if (error || authError) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#020408]">
        <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-6 text-center">
          <p className="text-red-400 text-sm font-mono">{String(error ?? authError)}</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex h-screen flex-col overflow-hidden bg-[#020408] text-white select-none"
      style={{
        backgroundImage:
          "linear-gradient(rgba(6,182,212,0.033) 1px,transparent 1px)," +
          "linear-gradient(90deg,rgba(6,182,212,0.033) 1px,transparent 1px)",
        backgroundSize: "44px 44px",
      }}
    >
      {/* ── Command header ─────────────────────────────────────────────────── */}
      <header className="flex flex-shrink-0 items-center justify-between border-b border-cyan-500/15 bg-black/80 px-5 py-2 backdrop-blur-sm z-20">
        {/* Brand + stats */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded border border-cyan-500/35 bg-cyan-950/50">
              <svg className="h-3.5 w-3.5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
              </svg>
            </div>
            <div>
              <span className="block text-[11px] font-black tracking-[0.22em] text-cyan-400">MACHANI</span>
              <span className="block text-[7px] font-medium uppercase tracking-[0.3em] text-gray-600">Retail Intelligence · Command Center</span>
            </div>
          </div>

          <div className="h-5 w-px bg-cyan-500/15" />

          {/* Live stats pills */}
          {[
            { label: "SUBJECTS",   val: String(predictions.length), col: predictions.length > 0 ? "text-cyan-400" : "text-gray-600" },
            { label: "BUY INTENT", val: String(buyCount),           col: buyCount > 0 ? "text-green-400" : "text-gray-600" },
            { label: "AVG CONF",   val: `${(avgProb * 100).toFixed(0)}%`, col: "text-white" },
            { label: "SESSION",    val: sessionStr,                 col: "text-white" },
          ].map(({ label, val, col }, i) => (
            <div key={i} className="flex items-center gap-1.5">
              {i > 0 && <div className="h-3 w-px bg-white/8" />}
              <span className="text-[7.5px] font-mono text-gray-600 uppercase tracking-[0.16em]">{label}</span>
              <span className={`text-[11px] font-bold font-mono ${col}`}>{val}</span>
            </div>
          ))}
        </div>

        {/* Right: intent level + live + signout */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 rounded border border-white/8 bg-black/40 px-2.5 py-1">
            <span className="text-[7px] font-mono text-gray-600 uppercase tracking-[0.2em]">Intent Level</span>
            <span className={`text-[9px] font-mono font-bold tracking-wider ${intentCol}`}>{intentLevel}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="relative">
              <div className={`h-2 w-2 rounded-full ${connected ? "bg-green-400" : "bg-amber-400"}`} />
              {connected && <div className="absolute inset-0 rounded-full bg-green-400 animate-ping opacity-50" />}
            </div>
            <span className="text-[9px] font-mono text-gray-500">{connected ? "LIVE" : "CONNECTING…"}</span>
            {connected && fps > 0 && <span className="text-[8.5px] font-mono text-gray-700 ml-0.5">{fps}fps</span>}
          </div>
          <button
            onClick={() => signOut(auth)}
            className="rounded border border-white/8 px-2.5 py-1 text-[8px] font-mono uppercase tracking-wider text-gray-500 hover:border-cyan-500/30 hover:text-cyan-400 transition-colors"
          >
            End Session
          </button>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0 gap-2.5 p-2.5">

        {/* ── Live feed ─────────────────────────────────────────────────────── */}
        <div className="relative flex-1 min-w-0 flex flex-col rounded-xl border border-cyan-500/20 bg-black/50 overflow-hidden"
          style={{ boxShadow: "0 0 30px rgba(6,182,212,0.06), inset 0 1px 0 rgba(6,182,212,0.07)" }}>

          {/* Feed label */}
          <div className="flex-shrink-0 flex items-center justify-between px-3 py-1.5 border-b border-cyan-500/12 bg-black/60">
            <div className="flex items-center gap-2">
              <div className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-cyan-400 animate-pulse" : "bg-gray-600"}`} />
              <span className="text-[8px] font-mono font-bold uppercase tracking-[0.24em] text-cyan-500/80">
                Live Surveillance · Sector A · Cam-01
              </span>
            </div>
            <div className="flex items-center gap-3">
              {buyCount > 0 && (
                <span className="text-[7.5px] font-mono text-green-400 animate-pulse uppercase tracking-widest">
                  ● {buyCount} High Intent
                </span>
              )}
              <span className="text-[7.5px] font-mono text-gray-700">640×480</span>
            </div>
          </div>

          {/* Video */}
          <div className="flex-1 relative min-h-0">
            {(!ready || !connected) && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-[#020408]">
                <div className="h-9 w-9 rounded-full border border-gray-800 border-t-cyan-400/80 animate-spin mb-3" />
                <p className="text-[9.5px] font-mono text-gray-600 uppercase tracking-widest">
                  {!ready ? "Acquiring camera feed…" : "Connecting to AI engine…"}
                </p>
              </div>
            )}

            <video
              ref={videoRef}
              autoPlay muted playsInline
              className="h-full w-full object-cover"
              style={{ transform: "scaleX(-1)" }}
            />

            <div className="absolute inset-0 overflow-hidden">
              <PredictionOverlay predictions={predictions} products={products} frameWidth={frameW} frameHeight={frameH} />
            </div>

            {/* Scan line */}
            {ready && connected && (
              <div className="pointer-events-none absolute inset-0 overflow-hidden">
                <div className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-cyan-400/25 to-transparent animate-scan" />
              </div>
            )}

            {/* Corner brackets */}
            <div className="pointer-events-none absolute inset-0">
              <div className="absolute left-2 top-2 h-4 w-4 border-l border-t border-cyan-500/55" />
              <div className="absolute right-2 top-2 h-4 w-4 border-r border-t border-cyan-500/55" />
              <div className="absolute left-2 bottom-2 h-4 w-4 border-l border-b border-cyan-500/55" />
              <div className="absolute right-2 bottom-2 h-4 w-4 border-r border-b border-cyan-500/55" />
            </div>
          </div>

          {/* Feed footer */}
          <div className="flex-shrink-0 flex items-center justify-between px-3 py-1.5 border-t border-cyan-500/12 bg-black/60">
            <div className="flex items-center gap-4">
              {connected && (
                <span className="text-[7.5px] font-mono uppercase tracking-widest text-cyan-500/80">◉ AI Active</span>
              )}
              {products.length > 0 && (
                <span className="text-[7.5px] font-mono text-amber-500 uppercase tracking-widest">
                  ◈ {products.length} Object{products.length > 1 ? "s" : ""} Locked
                </span>
              )}
            </div>
            <span className="text-[7.5px] font-mono text-gray-700">
              {predictions.length} tracked · {fps}fps · ByteTrack+MediaPipe
            </span>
          </div>
        </div>

        {/* ── Right column ───────────────────────────────────────────────────── */}
        <div className="flex w-[365px] flex-shrink-0 flex-col gap-2.5">

          {/* Subject Registry */}
          <Panel title="Subject Registry" badge={predictions.length > 0 ? `${predictions.length} active` : undefined}>
            {predictions.length === 0 ? (
              <p className="text-[8.5px] font-mono text-gray-700 py-4 text-center tracking-widest uppercase">
                No subjects detected
              </p>
            ) : (
              <div className="overflow-y-auto max-h-36">
                {sorted.map(p => <SubjectRow key={p.id} p={p} />)}
              </div>
            )}
          </Panel>

          {/* Behavioral Analysis */}
          <Panel
            title="Behavioral Analysis"
            badge={primary ? `Subject #${primary.id}` : undefined}
            className="flex-1 min-h-0"
          >
            {primary ? (
              <div className="flex-1 overflow-y-auto min-h-0">
                <PersonCard
                  p={primary}
                  isPrimary
                  expression={getExpression(primary)}
                  nearbyProduct={getNearbyProduct(primary, products)}
                />
                {secondary.length > 0 && (
                  <div className="px-3 pb-3 pt-2 space-y-1.5 border-t border-white/5">
                    <p className="text-[7.5px] font-mono font-bold uppercase tracking-[0.22em] text-gray-700">
                      Other subjects
                    </p>
                    {secondary.map(p => <MiniCard key={p.id} p={p} />)}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center flex-1 py-10">
                <p className="text-[8.5px] font-mono text-gray-700 tracking-widest uppercase">Awaiting subjects…</p>
              </div>
            )}
          </Panel>

          {/* Intel Feed */}
          <Panel title="Intel Feed" badge={`${events.length} events`} className="h-44 flex-shrink-0">
            <IntelFeed events={events} />
          </Panel>
        </div>
      </div>

      <canvas ref={captureCanvas} className="hidden" />
    </div>
  );
}
