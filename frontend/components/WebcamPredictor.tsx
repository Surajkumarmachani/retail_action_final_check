"use client";

import { useCallback, useEffect, useRef, useState, ReactNode } from "react";
import { signOut } from "firebase/auth";
import { auth } from "@/lib/firebase";
import { Prediction, ProductDetection, BAMEvent, BAMBand } from "@/lib/types";
import { useWebcam, FRAME_SEND_SCALE } from "@/hooks/useWebcam";
import { usePredictorSocket } from "@/hooks/useWebSocket";
import { useHolisticLandmarks } from "@/hooks/useHolisticLandmarks";
import { PredictionOverlay } from "./PredictionOverlay";
import { PersonCard, MiniCard } from "./PersonCard";
import { PostureTiltPanel } from "./PostureTiltPanel";

const STREAM_FPS = 15;
const STALE_MS   = 600;   // tighter — removes stale boxes faster

// ── Intel feed ────────────────────────────────────────────────────────────────

type EvType = "ENTRY" | "EXIT" | "OBJ" | "BAMP" | "BAMN";
interface IntelEvent { id: string; ts: number; type: EvType; body: string; tag?: string; }

const EV_STYLE: Record<EvType, string> = {
  ENTRY:  "text-cyan-400   border-cyan-500/40",
  EXIT:   "text-gray-500   border-gray-600/30",
  OBJ:    "text-amber-400  border-amber-500/40",
  BAMP:   "text-green-400  border-green-500/40",   // positive BAM event
  BAMN:   "text-red-400    border-red-500/40",      // negative BAM event
};
const EV_TAG: Record<EvType, string> = {
  ENTRY: "ENTR", EXIT: "EXIT", OBJ: "OBJ·", BAMP: "E", BAMN: "N",
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
            {ev.tag ?? EV_TAG[ev.type]}
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

const BAND_TEXT: Record<BAMBand, string> = {
  HOT:  "text-red-400",
  WARM: "text-amber-400",
  COLD: "text-blue-400",
};
const BAND_BAR: Record<BAMBand, string> = {
  HOT:  "bg-red-500",
  WARM: "bg-amber-500",
  COLD: "bg-blue-500",
};
const BAND_DOT: Record<BAMBand, string> = {
  HOT:  "bg-red-400 animate-pulse",
  WARM: "bg-amber-400",
  COLD: "bg-blue-400/60",
};

function SubjectRow({ p }: { p: Prediction }) {
  const band = (p.band ?? "COLD") as BAMBand;
  const bas  = p.bas ?? Math.round(p.prob * 100);
  const dwell = p.dwell_sec ?? 0;
  const m = Math.floor(dwell / 60), s = Math.floor(dwell % 60);
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 border-b border-white/[0.04] ${band === "HOT" ? "bg-red-950/20" : band === "WARM" ? "bg-amber-950/10" : ""}`}>
      <div className={`flex-shrink-0 h-1.5 w-1.5 rounded-full ${BAND_DOT[band]}`} />
      <span className="flex-shrink-0 w-7 font-mono text-[10px] font-bold text-gray-300">#{p.id}</span>
      <div className="flex-1 h-[3px] rounded-full bg-gray-800/80 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${BAND_BAR[band]}`}
          style={{ width: `${Math.min(100, bas)}%` }}
        />
      </div>
      <span className={`flex-shrink-0 w-8 text-right font-mono text-[10px] font-bold ${BAND_TEXT[band]}`}>
        {Math.round(bas)}
      </span>
      <span className={`flex-shrink-0 w-[46px] text-right font-mono text-[7.5px] font-bold ${BAND_TEXT[band]}`}>
        {band}
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

// ── Nearby-product detection (server products, display-res coords) ─────────────
// Finds the closest product overlapping/adjacent to a person box (within 90px).

interface NearbyProduct { label: string; conf: number; inHand: boolean; }

function getNearbyProduct(pred: Prediction, prods: ProductDetection[]): NearbyProduct | null {
  if (prods.length === 0) return null;
  const [px1, py1, px2, py2] = pred.box;
  const ph = py2 - py1;
  let best: (NearbyProduct & { score: number }) | null = null;
  const MARGIN = 90;
  for (const prod of prods) {
    if (prod.conf < 0.35) continue;
    const [ox1, oy1, ox2, oy2] = prod.box;
    const overlapX = Math.max(0, Math.min(px2 + MARGIN, ox2) - Math.max(px1 - MARGIN, ox1));
    const overlapY = Math.max(0, Math.min(py2 + MARGIN, oy2) - Math.max(py1 - MARGIN, oy1));
    if (overlapX <= 0 || overlapY <= 0) continue;
    const ocy    = (oy1 + oy2) / 2;
    const inHand = ocy > py1 + ph * 0.3;   // below top 30% of body = hand/torso zone
    const score  = prod.conf * (inHand ? 1.5 : 1.0);
    if (!best || score > best.score) best = { label: prod.label, conf: prod.conf, inHand, score };
  }
  return best ? { label: best.label, conf: best.conf, inHand: best.inHand } : null;
}

// ── Main component ────────────────────────────────────────────────────────────

export function WebcamPredictor() {
  const predMap       = useRef<Map<number, { pred: Prediction; ts: number }>>(new Map());
  const prevIds       = useRef<Set<number>>(new Set());
  // Per-subject event cooldowns
  const entrCooldown  = useRef<Map<number, number>>(new Map()); // id  → last ENTR ts
  const objCooldown   = useRef<Map<string,  number>>(new Map()); // label → last OBJ ts
  const prevProds     = useRef<ProductDetection[]>([]);          // products seen last frame
  // Delayed EXIT: only fire after subject is gone for EXIT_DELAY ms
  const disappearedAt = useRef<Map<number, number>>(new Map()); // id → disappeared ts
  const eventIdRef    = useRef(0);
  const fpsRef        = useRef(0);
  const startRef      = useRef(Date.now());

  const ENTR_COOLDOWN = 8_000;   // same subject won't spam ENTR within 8 s
  const OBJ_COOLDOWN  = 12_000;  // same object label won't spam within 12 s
  const EXIT_DELAY    = 2_500;   // subject must be gone for 2.5 s before EXIT fires

  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [products, setProducts]       = useState<ProductDetection[]>([]);
  const [sessionSec, setSessionSec]   = useState(0);
  const [fps, setFps]                 = useState(0);
  const [events, setEvents]           = useState<IntelEvent[]>([]);

  const { videoRef, captureCanvas, ready, error, captureFrame } = useWebcam();
  const meshCanvasRef = useRef<HTMLCanvasElement>(null);

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
  }, [pushEvent, ENTR_COOLDOWN]);

  // BAM behaviour events → Intel Feed (backend already cooldown-gates each type)
  const handleEvents = useCallback((evs: BAMEvent[]) => {
    for (const ev of evs) {
      const positive = ev.weight >= 0;
      const sign = positive ? "+" : "";
      pushEvent({
        ts:   ev.ts * 1000,
        type: positive ? "BAMP" : "BAMN",
        tag:  ev.type,
        body: `Subject #${ev.track_id} — ${ev.label} (${sign}${ev.weight.toFixed(0)})`,
      });
    }
  }, [pushEvent]);

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
          disappearedAt.current.delete(id);
        }
      }
    }, 500);
    return () => clearInterval(id);
  }, [pushEvent, EXIT_DELAY]);

  // Server product detections (YOLOv8). Upscale send-res (320×240) → display (640×480),
  // same as person boxes, and fire per-label "Object lock" Intel Feed events.
  const handleProducts = useCallback((prods: ProductDetection[]) => {
    const now = Date.now();
    const upscaled = prods.map(p => ({
      ...p,
      box: p.box.map(v => v / FRAME_SEND_SCALE) as ProductDetection["box"],
    }));
    const prevLabels = new Set(prevProds.current.map(p => p.label));
    for (const p of upscaled) {
      if (p.conf < 0.35) continue;
      const lastT = objCooldown.current.get(p.label) ?? 0;
      if (!prevLabels.has(p.label) && now - lastT > OBJ_COOLDOWN) {
        pushEvent({ ts: now, type: "OBJ", body: `Object lock: ${p.label} (${p.conf.toFixed(2)} conf)` });
        objCooldown.current.set(p.label, now);
      }
    }
    prevProds.current = upscaled;
    setProducts(upscaled);
  }, [pushEvent, OBJ_COOLDOWN]);

  const { connected, authError } = usePredictorSocket({
    onPredictions: handlePredictions,
    onProducts:    handleProducts,
    onEvents:      handleEvents,
    captureFrame,
    webcamReady: ready,
    fps: STREAM_FPS,
  });

  // MediaPipe enhancement layer: dense face mesh + hand mesh drawn on top of the
  // server skeleton. Purely additive — if it fails, the server layer still renders.
  const { ready: meshReady } = useHolisticLandmarks(videoRef, meshCanvasRef, connected && ready);

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
  const hotCount   = confirmed.filter(p => p.band === "HOT").length;
  const warmCount  = confirmed.filter(p => p.band === "WARM").length;
  const avgBas     = confirmed.length > 0
    ? confirmed.reduce((s, p) => s + (p.bas ?? p.prob * 100), 0) / confirmed.length : 0;

  const bandRank = (b?: string) => (b === "HOT" ? 0 : b === "WARM" ? 1 : 2);
  const sorted   = [...confirmed].sort((a, b) =>
    bandRank(a.band) !== bandRank(b.band)
      ? bandRank(a.band) - bandRank(b.band)
      : (b.bas ?? 0) - (a.bas ?? 0)
  );
  const primary   = sorted[0] ?? null;
  const secondary = sorted.slice(1);

  const mm = Math.floor(sessionSec / 60), ss = sessionSec % 60;
  const sessionStr = `${String(mm).padStart(2,"0")}:${String(ss).padStart(2,"0")}`;

  const intentLevel = hotCount >= 1 ? "HIGH"
    : warmCount >= 1 ? "MODERATE"
    : predictions.length > 0 ? "ACTIVE"
    : "STANDBY";
  const intentCol = hotCount >= 1 ? "text-red-400"
    : warmCount >= 1 ? "text-amber-400"
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
            { label: "SUBJECTS", val: String(predictions.length), col: predictions.length > 0 ? "text-cyan-400" : "text-gray-600" },
            { label: "HOT",      val: String(hotCount),           col: hotCount > 0 ? "text-red-400" : "text-gray-600" },
            { label: "AVG BAS",  val: `${avgBas.toFixed(0)}`,     col: "text-white" },
            { label: "SESSION",  val: sessionStr,                 col: "text-white" },
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
              {hotCount > 0 && (
                <span className="text-[7.5px] font-mono text-red-400 animate-pulse uppercase tracking-widest">
                  ● {hotCount} Hot Lead{hotCount > 1 ? "s" : ""}
                </span>
              )}
              {connected && ready && meshReady && (
                <span className="text-[7.5px] font-mono text-cyan-400 uppercase tracking-widest">◈ Mesh</span>
              )}
              <span className="text-[7.5px] font-mono text-gray-700">640×480</span>
            </div>
          </div>

          {/* Video */}
          <div className="flex-1 relative min-h-0 flex items-center justify-center">
            {(!ready || !connected) && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-[#020408]">
                <div className="h-9 w-9 rounded-full border border-gray-800 border-t-cyan-400/80 animate-spin mb-3" />
                <p className="text-[9.5px] font-mono text-gray-600 uppercase tracking-widest">
                  {!ready ? "Acquiring camera feed…" : "Connecting to AI engine…"}
                </p>
              </div>
            )}

            {/* Camera-aspect stage: the box matches the ACTUAL camera resolution so
                object-cover becomes a no-crop fill. Video + overlay canvas share this
                exact box, so the skeleton, person/product boxes, and video register 1:1. */}
            <div className="relative h-full max-w-full" style={{ aspectRatio: `${frameW} / ${frameH}` }}>
              <video
                ref={videoRef}
                autoPlay muted playsInline
                className="absolute inset-0 h-full w-full object-cover"
                style={{ transform: "scaleX(-1)" }}
              />

              {/* Server-rendered overlay: COCO-17 skeleton + person boxes + product boxes + gaze rays */}
              <div className="absolute inset-0 overflow-hidden">
                <PredictionOverlay predictions={predictions} products={products} frameWidth={frameW} frameHeight={frameH} />
              </div>

              {/* MediaPipe enhancement layer (on top): dense face mesh + hand mesh */}
              <canvas
                ref={meshCanvasRef}
                className="pointer-events-none absolute inset-0 h-full w-full"
              />

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
              {predictions.length} tracked · {fps}fps · YOLOv8-Pose + BAM
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
                  expression={null}
                  nearbyProduct={getNearbyProduct(primary, products)}
                />
                <PostureTiltPanel posture={primary.posture} />
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
