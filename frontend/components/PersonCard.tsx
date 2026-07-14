"use client";

import { useEffect, useRef, useState } from "react";
import { Prediction, BAMBand } from "@/lib/types";

interface NearbyProduct { label: string; conf: number; inHand: boolean; }
export type ExpressionName = "neutral" | "happy" | "sad" | "angry" | "surprised";
export interface Expression { dominant: ExpressionName; confidence: number; scores: Record<string, number>; }

// ── Expression metadata ───────────────────────────────────────────────────────

const EXPR_ICON: Record<ExpressionName, string> = {
  happy:     "😊",
  surprised: "😮",
  neutral:   "😐",
  sad:       "😟",
  angry:     "😠",
};

const EXPR_LABEL: Record<ExpressionName, string> = {
  happy:     "Positive — likely engaged",
  surprised: "Actively interested",
  neutral:   "Neutral / evaluating",
  sad:       "Hesitant",
  angry:     "Frustrated",
};

const EXPR_COLOR: Record<ExpressionName, { text: string; bar: string; bg: string; border: string }> = {
  happy:     { text: "text-green-300",  bar: "bg-green-400",  bg: "bg-green-950/30",  border: "border-green-500/25" },
  surprised: { text: "text-cyan-300",   bar: "bg-cyan-400",   bg: "bg-cyan-950/30",   border: "border-cyan-500/25" },
  neutral:   { text: "text-gray-300",   bar: "bg-gray-500",   bg: "bg-white/[0.025]", border: "border-white/8" },
  sad:       { text: "text-blue-300",   bar: "bg-blue-400",   bg: "bg-blue-950/20",   border: "border-blue-500/20" },
  angry:     { text: "text-red-300",    bar: "bg-red-400",    bg: "bg-red-950/20",    border: "border-red-500/20" },
};

// ── BAM band + event metadata ──────────────────────────────────────────────────

const BAND_META: Record<BAMBand, { text: string; ring: string; label: string; desc: string; icon: string }> = {
  HOT:  { text: "text-red-400",   ring: "#ef4444", label: "HOT LEAD", desc: "High buying affinity — alert staff", icon: "⚡" },
  WARM: { text: "text-amber-400", ring: "#f59e0b", label: "WARM",     desc: "Engaged — worth watching",           icon: "🛒" },
  COLD: { text: "text-blue-400",  ring: "#3b82f6", label: "COLD",     desc: "Browsing / low intent",              icon: "🔄" },
};

const EVENT_META: Record<string, { label: string; pos: boolean }> = {
  E01: { label: "Basket Placement", pos: true },  E02: { label: "Product Pickup",   pos: true },
  E03: { label: "Try-On / Demo",    pos: true },  E04: { label: "Hold Duration",    pos: true },
  E05: { label: "Comparison",       pos: true },  E06: { label: "Price Inspection", pos: true },
  E07: { label: "Staff Contact",    pos: true },  E08: { label: "Zone Dwell",       pos: true },
  E09: { label: "Repeat Return",    pos: true },  E10: { label: "Gaze Fixation",    pos: true },
  E11: { label: "Body Lean-In",     pos: true },  E12: { label: "Deceleration",     pos: true },
  E13: { label: "Reach Attempt",    pos: true },  E14: { label: "Companion",        pos: true },
  E15: { label: "Path Pattern",     pos: true },  E16: { label: "Zone Transition",  pos: true },
  E17: { label: "Phone at Shelf",   pos: true },  E18: { label: "Session Duration", pos: true },
  N01: { label: "Rapid Put-Back",   pos: false }, N02: { label: "Queue Abandon",    pos: false },
  N03: { label: "Walk-Past",        pos: false }, N04: { label: "Exit Movement",    pos: false },
};

const EMA_K = 0.6; // higher = faster response

// ── Hooks ─────────────────────────────────────────────────────────────────────

function useSmooth(raw: number) {
  const ref = useRef(raw);
  const [val, setVal] = useState(raw);
  useEffect(() => {
    ref.current = ref.current + EMA_K * (raw - ref.current);
    setVal(ref.current);
  }, [raw]);
  return val;
}

function useHistory(raw: number, maxPts = 24) {
  const hist = useRef<number[]>([raw]);
  const [snap, setSnap] = useState<number[]>([raw]);
  useEffect(() => {
    hist.current = [...hist.current.slice(-(maxPts - 1)), raw];
    setSnap([...hist.current]);
  }, [raw, maxPts]);
  return snap;
}

// ── Confidence sparkline (values 0-1) ──────────────────────────────────────────

function SparkLine({ pts, color, uid }: { pts: number[]; color: string; uid: string }) {
  if (pts.length < 2) return null;
  const W = 116, H = 26;
  const sy  = (v: number) => H - v * H;
  const sx  = (i: number) => (i / (pts.length - 1)) * W;
  const d   = pts.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
  const gid = `spk-${uid}`;
  const last = pts[pts.length - 1];
  const prev = pts[pts.length - 2] ?? last;
  const trend = last > prev ? "↑" : last < prev ? "↓" : "→";
  const trendCol = last > prev ? "#22c55e" : last < prev ? "#f87171" : "#9ca3af";
  return (
    <div className="flex items-center gap-2">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="overflow-visible flex-shrink-0">
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.22" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`${d} L${W},${H} L0,${H} Z`} fill={`url(#${gid})`} />
        <path d={d} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
        <circle cx={sx(pts.length - 1)} cy={sy(last)} r="2.5" fill={color} />
      </svg>
      <span className="text-[11px] font-mono font-bold flex-shrink-0" style={{ color: trendCol }}>
        {trend}
      </span>
    </div>
  );
}

// ── Large animated BAM Score ring (0-100) ──────────────────────────────────────

function BigRing({ bas, band }: { bas: number; band: BAMBand }) {
  const r  = 46;
  const c  = 2 * Math.PI * r;
  const pct = Math.round(bas);
  const meta = BAND_META[band];
  return (
    <div className="flex flex-col items-center">
      <svg width="108" height="108" viewBox="0 0 108 108">
        <defs>
          <filter id="gfx">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <circle cx="54" cy="54" r={r} fill="none" stroke="#1a2332" strokeWidth="7" />
        <circle
          cx="54" cy="54" r={r} fill="none"
          stroke={meta.ring}
          strokeWidth="7"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - Math.min(1, bas / 100))}
          strokeLinecap="round"
          transform="rotate(-90 54 54)"
          filter="url(#gfx)"
          style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(.4,0,.2,1), stroke .5s ease" }}
        />
        <text x="54" y="50" textAnchor="middle" fill="white" fontSize="22" fontWeight="800" fontFamily="monospace">
          {pct}
        </text>
        <text x="54" y="66" textAnchor="middle"
          fill={meta.ring}
          fontSize="7.5" fontWeight="700" letterSpacing="0.12em">
          {meta.label}
        </text>
      </svg>
      <span className="mt-0.5 text-[8px] font-mono uppercase tracking-[0.18em] text-gray-600">BAM Score</span>
    </div>
  );
}

// ── Behaviour event row ─────────────────────────────────────────────────────────

function EventRow({ code }: { code: string }) {
  const meta = EVENT_META[code] ?? { label: code, pos: true };
  return (
    <div className={`flex items-center gap-2.5 rounded-xl px-3 py-2 border transition-all duration-500 ${
      meta.pos ? "bg-green-950/40 border-green-500/25" : "bg-red-950/40 border-red-500/25"
    }`}>
      <span className={`flex-shrink-0 font-mono text-[8px] font-bold border rounded px-1.5 py-0.5 ${
        meta.pos ? "text-green-400 border-green-500/40" : "text-red-400 border-red-500/40"
      }`}>
        {code}
      </span>
      <span className={`text-[12px] font-medium leading-tight ${meta.pos ? "text-gray-100" : "text-red-200"}`}>
        {meta.label}
      </span>
      <div className={`ml-auto h-2 w-2 flex-shrink-0 rounded-full ${meta.pos ? "bg-green-400 animate-pulse" : "bg-red-400"}`} />
    </div>
  );
}

// ── Dwell timer ─────────────────────────────────────────────────────────────────

function DwellBadge({ sec }: { sec: number }) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return (
    <div className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-1">
      <svg className="h-3 w-3 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6l4 2m6-2a10 10 0 11-20 0 10 10 0 0120 0z"/>
      </svg>
      <span className="font-mono text-[11px] text-gray-300">
        {m > 0 ? `${m}m ` : ""}{s}s dwell
      </span>
    </div>
  );
}

// ── Mini card (secondary customers) ──────────────────────────────────────────

export function MiniCard({ p }: { p: Prediction }) {
  const band = (p.band ?? "COLD") as BAMBand;
  const meta = BAND_META[band];
  const bas  = useSmooth(p.bas ?? p.prob * 100);
  return (
    <div className={`flex items-center gap-3 rounded-xl border px-3 py-2.5 transition-all duration-500 ${
      band === "HOT" ? "border-red-500/20 bg-red-950/15" : band === "WARM" ? "border-amber-500/20 bg-amber-950/10" : "border-white/5 bg-white/[0.015]"
    }`}>
      <div className="flex-shrink-0">
        <div className={`h-8 w-8 rounded-full flex items-center justify-center text-xs font-bold ${
          band === "HOT" ? "bg-red-900/60 text-red-300" : band === "WARM" ? "bg-amber-900/50 text-amber-300" : "bg-gray-800 text-gray-400"
        }`}>
          #{p.id}
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <p className={`text-xs font-bold ${meta.text}`}>{meta.label}</p>
        <div className="mt-1 h-1 w-full rounded-full bg-gray-800 overflow-hidden">
          <div
            className="h-1 rounded-full"
            style={{ width: `${Math.min(100, bas)}%`, backgroundColor: meta.ring, transition: "width 0.8s ease" }}
          />
        </div>
      </div>
      <span className="flex-shrink-0 font-mono text-sm font-bold" style={{ color: meta.ring }}>
        {Math.round(bas)}
      </span>
    </div>
  );
}

// ── Primary analysis card ─────────────────────────────────────────────────────

export function PersonCard({
  p,
  isPrimary = false,
  expression = null,
  nearbyProduct = null,
}: {
  p: Prediction;
  isPrimary?: boolean;
  expression?: Expression | null;
  nearbyProduct?: NearbyProduct | null;
}) {
  const band = (p.band ?? "COLD") as BAMBand;
  const meta = BAND_META[band];
  const rawBas = p.bas ?? p.prob * 100;
  const bas  = useSmooth(rawBas);
  const hist = useHistory(bas / 100);
  const dwell = p.dwell_sec ?? 0;
  const events = p.top_events ?? [];

  // Flash on meaningful score change
  const prevB  = useRef(rawBas);
  const [flash, setFlash] = useState(false);
  useEffect(() => {
    if (Math.abs(rawBas - prevB.current) > 5) {
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 600);
      prevB.current = rawBas;
      return () => clearTimeout(t);
    }
    prevB.current = rawBas;
  }, [rawBas]);

  const isHot = band === "HOT";

  return (
    <div
      className={`overflow-hidden rounded-2xl border transition-all duration-500 ${
        isHot
          ? "border-red-500/30 bg-gradient-to-b from-red-950/30 to-[#030712]"
          : band === "WARM"
            ? "border-amber-500/25 bg-gradient-to-b from-amber-950/20 to-[#030712]"
            : "border-white/8 bg-gradient-to-b from-white/[0.03] to-[#030712]"
      } ${flash ? "ring-1 ring-white/20" : ""}`}
      style={isHot ? { boxShadow: "0 0 40px rgba(239,68,68,0.12), 0 0 80px rgba(239,68,68,0.05)" } : {}}
    >
      {/* Top section: ID + ring + dwell */}
      <div className="px-4 pt-4 pb-3">
        <div className="flex items-start justify-between mb-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] font-mono uppercase text-gray-600">Customer</span>
              <span className="text-base font-black text-white">#{p.id}</span>
              {isPrimary && isHot && (
                <span className="rounded-full bg-red-500 px-2 py-0.5 text-[8px] font-extrabold uppercase tracking-widest text-black">
                  TOP
                </span>
              )}
              {p.is_staff && (
                <span className="rounded-full bg-green-600 px-2 py-0.5 text-[8px] font-extrabold uppercase tracking-widest text-black">
                  STAFF
                </span>
              )}
            </div>
            <DwellBadge sec={dwell} />
          </div>

          <BigRing bas={bas} band={band} />
        </div>

        {/* BAM Score trend sparkline */}
        <div className="flex items-center justify-between rounded-xl bg-white/[0.025] border border-white/5 px-3 py-2 mt-1">
          <div className="flex flex-col gap-0.5">
            <span className="text-[9px] font-bold uppercase tracking-[0.16em] text-gray-600">Affinity Trend</span>
            <span className="text-[12px] font-mono font-bold" style={{ color: meta.ring }}>
              {Math.round(bas)} / 100
            </span>
          </div>
          <SparkLine pts={hist} color={meta.ring} uid={`${p.id}`} />
        </div>
      </div>

      {/* Facial expression */}
      {expression && (() => {
        const ec = EXPR_COLOR[expression.dominant];
        return (
          <div className="px-3 pb-2">
            <div className={`flex items-center gap-2.5 rounded-xl px-3 py-2 border transition-all duration-500 ${ec.bg} ${ec.border}`}>
              <span className="text-xl leading-none flex-shrink-0">{EXPR_ICON[expression.dominant]}</span>
              <div className="flex-1 min-w-0">
                <p className={`text-[11px] font-bold leading-tight ${ec.text}`}>
                  {EXPR_LABEL[expression.dominant]}
                </p>
                <div className="mt-1 h-1 rounded-full bg-gray-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${ec.bar}`}
                    style={{ width: `${expression.confidence * 100}%` }}
                  />
                </div>
              </div>
              <div className="flex-shrink-0 text-right">
                <p className="text-[8px] font-mono text-gray-600 uppercase tracking-wider">Facial</p>
                <p className={`text-[11px] font-mono font-bold ${ec.text}`}>
                  {Math.round(expression.confidence * 100)}%
                </p>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Nearby / held product */}
      {nearbyProduct && (
        <div className="px-3 pb-2">
          <div className="flex items-center gap-2.5 rounded-xl px-3 py-2 border border-amber-500/25 bg-amber-950/20">
            <span className="text-lg leading-none flex-shrink-0">{nearbyProduct.inHand ? "📦" : "🛍️"}</span>
            <div className="flex-1 min-w-0">
              <p className="text-[11px] font-bold leading-tight text-amber-300">
                {nearbyProduct.inHand ? "Holding" : "Looking at"}: {nearbyProduct.label}
              </p>
              <p className="text-[9px] text-gray-500 mt-0.5">Object detection {Math.round(nearbyProduct.conf * 100)}%</p>
            </div>
          </div>
        </div>
      )}

      {/* Behaviour events */}
      <div className="px-3 pb-3 space-y-1.5">
        <p className="px-1 mb-2 text-[9px] font-bold uppercase tracking-[0.2em] text-gray-600">
          Behaviour Signals
        </p>
        {events.length > 0 ? (
          events.map((code, i) => <EventRow key={`${code}-${i}`} code={code} />)
        ) : (
          <div className="rounded-xl px-3 py-3 border border-white/5 bg-white/[0.02]">
            <p className="text-[11px] text-gray-500">Analysing behaviour patterns…</p>
          </div>
        )}
      </div>

      {/* Band verdict highlight */}
      <div className={`mx-3 mb-3 rounded-xl p-3 transition-all duration-700 ${
        isHot
          ? "bg-red-950/60 border border-red-500/30"
          : band === "WARM"
            ? "bg-amber-950/40 border border-amber-500/25"
            : "bg-white/3 border border-white/5"
      }`}>
        <div className="flex items-center gap-2.5">
          <span className="text-2xl leading-none">{meta.icon}</span>
          <div className="flex-1 min-w-0">
            <p className={`text-sm font-bold ${meta.text}`}>{meta.label}</p>
            <p className="text-[10px] text-gray-600 mt-0.5">{meta.desc}</p>
          </div>
          {isHot && <div className="h-2.5 w-2.5 flex-shrink-0 rounded-full bg-red-400 animate-pulse" />}
        </div>
      </div>
    </div>
  );
}
