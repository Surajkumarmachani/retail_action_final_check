"use client";

import { useEffect, useRef, useState } from "react";
import { Prediction, PersonMetrics } from "@/lib/types";

interface NearbyProduct { label: string; conf: number; inHand: boolean; }
import { type FaceResult, type ExpressionName } from "@/hooks/useFaceExpression";

// ── Expression metadata ───────────────────────────────────────────────────────

const EXPR_ICON: Record<ExpressionName, string> = {
  happy:     "😊",
  surprised: "😮",
  neutral:   "😐",
  sad:       "😟",
  angry:     "😠",
  disgusted: "🤢",
  fearful:   "😰",
};

const EXPR_LABEL: Record<ExpressionName, string> = {
  happy:     "Positive — likely engaged",
  surprised: "Actively interested",
  neutral:   "Neutral / evaluating",
  sad:       "Hesitant",
  angry:     "Frustrated",
  disgusted: "Negative reaction",
  fearful:   "Uncertain",
};

const EXPR_COLOR: Record<ExpressionName, { text: string; bar: string; bg: string; border: string }> = {
  happy:     { text: "text-green-300",  bar: "bg-green-400",  bg: "bg-green-950/30",  border: "border-green-500/25" },
  surprised: { text: "text-cyan-300",   bar: "bg-cyan-400",   bg: "bg-cyan-950/30",   border: "border-cyan-500/25" },
  neutral:   { text: "text-gray-300",   bar: "bg-gray-500",   bg: "bg-white/[0.025]", border: "border-white/8" },
  sad:       { text: "text-blue-300",   bar: "bg-blue-400",   bg: "bg-blue-950/20",   border: "border-blue-500/20" },
  angry:     { text: "text-red-300",    bar: "bg-red-400",    bg: "bg-red-950/20",    border: "border-red-500/20" },
  disgusted: { text: "text-orange-300", bar: "bg-orange-400", bg: "bg-orange-950/20", border: "border-orange-500/20" },
  fearful:   { text: "text-purple-300", bar: "bg-purple-400", bg: "bg-purple-950/20", border: "border-purple-500/20" },
};

const DECAY_MS  = 1500; // signals stay green for 1.5 s after MediaPipe last saw them
const EMA_K     = 0.65; // higher = faster response to probability changes

// ── Hooks ─────────────────────────────────────────────────────────────────────

function useSmoothProb(raw: number) {
  const ref = useRef(raw);
  const [val, setVal] = useState(raw);
  useEffect(() => {
    ref.current = ref.current + EMA_K * (raw - ref.current);
    setVal(ref.current);
  }, [raw]);
  return val;
}

function useProbHistory(raw: number, maxPts = 24) {
  const hist = useRef<number[]>([raw]);
  const [snap, setSnap] = useState<number[]>([raw]);
  useEffect(() => {
    hist.current = [...hist.current.slice(-(maxPts - 1)), raw];
    setSnap([...hist.current]);
  }, [raw, maxPts]);
  return snap;
}

function useDecay(m: PersonMetrics | undefined) {
  const ts = useRef<Map<string, number>>(new Map());
  const [, tick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => tick(n => n + 1), 500);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    if (!m) return;
    const now = Date.now();
    const keys: (keyof PersonMetrics)[] = [
      "pose_detected", "arm_cross", "wrist_raised", "wrist_near_shelf",
      "holding_product", "relaxed_posture", "body_turned_away", "step_back",
      "face_detected", "looking_at_product", "hand_open", "hand_pinch",
    ];
    for (const k of keys) if (m[k]) ts.current.set(k, now);
    if ((m.hands_detected ?? 0) > 0) ts.current.set("hands", now);
  }, [m]);
  const now = Date.now();
  return (key: string, current: boolean | number) => {
    const ok = typeof current === "boolean" ? current : current > 0;
    return ok || now - (ts.current.get(key) ?? 0) < DECAY_MS;
  };
}

// ── Confidence sparkline ──────────────────────────────────────────────────────

function SparkLine({ pts, isBuy, uid }: { pts: number[]; isBuy: boolean; uid: string }) {
  if (pts.length < 2) return null;
  const W = 116, H = 26;
  const sy  = (v: number) => H - v * H;
  const sx  = (i: number) => (i / (pts.length - 1)) * W;
  const d   = pts.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
  const col = isBuy ? "#22c55e" : "#3b82f6";
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
            <stop offset="0%" stopColor={col} stopOpacity="0.22" />
            <stop offset="100%" stopColor={col} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`${d} L${W},${H} L0,${H} Z`} fill={`url(#${gid})`} />
        <path d={d} fill="none" stroke={col} strokeWidth="1.5" strokeLinejoin="round" />
        <circle cx={sx(pts.length - 1)} cy={sy(last)} r="2.5" fill={col} />
      </svg>
      <span className="text-[11px] font-mono font-bold flex-shrink-0" style={{ color: trendCol }}>
        {trend}
      </span>
    </div>
  );
}

// ── Large animated probability ring ──────────────────────────────────────────

function BigRing({ prob, isBuy }: { prob: number; isBuy: boolean }) {
  const r  = 46;
  const c  = 2 * Math.PI * r;
  const pct = Math.round(prob * 100);
  return (
    <div className="flex flex-col items-center">
      <svg width="108" height="108" viewBox="0 0 108 108">
        <defs>
          <filter id="gfx">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        {/* Track */}
        <circle cx="54" cy="54" r={r} fill="none" stroke="#1a2332" strokeWidth="7" />
        {/* Arc */}
        <circle
          cx="54" cy="54" r={r} fill="none"
          stroke={isBuy ? "#22c55e" : "#3b82f6"}
          strokeWidth="7"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - prob)}
          strokeLinecap="round"
          transform="rotate(-90 54 54)"
          filter="url(#gfx)"
          style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(.4,0,.2,1), stroke .5s ease" }}
        />
        {/* Percentage */}
        <text x="54" y="50" textAnchor="middle" fill="white" fontSize="20" fontWeight="800" fontFamily="monospace">
          {pct}%
        </text>
        <text x="54" y="64" textAnchor="middle"
          fill={isBuy ? "#22c55e" : "#3b82f6"}
          fontSize="7.5" fontWeight="700" letterSpacing="0.12em">
          {isBuy ? "LIKELY BUY" : "BROWSING"}
        </text>
      </svg>
    </div>
  );
}

// ── Behavioral signal row (plain English) ─────────────────────────────────────

function BehaviorRow({ icon, label, on, highlight = false }: {
  icon: string; label: string; on: boolean; highlight?: boolean;
}) {
  return (
    <div className={`flex items-center gap-3 rounded-xl px-3 py-2.5 transition-all duration-500 ${
      on
        ? highlight
          ? "bg-green-950/70 border border-green-500/30"
          : "bg-white/5 border border-white/8"
        : "opacity-25 border border-transparent"
    }`}>
      <span className="text-xl leading-none w-7 text-center flex-shrink-0">{icon}</span>
      <span className={`text-sm font-medium leading-tight ${on ? "text-gray-100" : "text-gray-600"}`}>
        {label}
      </span>
      {on && (
        <div className={`ml-auto h-2 w-2 flex-shrink-0 rounded-full ${
          highlight ? "bg-green-400 animate-pulse" : "bg-gray-400"
        }`} />
      )}
    </div>
  );
}

// ── Dwell timer (counts up live) ──────────────────────────────────────────────

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
  const isBuy  = p.verdict === "LIKELY BUY";
  const smooth = useSmoothProb(p.prob);
  return (
    <div className={`flex items-center gap-3 rounded-xl border px-3 py-2.5 transition-all duration-500 ${
      isBuy ? "border-green-500/20 bg-green-950/15" : "border-white/5 bg-white/[0.015]"
    }`}>
      <div className="flex-shrink-0">
        <div className={`h-8 w-8 rounded-full flex items-center justify-center text-xs font-bold ${
          isBuy ? "bg-green-900/60 text-green-300" : "bg-gray-800 text-gray-400"
        }`}>
          #{p.id}
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <p className={`text-xs font-bold ${isBuy ? "text-green-400" : "text-blue-400"}`}>{p.verdict}</p>
        <div className="mt-1 h-1 w-full rounded-full bg-gray-800 overflow-hidden">
          <div
            className={`h-1 rounded-full ${isBuy ? "bg-green-500" : "bg-blue-500"}`}
            style={{ width: `${smooth * 100}%`, transition: "width 0.8s ease" }}
          />
        </div>
      </div>
      <span className={`flex-shrink-0 font-mono text-sm font-bold ${isBuy ? "text-green-300" : "text-blue-300"}`}>
        {Math.round(smooth * 100)}%
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
  expression?: FaceResult | null;
  nearbyProduct?: NearbyProduct | null;
}) {
  const isBuy  = p.verdict === "LIKELY BUY";
  const m      = p.metrics;
  const sig    = useDecay(m);
  const smooth = useSmoothProb(p.prob);
  const hist   = useProbHistory(smooth);

  // Flash on meaningful probability change
  const prevP  = useRef(p.prob);
  const [flash, setFlash] = useState(false);
  useEffect(() => {
    if (Math.abs(p.prob - prevP.current) > 0.05) {
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 600);
      prevP.current = p.prob;
      return () => clearTimeout(t);
    }
    prevP.current = p.prob;
  }, [p.prob]);

  // Plain-English behavioral signals (the only 5 a CEO needs to see)
  const lookingAt  = sig("looking_at_product", m?.looking_at_product ?? false)
                  && sig("face_detected",       m?.face_detected      ?? false);
  const atShelf    = sig("wrist_near_shelf", m?.wrist_near_shelf ?? false)
                  || sig("holding_product",  m?.holding_product  ?? false);
  const relaxedOk  = sig("relaxed_posture", m?.relaxed_posture ?? false)
                  && !sig("body_turned_away", m?.body_turned_away ?? false);
  const examining  = atShelf && lookingAt;
  const movingAway = sig("body_turned_away", m?.body_turned_away ?? false)
                  || sig("step_back",        m?.step_back        ?? false);
  const highIntent = smooth >= 0.725;

  const dwell = p.dwell_sec ?? 0;

  return (
    <div
      className={`overflow-hidden rounded-2xl border transition-all duration-500 ${
        isBuy
          ? "border-green-500/30 bg-gradient-to-b from-green-950/30 to-[#030712]"
          : "border-white/8 bg-gradient-to-b from-white/[0.03] to-[#030712]"
      } ${flash ? "ring-1 ring-white/20" : ""}`}
      style={isBuy ? { boxShadow: "0 0 40px rgba(34,197,94,0.10), 0 0 80px rgba(34,197,94,0.05)" } : {}}
    >
      {/* Top section: ID + ring + dwell */}
      <div className="px-4 pt-4 pb-3">
        <div className="flex items-start justify-between mb-3">
          {/* Customer label */}
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] font-mono uppercase text-gray-600">Customer</span>
              <span className="text-base font-black text-white">#{p.id}</span>
              {isPrimary && isBuy && (
                <span className="rounded-full bg-green-500 px-2 py-0.5 text-[8px] font-extrabold uppercase tracking-widest text-black">
                  TOP
                </span>
              )}
            </div>
            <DwellBadge sec={dwell} />
          </div>

          {/* Probability ring */}
          <BigRing prob={smooth} isBuy={isBuy} />
        </div>

        {/* Window progress bar */}
        <div className="flex items-center gap-1.5">
          <span className="text-[9px] text-gray-700 font-mono w-16">Analysis</span>
          <div className="flex flex-1 items-center gap-0.5">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className={`h-1.5 flex-1 rounded-full transition-all duration-700 ${
                i < p.windows
                  ? isBuy ? "bg-green-500" : "bg-blue-500"
                  : "bg-gray-800"
              }`} />
            ))}
          </div>
          <span className="text-[9px] font-mono text-gray-600 w-10 text-right">{p.windows}×5s</span>
        </div>

        {/* Confidence trend sparkline */}
        <div className="flex items-center justify-between rounded-xl bg-white/[0.025] border border-white/5 px-3 py-2 mt-2">
          <div className="flex flex-col gap-0.5">
            <span className="text-[9px] font-bold uppercase tracking-[0.16em] text-gray-600">Confidence Trend</span>
            <span className={`text-[12px] font-mono font-bold ${isBuy ? "text-green-400" : "text-blue-400"}`}>
              {Math.round(smooth * 100)}%
            </span>
          </div>
          <SparkLine pts={hist} isBuy={isBuy} uid={`${p.id}`} />
        </div>
      </div>

      {/* Live metric chips */}
      {m && (
        <div className="px-3 pb-3 grid grid-cols-3 gap-1.5">
          {[
            { label: "Lean", val: `${Math.abs(m.lean_angle).toFixed(1)}°` },
            {
              label: "Gaze",
              val: `${Math.abs(m.head_yaw_proxy).toFixed(1)}°${m.head_yaw_proxy >= 0 ? "R" : "L"}`,
            },
            { label: "Hands", val: `${m.hands_detected}` },
          ].map(({ label, val }) => (
            <div key={label} className="rounded-lg bg-white/[0.03] border border-white/5 px-2 py-1.5 text-center">
              <p className="text-[9px] text-gray-600 uppercase tracking-wide mb-0.5">{label}</p>
              <p className="text-[12px] font-mono font-bold text-gray-200">{val}</p>
            </div>
          ))}
        </div>
      )}

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

      {/* Behavioral signals */}
      <div className="px-3 pb-3 space-y-1.5">
        <p className="px-1 mb-2 text-[9px] font-bold uppercase tracking-[0.2em] text-gray-600">
          Behaviour Signals
        </p>

        <BehaviorRow
          icon="👁️"
          label={nearbyProduct && lookingAt ? `Looking at: ${nearbyProduct.label}` : "Looking at product"}
          on={lookingAt}
          highlight={lookingAt}
        />
        <BehaviorRow
          icon={nearbyProduct?.inHand ? "📦" : "🤲"}
          label={
            nearbyProduct?.inHand
              ? `Holding: ${nearbyProduct.label} (${Math.round(nearbyProduct.conf * 100)}%)`
              : "Hands near / on item"
          }
          on={atShelf}
          highlight={atShelf && !!nearbyProduct}
        />
        <BehaviorRow
          icon="✨"
          label="Relaxed & interested"
          on={relaxedOk}
        />
        <BehaviorRow
          icon="🔍"
          label={
            examining && nearbyProduct
              ? `Examining: ${nearbyProduct.label}`
              : "Actively examining"
          }
          on={examining}
          highlight={examining}
        />
        {movingAway ? (
          <BehaviorRow icon="🚶" label="Moving away / hesitating" on={true} />
        ) : (
          <BehaviorRow icon="📍" label="Staying in area" on={dwell > 15} />
        )}
      </div>

      {/* Purchase intent highlight */}
      <div className={`mx-3 mb-3 rounded-xl p-3 transition-all duration-700 ${
        highIntent
          ? "bg-green-950/60 border border-green-500/30"
          : "bg-white/3 border border-white/5"
      }`}>
        <div className="flex items-center gap-2.5">
          <span className="text-2xl leading-none">{highIntent ? "⚡" : examining ? "🛒" : "🔄"}</span>
          <div className="flex-1 min-w-0">
            <p className={`text-sm font-bold ${highIntent ? "text-green-300" : "text-gray-400"}`}>
              {highIntent
                ? "High Purchase Intent"
                : examining
                  ? "Considering Purchase"
                  : "Still Analysing…"}
            </p>
            <p className="text-[10px] text-gray-600 mt-0.5">
              {highIntent
                ? "AI confidence above buy threshold"
                : examining
                  ? "Engaged with product"
                  : "Observing behaviour patterns"}
            </p>
          </div>
          {highIntent && <div className="h-2.5 w-2.5 flex-shrink-0 rounded-full bg-green-400 animate-pulse" />}
        </div>
      </div>
    </div>
  );
}
