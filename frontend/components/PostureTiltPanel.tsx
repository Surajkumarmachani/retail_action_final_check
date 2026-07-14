"use client";

// Posture & Tilt Analysis — semicircle gauges (Torso Tilt / Spine / Shoulder alignment)
// + a gaze-tilt classifier bar. Driven by backend-computed posture angles (reliable,
// no client model needed). Mirrors the reference "POSTURE & TILT ANALYSIS" panel.

import { type PostureAngles } from "@/lib/types";

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }

function Gauge({ label, value, range }: { label: string; value: number | undefined; range: number }) {
  const has = typeof value === "number";
  const v = clamp(has ? value! : 0, -range, range);
  const W = 92, H = 54, cx = W / 2, cy = H - 6, r = 36;
  const pt = (deg: number): [number, number] => {
    const a = (deg * Math.PI) / 180;
    return [cx + r * Math.cos(a), cy - r * Math.sin(a)];
  };
  // colored track segments (green centre → yellow → red edges)
  const N = 30;
  const segs = [];
  for (let i = 0; i < N; i++) {
    const d0 = 180 - (i / N) * 180;
    const d1 = 180 - ((i + 1) / N) * 180;
    const mid = (d0 + d1) / 2;
    const dev = Math.abs(90 - mid) / 90;          // 0 centre → 1 edge
    const col = dev < 0.34 ? "#22c55e" : dev < 0.67 ? "#eab308" : "#ef4444";
    const [x0, y0] = pt(d0), [x1, y1] = pt(d1);
    segs.push(<line key={i} x1={x0} y1={y0} x2={x1} y2={y1} stroke={col} strokeWidth={4} strokeOpacity={0.55} strokeLinecap="round" />);
  }
  const angle = 180 - ((v + range) / (2 * range)) * 180;  // -range→180°, 0→90°, +range→0°
  const [nx, ny] = pt(angle);
  return (
    <div className="flex flex-col items-center">
      <svg width={W} height={H} className="overflow-visible">
        {segs}
        {has && <>
          <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="#fff" strokeWidth={2} strokeLinecap="round" />
          <circle cx={cx} cy={cy} r={3} fill="#fff" />
        </>}
      </svg>
      <span className={`text-[11px] font-mono font-bold leading-none ${has ? "text-white" : "text-gray-600"}`}>
        {has ? `${value! >= 0 ? "+" : ""}${value!.toFixed(1)}°` : "—"}
      </span>
      <span className="mt-0.5 text-[7px] font-mono uppercase tracking-[0.1em] text-gray-500 text-center leading-tight">{label}</span>
    </div>
  );
}

function GazeClassifier({ deg }: { deg: number | undefined }) {
  const has = typeof deg === "number";
  const mag = clamp(has ? Math.abs(deg!) : 0, 0, 30);
  const pct = (mag / 30) * 100;
  const cls = mag < 6 ? "STRAIGHT" : mag < 15 ? "TILTED" : "SIGNIFICANT TILT";
  const col = mag < 6 ? "text-green-400" : mag < 15 ? "text-amber-400" : "text-red-400";
  return (
    <div className="mt-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[8px] font-mono uppercase tracking-[0.14em] text-gray-500">Gaze Classifier</span>
        <span className={`text-[9px] font-mono font-bold ${col}`}>{cls}{has ? ` (${Math.abs(deg!).toFixed(0)}°)` : ""}</span>
      </div>
      <div className="relative h-2 rounded-full overflow-hidden"
        style={{ background: "linear-gradient(90deg,#22c55e 0%,#22c55e 20%,#eab308 50%,#ef4444 100%)" }}>
        {has && (
          <div className="absolute top-1/2 h-3 w-[3px] -translate-y-1/2 rounded-full bg-white shadow"
            style={{ left: `calc(${pct}% - 1.5px)` }} />
        )}
      </div>
      <div className="mt-0.5 flex justify-between text-[6.5px] font-mono uppercase tracking-wide text-gray-600">
        <span>Straight</span><span>Tilted</span><span>Significant</span>
      </div>
    </div>
  );
}

export function PostureTiltPanel({ posture }: { posture?: PostureAngles }) {
  const p = posture ?? {};
  return (
    <div className="px-3 pb-3 pt-1 border-t border-white/5">
      <p className="mb-2 pt-2 text-[7.5px] font-mono font-bold uppercase tracking-[0.22em] text-gray-700">
        Posture &amp; Tilt Analysis
      </p>
      <div className="grid grid-cols-3 gap-1">
        <Gauge label="Torso Tilt"  value={p.torso_tilt}     range={30} />
        <Gauge label="Spine Angle" value={p.spine_angle}    range={30} />
        <Gauge label="Shoulder"    value={p.shoulder_align} range={20} />
      </div>
      <GazeClassifier deg={p.gaze_angle} />
    </div>
  );
}
