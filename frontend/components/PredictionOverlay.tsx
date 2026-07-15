"use client";

import { useEffect, useRef } from "react";
import { Prediction, ProductDetection, Keypoint } from "@/lib/types";

interface TrailPt { x: number; y: number; t: number; }
interface Props {
  predictions: Prediction[];
  products:    ProductDetection[];
  frameWidth:  number;
  frameHeight: number;
}

type Box = [number, number, number, number];
interface BoxState { from: Box; to: Box; startT: number; }
interface KpState  { from: Keypoint[]; to: Keypoint[]; startT: number; }

// BAM band → colour
const BAND_COL: Record<string, string> = {
  HOT:  "#ef4444",   // red
  WARM: "#f59e0b",   // amber
  COLD: "#3b82f6",   // blue
};

// This overlay draws everything from the backend payload (like yolo_demo.py):
// the green COCO-17 pose skeleton (from server keypoints), BAM person boxes/labels/BAS bars,
// trails, and cyan product boxes with labels.

// COCO-17 skeleton edges (matches the backend YOLOv8-pose keypoint order)
const COCO_EDGES: [number, number][] = [
  [0,1],[0,2],[1,3],[2,4],          // head: nose-eyes-ears
  [0,5],[0,6],                      // nose → shoulders
  [5,7],[7,9],[6,8],[8,10],         // arms
  [5,6],[5,11],[6,12],[11,12],      // torso
  [11,13],[13,15],[12,14],[14,16],  // legs
];

const LERP_MS = 120;

export function PredictionOverlay({ predictions, products, frameWidth, frameHeight }: Props) {
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const rafRef         = useRef(0);
  const predsRef       = useRef<Prediction[]>([]);
  const prodsRef       = useRef<ProductDetection[]>([]);
  const trailsRef      = useRef<Map<number, TrailPt[]>>(new Map());
  const boxStates      = useRef<Map<number, BoxState>>(new Map());
  const kpStates       = useRef<Map<number, KpState>>(new Map());    // keypoint interpolation
  const prodBoxStates  = useRef<Map<string, BoxState>>(new Map());   // keyed by label
  const phaseRef       = useRef(0);
  const sizeRef        = useRef({ w: frameWidth, h: frameHeight });

  useEffect(() => { sizeRef.current = { w: frameWidth, h: frameHeight }; }, [frameWidth, frameHeight]);

  // Update product ref + interpolation states (keyed by label)
  useEffect(() => {
    prodsRef.current = products;
    const now = Date.now();
    const pbs = prodBoxStates.current;
    const activeLabels = new Set(products.map(p => p.label));

    for (const p of products) {
      const prev = pbs.get(p.label);
      if (prev) {
        const elapsed = Math.min(1, (now - prev.startT) / 150);
        const lerped: Box = prev.from.map((v, i) => v + (prev.to[i] - v) * elapsed) as Box;
        pbs.set(p.label, { from: lerped, to: p.box, startT: now });
      } else {
        pbs.set(p.label, { from: p.box, to: p.box, startT: now });
      }
    }
    for (const label of pbs.keys()) if (!activeLabels.has(label)) pbs.delete(label);
  }, [products]);

  // Update predictions ref + trails + box/keypoint interpolation states
  useEffect(() => {
    predsRef.current = predictions;
    const now = Date.now();
    const trails = trailsRef.current;
    const boxes  = boxStates.current;
    const W = sizeRef.current.w || 640;
    const activeIds = new Set(predictions.map(p => p.id));

    for (const p of predictions) {
      const [x1, , x2] = p.box;
      const rawCy = (p.box[1] + p.box[3]) / 2;
      const mx = W - (x1 + x2) / 2;
      const trail = trails.get(p.id) ?? [];
      trail.push({ x: mx, y: rawCy, t: now });
      trails.set(p.id, trail.filter(pt => now - pt.t < 2500).slice(-24));

      // Box interpolation
      const prev = boxes.get(p.id);
      if (prev) {
        const elapsed = Math.min(1, (now - prev.startT) / LERP_MS);
        const lerped: Box = prev.from.map((v, i) => v + (prev.to[i] - v) * elapsed) as Box;
        boxes.set(p.id, { from: lerped, to: p.box, startT: now });
      } else {
        boxes.set(p.id, { from: p.box, to: p.box, startT: now });
      }

      // Keypoint interpolation — glides the skeleton between backend frames
      if (p.keypoints && p.keypoints.length) {
        const prevKp = kpStates.current.get(p.id);
        if (prevKp && prevKp.to.length === p.keypoints.length) {
          const el = Math.min(1, (now - prevKp.startT) / LERP_MS);
          const lerpedKp: Keypoint[] = p.keypoints.map((kp, i) => {
            const f = prevKp.from[i] ?? kp;
            const t = prevKp.to[i]   ?? kp;
            return [f[0] + (t[0] - f[0]) * el, f[1] + (t[1] - f[1]) * el, kp[2]] as Keypoint;
          });
          kpStates.current.set(p.id, { from: lerpedKp, to: p.keypoints, startT: now });
        } else {
          kpStates.current.set(p.id, { from: p.keypoints, to: p.keypoints, startT: now });
        }
      }
    }
    for (const id of trails.keys())          if (!activeIds.has(id)) trails.delete(id);
    for (const id of boxes.keys())           if (!activeIds.has(id)) boxes.delete(id);
    for (const id of kpStates.current.keys()) if (!activeIds.has(id)) kpStates.current.delete(id);
  }, [predictions]);

  useEffect(() => {
    function roundRect(
      ctx: CanvasRenderingContext2D,
      x: number, y: number, w: number, h: number, r: number,
    ) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.arcTo(x + w, y, x + w, y + r, r);
      ctx.lineTo(x + w, y + h - r);
      ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
      ctx.lineTo(x + r, y + h);
      ctx.arcTo(x, y + h, x, y + h - r, r);
      ctx.lineTo(x, y + r);
      ctx.arcTo(x, y, x + r, y, r);
      ctx.closePath();
    }

    function draw() {
      const canvas = canvasRef.current;
      const { w, h } = sizeRef.current;
      if (!canvas || !w || !h) { rafRef.current = requestAnimationFrame(draw); return; }
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, w, h);
      phaseRef.current += 0.025;
      const phase  = phaseRef.current;
      const preds  = predsRef.current;
      const prods  = prodsRef.current;
      const trails = trailsRef.current;
      const boxes  = boxStates.current;
      const now    = Date.now();

      // ── Product boxes (drawn first so persons render on top) ────────────
      for (const prod of prods) {
        if (prod.conf < 0.35) continue;   // match backend detection threshold

        // Lerp product box for smooth movement (150ms window, keyed by label)
        const pstate = prodBoxStates.current.get(prod.label);
        const rawProd: Box = pstate
          ? pstate.from.map((v, i) => v + (pstate.to[i] - v) * Math.min(1, (now - pstate.startT) / 70)) as Box
          : prod.box;
        const [_x1, _y1, _x2, _y2] = rawProd;
        const x1 = w - _x2;
        const x2 = w - _x1;
        const y1 = _y1;
        const y2 = _y2;
        const bw = x2 - x1;
        const bh = y2 - y1;
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        const cs = Math.min(bw, bh, 22);

        // Pulsing green fill
        const pa = 0.07 + Math.sin(phase * 3.5) * 0.03;
        ctx.fillStyle = `rgba(22,163,74,${pa})`;
        ctx.fillRect(x1, y1, bw, bh);

        // Solid dark-green border with glow
        ctx.save();
        ctx.shadowColor = "rgba(22,163,74,0.55)";
        ctx.shadowBlur  = 14;
        ctx.strokeStyle = "#16a34a";
        ctx.lineWidth   = 2;
        ctx.strokeRect(x1, y1, bw, bh);
        ctx.restore();

        // Animated corner brackets (green)
        const ca = 0.7 + Math.sin(phase * 2.2) * 0.3;
        ctx.save();
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth   = 3;
        ctx.lineCap     = "round";
        ctx.shadowColor = "#16a34a";
        ctx.shadowBlur  = 10;
        ctx.globalAlpha = ca;
        for (const [bx, by, dx, dy] of [
          [x1, y1, 1, 1], [x2, y1, -1, 1],
          [x1, y2, 1, -1], [x2, y2, -1, -1],
        ] as [number, number, number, number][]) {
          ctx.beginPath();
          ctx.moveTo(bx, by + dy * cs);
          ctx.lineTo(bx, by);
          ctx.lineTo(bx + dx * cs, by);
          ctx.stroke();
        }
        ctx.restore();

        // Centre crosshair dot
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
        ctx.fillStyle   = "rgba(134,239,172,0.95)";
        ctx.shadowColor = "#16a34a";
        ctx.shadowBlur  = 8;
        ctx.fill();
        ctx.restore();

        // Bold product name chip — e.g. "CUP 0.90"
        const label = `${prod.label.toUpperCase()}  ${prod.conf.toFixed(2)}`;
        ctx.font = "800 14px ui-monospace, monospace";
        const tw = ctx.measureText(label).width;
        const lx = x1;
        const ly = y1 > 32 ? y1 - 30 : y2 + 4;

        ctx.save();
        ctx.shadowColor = "rgba(22,163,74,0.6)";
        ctx.shadowBlur  = 8;
        roundRect(ctx, lx, ly, tw + 18, 24, 5);
        ctx.fillStyle   = "rgba(6,30,15,0.95)";
        ctx.fill();
        ctx.strokeStyle = "rgba(22,163,74,0.75)";
        ctx.lineWidth   = 1;
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = "#dcfce7";
        ctx.font      = "800 14px ui-monospace, monospace";
        ctx.fillText(label, lx + 9, ly + 16);
      }

      // ── Person boxes + skeleton ─────────────────────────────────────────
      for (const p of preds) {
        const band = p.band ?? "COLD";
        const col  = BAND_COL[band] ?? BAND_COL.COLD;   // band tint (BAS bar / label chip)
        const boxCol = "#16a34a";                        // dark-green person boundary
        const isHot = band === "HOT";

        // Lerp box coordinates for smooth motion
        const state = boxes.get(p.id);
        let rawBox = p.box;
        if (state) {
          const t = Math.min(1, (now - state.startT) / LERP_MS);
          rawBox = state.from.map((v, i) => v + (state.to[i] - v) * t) as typeof p.box;
        }
        const [_x1, _y1, _x2, _y2] = rawBox;

        // Mirror x so boxes sit over the CSS-mirrored video
        const x1 = w - _x2;
        const x2 = w - _x1;
        const y1 = _y1;
        const y2 = _y2;
        const bw = x2 - x1;
        const bh = y2 - y1;
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        const cs = Math.min(bw, bh, 24);

        // Subtle fill
        ctx.fillStyle = isHot ? "rgba(239,68,68,0.05)" : band === "WARM" ? "rgba(245,158,11,0.04)" : "rgba(59,130,246,0.03)";
        ctx.fillRect(x1, y1, bw, bh);

        // ── Green COCO-17 skeleton (interpolated server keypoints, normalised → mirrored px) ──
        const kpSt = kpStates.current.get(p.id);
        const kps: Keypoint[] | undefined = kpSt
          ? kpSt.from.map((f, i) => {
              const t = kpSt.to[i] ?? f;
              const el = Math.min(1, (now - kpSt.startT) / LERP_MS);
              return [f[0] + (t[0] - f[0]) * el, f[1] + (t[1] - f[1]) * el, t[2]] as Keypoint;
            })
          : p.keypoints;
        if (kps && kps.length >= 17) {
          const KX = (kp: Keypoint) => w - kp[0] * w;   // mirror to match CSS scaleX(-1)
          const KY = (kp: Keypoint) => kp[1] * h;

          // bones
          ctx.save();
          ctx.strokeStyle = "#22c55e";
          ctx.lineWidth   = 2.5;
          ctx.lineCap     = "round";
          ctx.shadowColor = "#22c55e";
          ctx.shadowBlur  = 4;
          for (const [a, b] of COCO_EDGES) {
            const ka = kps[a], kb = kps[b];
            if (!ka || !kb || ka[2] < 0.3 || kb[2] < 0.3) continue;
            ctx.beginPath();
            ctx.moveTo(KX(ka), KY(ka));
            ctx.lineTo(KX(kb), KY(kb));
            ctx.stroke();
          }
          // joints
          ctx.fillStyle = "#86efac";
          for (const kp of kps) {
            if (kp[2] < 0.3) continue;
            ctx.beginPath();
            ctx.arc(KX(kp), KY(kp), 3, 0, Math.PI * 2);
            ctx.fill();
          }
          ctx.restore();
        }

        // Trail
        const trail = trails.get(p.id) ?? [];
        if (trail.length > 1) {
          for (let i = 1; i < trail.length; i++) {
            const a = (i / trail.length) * 0.5;
            ctx.beginPath();
            ctx.strokeStyle = `${col}${Math.round(a * 255).toString(16).padStart(2, "0")}`;
            ctx.lineWidth = 2;
            ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
            ctx.lineTo(trail[i].x, trail[i].y);
            ctx.stroke();
          }
        }

        // Pulse rings (HOT only)
        if (isHot) {
          for (let ring = 0; ring < 3; ring++) {
            const t = ((phase + ring * 2.1) % 6.28) / 6.28;
            const r = Math.min(bw, bh) * 0.38 * (0.4 + t);
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(239,68,68,${(1 - t) * 0.28})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }

        // Glowing bounding box (dark-green boundary)
        ctx.save();
        ctx.shadowColor = boxCol;
        ctx.shadowBlur  = isHot ? 12 : 6;
        ctx.strokeStyle = boxCol;
        ctx.lineWidth   = isHot ? 2.5 : 2;
        ctx.strokeRect(x1, y1, bw, bh);
        ctx.restore();

        // Animated corner brackets (green)
        const ca = 0.65 + Math.sin(phase * 2.5 + p.id) * 0.35;
        ctx.save();
        ctx.strokeStyle = boxCol;
        ctx.lineWidth   = 3.5;
        ctx.lineCap     = "round";
        ctx.shadowColor = boxCol;
        ctx.shadowBlur  = 10;
        ctx.globalAlpha = ca;
        for (const [bx, by, dx, dy] of [
          [x1, y1, 1, 1], [x2, y1, -1, 1],
          [x1, y2, 1, -1], [x2, y2, -1, -1],
        ] as [number, number, number, number][]) {
          ctx.beginPath();
          ctx.moveTo(bx, by + dy * cs);
          ctx.lineTo(bx, by);
          ctx.lineTo(bx + dx * cs, by);
          ctx.stroke();
        }
        ctx.restore();

        // Label — "Person N  BAS  BAND" (bold)
        const bas   = Math.round(p.bas ?? (p.prob * 100));
        const label = `Person ${p.personNo ?? p.id}  ·  ${bas}  ${band}`;
        ctx.font = "800 13px ui-monospace, monospace";
        const tw = ctx.measureText(label).width;
        const lx = x1;
        const ly = y1 > 32 ? y1 - 30 : y2 + 4;

        ctx.save();
        ctx.shadowColor = boxCol;
        ctx.shadowBlur  = 8;
        roundRect(ctx, lx, ly, tw + 18, 24, 5);
        ctx.fillStyle = "rgba(6,30,15,0.95)";
        ctx.fill();
        ctx.strokeStyle = "rgba(22,163,74,0.7)";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = "#dcfce7";
        ctx.font      = "800 13px ui-monospace, monospace";
        ctx.fillText(label, lx + 9, ly + 16);

        // BAS bar at box bottom
        ctx.fillStyle = "rgba(0,0,0,0.5)";
        ctx.fillRect(x1, y2 - 3, bw, 3);
        ctx.save();
        ctx.shadowColor = col;
        ctx.shadowBlur  = 6;
        ctx.fillStyle   = col;
        ctx.fillRect(x1, y2 - 3, bw * Math.min(1, (p.bas ?? p.prob * 100) / 100), 3);
        ctx.restore();
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />;
}
