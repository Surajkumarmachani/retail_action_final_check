"use client";

import { useEffect, useRef } from "react";
import { Prediction, ProductDetection } from "@/lib/types";

interface TrailPt { x: number; y: number; t: number; }
interface Props {
  predictions: Prediction[];
  products:    ProductDetection[];
  frameWidth:  number;
  frameHeight: number;
}

type Box = [number, number, number, number];
interface BoxState { from: Box; to: Box; startT: number; }

export function PredictionOverlay({ predictions, products, frameWidth, frameHeight }: Props) {
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const rafRef         = useRef(0);
  const predsRef       = useRef<Prediction[]>([]);
  const prodsRef       = useRef<ProductDetection[]>([]);
  const trailsRef      = useRef<Map<number, TrailPt[]>>(new Map());
  const boxStates      = useRef<Map<number, BoxState>>(new Map());
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

  // Update predictions ref + trails + box interpolation states
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

      const prev = boxes.get(p.id);
      if (prev) {
        // Snap "from" to wherever the lerp was, start a new lerp to new target
        const elapsed = Math.min(1, (now - prev.startT) / 120);
        const lerped: Box = prev.from.map((v, i) => v + (prev.to[i] - v) * elapsed) as Box;
        boxes.set(p.id, { from: lerped, to: p.box, startT: now });
      } else {
        boxes.set(p.id, { from: p.box, to: p.box, startT: now });
      }
    }
    for (const id of trails.keys()) if (!activeIds.has(id)) trails.delete(id);
    for (const id of boxes.keys())  if (!activeIds.has(id)) boxes.delete(id);
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
        if (prod.conf < 0.75) continue;   // skip low-confidence detections

        // Lerp product box for smooth movement (150ms window, keyed by label)
        const pstate = prodBoxStates.current.get(prod.label);
        const rawProd: Box = pstate
          ? pstate.from.map((v, i) => v + (pstate.to[i] - v) * Math.min(1, (now - pstate.startT) / 150)) as Box
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

        // Pulsing amber fill
        const pa = 0.06 + Math.sin(phase * 3.5) * 0.025;
        ctx.fillStyle = `rgba(245,158,11,${pa})`;
        ctx.fillRect(x1, y1, bw, bh);

        // Dashed amber border with glow
        ctx.save();
        ctx.shadowColor = "rgba(245,158,11,0.5)";
        ctx.shadowBlur  = 16;
        ctx.strokeStyle = "rgba(245,158,11,0.8)";
        ctx.lineWidth   = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x1, y1, bw, bh);
        ctx.setLineDash([]);
        ctx.restore();

        // Animated corner brackets
        const ca = 0.7 + Math.sin(phase * 2.2) * 0.3;
        ctx.save();
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth   = 2.5;
        ctx.lineCap     = "round";
        ctx.shadowColor = "#f59e0b";
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
        ctx.fillStyle   = "rgba(251,191,36,0.9)";
        ctx.shadowColor = "#f59e0b";
        ctx.shadowBlur  = 8;
        ctx.fill();
        ctx.restore();

        // Vertical scan line sweeping through box
        const scanY = y1 + ((phase * 40) % bh);
        ctx.save();
        ctx.globalAlpha = 0.22;
        const sg = ctx.createLinearGradient(x1, scanY - 8, x1, scanY + 8);
        sg.addColorStop(0,   "rgba(245,158,11,0)");
        sg.addColorStop(0.5, "rgba(245,158,11,1)");
        sg.addColorStop(1,   "rgba(245,158,11,0)");
        ctx.fillStyle = sg;
        ctx.fillRect(x1, scanY - 8, bw, 16);
        ctx.restore();

        // Label chip
        const label = `${prod.label}  ${Math.round(prod.conf * 100)}%`;
        ctx.font = "700 11px ui-monospace, monospace";
        const tw = ctx.measureText(label).width;
        const lx = x1;
        const ly = y1 > 30 ? y1 - 28 : y2 + 4;

        ctx.save();
        ctx.shadowColor = "rgba(245,158,11,0.6)";
        ctx.shadowBlur  = 8;
        roundRect(ctx, lx, ly, tw + 16, 22, 4);
        ctx.fillStyle   = "rgba(92,40,5,0.95)";
        ctx.fill();
        ctx.strokeStyle = "rgba(245,158,11,0.6)";
        ctx.lineWidth   = 1;
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = "#fcd34d";
        ctx.font      = "700 11px ui-monospace, monospace";
        ctx.fillText(label, lx + 8, ly + 15);
      }

      // ── Person boxes ────────────────────────────────────────────────────
      for (const p of preds) {
        const isBuy = p.verdict === "LIKELY BUY";
        const col   = isBuy ? "#22c55e" : "#3b82f6";

        // Lerp box coordinates for smooth motion (120ms interpolation window)
        const state = boxes.get(p.id);
        let rawBox = p.box;
        if (state) {
          const t = Math.min(1, (now - state.startT) / 120);
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
        ctx.fillStyle = isBuy ? "rgba(34,197,94,0.05)" : "rgba(59,130,246,0.03)";
        ctx.fillRect(x1, y1, bw, bh);

        // Trail
        const trail = trails.get(p.id) ?? [];
        if (trail.length > 1) {
          for (let i = 1; i < trail.length; i++) {
            const a = (i / trail.length) * 0.5;
            ctx.beginPath();
            ctx.strokeStyle = isBuy ? `rgba(34,197,94,${a})` : `rgba(59,130,246,${a})`;
            ctx.lineWidth = 2;
            ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
            ctx.lineTo(trail[i].x, trail[i].y);
            ctx.stroke();
          }
          ctx.beginPath();
          ctx.arc(cx, cy, 3, 0, Math.PI * 2);
          ctx.fillStyle = col;
          ctx.fill();
        }

        // Pulse rings (BUY only)
        if (isBuy) {
          for (let ring = 0; ring < 3; ring++) {
            const t = ((phase + ring * 2.1) % 6.28) / 6.28;
            const r = Math.min(bw, bh) * 0.38 * (0.4 + t);
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(34,197,94,${(1 - t) * 0.28})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }

        // Glowing bounding box
        ctx.save();
        ctx.shadowColor = col;
        ctx.shadowBlur  = isBuy ? 14 : 6;
        ctx.strokeStyle = col;
        ctx.lineWidth   = isBuy ? 2 : 1.5;
        ctx.strokeRect(x1, y1, bw, bh);
        ctx.restore();

        // Animated corner brackets
        const ca = 0.65 + Math.sin(phase * 2.5 + p.id) * 0.35;
        ctx.save();
        ctx.strokeStyle = col;
        ctx.lineWidth   = 3;
        ctx.lineCap     = "round";
        ctx.shadowColor = col;
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

        // Label (text is NOT mirrored — canvas itself is un-mirrored)
        const pct   = `${(p.prob * 100).toFixed(0)}%`;
        const label = `#${p.id}  ${pct}  ${p.verdict}`;
        ctx.font = "bold 11px ui-monospace, monospace";
        const tw = ctx.measureText(label).width;
        const lx = x1;
        const ly = y1 > 30 ? y1 - 28 : y2 + 4;

        ctx.save();
        ctx.shadowColor = col;
        ctx.shadowBlur  = 8;
        roundRect(ctx, lx, ly, tw + 14, 22, 4);
        ctx.fillStyle = isBuy ? "rgba(20,83,45,0.95)" : "rgba(30,58,138,0.95)";
        ctx.fill();
        ctx.restore();

        ctx.fillStyle = "#fff";
        ctx.font      = "bold 11px ui-monospace, monospace";
        ctx.fillText(label, lx + 7, ly + 14);

        // Probability bar at box bottom
        ctx.fillStyle = "rgba(0,0,0,0.5)";
        ctx.fillRect(x1, y2 - 3, bw, 3);
        ctx.save();
        ctx.shadowColor = col;
        ctx.shadowBlur  = 6;
        ctx.fillStyle   = col;
        ctx.fillRect(x1, y2 - 3, bw * p.prob, 3);
        ctx.restore();
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />;
}
