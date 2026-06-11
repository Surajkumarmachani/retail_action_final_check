"use client";

import { useEffect, useRef } from "react";
import { Prediction } from "@/lib/types";

interface TrailPt { x: number; y: number; t: number; }
interface Props { predictions: Prediction[]; frameWidth: number; frameHeight: number; }

export function PredictionOverlay({ predictions, frameWidth, frameHeight }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef    = useRef(0);
  const predsRef  = useRef<Prediction[]>([]);
  const trailsRef = useRef<Map<number, TrailPt[]>>(new Map());
  const phaseRef  = useRef(0);
  const sizeRef   = useRef({ w: frameWidth, h: frameHeight });

  useEffect(() => { sizeRef.current = { w: frameWidth, h: frameHeight }; }, [frameWidth, frameHeight]);

  // Update predictions ref + trails (mirror x so they match the CSS-mirrored video)
  useEffect(() => {
    predsRef.current = predictions;
    const now = Date.now();
    const trails = trailsRef.current;
    const W = sizeRef.current.w || 640;
    const activeIds = new Set(predictions.map(p => p.id));

    for (const p of predictions) {
      const [x1, , x2, y1y2] = p.box;
      const rawCy = (p.box[1] + p.box[3]) / 2;
      // Mirror centre x to match the CSS-mirrored video element
      const mx = W - (x1 + x2) / 2;
      const trail = trails.get(p.id) ?? [];
      trail.push({ x: mx, y: rawCy, t: now });
      trails.set(p.id, trail.filter(pt => now - pt.t < 2500).slice(-24));
    }
    for (const id of trails.keys()) if (!activeIds.has(id)) trails.delete(id);
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
      const trails = trailsRef.current;

      for (const p of preds) {
        const [_x1, _y1, _x2, _y2] = p.box;
        const isBuy = p.verdict === "LIKELY BUY";
        const col   = isBuy ? "#22c55e" : "#3b82f6";

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
