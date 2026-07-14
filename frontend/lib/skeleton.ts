// lib/skeleton.ts
// Pure canvas drawing for MediaPipe Holistic landmarks — body posture skeleton,
// finger-level hand skeletons, eye/iris tracking, gaze vector, and floating angle
// annotations. Also computes posture metrics (angles, lean, gaze focus) in one pass.
//
// Coordinates: landmarks are normalised [0,1] of the RAW (un-mirrored) camera frame.
// The video is CSS-mirrored (scaleX(-1)); this overlay canvas is NOT CSS-mirrored, so
// x is flipped in code with (1 - x) * w to register with the mirrored video — exactly
// the convention PredictionOverlay uses.

export interface LM { x: number; y: number; z: number; visibility?: number }

/** A client-side detected object. box is [x1,y1,x2,y2] normalised (0-1) in RAW frame coords. */
export interface ClientObject {
  label: string;
  score: number;
  box:   [number, number, number, number];
}

/** A MediaPipe connection: either {start,end} (Tasks API) or [a,b]. */
export type Conn = { start: number; end: number } | [number, number];

export interface HolisticParts {
  pose?:     LM[];        // 33 BlazePose
  hands?:    LM[][];      // 0-2 hands × 21 landmarks (from dedicated HandLandmarker)
  face?:     LM[];        // ~478 face mesh incl. iris
  objects?:  ClientObject[]; // real-time detected products
  faceTess?: Conn[];      // FaceLandmarker.FACE_LANDMARKS_TESSELATION (~2600 edges) — the mesh grid
}

function connAB(c: Conn): [number, number] {
  return Array.isArray(c) ? [c[0], c[1]] : [c.start, c.end];
}

export interface PostureMetrics {
  headTiltDeg:      number;        // head/eye-line tilt from horizontal
  leanDeg:          number;        // torso lean from vertical
  leftElbowDeg:     number | null;
  rightElbowDeg:    number | null;
  leftKneeDeg:      number | null;
  rightKneeDeg:     number | null;
  interactionAngleDeg: number | null; // most-bent arm (reaching / holding)
  gazeFocus:        number;        // 0-1 (iris centred + facing camera)
  handsVisible:     number;        // 0 / 1 / 2
  bodyVisible:      boolean;
  faceVisible:      boolean;
}

// ── BlazePose 33-point connections ──────────────────────────────────────────────
const POSE_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
  [11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  [11,23],[12,24],[23,24],
  [23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32],
];
// Major joints to emphasise with larger dots
const POSE_MAJOR = [11,12,13,14,15,16,23,24,25,26,27,28,0];

// ── Hand 21-point connections ───────────────────────────────────────────────────
const HAND_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[0,17],[17,18],[18,19],[19,20],
];

// ── Face connection sets (478-pt refined mesh) ──────────────────────────────────
const FACE_OVAL: [number, number][] = [
  [10,338],[338,297],[297,332],[332,284],[284,251],[251,389],[389,356],[356,454],
  [454,323],[323,361],[361,288],[288,397],[397,365],[365,379],[379,378],[378,400],
  [400,377],[377,152],[152,148],[148,176],[176,149],[149,150],[150,136],[136,172],
  [172,58],[58,132],[132,93],[93,234],[234,127],[127,162],[162,21],[21,54],
  [54,103],[103,67],[67,109],[109,10],
];
const LEFT_EYE: [number, number][] = [
  [263,249],[249,390],[390,373],[373,374],[374,380],[380,381],[381,382],[382,362],
  [263,466],[466,388],[388,387],[387,386],[386,385],[385,384],[384,398],[398,362],
];
const RIGHT_EYE: [number, number][] = [
  [33,7],[7,163],[163,144],[144,145],[145,153],[153,154],[154,155],[155,133],
  [33,246],[246,161],[161,160],[160,159],[159,158],[158,157],[157,173],[173,133],
];
const LEFT_IRIS: [number, number][]  = [[474,475],[475,476],[476,477],[477,474]];
const RIGHT_IRIS: [number, number][] = [[469,470],[470,471],[471,472],[472,469]];
const LEFT_BROW: [number, number][]  = [[276,283],[283,282],[282,295],[295,285],[300,293],[293,334],[334,296],[296,336]];
const RIGHT_BROW: [number, number][] = [[46,53],[53,52],[52,65],[65,55],[70,63],[63,105],[105,66],[66,107]];
const LIPS: [number, number][] = [
  [61,146],[146,91],[91,181],[181,84],[84,17],[17,314],[314,405],[405,321],[321,375],[375,291],
  [61,185],[185,40],[40,39],[39,37],[37,0],[0,267],[267,269],[269,270],[270,409],[409,291],
];
const L_IRIS_CENTER = 473;
const R_IRIS_CENTER = 468;

// ── Palette ──────────────────────────────────────────────────────────────────────
const C_BONE   = "#22d3ee";   // cyan
const C_JOINT  = "#67e8f9";
const C_FACE   = "rgba(34,211,238,0.55)";
const C_HAND   = "#5eead4";   // teal
const C_GAZE   = "#f59e0b";   // amber gaze vector
const VIS_MIN  = 0.4;

// ── Geometry helpers ──────────────────────────────────────────────────────────────
function angleAt(a: LM, b: LM, c: LM, W: number, H: number): number {
  const bax = (a.x - b.x) * W, bay = (a.y - b.y) * H;
  const bcx = (c.x - b.x) * W, bcy = (c.y - b.y) * H;
  const dot = bax * bcx + bay * bcy;
  const mag = Math.hypot(bax, bay) * Math.hypot(bcx, bcy) + 1e-6;
  return Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180 / Math.PI;
}
function tiltFromVertical(top: LM, bottom: LM, W: number, H: number): number {
  const dx = (bottom.x - top.x) * W, dy = (bottom.y - top.y) * H;
  return Math.abs(Math.atan2(dx, dy) * 180 / Math.PI); // 0 = perfectly vertical
}
function vis(p: LM | undefined): boolean {
  return !!p && (p.visibility === undefined || p.visibility >= VIS_MIN) && !(p.x === 0 && p.y === 0);
}

// ── Main draw ──────────────────────────────────────────────────────────────────────
export interface DrawOpts { drawPose?: boolean; drawObjects?: boolean }

export function drawSkeleton(
  canvas: HTMLCanvasElement,
  video:  HTMLVideoElement,
  parts:  HolisticParts,
  opts:   DrawOpts = {},
): PostureMetrics {
  const { drawPose = true, drawObjects = true } = opts;
  const w = video.videoWidth, h = video.videoHeight;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, w, h);

  const X = (p: LM) => (1 - p.x) * w;   // mirror flip
  const Y = (p: LM) => p.y * h;

  const { pose, hands, face, objects, faceTess } = parts;

  const metrics: PostureMetrics = {
    headTiltDeg: 0, leanDeg: 0,
    leftElbowDeg: null, rightElbowDeg: null, leftKneeDeg: null, rightKneeDeg: null,
    interactionAngleDeg: null, gazeFocus: 0,
    handsVisible: hands?.length ?? 0,
    bodyVisible: !!pose, faceVisible: !!face,
  };

  // ── Real-time object boxes (drawn first, behind the skeleton) ────────────────
  if (drawObjects && objects && objects.length) {
    for (const o of objects) {
      const [ox1, oy1, ox2, oy2] = o.box;
      const x1 = (1 - ox2) * w, x2 = (1 - ox1) * w;   // mirror flip
      const y1 = oy1 * h, y2 = oy2 * h;
      const bw = x2 - x1, bh = y2 - y1;
      const cs = Math.min(bw, bh, 20);

      ctx.save();
      ctx.fillStyle = "rgba(245,158,11,0.06)";
      ctx.fillRect(x1, y1, bw, bh);
      ctx.shadowColor = "rgba(245,158,11,0.5)"; ctx.shadowBlur = 12;
      ctx.strokeStyle = "rgba(245,158,11,0.85)"; ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]); ctx.strokeRect(x1, y1, bw, bh); ctx.setLineDash([]);
      // corner brackets
      ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2.5; ctx.lineCap = "round";
      for (const [bx, by, dx, dy] of [
        [x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1],
      ] as [number, number, number, number][]) {
        ctx.beginPath();
        ctx.moveTo(bx, by + dy * cs); ctx.lineTo(bx, by); ctx.lineTo(bx + dx * cs, by);
        ctx.stroke();
      }
      ctx.restore();

      // label chip
      const label = `${o.label}  ${Math.round(o.score * 100)}%`;
      ctx.font = "700 11px ui-monospace, monospace";
      const tw = ctx.measureText(label).width;
      const ly = y1 > 24 ? y1 - 20 : y2 + 2;
      ctx.save();
      ctx.fillStyle = "rgba(92,40,5,0.92)"; ctx.strokeStyle = "rgba(245,158,11,0.6)"; ctx.lineWidth = 1;
      ctx.fillRect(x1, ly, tw + 14, 18); ctx.strokeRect(x1, ly, tw + 14, 18);
      ctx.fillStyle = "#fcd34d"; ctx.fillText(label, x1 + 7, ly + 13);
      ctx.restore();
    }
  }

  // ── Face mesh: oval, brows, lips, eyes, iris ──────────────────────────────────
  if (face && face.length >= 468) {
    const strokeSet = (set: [number, number][], color: string, lw: number) => {
      ctx.strokeStyle = color; ctx.lineWidth = lw;
      ctx.beginPath();
      for (const [a, b] of set) {
        const pa = face[a], pb = face[b];
        if (!pa || !pb) continue;
        ctx.moveTo(X(pa), Y(pa)); ctx.lineTo(X(pb), Y(pb));
      }
      ctx.stroke();
    };
    // Head tilt from the eye line (true head tilt — computed in raw coords so the
    // mirror flip doesn't invert it). Outer eye corners: 33 (subject R), 263 (subject L).
    const eOutR = face[33], eOutL = face[263];
    if (eOutR && eOutL)
      metrics.headTiltDeg = Math.abs(
        Math.atan2((eOutL.y - eOutR.y) * h, (eOutL.x - eOutR.x) * w) * 180 / Math.PI,
      );

    // Dense face-mesh tessellation grid (the "grid" over the face). Drawn faint + first
    // so the highlighted eyes/iris/oval/lips render on top.
    if (faceTess && faceTess.length) {
      ctx.save();
      ctx.strokeStyle = "rgba(103,232,249,0.16)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      for (const c of faceTess) {
        const [a, b] = connAB(c);
        const pa = face[a], pb = face[b];
        if (!pa || !pb) continue;
        ctx.moveTo(X(pa), Y(pa)); ctx.lineTo(X(pb), Y(pb));
      }
      ctx.stroke();
      ctx.restore();
    }

    ctx.save();
    ctx.shadowColor = C_BONE; ctx.shadowBlur = 3;
    strokeSet(FACE_OVAL, C_FACE, 1);
    strokeSet(LEFT_BROW, C_FACE, 1);
    strokeSet(RIGHT_BROW, C_FACE, 1);
    strokeSet(LIPS, C_FACE, 1);
    // eyes brighter (eye tracking emphasis)
    strokeSet(LEFT_EYE,  C_BONE, 1.6);
    strokeSet(RIGHT_EYE, C_BONE, 1.6);
    strokeSet(LEFT_IRIS, "#a5f3fc", 1.6);
    strokeSet(RIGHT_IRIS, "#a5f3fc", 1.6);
    ctx.restore();

    // Iris centre dots + gaze vector
    const li = face[L_IRIS_CENTER], ri = face[R_IRIS_CENTER];
    if (li && ri) {
      ctx.save();
      ctx.fillStyle = "#e0fbff"; ctx.shadowColor = C_BONE; ctx.shadowBlur = 6;
      for (const c of [li, ri]) { ctx.beginPath(); ctx.arc(X(c), Y(c), 3, 0, Math.PI * 2); ctx.fill(); }
      ctx.restore();

      // Per-eye iris offset within the eye socket → gaze DIRECTION + focus.
      // (cornerIn, cornerOut) are the inner/outer corners of that eye.
      const eyeGaze = (irisC: LM, cornerIn: LM, cornerOut: LM) => {
        if (!cornerIn || !cornerOut) return { ox: 0, oy: 0, focus: 0.5 };
        const cx = (cornerIn.x + cornerOut.x) / 2, cy = (cornerIn.y + cornerOut.y) / 2;
        const eyeW = Math.abs(cornerOut.x - cornerIn.x) + 1e-4;
        const ox = (irisC.x - cx) / eyeW;              // signed, ~ −0.5..0.5
        const oy = (irisC.y - cy) / (eyeW * 0.6);
        const focus = Math.max(0, 1 - (Math.abs(ox) + Math.abs(oy)));  // 1 = looking straight
        return { ox, oy, focus };
      };
      const gL = eyeGaze(li, face[362], face[263]);   // subject's left eye
      const gR = eyeGaze(ri, face[133], face[33]);    // subject's right eye
      metrics.gazeFocus = Math.max(0, Math.min(1, (gL.focus + gR.focus) / 2));

      // Gaze vector points where the eyes actually look (from iris offset). Centred iris
      // (looking at camera) → tiny arrow; looking aside → longer arrow in that direction.
      const eyeMidX = (X(li) + X(ri)) / 2, eyeMidY = (Y(li) + Y(ri)) / 2;
      const gxN = (gL.ox + gR.ox) / 2, gyN = (gL.oy + gR.oy) / 2;
      const S = 320;                                   // amplify small offsets to a visible arrow
      const gx = eyeMidX - gxN * S;                    // −x: raw → mirrored display
      const gy = eyeMidY + gyN * S;
      ctx.save();
      ctx.strokeStyle = C_GAZE; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.shadowColor = C_GAZE; ctx.shadowBlur = 6;
      ctx.beginPath(); ctx.moveTo(eyeMidX, eyeMidY); ctx.lineTo(gx, gy); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(gx, gy, 3, 0, Math.PI * 2); ctx.fillStyle = C_GAZE; ctx.fill();
      ctx.restore();
    }
  }

  // ── Pose skeleton ──────────────────────────────────────────────────────────────
  if (drawPose && pose && pose.length >= 33) {
    ctx.save();
    ctx.strokeStyle = C_BONE; ctx.lineWidth = 3; ctx.lineCap = "round";
    ctx.shadowColor = C_BONE; ctx.shadowBlur = 6;
    ctx.beginPath();
    for (const [a, b] of POSE_CONNECTIONS) {
      const pa = pose[a], pb = pose[b];
      if (!vis(pa) || !vis(pb)) continue;
      ctx.moveTo(X(pa), Y(pa)); ctx.lineTo(X(pb), Y(pb));
    }
    ctx.stroke();
    ctx.restore();

    // Joint dots
    ctx.save();
    ctx.shadowColor = C_BONE; ctx.shadowBlur = 5;
    for (let i = 0; i < pose.length; i++) {
      const p = pose[i];
      if (!vis(p)) continue;
      const major = POSE_MAJOR.includes(i);
      ctx.beginPath(); ctx.arc(X(p), Y(p), major ? 4.5 : 2.5, 0, Math.PI * 2);
      ctx.fillStyle = major ? "#ffffff" : C_JOINT; ctx.fill();
      if (major) { ctx.beginPath(); ctx.arc(X(p), Y(p), 4.5, 0, Math.PI * 2); ctx.strokeStyle = C_BONE; ctx.lineWidth = 1.5; ctx.stroke(); }
    }
    ctx.restore();

    // ── Metrics from pose ──────────────────────────────────────────────────────
    const [LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA] =
      [11,12,13,14,15,16,23,24,25,26,27,28].map(i => pose[i]);
    const shoulderMid: LM = { x: (LS.x+RS.x)/2, y: (LS.y+RS.y)/2, z: 0 };
    const hipMid: LM      = { x: (LH.x+RH.x)/2, y: (LH.y+RH.y)/2, z: 0 };

    if (vis(LS) && vis(RS) && vis(LH) && vis(RH))
      metrics.leanDeg = tiltFromVertical(shoulderMid, hipMid, w, h);
    // headTiltDeg is computed from the eye line in the face block (true head tilt).

    if (vis(LS)&&vis(LE)&&vis(LW)) metrics.leftElbowDeg  = angleAt(LS, LE, LW, w, h);
    if (vis(RS)&&vis(RE)&&vis(RW)) metrics.rightElbowDeg = angleAt(RS, RE, RW, w, h);
    if (vis(LH)&&vis(LK)&&vis(LA)) metrics.leftKneeDeg   = angleAt(LH, LK, LA, w, h);
    if (vis(RH)&&vis(RK)&&vis(RA)) metrics.rightKneeDeg  = angleAt(RH, RK, RA, w, h);

    // Interaction angle = the most bent (reaching) elbow
    const elbows = [metrics.leftElbowDeg, metrics.rightElbowDeg].filter((v): v is number => v !== null);
    if (elbows.length) metrics.interactionAngleDeg = Math.min(...elbows);

    // ── Floating angle annotation chips ──────────────────────────────────────────
    if (metrics.leanDeg)
      chip(ctx, X(shoulderMid) + 12, Y(shoulderMid), `LEAN ${metrics.leanDeg.toFixed(1)}°`, w);
    if (metrics.faceVisible && vis(pose[0]))
      chip(ctx, X(pose[0]) + 14, Y(pose[0]) - 10, `HEAD ${metrics.headTiltDeg.toFixed(1)}°`, w);
    if (metrics.leftElbowDeg && vis(LE))
      chip(ctx, X(LE) + 10, Y(LE), `${metrics.leftElbowDeg.toFixed(0)}°`, w);
    if (metrics.rightElbowDeg && vis(RE))
      chip(ctx, X(RE) + 10, Y(RE), `${metrics.rightElbowDeg.toFixed(0)}°`, w);
    if (metrics.leftKneeDeg && vis(LK))
      chip(ctx, X(LK) + 10, Y(LK), `${metrics.leftKneeDeg.toFixed(0)}°`, w);
    if (metrics.rightKneeDeg && vis(RK))
      chip(ctx, X(RK) + 10, Y(RK), `${metrics.rightKneeDeg.toFixed(0)}°`, w);
  }

  // ── Hands (from dedicated HandLandmarker) ────────────────────────────────────────
  for (const hand of hands ?? []) {
    if (!hand || hand.length < 21) continue;
    ctx.save();
    ctx.strokeStyle = C_HAND; ctx.lineWidth = 2; ctx.lineCap = "round";
    ctx.shadowColor = C_HAND; ctx.shadowBlur = 4;
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = hand[a], pb = hand[b];
      if (!pa || !pb) continue;
      ctx.moveTo(X(pa), Y(pa)); ctx.lineTo(X(pb), Y(pb));
    }
    ctx.stroke();
    ctx.fillStyle = "#ccfbf1";
    for (const p of hand) { ctx.beginPath(); ctx.arc(X(p), Y(p), 2.4, 0, Math.PI * 2); ctx.fill(); }
    ctx.restore();
  }

  return metrics;
}

// ── Annotation chip ────────────────────────────────────────────────────────────────
function chip(ctx: CanvasRenderingContext2D, x: number, y: number, text: string, w: number) {
  ctx.save();
  ctx.font = "600 11px ui-monospace, monospace";
  const tw = ctx.measureText(text).width;
  let bx = x, by = y - 9;
  if (bx + tw + 12 > w) bx = x - tw - 24;   // keep in-frame
  if (by < 2) by = 2;
  ctx.fillStyle = "rgba(3,10,18,0.82)";
  ctx.strokeStyle = "rgba(34,211,238,0.5)";
  ctx.lineWidth = 1;
  const rx = bx, ry = by, rw = tw + 12, rh = 18, r = 4;
  ctx.beginPath();
  ctx.moveTo(rx + r, ry);
  ctx.arcTo(rx + rw, ry, rx + rw, ry + rh, r);
  ctx.arcTo(rx + rw, ry + rh, rx, ry + rh, r);
  ctx.arcTo(rx, ry + rh, rx, ry, r);
  ctx.arcTo(rx, ry, rx + rw, ry, r);
  ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#67e8f9";
  ctx.fillText(text, rx + 6, ry + 13);
  ctx.restore();
}
