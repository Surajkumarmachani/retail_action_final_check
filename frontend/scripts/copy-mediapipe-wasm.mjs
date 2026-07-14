// Copies the @mediapipe/tasks-vision WASM runtime into /public/mediapipe/wasm so it is
// served same-origin (required because next.config.js sets COEP: require-corp), and downloads
// the Holistic model .task into /public/mediapipe. Runs on postinstall so it survives clean
// installs / CI. The wasm files are NOT importable (not in the package "exports" map) — they
// must be physically copied.

import { existsSync, mkdirSync, copyFileSync, readdirSync, statSync, createWriteStream, unlinkSync, renameSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import https from "node:https";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root      = join(__dirname, "..");
const srcWasm   = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const dstWasm   = join(root, "public", "mediapipe", "wasm");
const dstModels = join(root, "public", "mediapipe");

const MODELS = [
  {
    file: "holistic_landmarker.task",
    url:  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
  },
  {
    // Blendshape fallback used by useHolisticLandmarks when Holistic yields no blendshapes.
    file: "face_landmarker.task",
    url:  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
  },
  {
    // Dedicated hand landmarker — crisper finger tracking than Holistic's hand sub-model.
    file: "hand_landmarker.task",
    url:  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
  },
  {
    // Client-side real-time object detector (EfficientDet-Lite2, COCO-80).
    file: "efficientdet_lite2.tflite",
    url:  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/latest/efficientdet_lite2.tflite",
  },
];

function ensureDir(d) { if (!existsSync(d)) mkdirSync(d, { recursive: true }); }

function copyWasm() {
  if (!existsSync(srcWasm)) {
    console.warn("[mediapipe] wasm source not found (is @mediapipe/tasks-vision installed?) — skipping copy");
    return;
  }
  ensureDir(dstWasm);
  let n = 0;
  for (const f of readdirSync(srcWasm)) {
    const s = join(srcWasm, f);
    if (statSync(s).isFile()) { copyFileSync(s, join(dstWasm, f)); n++; }
  }
  console.log(`[mediapipe] copied ${n} wasm file(s) → public/mediapipe/wasm`);
}

function download(url, dest) {
  // Download to a temp file and rename on success so an interrupted transfer never
  // leaves a truncated .task on disk (which would later be served as if valid).
  const tmp = dest + ".part";
  return new Promise((resolve, reject) => {
    const cleanup = () => { try { if (existsSync(tmp)) unlinkSync(tmp); } catch { /* ignore */ } };
    const fail = (err) => { cleanup(); reject(err); };
    const go = (u, redirects = 0) => {
      const req = https.get(u, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          if (redirects > 5) return fail(new Error("too many redirects"));
          return go(res.headers.location, redirects + 1);
        }
        if (res.statusCode !== 200) { res.resume(); return fail(new Error(`HTTP ${res.statusCode} for ${u}`)); }
        const out = createWriteStream(tmp);
        res.on("error", fail);
        out.on("error", fail);
        out.on("finish", () => out.close(() => {
          try { renameSync(tmp, dest); resolve(); }
          catch (e) { fail(e); }
        }));
        res.pipe(out);
      });
      req.on("error", fail);
    };
    go(url);
  });
}

async function downloadModels() {
  ensureDir(dstModels);
  for (const m of MODELS) {
    const dest = join(dstModels, m.file);
    if (existsSync(dest) && statSync(dest).size > 100_000) {
      console.log(`[mediapipe] ${m.file} already present — skipping`);
      continue;
    }
    try {
      console.log(`[mediapipe] downloading ${m.file} …`);
      await download(m.url, dest);
      console.log(`[mediapipe] saved ${m.file} (${(statSync(dest).size / 1e6).toFixed(1)} MB)`);
    } catch (e) {
      console.warn(`[mediapipe] failed to download ${m.file}: ${e.message}`);
    }
  }
}

copyWasm();
await downloadModels();
