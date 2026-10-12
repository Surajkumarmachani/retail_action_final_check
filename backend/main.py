# main.py – FastAPI WebSocket server
#
# Architecture overview
# ─────────────────────
# 1. Browser sends Firebase ID token as first WS message → backend verifies.
# 2. Browser streams webcam frames (Base64 JPEG) at ~15 FPS via WebSocket.
# 3. Blocking CV work (YOLO+ByteTrack+MediaPipe) is offloaded to a
#    ThreadPoolExecutor so the asyncio event loop stays responsive.
# 4. Per-connection SessionManager maintains one LiveCustomerSession per
#    track_id.  Every 5 real-seconds a session flushes one 41-d window dict.
# 5. When a flush occurs, PurchasePredictor.predict() runs the Transformer
#    over the full accumulated window sequence and returns a probability.
# 6. After every frame, the server streams a JSON array of bounding-box +
#    probability objects back to the browser.
#
# Scaling past the GIL
# ─────────────────────
# ThreadPoolExecutor works well for 1–4 concurrent streams because OpenCV,
# NumPy, and PyTorch release the GIL during their heavy operations.  For
# higher concurrency, spawn multiple uvicorn workers:
#   uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
# Each worker process has its own ByteTrack state (correct – no sharing needed).
# If you need to share a single GPU across workers, a Redis message-queue
# (frames in → predictions out) is the standard pattern.

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Load backend/.env before anything reads os.environ (pipeline/bam_pipeline reads
# BAM_* at import time). No-op in Docker/Cloud Run, where .env isn't shipped.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from firebase_auth import init_firebase, verify_id_token
from pipeline.bam_pipeline import BAMTracker, decode_frame, process_frame_bam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Retail Purchase Predictor", version="1.0.0")

ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,https://your-app.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (initialised at startup) ─────────────────────────────────────────

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./artifacts")
POSE_PATH   = os.path.join(ARTIFACTS_DIR, "yolov8n-pose.pt")   # persons + skeleton + IDs
DETECT_PATH = os.path.join(ARTIFACTS_DIR, "yolov8n.pt")        # product detection
STORE_FORMAT = os.environ.get("BAM_STORE_FORMAT", "lifestyle")

executor: ThreadPoolExecutor | None = None


@app.on_event("startup")
async def startup() -> None:
    global executor
    init_firebase()
    executor = ThreadPoolExecutor(
        max_workers=int(os.environ.get("CV_WORKERS", "2")),
        thread_name_prefix="cv-worker",
    )
    logger.info("BAM server ready.  Listening for WebSocket connections.")


@app.on_event("shutdown")
async def shutdown() -> None:
    if executor:
        executor.shutdown(wait=False)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "engine": "BAM", "store_format": STORE_FORMAT}


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/predict")
async def ws_predict(ws: WebSocket) -> None:
    await ws.accept()
    uid = None

    # ── Step 1: auth handshake ─────────────────────────────────────────────
    try:
        raw   = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        msg   = json.loads(raw)
        token = msg.get("token", "")
        if not token:
            raise ValueError("Missing token in auth message")

        decoded = await verify_id_token(token)
        uid     = decoded["uid"]
        logger.info("Connection accepted  uid=%s", uid)
        await ws.send_text(json.dumps({"type": "auth_ok", "uid": uid}))

    except Exception as exc:
        logger.warning("Auth rejected: %s", exc)
        await ws.send_text(json.dumps({"type": "auth_error", "detail": str(exc)}))
        await ws.close()
        return

    # ── Step 2: per-connection state ───────────────────────────────────────
    tracker = BAMTracker(POSE_PATH, DETECT_PATH, store_format=STORE_FORMAT)
    loop    = asyncio.get_event_loop()
    last_gc = time.monotonic()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "frame":
                continue

            frame = decode_frame(msg.get("data", ""))
            if frame is None:
                continue

            # ── Offload blocking CV + BAM scoring to thread pool ────────────
            persons, products, events = await loop.run_in_executor(
                executor, process_frame_bam, frame, tracker
            )

            # ── Periodic eviction of idle tracks (privacy §1.1) ─────────────
            now = time.monotonic()
            if now - last_gc > 5.0:
                tracker.engine.evict_expired_sessions(session_timeout_s=30.0)
                last_gc = now

            # ── Stream BAM predictions back to the browser ──────────────────
            await ws.send_text(json.dumps({
                "type":     "predictions",
                "data":     persons,
                "products": products,
                "events":   events,
            }))

    except WebSocketDisconnect:
        logger.info("Client disconnected  uid=%s  active_tracks=%d",
                    uid, len(tracker.states))
    finally:
        tracker.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
