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

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from constants import GC_TIMEOUT_SECONDS
from firebase_auth import init_firebase, verify_id_token
from model.predictor import PurchasePredictor
from pipeline.cv_pipeline import PerConnectionTracker, decode_frame, process_frame
from pipeline.session_manager import SessionManager

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
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "transformer_behaviour_best.joblib")
SCALER_PATH   = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
YOLO_PATH     = os.path.join(ARTIFACTS_DIR, "yolov8s.pt")

predictor: PurchasePredictor | None = None
executor:  ThreadPoolExecutor | None = None


@app.on_event("startup")
async def startup() -> None:
    global predictor, executor
    init_firebase()
    predictor = PurchasePredictor(MODEL_PATH, SCALER_PATH)
    executor  = ThreadPoolExecutor(
        max_workers=int(os.environ.get("CV_WORKERS", "2")),
        thread_name_prefix="cv-worker",
    )
    logger.info("Server ready.  Listening for WebSocket connections.")


@app.on_event("shutdown")
async def shutdown() -> None:
    if executor:
        executor.shutdown(wait=False)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": predictor is not None}


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
    tracker     = PerConnectionTracker(YOLO_PATH)
    session_mgr = SessionManager()
    loop        = asyncio.get_event_loop()
    last_gc     = time.monotonic()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "frame":
                continue

            frame = decode_frame(msg.get("data", ""))
            if frame is None:
                continue

            # ── Offload blocking CV to thread pool ─────────────────────────
            observations = await loop.run_in_executor(
                executor, process_frame, frame, tracker
            )

            # ── Update sessions; run inference only on window flush ─────────
            response_items = []
            for obs in observations:
                tid  = obs["track_id"]
                sess = session_mgr.get_or_create(tid)
                sess.add_observation(obs)

                if sess.maybe_flush():
                    sess.last_prob    = predictor.predict(sess.windows)
                    sess.last_verdict = predictor.verdict(sess.last_prob)

                response_items.append({
                    "id":      tid,
                    "box":     obs["box"],
                    "prob":    round(sess.last_prob, 4),
                    "verdict": sess.last_verdict,
                    "windows": len(sess.windows),
                    "dwell_sec": round(time.monotonic() - sess.session_start, 1),
                    "metrics": {
                        "pose_detected":      float(obs.get("pose_detected", 0)) > 0.5,
                        "lean_angle":         round(float(obs.get("lean_angle", 0)), 1),
                        "arm_cross":          bool(obs.get("arm_cross", False)),
                        "wrist_raised":       bool(obs.get("wrist_raised", False)),
                        "wrist_near_shelf":   bool(obs.get("wrist_near_shelf", False)),
                        "holding_product":    bool(obs.get("holding_product", False)),
                        "relaxed_posture":    bool(obs.get("relaxed_posture", False)),
                        "body_turned_away":   bool(obs.get("body_turned_away", False)),
                        "face_detected":      float(obs.get("face_detected", 0)) > 0.5,
                        "looking_at_product": bool(obs.get("looking_at_product", False)),
                        "head_yaw_proxy":     round(float(obs.get("head_yaw_proxy", 0)), 1),
                        "hands_detected":     int(obs.get("hands_detected", 0)),
                        "hand_open":          bool(obs.get("hand_open", False)),
                        "hand_pinch":         bool(obs.get("hand_pinch", False)),
                        "near_others":        bool(obs.get("near_others", False)),
                        "group_size":         int(obs.get("group_size", 1)),
                        "step_back":          bool(obs.get("step_back", False)),
                        "head_pitch_proxy":   round(float(obs.get("head_pitch_proxy", 0)), 1),
                    },
                })

            # ── Periodic garbage collection of vanished tracks ──────────────
            now = time.monotonic()
            if now - last_gc > 1.0:
                removed  = session_mgr.garbage_collect()
                last_gc  = now
                if removed:
                    logger.debug("GC: removed %d stale track(s)", removed)

            # ── Stream predictions back to the browser ──────────────────────
            await ws.send_text(json.dumps({
                "type": "predictions",
                "data": response_items,
            }))

    except WebSocketDisconnect:
        logger.info("Client disconnected  uid=%s  active_tracks=%d",
                    uid, session_mgr.active_count)
    finally:
        tracker.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
