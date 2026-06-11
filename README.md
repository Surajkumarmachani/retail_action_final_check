# Retail Purchase Predictor – Live Webcam App

Real-time purchase-intent prediction from a webcam feed.

```
Browser (Next.js + Vercel)
    │  Firebase Auth ID token (WS handshake)
    │  Base64 JPEG frames  →
    │                       ← JSON predictions {id, box, prob, verdict}
    │
Backend (FastAPI + PyTorch, Render/Railway/Fly/GPU VM)
    ├── YOLOv8 + ByteTrack  (person detection + tracking)
    ├── MediaPipe            (pose, head, hands every 3rd frame)
    ├── LiveCustomerSession  (two-level window buffer per track)
    └── BehaviourTransformer (41-d × T-window sequence → purchase prob)
```

---

## Project layout

```
├── backend/
│   ├── main.py                    FastAPI WebSocket server
│   ├── constants.py               Hyperparameters + 41-feature column order
│   ├── firebase_auth.py           ID-token verification
│   ├── model/
│   │   ├── behaviour_transformer.py   PyTorch model class (must match training)
│   │   └── predictor.py               Load weights + scaler, run inference
│   ├── pipeline/
│   │   ├── cv_pipeline.py             YOLO+MediaPipe per-frame extraction
│   │   ├── session.py                 LiveCustomerSession (two-level buffer)
│   │   └── session_manager.py         Active-session dict + GC
│   ├── export_scaler.py           One-time scaler export from training data
│   ├── artifacts/                 ← PUT YOUR .joblib + .pt FILES HERE
│   │   ├── transformer_behaviour_best.joblib
│   │   ├── scaler.joblib
│   │   └── yolov8s.pt
│   ├── requirements.txt
│   └── Dockerfile
└── frontend/
    ├── app/page.tsx               Auth gate → WebcamPredictor
    ├── components/
    │   ├── WebcamPredictor.tsx    Video + overlay + stats
    │   ├── PredictionOverlay.tsx  Canvas drawing (no server-side rendering)
    │   └── AuthGate.tsx           Email/Google sign-in
    ├── hooks/
    │   ├── useWebcam.ts           getUserMedia + frame capture
    │   └── useWebSocket.ts        Authenticated WS + streaming loop
    └── lib/
        ├── firebase.ts            Firebase client SDK singleton
        └── types.ts               Shared TypeScript types
```

---

## Setup

### 1. Export the StandardScaler from your notebook

The model was trained on standardised features.  Export the scaler once:

```python
# In your training notebook (after fitting the scaler):
import joblib
joblib.dump(scaler, "scaler.joblib")
```

Or use the helper script (needs access to your training CSV):

```bash
cd backend
python export_scaler.py --csv /path/to/training_features.csv --out ./artifacts/scaler.joblib
```

### 2. Place model artifacts

```
backend/artifacts/
    transformer_behaviour_best.joblib   ← your trained model state_dict
    scaler.joblib                       ← exported above
    yolov8s.pt                          ← download: yolo export or ultralytics auto-download
```

### 3. Backend

```bash
cd backend
cp .env.example .env          # fill in FIREBASE_CREDENTIALS_JSON
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Deploy** (Render/Railway/Fly.io):
```bash
docker build -t retail-predictor .
# Mount /app/artifacts and set FIREBASE_CREDENTIALS_JSON env var
```

### 4. Frontend

```bash
cd frontend
cp .env.local.example .env.local    # fill in Firebase keys + WS_URL
npm install
npm run dev                          # http://localhost:3000
```

**Deploy to Vercel**: connect the `frontend/` directory, set all `NEXT_PUBLIC_*` env vars.

---

## How the temporal windowing works

### Notebook (frame-count based)
```
30 FPS × 5 s = 150 frames → sample every 3 → 50 MediaPipe calls per window
```
This breaks at any frame rate other than 30 FPS.

### Live server (wall-clock based)
```
LiveCustomerSession.maybe_flush() checks time.monotonic()
→ flush when elapsed ≥ WINDOW_SECONDS (5.0)
→ correct at 10 FPS, 15 FPS, 30 FPS, or variable FPS
```

### Two-level buffer

```
Per-frame observations ──[every frame]──► level-1 frame_obs[]
                                                │
                                         [every 5 seconds]
                                                ▼
                            level-2 windows[]  (one 41-d dict per flush)
                                                │
                                         [on each flush]
                                                ▼
                            BehaviourTransformer(windows[-MAX_SEQ:])
                                                │
                                                ▼
                                      purchase probability
```

Predictions **sharpen** the longer a person is observed because each new window
adds evidence the Transformer can attend over.  Early in a session (1–2 windows)
the probability is low-confidence; after 5+ windows it stabilises.

---

## Inference threshold

`PURCHASE_THRESHOLD = 0.725`  (Youden's J from training ROC curve)

Change it in `backend/constants.py`.  Higher → fewer false positives.
Lower → catch more buyers but more false alarms.

---

## Scaling

| Connections | Recommendation |
|-------------|---------------|
| 1–4         | `CV_WORKERS=2` ThreadPoolExecutor (default) |
| 5–20        | `uvicorn --workers 4` (multiple processes, each with own ByteTrack) |
| 20+         | Redis queue: frames → worker pool → predictions |

For GPU inference, set `device="cuda"` in `PurchasePredictor.__init__`.
The Transformer is tiny (128-dim, 4 layers) so CPU is usually adequate.
