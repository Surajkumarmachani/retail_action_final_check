# MACHANI – Retail Actions

A live webcam app that watches shoppers in a store and estimates how likely each person is to
buy something. Each person is detected, tracked and drawn with a skeleton. The app picks up
actions like reaching for a product, picking it up, or lingering at a shelf, and turns them
into a purchase-intent score that updates in real time.

```
Browser (Next.js)                                   Backend (FastAPI, Python 3.11)
──────────────────                                  ──────────────────────────────
Firebase sign-in  ──── ID token (first WS msg) ───►  verify token (Google public keys)
Webcam frames     ──── JPEG over WebSocket ───────►  YOLOv8-pose + ByteTrack (people, IDs)
                                                    YOLOv8 (products)
                                                    BAM event detector + scoring engine
Overlay + cards   ◄─── JSON per frame ─────────────  {person no., box, skeleton, score, events}

Runs in the browser too (no server round trip):
  MediaPipe holistic landmarks (face/hands) + EfficientDet product boxes
```

**BAM** (Behaviour Action Model) is the rule/score engine in `backend/bam/`. It replaced the
earlier Transformer model. The original notebook and the old model files are kept in `docs/`
for reference only, and the running app doesn't use them.

---

## Project layout

```
backend/
  main.py                 FastAPI app: /health and the /ws/predict WebSocket
  firebase_auth.py        Verifies Firebase ID tokens (PyJWT + Google JWKs, no Admin SDK)
  bam/                    Event detector, scoring engine, data models
  pipeline/bam_pipeline.py  Per-frame YOLO pose + detection + tracking → BAM
  Dockerfile, requirements-deploy.txt, deploy-cloudrun.sh   Deployment
frontend/
  app/page.tsx            Sign-in gate → live predictor
  components/             WebcamPredictor, PredictionOverlay, PersonCard, PostureTiltPanel, AuthGate
  hooks/                  useWebcam, useWebSocket, useHolisticLandmarks, useObjectDetector
  lib/                    firebase.ts, skeleton.ts, types.ts
  public/mediapipe/       Self-hosted MediaPipe models + WASM (served as static files)
docs/                     Research notebook, BAM prototype scripts, screenshots
render.yaml, DEPLOY.md    Hosting notes (Cloud Run or Render for backend, Vercel for frontend)
```

---

## Running it locally

**Requirements:** Python 3.11, Node 18+, and access to the Firebase project.

### Backend

```bash
cd backend
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # defaults work for local dev
uvicorn main:app --reload --port 8000
```

On first start, `yolov8n.pt` and `yolov8n-pose.pt` download automatically into
`backend/artifacts/` (they're git-ignored because of their size). Check it's up at
http://localhost:8000/health.

### Frontend

```bash
cd frontend
cp .env.local.example .env.local   # fill in the Firebase web config (see below)
npm install                        # also copies MediaPipe WASM into public/
npm run dev                        # http://localhost:3000
```

The Firebase values come from **Firebase Console → Project settings → General → Your apps →
Web app → SDK setup and configuration**.

---

## Handover checklist (one-time setup)

Things that live outside the code and have to be done in the consoles:

1. **Firebase project access.** The app uses the Firebase project `retail-actions`. Ask for
   Owner access to be transferred or added in Firebase Console → Project settings → Users and
   permissions. The same project is used on Google Cloud (Cloud Run).
2. **Enable sign-in methods.** Firebase Console → Authentication → Sign-in method → enable
   **Email/Password** and **Google**. Google also needs a *project support email*.
3. **Authorized domains.** Authentication → Settings → Authorized domains must list every
   domain the frontend runs on: `localhost` plus the deployed one (e.g. `your-app.vercel.app`).
   Google sign-in fails on any domain that isn't listed.
4. **Frontend env vars.** Put the six `NEXT_PUBLIC_FIREBASE_*` values and `NEXT_PUBLIC_WS_URL`
   in `.env.local` for local runs, and in Vercel → Settings → Environment Variables for
   production. In production the WS URL must start with `wss://`.
5. **Backend env vars.** `FIREBASE_PROJECT_ID`, `CORS_ORIGINS` (exact frontend URL),
   `CV_WORKERS`. No secret files are needed in production.
6. **(Recommended) Restrict the Firebase web API key.** Google Cloud Console → APIs &
   Services → Credentials → the "Browser key" → *Application restrictions: Websites*. Add your
   frontend domains **and** `retail-actions.firebaseapp.com/*`, or Google sign-in will break.

> **About the API key in the sign-in URL.** When you click "Continue with Google", the popup
> URL includes `apiKey=AIza…`. That is normal. A Firebase *web* API key is a public identifier
> that ships inside every Firebase web app, and it is not a password. Access is protected by
> Firebase Auth, the authorized-domains list and (optionally) the key restriction in step 6.
> The secret is `backend/serviceAccountKey.json`, which is git-ignored and never committed.

Full deployment steps are in [DEPLOY.md](DEPLOY.md).

---

## Configuration reference

| Where    | Variable                          | Purpose |
|----------|-----------------------------------|---------|
| frontend | `NEXT_PUBLIC_FIREBASE_*` (6 vars) | Firebase web app config |
| frontend | `NEXT_PUBLIC_WS_URL`              | Backend WebSocket, e.g. `ws://localhost:8000/ws/predict` |
| backend  | `FIREBASE_PROJECT_ID`             | Project whose ID tokens are accepted |
| backend  | `CORS_ORIGINS`                    | Comma-separated allowed frontend origins |
| backend  | `CV_WORKERS`                      | CV thread pool size (default 2) |
| backend  | `ARTIFACTS_DIR`                   | YOLO weights folder (default `./artifacts`) |
| backend  | `BAM_STORE_FORMAT`                | Store layout for scoring (default `lifestyle`) |

---

## Known limitations

- Each WebSocket connection loads its own YOLO models, so memory grows with the number of
  people watching at once. Plan for about 2 GB RAM per backend instance.
- The backend runs on CPU (about 10–15 fps for pose). A GPU host would be faster, and the
  Dockerfile works there if you swap the torch line for a CUDA wheel.
- `docs/` contains large training files that are git-ignored (`window_features.*`, `*.pt`).
  You only need them to retrain, not to run the app.

---

## Work log

Roughly how the time went, worked out from commit dates and file history. These are
honest estimates, not timesheet-exact numbers.

| When | What I worked on | Approx. hours |
|------|------------------|---------------|
| Early June 2026 | Research and model work: feature engineering from shopper video, training and evaluating the purchase-prediction models in the notebook | ~24 h |
| 9 – 12 June | First working version: FastAPI WebSocket server, YOLO + MediaPipe pipeline, Firebase login, Next.js webcam UI | ~20 h |
| 15 – 16 June | Cleanup and refactor, plus fixing rough edges found in the first round of testing | ~8 h |
| 25 June | Overlay and person-card UI improvements | ~6 h |
| 14 July | Replaced the Transformer with the BAM engine: ported the event detector and scoring engine, added the skeleton overlay, posture panel and self-hosted MediaPipe models | ~14 h |
| 15 July | Moved product detection into the browser, stabilised person numbering and tracking, tuned the WebSocket frame loop | ~8 h |
| 15 July – 6 Aug | Deployment setup (Docker, Cloud Run and Render configs), lighter token verification without the Admin SDK | ~8 h |
| 7 Oct | Handover: fixed Google sign-in popup, clearer login errors, documentation | ~3 h |
| **Total** | | **~91 h** |

---

## Contact

Built by Vinod Maurya (vinod.m@machanigroup.com). Reach out if anything in the setup is unclear.
