# Deploying MACHANI

**Frontend → Vercel.  Backend → Google Cloud Run** (same GCP project as your Firebase
project `retail-actions`).

> **Why not Firebase Hosting / Cloud Functions?** Neither can run this backend:
> Hosting is static-only; Cloud Functions has no persistent WebSockets and can't fit
> PyTorch/OpenCV (~2 GB) or use a GPU. **Cloud Run** is the container service inside the
> same Google Cloud project — it runs the Docker image, supports WebSockets, and is the
> correct "on Firebase/GCP" target. (Firebase Hosting can even `rewrite` to it later.)

---

## 1. Backend on Cloud Run (Firebase/GCP)  ← recommended

The Docker image is already built + tested. Deploy with the helper script:

```bash
cd backend
CORS_ORIGINS="https://your-app.vercel.app" bash deploy-cloudrun.sh
```

That runs `gcloud run deploy --source .` (Cloud Build builds the Dockerfile — no local
Docker needed), with the right flags for WebSockets:
`--memory 2Gi --cpu 2 --timeout 3600 --no-cpu-throttling --concurrency 4`
and env vars `FIREBASE_PROJECT_ID=retail-actions`, `CORS_ORIGINS`, `CV_WORKERS=2`.

Prereqs: `gcloud` CLI installed + `gcloud auth login`. First deploy enables the
run/cloudbuild/artifactregistry APIs automatically.

After it finishes it prints a URL like `https://machani-backend-xxxxx-uc.a.run.app`.

**Notes**
- **Memory/concurrency:** each WebSocket connection loads its own YOLO models, so RAM grows
  per concurrent viewer. `--concurrency 4` + `--memory 2Gi` suits a small demo; raise memory
  or lower concurrency for more simultaneous users, and Cloud Run scales out via instances.
- **WebSocket length:** Cloud Run caps a connection at 60 min (`--timeout 3600`); the frontend
  auto-reconnects, so this is invisible in use.
- **Cold start:** `--min-instances 0` sleeps when idle (cheap) with a few-seconds cold start.
  Set `--min-instances 1` for always-warm.
- **GPU (optional):** Cloud Run supports NVIDIA L4 in some regions — add `--gpu 1
  --gpu-type nvidia-l4` and switch the Dockerfile's torch line to a CUDA wheel.

Then jump to **section 2** (Vercel) using the printed Cloud Run URL.

---

## 1b. Alternative — Backend on Render

Files already in the repo: `backend/Dockerfile`, `backend/requirements-deploy.txt`,
`backend/.dockerignore`, `render.yaml`.

1. Push the repo to GitHub.
2. Render → **New → Blueprint** → select this repo. It reads `render.yaml` and creates
   the `machani-backend` web service (Docker, health check `/health`).
3. Set env vars (Render dashboard → the service → Environment):
   - `FIREBASE_PROJECT_ID` = `retail-actions`  (already in render.yaml)
   - `CORS_ORIGINS` = `https://<your-app>.vercel.app`  (your Vercel URL — set this)
   - `CV_WORKERS` = `2`
4. Deploy. First build takes a few minutes (installs torch, downloads YOLO weights).
5. Note the service URL, e.g. `https://machani-backend.onrender.com`.

**Plan / RAM:** torch + YOLO need ~1–1.5 GB RAM → use the **Standard** (2 GB) plan.
Free/Starter (512 MB) will likely OOM.

**GPU:** Render is CPU-only. That's fine now — pose runs ~10–15fps on CPU, and product
detection runs client-side (in the browser). For higher server fps, use a GPU host
(Fly.io GPU / RunPod) — the Dockerfile works there too (swap the CPU torch line for CUDA).

---

## 2. Frontend on Vercel

Set these Environment Variables in the Vercel project (Settings → Environment Variables):

- `NEXT_PUBLIC_WS_URL` = `wss://machani-backend.onrender.com/ws/predict`
  **Must be `wss://`** (secure) — Vercel is HTTPS, so plain `ws://` is blocked as mixed content.
- All the `NEXT_PUBLIC_FIREBASE_*` vars from your local `.env.local`.

Redeploy the frontend after setting them.

Then add the Vercel domain to **Firebase Console → Authentication → Settings → Authorized
domains**, or Google sign-in will fail on the deployed site.

The MediaPipe model files under `frontend/public/mediapipe/` are served by Vercel as static
assets (client-side product/mesh detection) — no backend involved.

---

## 3. Quick checks after deploy

- `https://machani-backend.onrender.com/health` → `{"status":"ok","engine":"BAM",...}`
- Open the Vercel app, sign in, allow camera → skeleton + product boxes should appear.
- If the WebSocket fails: check `CORS_ORIGINS` matches the exact Vercel origin and the URL is `wss://`.

## Notes
- No secret file is shipped — the backend verifies Firebase ID tokens using Google's public
  keys + `FIREBASE_PROJECT_ID` only.
- On the Free plan the service sleeps when idle; the first request after idle has a cold start
  (~30 s + model load). Standard/always-on avoids this.
