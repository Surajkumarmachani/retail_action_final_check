#!/usr/bin/env bash
# Deploy the MACHANI BAM backend to Google Cloud Run.
# Cloud Run lives in the SAME Google Cloud project as your Firebase project — this is the
# "deploy on Firebase" path for a container backend (Firebase Hosting/Functions can't run
# a persistent-WebSocket PyTorch service).
#
# Prereqs (one time):
#   1. Install gcloud CLI + `gcloud auth login`
#   2. It uses the Dockerfile in ./backend via Cloud Build (no local Docker needed)
#
# Usage:
#   cd backend
#   CORS_ORIGINS="https://your-app.vercel.app" bash deploy-cloudrun.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-retail-actions}"          # your Firebase/GCP project id
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-machani-backend}"
CORS_ORIGINS="${CORS_ORIGINS:-https://your-app.vercel.app}"

echo "Deploying '$SERVICE' to project '$PROJECT_ID' ($REGION)…"

# Enable the APIs Cloud Run needs (idempotent)
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project "$PROJECT_ID"

# Build from ./backend (uses the Dockerfile) and deploy.
# --no-cpu-throttling keeps the CPU allocated for the whole WebSocket connection.
# --timeout 3600 = max WS connection length (Cloud Run caps at 60 min; client auto-reconnects).
gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --no-cpu-throttling \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "FIREBASE_PROJECT_ID=${PROJECT_ID},CORS_ORIGINS=${CORS_ORIGINS},CV_WORKERS=2"

echo
echo "Done. Service URL is printed above."
echo "Set the frontend (Vercel) env var:"
echo "  NEXT_PUBLIC_WS_URL = wss://<that-url-without-https>/ws/predict"
