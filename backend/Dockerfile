# BAM backend — FastAPI + WebSocket + YOLOv8 (CPU). Deploy target: Render (Docker).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARTIFACTS_DIR=/app/artifacts \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# System libs required by OpenCV + Torch OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (avoids the ~2.5GB CUDA wheel), then the rest
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-deploy.txt

# App code (.dockerignore trims venv, unused big models, secrets)
COPY . .

# Fetch the two YOLO weights at build time (they are git-ignored, so not in the repo).
# ultralytics resolves the correct download URL; if COPY already provided them, it's a no-op.
RUN mkdir -p artifacts && cd artifacts && \
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')" && \
    ls -la yolov8n.pt yolov8n-pose.pt

EXPOSE 8000
# Render provides $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
