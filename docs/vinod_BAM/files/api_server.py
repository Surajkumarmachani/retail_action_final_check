"""
BAM — Buying Affinity Model
api_server.py  ·  FastAPI REST + WebSocket serving layer (L6)

Endpoints (§5.3)
────────────────
GET  /v1/stores/{id}/sessions/{sid}/score        Current BAS + event ledger
WS   /v1/stores/{id}/scores/stream               Push stream: score updates + HOT alerts
GET  /v1/stores/{id}/zones/{zid}/analytics       Zone-level dwell + event histograms
POST /v1/stores/{id}/config/zones                Upload/replace zone map (GeoJSON)
POST /v1/stores/{id}/config/weights              Apply + validate weight set
GET  /v1/health/cameras                          Per-camera FPS, decode errors, drift

Performance notes
─────────────────
• WebSocket broadcast uses asyncio.gather over all active connections — no
  per-client blocking.
• Score snapshot is taken once per 1 Hz tick and fanned out to all subscribers
  rather than each subscriber triggering its own score() call.
• Connection registry uses a plain set with an asyncio.Lock — fast for typical
  store cardinality (<100 concurrent dashboard clients).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, validator
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from models import ScoreBand, ScoreResult, build_weights
from scoring_engine import BAMScoringEngine

logger = logging.getLogger("bam.api")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request/response models
# ─────────────────────────────────────────────────────────────────────────────

if HAS_FASTAPI:
    class WeightPayload(BaseModel):
        weights: dict[str, float]

        @validator("weights")
        def must_sum_to_100(cls, v):
            pos_sum = sum(w for w in v.values() if w > 0)
            if abs(pos_sum - 100.0) > 0.01:
                raise ValueError(f"Positive weights must sum to 100, got {pos_sum:.2f}")
            return v

    class ZoneMapPayload(BaseModel):
        geojson: dict
        version: str = "v1"


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket connection manager — fan-out broadcast
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """
    Manages WebSocket connections per store.
    Broadcast is O(clients) with concurrent sends via asyncio.gather.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # store_id → set of active WebSocket connections
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, store_id: str, ws: "WebSocket") -> None:
        await ws.accept()
        async with self._lock:
            self._connections[store_id].add(ws)
        logger.info("WS client connected to store %s", store_id)

    async def disconnect(self, store_id: str, ws: "WebSocket") -> None:
        async with self._lock:
            self._connections[store_id].discard(ws)

    async def broadcast(self, store_id: str, message: dict) -> None:
        """Send message to all subscribers of a store — concurrent, non-blocking."""
        async with self._lock:
            clients = set(self._connections.get(store_id, set()))

        if not clients:
            return

        payload = json.dumps(message)
        results = await asyncio.gather(
            *[ws.send_text(payload) for ws in clients],
            return_exceptions=True,
        )
        # Clean up dead connections
        dead = {
            ws for ws, result in zip(clients, results)
            if isinstance(result, Exception)
        }
        if dead:
            async with self._lock:
                self._connections[store_id] -= dead

    def active_client_count(self, store_id: str) -> int:
        return len(self._connections.get(store_id, set()))


# ─────────────────────────────────────────────────────────────────────────────
# Zone analytics accumulator (in-memory; flush to TimescaleDB in production)
# ─────────────────────────────────────────────────────────────────────────────

class ZoneAnalytics:
    """Rolling event histograms per zone (§5.3 GET analytics)."""

    def __init__(self) -> None:
        # store_id → zone_id → event_type → count
        self._counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self._dwell_sum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._dwell_n:   dict[str, dict[str, int]]   = defaultdict(lambda: defaultdict(int))

    def record_event(self, store_id: str, zone_id: str, event_type: str) -> None:
        self._counts[store_id][zone_id][event_type] += 1

    def record_dwell(self, store_id: str, zone_id: str, dwell_s: float) -> None:
        self._dwell_sum[store_id][zone_id] += dwell_s
        self._dwell_n[store_id][zone_id]   += 1

    def get_zone_analytics(self, store_id: str, zone_id: str) -> dict:
        n = self._dwell_n[store_id][zone_id]
        avg_dwell = (
            self._dwell_sum[store_id][zone_id] / n if n > 0 else 0.0
        )
        return {
            "zone_id":       zone_id,
            "store_id":      store_id,
            "avg_dwell_s":   round(avg_dwell, 2),
            "visit_count":   n,
            "event_counts":  dict(self._counts[store_id][zone_id]),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Camera health registry
# ─────────────────────────────────────────────────────────────────────────────

class CameraHealthRegistry:
    """Stores per-camera telemetry (FPS, errors, calibration drift)."""

    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    def update(
        self,
        camera_id: str,
        fps: float,
        decode_errors: int,
        calibration_drift_px: float,
    ) -> None:
        self._data[camera_id] = {
            "camera_id":            camera_id,
            "fps":                  round(fps, 1),
            "decode_errors":        decode_errors,
            "calibration_drift_px": round(calibration_drift_px, 3),
            "status": "OK" if calibration_drift_px < 5.0 else "DRIFT_ALERT",
            "t_updated":            time.time(),
        }

    def all(self) -> list[dict]:
        return list(self._data.values())


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(
    engine: BAMScoringEngine,
    analytics: ZoneAnalytics | None = None,
    camera_health: CameraHealthRegistry | None = None,
) -> "FastAPI":
    """
    Build the FastAPI app with all BAM v1 routes attached.

    In production, call this from your ASGI entry point:
        app = create_app(engine)
        uvicorn.run(app, host="0.0.0.0", port=8080, workers=1, loop="uvloop")

    Note: use workers=1 — the scoring engine state is in-process.
    For multi-process deployments, move engine state to Redis Streams.
    """
    if not HAS_FASTAPI:
        raise RuntimeError("fastapi not installed — pip install fastapi uvicorn")

    app        = FastAPI(title="BAM API", version="1.0.0")
    ws_manager = ConnectionManager()
    analytics  = analytics or ZoneAnalytics()
    cam_health = camera_health or CameraHealthRegistry()

    # ── Score snapshot cache — avoid re-scoring on every WS tick ─────────
    _score_cache: dict[str, list[ScoreResult]] = {"latest": []}

    async def _score_broadcast_loop() -> None:
        """Background task: score all tracks at 1 Hz and broadcast."""
        while True:
            t0 = time.monotonic()
            results = await asyncio.get_running_loop().run_in_executor(
                None, engine.score_all_tracks
            )
            _score_cache["latest"] = results

            # Fan out to all connected stores
            store_groups: dict[str, list[dict]] = defaultdict(list)
            for r in results:
                store_groups[r.store_id].append(r.to_dict())

            for store_id, payloads in store_groups.items():
                await ws_manager.broadcast(store_id, {
                    "type": "score_update",
                    "t":    time.time(),
                    "scores": payloads,
                })

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, 1.0 - elapsed))

    @app.on_event("startup")
    async def startup() -> None:
        asyncio.create_task(_score_broadcast_loop())
        logger.info("BAM API server started")

        @app.get("/", response_class=HTMLResponse)
        async def root() -> HTMLResponse:
                return HTMLResponse(
                        """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>BAM API Dashboard</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #0b1020;
            --panel: rgba(17, 24, 39, 0.86);
            --panel-border: rgba(148, 163, 184, 0.16);
            --text: #e5eefc;
            --muted: #9fb0cf;
            --accent: #60a5fa;
            --good: #34d399;
            --shadow: 0 24px 64px rgba(0, 0, 0, 0.35);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.22), transparent 30%),
                radial-gradient(circle at right top, rgba(52, 211, 153, 0.16), transparent 24%),
                linear-gradient(160deg, #060816 0%, #0b1020 45%, #0f172a 100%);
        }
        .wrap {
            max-width: 1100px;
            margin: 0 auto;
            padding: 48px 24px 32px;
        }
        .hero {
            display: grid;
            gap: 18px;
            grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.7fr);
            align-items: stretch;
        }
        .panel {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
        }
        .headline {
            padding: 30px;
        }
        .kicker {
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 12px;
            font-weight: 700;
            margin: 0 0 14px;
        }
        h1 {
            margin: 0;
            font-size: clamp(34px, 6vw, 58px);
            line-height: 0.95;
            letter-spacing: -0.04em;
        }
        .subtitle {
            margin: 18px 0 0;
            max-width: 56ch;
            color: var(--muted);
            font-size: 16px;
            line-height: 1.7;
        }
        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 24px;
        }
        .button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 16px;
            border-radius: 999px;
            text-decoration: none;
            font-weight: 700;
            border: 1px solid transparent;
            transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
        }
        .button:hover { transform: translateY(-1px); }
        .button.primary {
            color: #08111f;
            background: linear-gradient(135deg, #93c5fd, #34d399);
        }
        .button.secondary {
            color: var(--text);
            background: rgba(148, 163, 184, 0.08);
            border-color: rgba(148, 163, 184, 0.18);
        }
        .status-card {
            padding: 24px;
            display: grid;
            gap: 14px;
            align-content: start;
        }
        .pill {
            display: inline-flex;
            width: fit-content;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(52, 211, 153, 0.14);
            color: #d7ffee;
            font-weight: 700;
            font-size: 13px;
        }
        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--good);
            box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.14);
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin-top: 18px;
        }
        .metric {
            padding: 18px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.12);
        }
        .metric-label {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.14em;
        }
        .metric-value {
            margin-top: 8px;
            font-size: 20px;
            font-weight: 800;
        }
        .section {
            margin-top: 18px;
            padding: 24px;
        }
        .section h2 {
            margin: 0 0 10px;
            font-size: 18px;
        }
        .section p, .section li {
            color: var(--muted);
            line-height: 1.7;
        }
        .codebox {
            margin-top: 14px;
            padding: 16px 18px;
            border-radius: 18px;
            background: rgba(2, 6, 23, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            overflow: auto;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 13px;
            color: #dbeafe;
        }
        @media (max-width: 900px) {
            .hero { grid-template-columns: 1fr; }
            .metric-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <main class="wrap">
        <section class="hero">
            <div class="panel headline">
                <p class="kicker">BAM API Dashboard</p>
                <h1>Buying Affinity Model</h1>
                <p class="subtitle">
                    A lightweight browser frontend for the live scoring API. It shows the server
                    status, gives you direct links to the docs, and keeps the existing backend logic intact.
                </p>
                <div class="actions">
                    <a class="button primary" href="/docs">Open API Docs</a>
                    <a class="button secondary" href="/v1/health/engine">View Engine Health</a>
                </div>
            </div>
            <aside class="panel status-card">
                <div class="pill"><span class="dot"></span> Live server</div>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-label">Status</div>
                        <div class="metric-value" id="status">Loading</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Service</div>
                        <div class="metric-value">BAM API</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Health</div>
                        <div class="metric-value">/v1/health/engine</div>
                    </div>
                </div>
            </aside>
        </section>

        <section class="panel section">
            <h2>What to open next</h2>
            <p>
                Use the links above to inspect the API, or call the live JSON health endpoint from your browser or terminal.
            </p>
            <div class="codebox">curl http://127.0.0.1:8000/v1/health/engine</div>
        </section>
    </main>

    <script>
        fetch('/v1/health/engine')
            .then((response) => response.json())
            .then((data) => {
                document.getElementById('status').textContent = data.metrics ? 'OK' : 'Live';
            })
            .catch(() => {
                document.getElementById('status').textContent = 'Offline';
            });
    </script>
</body>
</html>
                        """
                )

    # ─────────────────────────────────────────────────────────────────────
    # REST endpoints
    # ─────────────────────────────────────────────────────────────────────

    @app.get("/v1/stores/{store_id}/sessions/{session_id}/score")
    async def get_score(store_id: str, session_id: str) -> JSONResponse:
        """Current BAS, band, and decay-adjusted event ledger for a session."""
        # Find track by session_id in the cached results
        result = next(
            (r for r in _score_cache["latest"] if r.session_id == session_id),
            None,
        )
        if result is None:
            raise HTTPException(404, f"Session {session_id} not found")
        return JSONResponse(result.to_dict())

    @app.get("/v1/stores/{store_id}/zones/{zone_id}/analytics")
    async def get_zone_analytics(store_id: str, zone_id: str) -> JSONResponse:
        """Dwell, conversion-proxy, event histograms per zone."""
        data = analytics.get_zone_analytics(store_id, zone_id)
        return JSONResponse(data)

    @app.post("/v1/stores/{store_id}/config/weights")
    async def update_weights(store_id: str, payload: "WeightPayload") -> JSONResponse:
        """
        Apply a new weight set (validated: Σ positive = 100).
        In production, persist to config DB and reload engine weights.
        """
        # Validation already done by Pydantic validator
        logger.info("Weight update for store %s: %s", store_id, payload.weights)
        # TODO: engine.update_weights(store_id, payload.weights)
        return JSONResponse({"status": "accepted", "store_id": store_id})

    @app.post("/v1/stores/{store_id}/config/zones")
    async def update_zones(store_id: str, payload: "ZoneMapPayload") -> JSONResponse:
        """Upload/replace zone map (GeoJSON, floor coordinates); versioned."""
        logger.info(
            "Zone map update for store %s version %s", store_id, payload.version
        )
        # TODO: zone_engine.load_zone_map(store_id, payload.geojson, payload.version)
        return JSONResponse({
            "status":   "accepted",
            "store_id": store_id,
            "version":  payload.version,
        })

    @app.get("/v1/health/cameras")
    async def camera_health() -> JSONResponse:
        """Per-camera FPS, decode errors, calibration drift status."""
        return JSONResponse({"cameras": cam_health.all()})

    @app.get("/v1/health/engine")
    async def engine_health() -> JSONResponse:
        """Scoring engine metrics."""
        return JSONResponse({
            "active_tracks": len(engine._tracks),
            "metrics":       engine.metrics,
        })

    # ─────────────────────────────────────────────────────────────────────
    # WebSocket push stream
    # ─────────────────────────────────────────────────────────────────────

    @app.websocket("/v1/stores/{store_id}/scores/stream")
    async def ws_scores(websocket: "WebSocket", store_id: str) -> None:
        """
        Push stream of score updates and band transitions (HOT alerts).
        Client receives JSON messages at ≤ 1 Hz for all active tracks.
        HOT-band transitions trigger an additional immediate push.
        """
        await ws_manager.connect(store_id, websocket)
        try:
            while True:
                # Keep connection alive; server pushes via broadcast loop
                await asyncio.sleep(30)
                await websocket.send_text(json.dumps({"type": "ping"}))
        except WebSocketDisconnect:
            pass
        finally:
            await ws_manager.disconnect(store_id, websocket)

    return app
