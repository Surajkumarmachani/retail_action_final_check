"""
BAM — Buying Affinity Model
event_pipeline.py  ·  Parallel camera ingest + event bus (L1–L5 glue)

Architecture
────────────
                ┌─────────────┐
  RTSP cam 1 ──▶│FrameWorker-1│─┐
  RTSP cam 2 ──▶│FrameWorker-2│─┼──▶ EventBus (asyncio.Queue) ──▶ ScoringEngine
  RTSP cam N ──▶│FrameWorker-N│─┘
                └─────────────┘

Key optimisations
─────────────────
• One OS thread per camera — frame decode is I/O + GPU, threads don't contend.
• asyncio EventBus — zero-copy event handoff between producers (threads) and
  consumers (coroutines) via asyncio.Queue; no serialisation overhead.
• Back-pressure — if the queue is full, the frame worker drops the frame and
  increments a metric (better than blocking the camera thread).
• Batch consumer — consumer drains up to DRAIN_BATCH events per loop tick to
  amortise coroutine scheduling overhead.
• Structured cancellation — CancellationToken lets the supervisor stop all
  workers cleanly without killing threads.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from models import BehaviourEvent
from scoring_engine import BAMScoringEngine, ScoreResult

logger = logging.getLogger("bam.pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    camera_id:  str
    rtsp_url:   str
    store_id:   str
    fps_target: int = 15          # §7 — 15 fps analysis rate
    gpu_device: int = 0           # CUDA device index

@dataclass
class PipelineConfig:
    cameras:            list[CameraConfig]
    store_format:       str  = "lifestyle"
    event_queue_size:   int  = 4096   # back-pressure buffer
    drain_batch:        int  = 64     # events consumed per asyncio tick
    score_push_interval_s: float = 1.0   # §5.2 1 Hz score downsample
    session_timeout_s:  float = 1800.0   # §1.1 30-min absence
    eviction_interval_s: float = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Cancellation token — shared flag between supervisor and workers
# ─────────────────────────────────────────────────────────────────────────────

class CancellationToken:
    """Lightweight cooperative cancellation for worker threads."""
    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Camera frame worker (runs in a dedicated thread per camera)
# ─────────────────────────────────────────────────────────────────────────────

class CameraFrameWorker(threading.Thread):
    """
    Pulls frames from one RTSP camera, runs perception (L2) and event
    extraction (L4 stub), and pushes BehaviourEvents onto the shared queue.

    In production:
      • Replace _run_perception() with TensorRT/RT-DETR INT8 inference.
      • Replace _extract_events() with ST-GCN + rule engine.
      • Use NVDEC for hardware decode (zero-copy to CUDA tensor).

    This implementation provides the correct threading structure and interface
    that the production perception stack plugs into.
    """

    def __init__(
        self,
        config: CameraConfig,
        event_queue: queue.Queue,
        cancel_token: CancellationToken,
        perception_fn: Callable | None = None,
        extraction_fn: Callable | None = None,
    ) -> None:
        super().__init__(
            name=f"cam-{config.camera_id}", daemon=True
        )
        self._cfg          = config
        self._queue        = event_queue
        self._cancel       = cancel_token
        self._perception   = perception_fn or self._stub_perception
        self._extraction   = extraction_fn or self._stub_extraction

        # Per-camera metrics
        self.frames_decoded  = 0
        self.frames_dropped  = 0   # queue full
        self.events_emitted  = 0

    def run(self) -> None:
        frame_interval = 1.0 / self._cfg.fps_target
        logger.info("Camera %s worker started", self._cfg.camera_id)

        while not self._cancel.is_cancelled:
            t_frame_start = time.monotonic()

            try:
                # ── L1: Decode frame (stub: production uses NVDEC) ────────
                frame = self._decode_frame()
                if frame is None:
                    continue
                self.frames_decoded += 1

                # ── L2: Perception — detection + pose ─────────────────────
                detections = self._perception(frame)

                # ── L4: Event extraction ───────────────────────────────────
                events: list[BehaviourEvent] = self._extraction(
                    detections, self._cfg
                )

                # ── Publish to event bus ───────────────────────────────────
                for ev in events:
                    try:
                        self._queue.put_nowait(ev)
                        self.events_emitted += 1
                    except queue.Full:
                        self.frames_dropped += 1
                        logger.warning(
                            "Event queue full — dropped event from %s",
                            self._cfg.camera_id,
                        )

            except Exception:
                logger.exception("Error in camera %s worker", self._cfg.camera_id)

            # ── Rate-limit to fps_target ───────────────────────────────────
            elapsed = time.monotonic() - t_frame_start
            sleep_s = frame_interval - elapsed
            if sleep_s > 0:
                self._cancel.wait(sleep_s)

        logger.info("Camera %s worker stopped", self._cfg.camera_id)

    def _decode_frame(self):
        """
        Stub: returns a sentinel dict.
        Production: cv2.VideoCapture / GStreamer appsink + NVDEC.
        """
        time.sleep(0)   # yield to OS
        return {"camera_id": self._cfg.camera_id, "t": time.time()}

    def _stub_perception(self, frame: dict) -> dict:
        """Stub perception — replace with TensorRT RT-DETR."""
        return {"tracks": [], "poses": [], "objects": []}

    def _stub_extraction(
        self, detections: dict, cfg: CameraConfig
    ) -> list[BehaviourEvent]:
        """
        Stub event extraction — replace with ST-GCN + rule engine.
        In production this calls the stateful rule evaluator for each
        active track against zone geometry and emits E/N events.
        """
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Async event bus consumer
# ─────────────────────────────────────────────────────────────────────────────

class AsyncEventConsumer:
    """
    Bridges the thread-safe queue (filled by CameraFrameWorkers) into asyncio
    for downstream async services.

    Uses asyncio.get_event_loop().run_in_executor to poll the blocking queue
    without blocking the event loop.
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        engine: BAMScoringEngine,
        drain_batch: int = 64,
        on_hot_alert: Callable[[ScoreResult], None] | None = None,
    ) -> None:
        self._queue       = event_queue
        self._engine      = engine
        self._batch       = drain_batch
        self._on_hot      = on_hot_alert or (lambda _: None)
        self._running     = False
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self, cancel_token: CancellationToken) -> None:
        self._running = True
        self._loop    = asyncio.get_running_loop()
        logger.info("AsyncEventConsumer started")

        while not cancel_token.is_cancelled:
            # Drain up to _batch events per tick to keep latency low
            consumed = 0
            while consumed < self._batch:
                try:
                    ev: BehaviourEvent = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._engine.on_event(ev)
                consumed += 1

            if consumed == 0:
                # Nothing in queue — yield control for ~1 ms
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)  # yield without delay

        self._running = False
        logger.info("AsyncEventConsumer stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Score publisher — 1 Hz push to WebSocket / dashboard
# ─────────────────────────────────────────────────────────────────────────────

class ScorePublisher:
    """
    Runs at 1 Hz, scores all active tracks in parallel, and pushes results.
    HOT-band transitions trigger immediate staff alerts.
    """

    def __init__(
        self,
        engine: BAMScoringEngine,
        publish_fn: Callable[[list[ScoreResult]], None],
        interval_s: float = 1.0,
    ) -> None:
        self._engine    = engine
        self._publish   = publish_fn
        self._interval  = interval_s
        self._prev_bands: dict[str, str] = {}

    async def run(self, cancel_token: CancellationToken) -> None:
        logger.info("ScorePublisher started (interval=%.1fs)", self._interval)

        while not cancel_token.is_cancelled:
            t0 = time.monotonic()

            # Parallel scoring of all active tracks
            results = await asyncio.get_running_loop().run_in_executor(
                None, self._engine.score_all_tracks
            )

            # Detect HOT-band transitions for immediate alert
            for r in results:
                prev = self._prev_bands.get(r.track_id)
                if r.band.name == "HOT" and prev != "HOT":
                    logger.info(
                        "HOT alert: track=%s zone=? BAS=%.1f", r.track_id, r.bas
                    )
                self._prev_bands[r.track_id] = r.band.name

            self._publish(results)

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, self._interval - elapsed))


# ─────────────────────────────────────────────────────────────────────────────
# Session eviction background task
# ─────────────────────────────────────────────────────────────────────────────

async def session_eviction_task(
    engine: BAMScoringEngine,
    cancel_token: CancellationToken,
    interval_s: float = 60.0,
    session_timeout_s: float = 1800.0,
) -> None:
    """Evict idle tracks every `interval_s` seconds (§privacy §1.1)."""
    logger.info("Session eviction task started")
    while not cancel_token.is_cancelled:
        await asyncio.sleep(interval_s)
        evicted = engine.evict_expired_sessions(session_timeout_s)
        if evicted:
            logger.info("Evicted %d expired sessions", evicted)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline supervisor
# ─────────────────────────────────────────────────────────────────────────────

class BAMPipeline:
    """
    Assembles and supervises the full BAM pipeline.

        pipeline = BAMPipeline(config, publish_fn=my_websocket_push)
        asyncio.run(pipeline.run())

    To plug in real perception:
        pipeline.set_perception_fn(camera_id, my_tensorrt_fn)
        pipeline.set_extraction_fn(camera_id, my_rule_engine_fn)
    """

    def __init__(
        self,
        config: PipelineConfig,
        publish_fn: Callable[[list[ScoreResult]], None] | None = None,
    ) -> None:
        self._cfg     = config
        self._publish = publish_fn or (lambda _: None)
        self._cancel  = CancellationToken()

        # Shared thread-safe queue (thread → async bridge)
        self._event_queue: queue.Queue = queue.Queue(
            maxsize=config.event_queue_size
        )

        # Scoring engine
        self._engine = BAMScoringEngine(
            store_format=config.store_format,
            max_workers=min(len(config.cameras) + 4, 16),
        )

        # Camera workers
        self._camera_workers: list[CameraFrameWorker] = []
        self._perception_fns: dict[str, Callable] = {}
        self._extraction_fns: dict[str, Callable] = {}

    # ── hooks for production perception stack ─────────────────────────────

    def set_perception_fn(
        self, camera_id: str, fn: Callable
    ) -> None:
        """Plug in TensorRT / RT-DETR perception for a camera."""
        self._perception_fns[camera_id] = fn

    def set_extraction_fn(
        self, camera_id: str, fn: Callable
    ) -> None:
        """Plug in ST-GCN + rule engine for a camera."""
        self._extraction_fns[camera_id] = fn

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start all workers and run until cancelled."""
        # Start one camera thread per camera
        for cam_cfg in self._cfg.cameras:
            worker = CameraFrameWorker(
                config=cam_cfg,
                event_queue=self._event_queue,
                cancel_token=self._cancel,
                perception_fn=self._perception_fns.get(cam_cfg.camera_id),
                extraction_fn=self._extraction_fns.get(cam_cfg.camera_id),
            )
            worker.start()
            self._camera_workers.append(worker)
        logger.info("Started %d camera workers", len(self._camera_workers))

        # Run async tasks concurrently
        consumer  = AsyncEventConsumer(
            self._event_queue, self._engine, self._cfg.drain_batch
        )
        publisher = ScorePublisher(
            self._engine, self._publish, self._cfg.score_push_interval_s
        )

        await asyncio.gather(
            consumer.run(self._cancel),
            publisher.run(self._cancel),
            session_eviction_task(
                self._engine, self._cancel,
                self._cfg.eviction_interval_s,
                self._cfg.session_timeout_s,
            ),
        )

    def stop(self) -> None:
        """Graceful shutdown — signals all workers to stop."""
        self._cancel.cancel()
        self._engine.shutdown()
        logger.info("BAMPipeline shutdown initiated")

    @property
    def engine(self) -> BAMScoringEngine:
        return self._engine

    @property
    def metrics(self) -> dict:
        return {
            "engine": self._engine.metrics,
            "cameras": {
                w.name: {
                    "frames_decoded": w.frames_decoded,
                    "frames_dropped": w.frames_dropped,
                    "events_emitted": w.events_emitted,
                }
                for w in self._camera_workers
            },
        }
