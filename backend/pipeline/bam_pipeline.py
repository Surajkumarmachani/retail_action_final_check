# pipeline/bam_pipeline.py
# BAM (Buying Affinity Model) per-frame pipeline.
#
# Replaces the BehaviourTransformer purchase-predictor with the event-based
# BAM scoring engine.  Per frame:
#   1. YOLOv8n-pose .track()  → persons: boxes + track IDs + 17-keypoint skeleton
#   2. YOLOv8n detect         → products (all non-person COCO classes, cached)
#   3. YOLOEventDetector      → BehaviourEvents (E01–E18, N01–N04) per person
#   4. BAMScoringEngine       → BAS (0-100) + band (COLD/WARM/HOT) per track
#
# Thread-safety: one BAMTracker per WebSocket connection so ByteTrack state,
# track-state registry, and the scoring engine are isolated per stream.

from __future__ import annotations

import base64
import logging
import math
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from bam.event_detector import (
    DetectedObject, TrackState, YOLOEventDetector, EVENT_LABELS,
)
from bam.models import EVENT_WEIGHTS, ScoreBand
from bam.scoring_engine import BAMScoringEngine
from constants import SAMPLE_EVERY

logger = logging.getLogger(__name__)

# Inference device: Apple GPU (MPS) is ~6.5× faster than CPU on M-series.
# Falls back to CUDA, then CPU. Overridable via BAM_DEVICE env var.
import os
def _pick_device() -> str:
    forced = os.environ.get("BAM_DEVICE")
    if forced:
        return forced
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _pick_device()
INFER_IMGSZ  = int(os.environ.get("BAM_IMGSZ", "320"))       # pose (matches sent frame width)
DETECT_IMGSZ = int(os.environ.get("BAM_DETECT_IMGSZ", "448"))  # seg/products — larger = better recall
SEG_EVERY    = int(os.environ.get("BAM_SEG_EVERY", "1"))       # segment every frame (fast on MPS) for instant product ID
logger.info("BAM device=%s  pose_imgsz=%d  detect_imgsz=%d  seg_every=%d",
            DEVICE, INFER_IMGSZ, DETECT_IMGSZ, SEG_EVERY)

# COCO class names (index → label); class 0 = person handled by the pose model
_COCO_NAMES: Dict[int, str] = {
    0:"person",1:"bicycle",2:"car",3:"motorcycle",4:"airplane",5:"bus",6:"train",
    7:"truck",8:"boat",9:"traffic light",10:"fire hydrant",11:"stop sign",
    12:"parking meter",13:"bench",14:"bird",15:"cat",16:"dog",17:"horse",18:"sheep",
    19:"cow",20:"elephant",21:"bear",22:"zebra",23:"giraffe",24:"backpack",
    25:"umbrella",26:"handbag",27:"tie",28:"suitcase",29:"frisbee",30:"skis",
    31:"snowboard",32:"sports ball",33:"kite",34:"baseball bat",35:"baseball glove",
    36:"skateboard",37:"surfboard",38:"tennis racket",39:"bottle",40:"wine glass",
    41:"cup",42:"fork",43:"knife",44:"spoon",45:"bowl",46:"banana",47:"apple",
    48:"sandwich",49:"orange",50:"broccoli",51:"carrot",52:"hot dog",53:"pizza",
    54:"donut",55:"cake",56:"chair",57:"couch",58:"potted plant",59:"bed",
    60:"dining table",61:"toilet",62:"tv",63:"laptop",64:"mouse",65:"remote",
    66:"keyboard",67:"cell phone",68:"microwave",69:"oven",70:"toaster",71:"sink",
    72:"refrigerator",73:"book",74:"clock",75:"vase",76:"scissors",77:"teddy bear",
    78:"hair drier",79:"toothbrush",
}


# ── Frame decode ────────────────────────────────────────────────────────────

def decode_frame(b64_data: str):
    """Decode a Base64-encoded JPEG/PNG string into an OpenCV BGR frame."""
    try:
        data  = base64.b64decode(b64_data)
        arr   = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.warning("Frame decode error: %s", exc)
        return None


# ── Staff classifier (green torso heuristic) ─────────────────────────────────

def _is_staff_green(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    torso_y2 = y1 + int((y2 - y1) * 0.4)
    if torso_y2 <= y1 or x2 <= x1:
        return False
    crop = frame[y1:torso_y2, x1:x2]
    if crop.size == 0:
        return False
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    return cv2.countNonZero(mask) / (crop.shape[0] * crop.shape[1] + 1e-6) > 0.15


# ── Per-connection tracker ───────────────────────────────────────────────────

class BAMTracker:
    """
    Owns per-connection CV models + BAM state.
    Call close() when the WebSocket drops.
    """

    def __init__(
        self,
        pose_path: str,
        detect_path: str,
        store_format: str = "lifestyle",
    ) -> None:
        from ultralytics import YOLO
        self.pose   = YOLO(pose_path)     # yolov8n-pose.pt  (persons + skeleton + IDs)
        self.detect = YOLO(detect_path)   # yolov8n.pt        (product detection — light & fast)
        self.event_detector = YOLOEventDetector()
        self.engine = BAMScoringEngine(store_format=store_format, max_workers=4)

        self.states: Dict[str, TrackState] = {}          # tid_str → TrackState
        self.recent_events: Dict[str, deque] = {}        # tid_str → deque[(ts, etype)]
        self.frame_count = 0
        self.last_products: List[Dict[str, Any]] = []

        # Warm up both models on the target device (first MPS/CUDA call compiles kernels)
        try:
            warm = np.zeros((INFER_IMGSZ, INFER_IMGSZ, 3), dtype=np.uint8)
            self.pose.predict(warm, device=DEVICE, imgsz=INFER_IMGSZ, verbose=False)
            self.detect.predict(warm, device=DEVICE, imgsz=DETECT_IMGSZ, verbose=False)
        except Exception as exc:
            logger.warning("Model warmup failed (continuing): %s", exc)

    def close(self) -> None:
        self.engine.shutdown()


# ── Main pipeline entry point (runs in ThreadPoolExecutor) ───────────────────

def process_frame_bam(
    frame: np.ndarray,
    tracker: BAMTracker,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (persons, products, new_events).

    persons     : one dict per tracked person — box (px), keypoints (norm),
                  bas, band, dwell_sec, top_events, is_staff.
    products    : one dict per non-person object — box (px), label, conf.
    new_events  : BehaviourEvents fired this frame — {track_id, type, label,
                  weight, conf, ts} for the Intel Feed.
    """
    tracker.frame_count += 1
    do_detect = (tracker.frame_count % SEG_EVERY == 0)
    h, w = frame.shape[:2]
    now  = time.time()

    # ── 1. Person tracking + pose (single pass) ──────────────────────────────
    # conf=0.5 filters low-confidence false positives (e.g. a held mouse/phone or a
    # bare hand being mis-detected as a separate person → ghost "subjects").
    pose_res = tracker.pose.track(
        frame, persist=True, classes=[0], conf=0.5, verbose=False,
        tracker="bytetrack.yaml", device=DEVICE, imgsz=INFER_IMGSZ,
    )

    # ── 2. Product detection (every frame for snappy product ID) ─────────────
    if do_detect:
        tracker.last_products = _detect_products(frame, tracker)
    products = tracker.last_products

    persons: List[Dict[str, Any]] = []
    new_events: List[Dict[str, Any]] = []

    r = pose_res[0] if pose_res else None
    if r is None or r.boxes is None or r.boxes.id is None or r.keypoints is None:
        return persons, products, new_events

    boxes      = r.boxes
    track_ids  = boxes.id.cpu().numpy().astype(int)
    xyxy       = boxes.xyxy.cpu().numpy()
    kpts_xyn   = r.keypoints.xyn.cpu().numpy()   # (N,17,2) normalised
    kpts_conf  = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

    # Build DetectedObjects for the event detector (products only, normalised centres)
    detected_objects: List[DetectedObject] = []
    for p in products:
        px1, py1, px2, py2 = p["box"]
        detected_objects.append(DetectedObject(
            category=p["label"],
            score=p["conf"],
            bbox=(px1, py1, px2, py2),
            center_norm=(((px1 + px2) / 2) / w, ((py1 + py2) / 2) / h),
        ))

    # First pass: refresh every track's state so social detectors see all peers
    for i, tid in enumerate(track_ids):
        tid_str = f"trk_{int(tid)}"
        state = tracker.states.get(tid_str)
        if state is None:
            state = TrackState(
                track_id=tid_str,
                session_id=f"ses_{int(tid)}_{int(now)}",
                store_id="WEBCAM-DEMO",
            )
            tracker.states[tid_str] = state
            tracker.recent_events[tid_str] = deque(maxlen=6)
        x1, y1, x2, y2 = (int(v) for v in xyxy[i])
        state.is_staff = _is_staff_green(frame, x1, y1, x2, y2)

    # Second pass: run detectors + scoring per person
    for i, tid in enumerate(track_ids):
        tid_str = f"trk_{int(tid)}"
        state   = tracker.states[tid_str]
        x1, y1, x2, y2 = (int(v) for v in xyxy[i])

        kconf_i = kpts_conf[i] if kpts_conf is not None else None
        fired = tracker.event_detector.detect(
            kpts_xyn[i], kconf_i,
            state, tracker.states,
            w, h,
            detected_objects,
        )
        for ev in fired:
            tracker.engine.on_event(ev)
            wt = EVENT_WEIGHTS.get(ev.event_type, 0.0)
            tracker.recent_events[tid_str].append((now, ev.event_type))
            new_events.append({
                "track_id": int(tid),
                "type":     ev.event_type,
                "label":    EVENT_LABELS.get(ev.event_type, "?"),
                "weight":   wt,
                "conf":     round(ev.confidence, 2),
                "ts":       now,
            })

        # Score this track
        result = tracker.engine.score(tid_str, state.session_id, "WEBCAM-DEMO")
        bas, band = result.bas, result.band

        # Recent distinct event codes (most-recent first) for the card
        seen: set = set()
        top_events: List[str] = []
        for _, et in reversed(tracker.recent_events[tid_str]):
            if et not in seen:
                seen.add(et)
                top_events.append(et)
            if len(top_events) >= 3:
                break

        # Keypoints for the frontend skeleton: [xn, yn, conf] × 17 (normalised)
        kp_out = []
        for j in range(kpts_xyn.shape[1]):
            kx, ky = float(kpts_xyn[i][j][0]), float(kpts_xyn[i][j][1])
            kc = float(kconf_i[j]) if kconf_i is not None else 0.0
            kp_out.append([round(kx, 4), round(ky, 4), round(kc, 3)])

        posture = _posture_metrics(kpts_xyn[i], kconf_i, w, h)

        persons.append({
            "id":        int(tid),
            "box":       [x1, y1, x2, y2],
            "keypoints": kp_out,
            "posture":   posture,
            "bas":       round(bas, 1),
            "band":      band.name,
            "dwell_sec": round(now - state.first_seen_t, 1) if state.first_seen_t else 0.0,
            "is_staff":  state.is_staff,
            "top_events": top_events,
            # Backward-compat fields for the existing dashboard UI
            "prob":      round(bas / 100.0, 4),
            "verdict":   "LIKELY BUY" if band == ScoreBand.HOT else "BROWSING",
        })

    return persons, products, new_events


# ── Posture / tilt metrics from COCO-17 keypoints ─────────────────────────────

_COCO = {"nose": 0, "l_eye": 1, "r_eye": 2, "l_sh": 5, "r_sh": 6, "l_hip": 11, "r_hip": 12}

def _posture_metrics(kp_xyn: np.ndarray, kconf, w: int, h: int) -> Dict[str, Any]:
    """
    Torso tilt, spine angle, shoulder alignment, and gaze (head) tilt in degrees,
    derived from the pose keypoints. Computed in pixel space (x*w, y*h) so aspect
    ratio doesn't distort angles. Returns only the metrics whose keypoints are visible.
    """
    def ok(i: int) -> bool:
        return kconf is None or float(kconf[i]) >= 0.3
    def P(i: int):
        return (float(kp_xyn[i][0]) * w, float(kp_xyn[i][1]) * h)
    def mid(a, b):
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    out: Dict[str, Any] = {}

    if ok(_COCO["l_sh"]) and ok(_COCO["r_sh"]):
        ls, rs = P(_COCO["l_sh"]), P(_COCO["r_sh"])
        # shoulder line tilt from horizontal (0 = level)
        out["shoulder_align"] = round(math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0])), 1)

        if ok(_COCO["l_hip"]) and ok(_COCO["r_hip"]):
            sm = mid(ls, rs)
            hm = mid(P(_COCO["l_hip"]), P(_COCO["r_hip"]))
            # torso tilt from vertical (0 = upright); +ve = leaning to image-right
            out["torso_tilt"] = round(math.degrees(math.atan2(sm[0] - hm[0], hm[1] - sm[1])), 1)
            if ok(_COCO["nose"]):
                nose = P(_COCO["nose"])
                # spine/neck angle: nose relative to shoulder mid, from vertical
                out["spine_angle"] = round(math.degrees(math.atan2(nose[0] - sm[0], sm[1] - nose[1])), 1)

    if ok(_COCO["l_eye"]) and ok(_COCO["r_eye"]):
        le, re = P(_COCO["l_eye"]), P(_COCO["r_eye"])
        out["gaze_angle"] = round(math.degrees(math.atan2(re[1] - le[1], re[0] - le[0])), 1)

    return out


# ── Product detection (non-person COCO classes) ───────────────────────────────

def _detect_products(
    frame: np.ndarray,
    tracker: BAMTracker,
    conf_threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    """Run YOLO detect for all non-person classes. Returns {box, class_id, label, conf}."""
    h, w = frame.shape[:2]
    try:
        res = tracker.detect.predict(
            frame, classes=list(range(1, 80)), conf=conf_threshold,
            verbose=False, device=DEVICE, imgsz=DETECT_IMGSZ,
        )
    except Exception as exc:
        logger.warning("Product detection error: %s", exc)
        return []

    r = res[0] if res else None
    if r is None or r.boxes is None:
        return []

    xyxy    = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    confs   = r.boxes.conf.cpu().numpy()

    products: List[Dict[str, Any]] = []
    for box, cls_id, conf in zip(xyxy, cls_ids, confs):
        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2])); y2 = min(h, int(box[3]))
        products.append({
            "box":      [x1, y1, x2, y2],
            "class_id": int(cls_id),
            "label":    _COCO_NAMES.get(int(cls_id), f"class_{cls_id}"),
            "conf":     round(float(conf), 2),
        })
    return products
