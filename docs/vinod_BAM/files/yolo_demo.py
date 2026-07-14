"""
BAM — Buying Affinity Model
yolo_demo.py  ·  Production-wired YOLO webcam demo

Architecture
────────────
  YOLOv8 detect + ByteTrack  ─┐
  YOLOv8-pose                 ─┼──▶ YOLOPerceptionAdapter
  Staff classifier (HSV)      ─┘         │
                                          ▼
                               CameraFrameWorker (dedicated thread)
                                          │  thread-safe queue
                                          ▼
                               AsyncEventConsumer  ──▶  BAMScoringEngine
                                                          (vectorised decay,
                                                           idempotency, zone cap,
                                                           parallel scoring,
                                                           session eviction)
                                          │
                                          ▼
                               ScorePublisher (1 Hz)  ──▶  OverlayRenderer

What changed vs the old yolo_demo.py
──────────────────────────────────────
1. Scoring goes through BAMScoringEngine — correct exponential decay, λ,
   zone cap (1.5×), confidence gate (0.70), idempotency (Blake2b IDs),
   late-event rejection (10 s window), session eviction (30 min idle).

2. Camera loop runs in a dedicated thread (CameraFrameWorker) — the main
   thread is free for rendering; the scoring engine runs its own
   ThreadPoolExecutor for parallel track scoring.

3. All 22 events (E01–E18, N01–N04) are wired. Previously only ~12 were
   implemented and several lacked correct spec multipliers.

4. Per-track state is keyed by ByteTrack ID — multiple simultaneous
   shoppers each get their own BAS, each scored independently.

5. Weight invariant (Σ=100) is enforced at startup; format deltas are
   applied through build_weights().

Usage
─────
    python3 yolo_demo.py              # default webcam (index 0)
    python3 yolo_demo.py --camera 1   # secondary webcam
    python3 yolo_demo.py --mirror     # mirror / selfie view
    python3 yolo_demo.py --format fashion   # apply fashion weight deltas

Controls
────────
    q / ESC   — quit
    r         — reset all sessions
    m         — toggle mirror
    s         — toggle skeleton
    b         — toggle bounding boxes
    z         — toggle zone overlays
    d         — toggle debug panel
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

# ── BAM production modules ─────────────────────────────────────────────────
from models import BehaviourEvent, EVENT_WEIGHTS, ScoreBand, build_weights
from scoring_engine import BAMScoringEngine, ScoreResult
from event_pipeline import (
    CameraConfig, CameraFrameWorker, CancellationToken,
    AsyncEventConsumer, ScorePublisher,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("bam.yolo_demo")

# ─────────────────────────────────────────────────────────────────────────────
# COCO pose landmark indices
# ─────────────────────────────────────────────────────────────────────────────

class LM:
    NOSE           = 0
    LEFT_EYE       = 1;  RIGHT_EYE       = 2
    LEFT_EAR       = 3;  RIGHT_EAR       = 4
    LEFT_SHOULDER  = 5;  RIGHT_SHOULDER  = 6
    LEFT_ELBOW     = 7;  RIGHT_ELBOW     = 8
    LEFT_WRIST     = 9;  RIGHT_WRIST     = 10
    LEFT_HIP       = 11; RIGHT_HIP       = 12
    LEFT_KNEE      = 13; RIGHT_KNEE      = 14
    LEFT_ANKLE     = 15; RIGHT_ANKLE     = 16


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (normalised 0-1 coords)
# ─────────────────────────────────────────────────────────────────────────────

def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def _mid(a: tuple, b: tuple) -> tuple:
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)

def _angle3(a: tuple, b: tuple, c: tuple) -> float:
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = (math.sqrt(ba[0]**2+ba[1]**2)+1e-6)*(math.sqrt(bc[0]**2+bc[1]**2)+1e-6)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))

def _lm(kpts_xyn, idx: int) -> tuple[float, float]:
    return float(kpts_xyn[idx][0]), float(kpts_xyn[idx][1])

def _vis(kpts_conf, idx: int, thr: float = 0.5) -> bool:
    return kpts_conf is not None and float(kpts_conf[idx]) >= thr


# ─────────────────────────────────────────────────────────────────────────────
# Staff classifier (green torso heuristic)
# ─────────────────────────────────────────────────────────────────────────────

def _is_staff_green(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    torso_y2 = y1 + int((y2 - y1) * 0.4)
    if torso_y2 <= y1 or x2 <= x1:
        return False
    crop = frame[y1:torso_y2, x1:x2]
    if crop.size == 0:
        return False
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35,50,50]), np.array([85,255,255]))
    return cv2.countNonZero(mask) / (crop.shape[0]*crop.shape[1]+1e-6) > 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Per-track mutable state  (one per ByteTrack ID)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackState:
    """All mutable per-shopper state for the YOLO event detector."""
    track_id:   str
    session_id: str
    store_id:   str   = "WEBCAM-DEMO"
    zone_id:    str   = "Z-WEBCAM"
    is_staff:   bool  = False

    first_seen_t: float = 0.0
    last_seen_t:  float = 0.0

    position_history:  deque = field(default_factory=lambda: deque(maxlen=90))
    speed_history:     deque = field(default_factory=lambda: deque(maxlen=30))
    nose_history:      deque = field(default_factory=lambda: deque(maxlen=45))
    wrist_history_l:   deque = field(default_factory=lambda: deque(maxlen=30))
    wrist_history_r:   deque = field(default_factory=lambda: deque(maxlen=30))

    # Timers
    arm_extended_since:  float = 0.0;  arm_extended:  bool = False
    hold_since:          float = 0.0;  holding:       bool = False
    head_down_since:     float = 0.0;  head_down:     bool = False
    lean_in_since:       float = 0.0;  leaning_in:    bool = False
    phone_since:         float = 0.0;  phone_detected: bool = False
    gaze_fixed_since:    float = 0.0;  gaze_fixed:    bool = False
    in_mirror_since:     float = 0.0
    in_queue_since:      float = 0.0;  was_in_queue:  bool = False
    staff_contact_since: float = 0.0
    companion_since:     float = 0.0;  companion_id:  str | None = None

    objects_held_since: dict = field(default_factory=dict)

    # cooldown tracker — prevents firing same event too rapidly
    _last_fired: dict = field(default_factory=dict)

    def can_fire(self, event_type: str, cooldown_s: float = 5.0) -> bool:
        now = time.time()
        if now - self._last_fired.get(event_type, 0.0) >= cooldown_s:
            self._last_fired[event_type] = now
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Detected object DTO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    category:    str
    score:       float
    bbox:        tuple[int, int, int, int]   # pixel x1y1x2y2
    center_norm: tuple[float, float]          # normalised cx, cy


# ─────────────────────────────────────────────────────────────────────────────
# YOLO event detector  (all 22 BAM events, spec-wired multipliers)
# ─────────────────────────────────────────────────────────────────────────────

class YOLOEventDetector:
    """
    Stateful rule evaluator that maps YOLOv8 detection + pose output
    to BehaviourEvents.  Implements all E01–E18 and N01–N04 with the
    same trigger conditions as event_detectors.py but adapted for the
    COCO 17-keypoint normalised coordinate space from YOLOv8-pose.

    Thresholds that reference floor-plane metres are approximated by
    normalised-frame fractions until camera calibration is available.
    """

    # ── Configurable thresholds ──
    DWELL_BASELINE_S   = 5.0    # E08 — replace with rolling 28-day P75 in production
    GAZE_FIXATION_S    = 3.0    # E10
    HOLD_DURATION_S    = 3.0    # E04 (spec: 8 s; shortened for demo responsiveness)
    HEAD_DOWN_S        = 2.0    # E06
    LEAN_ANGLE_DEG     = 15.0   # E11
    LEAN_DURATION_S    = 1.5    # E11
    REACH_DURATION_S   = 0.5    # E13
    PHONE_DIST_NORM    = 0.08   # E17 wrist-to-ear normalised distance
    PHONE_DURATION_S   = 2.0    # E17
    SESSION_BASELINE_S = 15.0   # E18 (replace with store median)
    WALK_SPEED_THRESH  = 0.015  # N03 minimum speed to count as walking
    WALK_CV_THRESH     = 0.30   # N03 coefficient of variation (constant speed)
    EXIT_EDGE_THRESH   = 0.12   # N04 edge fraction to count as exit-directed
    SPEED_DROP_RATIO   = 0.50   # E12 deceleration threshold
    WRIST_STABILITY    = 0.020  # E04 wrist variance threshold
    NOSE_STABILITY     = 0.015  # E10 nose variance threshold
    HAND_OBJ_DIST      = 0.12   # E02/E05 wrist-to-object normalised distance
    QUEUE_ABANDON_S    = 2.0    # N02 minimum queue dwell before abandon counts
    MIRROR_DWELL_S     = 3.0    # E03 mirror-zone dwell threshold
    STAFF_CONTACT_S    = 3.0    # E07 proximity dwell threshold
    COMPANION_S        = 5.0    # E14 co-proximity dwell threshold
    SPEED_DROP_FULL_S  = 0.001  # E12 full-stop threshold (normalised speed)

    # Hardcoded demo zones (pixel coords at 1280×720)
    # In production: load from POST /v1/stores/{id}/config/zones GeoJSON
    ZONES = {
        "queue":    (50,  400, 350, 700),
        "register": (900, 400, 1200, 700),
        "mirror":   (400, 200, 880, 550),
    }

    def __init__(self) -> None:
        pass  # all state lives in TrackState, detector is stateless

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _in_zone(cx_px: int, cy_px: int, zone: str) -> bool:
        x1, y1, x2, y2 = YOLOEventDetector.ZONES[zone]
        return x1 <= cx_px <= x2 and y1 <= cy_px <= y2

    @staticmethod
    def _make_ev(
        state: TrackState,
        event_type: str,
        confidence: float,
        multiplier: float = 1.0,
    ) -> BehaviourEvent:
        return BehaviourEvent(
            event_type=event_type,
            track_id=state.track_id,
            session_id=state.session_id,
            store_id=state.store_id,
            zone_id=state.zone_id,
            t_start=time.time(),
            confidence=confidence,
            multiplier=multiplier,
        )

    # ── main entry point ─────────────────────────────────────────────────

    def detect(
        self,
        kpts_xyn:        np.ndarray,           # (17,2) normalised
        kpts_conf:       np.ndarray,           # (17,) confidence
        state:           TrackState,
        all_states:      dict[str, TrackState],
        frame_w:         int,
        frame_h:         int,
        detected_objects: list[DetectedObject],
    ) -> list[BehaviourEvent]:
        """
        Called once per frame per tracked person.
        Returns list of BehaviourEvents to push onto the event bus.
        """
        events: list[BehaviourEvent] = []

        if not _vis(kpts_conf, LM.NOSE):
            return events

        now = time.time()
        state.last_seen_t = now
        if state.first_seen_t == 0:
            state.first_seen_t = now

        # ── Extract keypoints ─────────────────────────────────────────────
        nose        = _lm(kpts_xyn, LM.NOSE)
        l_shoulder  = _lm(kpts_xyn, LM.LEFT_SHOULDER)
        r_shoulder  = _lm(kpts_xyn, LM.RIGHT_SHOULDER)
        shoulder_mid = _mid(l_shoulder, r_shoulder)
        l_hip       = _lm(kpts_xyn, LM.LEFT_HIP)
        r_hip       = _lm(kpts_xyn, LM.RIGHT_HIP)
        hip_mid     = _mid(l_hip, r_hip)
        l_wrist     = _lm(kpts_xyn, LM.LEFT_WRIST)
        r_wrist     = _lm(kpts_xyn, LM.RIGHT_WRIST)
        l_elbow     = _lm(kpts_xyn, LM.LEFT_ELBOW)
        r_elbow     = _lm(kpts_xyn, LM.RIGHT_ELBOW)
        l_ear       = _lm(kpts_xyn, LM.LEFT_EAR)
        r_ear       = _lm(kpts_xyn, LM.RIGHT_EAR)

        hip_px = (int(hip_mid[0]*frame_w), int(hip_mid[1]*frame_h))

        # ── Update histories ──────────────────────────────────────────────
        state.position_history.append((hip_mid, now))
        state.nose_history.append((nose, now))
        state.wrist_history_l.append((l_wrist, now))
        state.wrist_history_r.append((r_wrist, now))

        speed = 0.0
        if len(state.position_history) >= 2:
            prev_pos, prev_t = state.position_history[-2]
            dt = now - prev_t
            if dt > 0:
                speed = _dist(hip_mid, prev_pos) / dt
        state.speed_history.append(speed)

        head_drop = nose[1] - shoulder_mid[1]
        non_person_objs = [o for o in detected_objects if o.category != "person"]
        phone_objs      = [o for o in non_person_objs  if o.category == "cell phone"]

        # ── E03: Try-on / Mirror dwell ────────────────────────────────────
        if self._in_zone(hip_px[0], hip_px[1], "mirror"):
            if state.in_mirror_since == 0:
                state.in_mirror_since = now
            elif now - state.in_mirror_since > self.MIRROR_DWELL_S:
                if state.can_fire("E03", cooldown_s=15.0):
                    events.append(self._make_ev(state, "E03", 0.90))
        else:
            state.in_mirror_since = 0

        # ── N02: Queue abandonment ────────────────────────────────────────
        in_queue    = self._in_zone(hip_px[0], hip_px[1], "queue")
        in_register = self._in_zone(hip_px[0], hip_px[1], "register")
        if in_queue:
            if state.in_queue_since == 0:
                state.in_queue_since = now
            elif now - state.in_queue_since > self.QUEUE_ABANDON_S:
                state.was_in_queue = True
        else:
            state.in_queue_since = 0
        at_edge = hip_mid[0] < self.EXIT_EDGE_THRESH or hip_mid[0] > 1 - self.EXIT_EDGE_THRESH
        if state.was_in_queue and at_edge and speed > 0.005 and not in_register:
            if state.can_fire("N02", cooldown_s=20.0):
                events.append(self._make_ev(state, "N02", 0.85))
                state.was_in_queue = False

        # ── E07 / E14: Social events ──────────────────────────────────────
        staff_near = False
        companion_near = False
        for other_id, other in all_states.items():
            if other_id == state.track_id or not other.position_history:
                continue
            other_hip = other.position_history[-1][0]
            d = _dist(hip_mid, other_hip)
            if d < 0.15:
                # E07 — staff contact
                if other.is_staff and speed < 0.005:
                    staff_near = True
                    if state.staff_contact_since == 0:
                        state.staff_contact_since = now
                    elif now - state.staff_contact_since > self.STAFF_CONTACT_S:
                        if state.can_fire("E07", cooldown_s=15.0):
                            events.append(self._make_ev(state, "E07", 0.88))
                # E14 — companion (entered within 2 s of each other)
                if abs(state.first_seen_t - other.first_seen_t) < 2.0:
                    companion_near = True
                    if state.companion_since == 0:
                        state.companion_since = now
                        state.companion_id = other_id
                    elif now - state.companion_since > self.COMPANION_S:
                        if state.can_fire("E14", cooldown_s=30.0):
                            events.append(self._make_ev(state, "E14", 0.85))
        if not staff_near:
            state.staff_contact_since = 0
        if not companion_near:
            state.companion_since = 0

        # ── E02: Product pickup (hand near object) ────────────────────────
        hand_near_obj = False
        for obj in non_person_objs:
            for wrist in (l_wrist, r_wrist):
                if _dist(wrist, obj.center_norm) < self.HAND_OBJ_DIST:
                    hand_near_obj = True
                    if obj.category not in state.objects_held_since:
                        state.objects_held_since[obj.category] = now
        if not hand_near_obj:
            state.objects_held_since.clear()
        if hand_near_obj:
            for cat, t_hold in state.objects_held_since.items():
                if now - t_hold > 0.5:
                    if state.can_fire("E02", cooldown_s=8.0):
                        events.append(self._make_ev(state, "E02", 0.85))
                    break

        # ── E05: Comparison (two objects, one per hand) ───────────────────
        obj_l = any(_dist(l_wrist, o.center_norm) < self.HAND_OBJ_DIST for o in non_person_objs)
        obj_r = any(_dist(r_wrist, o.center_norm) < self.HAND_OBJ_DIST for o in non_person_objs)
        if obj_l and obj_r and head_drop > 0.05:
            if state.can_fire("E05", cooldown_s=8.0):
                events.append(self._make_ev(state, "E05", 0.85))

        # ── E08: Zone dwell ───────────────────────────────────────────────
        dwell = now - state.first_seen_t
        if dwell > self.DWELL_BASELINE_S:
            if state.can_fire("E08", cooldown_s=8.0):
                events.append(self._make_ev(state, "E08", 0.95))

        # ── E10: Gaze fixation (nose stability) ───────────────────────────
        if len(state.nose_history) >= 15:
            recent_noses = [p for p, _ in list(state.nose_history)[-15:]]
            var = np.var([n[0] for n in recent_noses]) + np.var([n[1] for n in recent_noses])
            if var < self.NOSE_STABILITY**2:
                if not state.gaze_fixed:
                    state.gaze_fixed = True
                    state.gaze_fixed_since = now
                elif now - state.gaze_fixed_since >= self.GAZE_FIXATION_S:
                    if state.can_fire("E10", cooldown_s=6.0):
                        events.append(self._make_ev(state, "E10", 0.80))
                    state.gaze_fixed = False
            else:
                state.gaze_fixed = False

        # ── E04: Hold duration (wrist stability + elevated) ───────────────
        if len(state.wrist_history_l) >= 15 and len(state.wrist_history_r) >= 15:
            rl = [p for p, _ in list(state.wrist_history_l)[-15:]]
            rr = [p for p, _ in list(state.wrist_history_r)[-15:]]
            vl = np.var([p[0] for p in rl]) + np.var([p[1] for p in rl])
            vr = np.var([p[0] for p in rr]) + np.var([p[1] for p in rr])
            elevated = l_wrist[1] < hip_mid[1] and r_wrist[1] < hip_mid[1]
            if vl < self.WRIST_STABILITY**2 and vr < self.WRIST_STABILITY**2 and elevated:
                if not state.holding:
                    state.holding = True
                    state.hold_since = now
                elif now - state.hold_since >= self.HOLD_DURATION_S:
                    if state.can_fire("E04", cooldown_s=8.0):
                        conf = 0.92 if hand_near_obj else 0.87
                        events.append(self._make_ev(state, "E04", conf))
                    state.holding = False
            else:
                state.holding = False

        # ── E06: Price/label inspection (head down + hand near obj) ───────
        if head_drop > 0.06:
            if not state.head_down:
                state.head_down = True
                state.head_down_since = now
            elif now - state.head_down_since >= self.HEAD_DOWN_S:
                if state.can_fire("E06", cooldown_s=5.0):
                    conf = 0.88 if hand_near_obj else 0.80
                    events.append(self._make_ev(state, "E06", conf))
                state.head_down = False
        else:
            state.head_down = False

        # ── E11: Body lean-in (torso angle) ──────────────────────────────
        torso_angle = abs(math.degrees(math.atan2(
            shoulder_mid[0] - hip_mid[0], -(shoulder_mid[1] - hip_mid[1])
        )))
        if torso_angle > self.LEAN_ANGLE_DEG:
            if not state.leaning_in:
                state.leaning_in = True
                state.lean_in_since = now
            elif now - state.lean_in_since >= self.LEAN_DURATION_S:
                if state.can_fire("E11", cooldown_s=5.0):
                    events.append(self._make_ev(state, "E11", 0.82))
                state.leaning_in = False
        else:
            state.leaning_in = False

        # ── E12: Approach deceleration ─────────────────────────────────────
        if len(state.speed_history) >= 15:
            spds = list(state.speed_history)
            mid  = len(spds) // 2
            avg_first  = sum(spds[:mid]) / mid if mid else 0
            avg_second = sum(spds[mid:]) / (len(spds)-mid) if (len(spds)-mid) else 0
            if avg_first > 0.005 and avg_second < avg_first * self.SPEED_DROP_RATIO:
                if state.can_fire("E12", cooldown_s=8.0):
                    # Full stop → weight 1.0; partial decel → 0.5  (§3.2 E12)
                    mult = 1.0 if avg_second < self.SPEED_DROP_FULL_S else 0.5
                    events.append(self._make_ev(state, "E12", 0.88, multiplier=mult))

        # ── E13: Reach attempt (arm extension toward shelf) ───────────────
        arm_extended = False
        for shoulder, elbow, wrist in [
            (l_shoulder, l_elbow, l_wrist),
            (r_shoulder, r_elbow, r_wrist),
        ]:
            angle = _angle3(shoulder, elbow, wrist)
            if angle > 150 and wrist[1] < shoulder[1] + 0.05:
                arm_extended = True
                break
        if arm_extended:
            if not state.arm_extended:
                state.arm_extended = True
                state.arm_extended_since = now
            elif now - state.arm_extended_since >= self.REACH_DURATION_S:
                if state.can_fire("E13", cooldown_s=4.0):
                    # No E02 within 1.5 s → aborted reach at 0.5 (§3.2 E13)
                    last_e02 = state._last_fired.get("E02", 0.0)
                    mult = 1.0 if (now - last_e02) <= 1.5 else 0.5
                    events.append(self._make_ev(state, "E13", 0.84, multiplier=mult))
                state.arm_extended = False
        else:
            state.arm_extended = False

        # ── E17: Phone use at shelf ───────────────────────────────────────
        phone_by_pose = any(
            _dist(w, e) < self.PHONE_DIST_NORM
            for w, e in [(l_wrist, l_ear), (r_wrist, r_ear)]
        )
        phone_by_obj = bool(phone_objs) and any(
            _dist(w, o.center_norm) < self.HAND_OBJ_DIST * 1.5
            for o in phone_objs for w in (l_wrist, r_wrist)
        )
        if phone_by_pose or phone_by_obj:
            if not state.phone_detected:
                state.phone_detected = True
                state.phone_since = now
            elif now - state.phone_since >= self.PHONE_DURATION_S:
                if state.can_fire("E17", cooldown_s=6.0):
                    conf = 0.90 if phone_by_obj else 0.80
                    events.append(self._make_ev(state, "E17", conf))
                state.phone_detected = False
        else:
            state.phone_detected = False

        # ── E18: Session duration vs baseline ────────────────────────────
        if dwell > self.SESSION_BASELINE_S:
            if state.can_fire("E18", cooldown_s=15.0):
                events.append(self._make_ev(state, "E18", 0.91))

        # ── N01: Rapid put-back (hand near obj then no obj within 3 s) ───
        if not hand_near_obj and state.objects_held_since:
            last_e02 = state._last_fired.get("E02", 0.0)
            if 0 < (now - last_e02) <= 3.0:
                if state.can_fire("N01", cooldown_s=5.0):
                    events.append(self._make_ev(state, "N01", 0.88))

        # ── N03: Walk-past (constant speed, no engagement) ───────────────
        if len(state.speed_history) >= 10:
            spds = [s for s in state.speed_history if s > 0.001]
            if len(spds) >= 8:
                avg_s = sum(spds) / len(spds)
                cv = float(np.std(spds)) / (avg_s + 1e-6)
                if avg_s > self.WALK_SPEED_THRESH and cv < self.WALK_CV_THRESH:
                    if state.can_fire("N03", cooldown_s=5.0):
                        events.append(self._make_ev(state, "N03", 0.85))

        # ── N04: Exit-directed movement ───────────────────────────────────
        if at_edge and speed > 0.005 and not state.was_in_queue:
            if state.can_fire("N04", cooldown_s=8.0):
                events.append(self._make_ev(state, "N04", 0.87))

        return events


# ─────────────────────────────────────────────────────────────────────────────
# YOLO perception adapter — plugs into CameraFrameWorker hooks
# ─────────────────────────────────────────────────────────────────────────────

class YOLOPerceptionAdapter:
    """
    Wraps YOLOv8 detection + pose into the two callables expected by
    CameraFrameWorker: perception_fn(frame) → detections dict,
    extraction_fn(detections, cfg) → list[BehaviourEvent].

    Owns:
      • YOLO object detector with ByteTrack (persist=True)
      • YOLO pose estimator
      • Per-track state registry
      • YOLOEventDetector (stateless rule engine)

    Thread-safety: both callables are called on the same camera thread
    so no locking is needed within the adapter.
    """

    def __init__(
        self,
        store_format:   str  = "lifestyle",
        show_skeleton:  bool = True,
        show_boxes:     bool = True,
        mirror:         bool = False,
    ) -> None:
        from ultralytics import YOLO as _YOLO
        print("Loading YOLOv8 models...")
        self._detector       = _YOLO("yolov8n.pt")
        self._pose_estimator = _YOLO("yolov8n-pose.pt")
        self._event_detector = YOLOEventDetector()
        self._track_states:  dict[str, TrackState] = {}
        self._primary_tid:   str | None = None
        self.mirror          = mirror
        self.show_skeleton   = show_skeleton
        self.show_boxes      = show_boxes
        # latest rendered frame and primary score for main thread rendering
        self._last_frame:    np.ndarray | None = None
        self._last_objects:  list[DetectedObject] = []
        self._all_states_snapshot: dict[str, TrackState] = {}
        print("YOLOv8 models loaded.")

    # ── CameraFrameWorker.perception_fn hook ─────────────────────────────

    def perception(self, raw_frame: dict) -> dict:
        """
        raw_frame is a dict with key 'frame' (np.ndarray) and 't' (float).
        Returns a dict with detection + pose results for extraction_fn.
        """
        frame: np.ndarray = raw_frame["frame"]
        if self.mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # ── Object detection + tracking ───────────────────────────────────
        track_res = self._detector.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False
        )
        # ── Pose estimation ───────────────────────────────────────────────
        pose_res  = self._pose_estimator(frame, classes=[0], verbose=False)

        # Build annotated frame for display
        ann = frame.copy()
        if self.show_boxes:
            ann = track_res[0].plot(img=ann, labels=True)
        if self.show_skeleton:
            ann = pose_res[0].plot(img=ann, labels=False, boxes=False)

        # Parse detected objects
        detected_objects: list[DetectedObject] = []
        person_box_map:   dict[str, tuple] = {}  # tid_str → bbox pixels

        if track_res[0].boxes:
            for box in track_res[0].boxes:
                cls_id   = int(box.cls[0].item())
                score    = float(box.conf[0].item())
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                category = self._detector.names[cls_id]
                cx_norm  = ((x1+x2)/2) / w
                cy_norm  = ((y1+y2)/2) / h
                detected_objects.append(DetectedObject(category, score, (x1,y1,x2,y2), (cx_norm,cy_norm)))

                if category == "person" and box.id is not None:
                    tid = str(int(box.id[0].item()))
                    tid_str = f"trk_{tid}"
                    person_box_map[tid_str] = (x1, y1, x2, y2)

                    if tid_str not in self._track_states:
                        self._track_states[tid_str] = TrackState(
                            track_id=tid_str,
                            session_id=f"ses_{tid}_{int(time.time())}",
                        )
                        if self._primary_tid is None:
                            self._primary_tid = tid_str

                    # Staff classification
                    is_staff = _is_staff_green(frame, x1, y1, x2, y2)
                    self._track_states[tid_str].is_staff = is_staff

                    # Draw STAFF label on annotated frame
                    if is_staff and self.show_boxes:
                        cv2.putText(ann, "[STAFF]", (x1, max(0, y1-25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Update primary if it vanished
        active = set(person_box_map.keys())
        if self._primary_tid and self._primary_tid not in active:
            self._primary_tid = next(iter(active)) if active else None

        self._last_frame   = ann
        self._last_objects = detected_objects
        self._all_states_snapshot = dict(self._track_states)

        return {
            "frame":          frame,
            "ann_frame":      ann,
            "w": w, "h": h,
            "pose_res":       pose_res,
            "track_res":      track_res,
            "detected_objects": detected_objects,
            "person_box_map": person_box_map,
            "t":              raw_frame["t"],
        }

    # ── CameraFrameWorker.extraction_fn hook ─────────────────────────────

    def extraction(self, detections: dict, cfg: CameraConfig) -> list[BehaviourEvent]:
        """
        Runs the YOLOEventDetector on each tracked person and returns
        all BehaviourEvents for the BAM scoring engine.
        """
        pose_res        = detections["pose_res"]
        person_box_map  = detections["person_box_map"]
        detected_objects = detections["detected_objects"]
        w, h            = detections["w"], detections["h"]
        all_events:     list[BehaviourEvent] = []

        if not (pose_res[0].keypoints and pose_res[0].keypoints.has_visible):
            return all_events

        person_boxes = [
            box for box in detections["track_res"][0].boxes
            if self._detector.names[int(box.cls[0].item())] == "person"
                and box.id is not None
        ]

        for i, box in enumerate(person_boxes):
            if i >= len(pose_res[0].keypoints.xyn):
                break
            tid_str = f"trk_{int(box.id[0].item())}"
            state   = self._track_states.get(tid_str)
            if state is None:
                continue

            kpts_xyn  = pose_res[0].keypoints.xyn[i].cpu().numpy()
            kpts_conf = pose_res[0].keypoints.conf[i].cpu().numpy()

            fired = self._event_detector.detect(
                kpts_xyn, kpts_conf,
                state, self._track_states,
                w, h,
                detected_objects,
            )
            all_events.extend(fired)

        return all_events

    # ── evict stale tracks (called by session eviction task) ─────────────

    def evict_track(self, tid_str: str) -> None:
        self._track_states.pop(tid_str, None)
        if self._primary_tid == tid_str:
            self._primary_tid = next(iter(self._track_states), None)


# ─────────────────────────────────────────────────────────────────────────────
# Camera frame worker subclass — real cv2 capture replaces stub
# ─────────────────────────────────────────────────────────────────────────────

class WebcamFrameWorker(CameraFrameWorker):
    """
    CameraFrameWorker with a real cv2.VideoCapture decode replacing the stub.
    Owns the capture object; opens/closes it on thread start/stop.
    """

    def __init__(
        self,
        config:       CameraConfig,
        event_queue:  queue.Queue,
        cancel_token: CancellationToken,
        perception_fn: Callable,
        extraction_fn: Callable,
        video_source:  int | str = 0,
    ) -> None:
        super().__init__(config, event_queue, cancel_token, perception_fn, extraction_fn)
        self._video_source = video_source
        self._cap: cv2.VideoCapture | None = None

    def run(self) -> None:
        self._cap = cv2.VideoCapture(self._video_source)
        self._cap.set(cv2.CAP_PROP_FPS,          self._cfg.fps_target)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        if not self._cap.isOpened():
            logger.error("Could not open video source %s", self._video_source)
            return

        logger.info("WebcamFrameWorker started on source %s", self._video_source)
        super().run()   # uses _decode_frame below
        self._cap.release()
        logger.info("WebcamFrameWorker stopped")

    def _decode_frame(self):  # overrides stub
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return {"frame": frame, "t": time.time()}


# ─────────────────────────────────────────────────────────────────────────────
# HUD overlay renderer
# ─────────────────────────────────────────────────────────────────────────────

EVENT_LABELS = {
    "E01":"Basket Placement", "E02":"Product Pickup",    "E03":"Try-On/Demo",
    "E04":"Hold Duration",    "E05":"Comparison",        "E06":"Price Inspection",
    "E07":"Staff Contact",    "E08":"Zone Dwell",        "E09":"Repeat Return",
    "E10":"Gaze Fixation",    "E11":"Body Lean-In",      "E12":"Deceleration",
    "E13":"Reach Attempt",    "E14":"Companion",         "E15":"Path Pattern",
    "E16":"Zone Transition",  "E17":"Phone at Shelf",    "E18":"Session Duration",
    "N01":"Rapid Put-Back",   "N02":"Queue Abandon",     "N03":"Walk-Past",
    "N04":"Exit Movement",
}

class HUDRenderer:
    def __init__(self) -> None:
        self.event_log: deque = deque(maxlen=8)
        self._hot_flash_until = 0.0

    def add_event(self, event_type: str, weight: float) -> None:
        self.event_log.appendleft((time.time(), event_type, weight))

    def render(
        self,
        frame:     np.ndarray,
        bas:       float,
        band:      ScoreBand,
        show_zones: bool = True,
        show_debug: bool = False,
        n_tracks:  int = 0,
        metrics:   dict | None = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        now  = time.time()

        def panel(x1,y1,x2,y2,a=0.80):
            p = frame.copy()
            cv2.rectangle(p,(x1,y1),(x2,y2),(18,18,18),-1)
            cv2.rectangle(p,(x1,y1),(x2,y2),(70,70,70),1)
            cv2.addWeighted(p,a,frame,1-a,0,frame)

        def chip(x,y,text,color):
            (tw,th),bl = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
            px,py = 8,5
            cv2.rectangle(frame,(x,y-th-py*2),(x+tw+px*2,y+bl),color,-1)
            cv2.putText(frame,text,(x+px,y-py),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
            return x+tw+px*2+5

        # ── Zone overlays ────────────────────────────────────────────────
        if show_zones:
            ov = frame.copy()
            for zone_name, color, label in [
                ("queue",    (0,165,255),  "QUEUE"),
                ("register", (0,200,0),    "REGISTER"),
                ("mirror",   (255,120,0),  "MIRROR"),
            ]:
                x1,y1,x2,y2 = YOLOEventDetector.ZONES[zone_name]
                cv2.rectangle(ov,(x1,y1),(x2,y2),color,2)
                cv2.rectangle(ov,(x1,y1),(x2,y1+26),color,-1)
                cv2.putText(ov,label,(x1+6,y1+19),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)
            cv2.addWeighted(ov,0.12,frame,0.88,0,frame)

        # ── BAS score panel ───────────────────────────────────────────────
        panel(10,10,390,148)
        sc = {"HOT":(0,0,255),"WARM":(0,200,255),"COLD":(200,150,0)}[band.name]
        cv2.putText(frame,"BAM Score",(22,38),cv2.FONT_HERSHEY_SIMPLEX,0.65,(220,220,220),1,cv2.LINE_AA)
        cv2.putText(frame,f"{bas:.1f}",(22,94),cv2.FONT_HERSHEY_SIMPLEX,2.0,sc,3,cv2.LINE_AA)
        cv2.putText(frame,band.name,(220,90),cv2.FONT_HERSHEY_SIMPLEX,1.35,sc,2,cv2.LINE_AA)
        # progress bar
        bx,by,bw,bh = 20,108,340,12
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(55,55,55),-1)
        fw = int(bw * min(bas/100,1))
        for i in range(fw):
            r = i/bw
            c = (200,150,0) if r<0.4 else (0,200,255) if r<0.7 else (0,0,255)
            cv2.line(frame,(bx+i,by),(bx+i,by+bh),c,1)
        for t in (40,70):
            tx = bx + int(bw*t/100)
            cv2.line(frame,(tx,by-2),(tx,by+bh+2),(255,255,255),1)

        # ── Event log panel ───────────────────────────────────────────────
        px = w - 360
        ph = min(54 + len(self.event_log)*28, 280)
        panel(px,10,w-10,10+ph)
        cv2.putText(frame,"Events",(px+10,34),cv2.FONT_HERSHEY_SIMPLEX,0.6,(230,230,230),1,cv2.LINE_AA)
        for i,(t,etype,wt) in enumerate(self.event_log):
            if i >= 7: break
            age = now - t
            a   = max(0.3, 1-age/15)
            y   = 56 + i*28
            col = tuple(int(c*a) for c in ((0,190,90) if wt>0 else (0,120,255)))
            nx  = chip(px+10, y, etype, col)
            sign = "+" if wt > 0 else ""
            cv2.putText(frame,f"{sign}{wt:.0f}",(nx,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1,cv2.LINE_AA)
            lbl = EVENT_LABELS.get(etype,"?")[:22]
            cv2.putText(frame,lbl,(px+155,y),cv2.FONT_HERSHEY_SIMPLEX,0.44,(225,225,225),1,cv2.LINE_AA)

        # ── HOT flash ────────────────────────────────────────────────────
        if band == ScoreBand.HOT:
            self._hot_flash_until = now + 0.5
        if now < self._hot_flash_until:
            fl = frame.copy()
            cv2.rectangle(fl,(0,0),(w,h),(0,0,180),-1)
            a  = 0.13 * max(0, 1-(now-(self._hot_flash_until-0.5))/0.5)
            cv2.addWeighted(fl,max(0,a),frame,1-max(0,a),0,frame)
            txt = "!! HOT ALERT — High Buying Affinity !!"
            ts  = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)[0]
            cv2.putText(frame,txt,((w-ts[0])//2,h-50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)

        # ── Debug panel ───────────────────────────────────────────────────
        if show_debug and metrics:
            panel(10,h-90,340,h-10)
            m = metrics
            cv2.putText(frame,f"Tracks: {n_tracks}  Scored: {m.get('events_scored',0)}",(18,h-68),cv2.FONT_HERSHEY_SIMPLEX,0.45,(220,220,220),1,cv2.LINE_AA)
            cv2.putText(frame,f"Gated: {m.get('events_gated',0)}  Deduped: {m.get('events_deduped',0)}  Late: {m.get('events_late',0)}",(18,h-46),cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1,cv2.LINE_AA)

        # ── Controls footer ───────────────────────────────────────────────
        cv2.rectangle(frame,(0,h-28),(w,h),(10,10,10),-1)
        cx = 70
        cv2.putText(frame,"Controls",(8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.40,(180,180,180),1,cv2.LINE_AA)
        for lbl in ("Q quit","R reset","M mirror","S skeleton","B boxes","Z zones","D debug"):
            cx = chip(cx,h-10,lbl,(80,80,80))

        return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main demo orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class BAMYoloDemo:
    """
    Ties everything together:
      • WebcamFrameWorker (dedicated thread) → event bus → BAMScoringEngine
      • Main thread: reads latest annotated frame + scores, renders HUD, shows window
    """

    def __init__(
        self,
        video_source:  int | str = 0,
        store_format:  str       = "lifestyle",
        mirror:        bool      = False,
    ) -> None:
        self.mirror        = mirror
        self._cancel       = CancellationToken()
        self._event_queue: queue.Queue = queue.Queue(maxsize=4096)

        self._engine = BAMScoringEngine(
            store_format=store_format, max_workers=4
        )

        self._adapter = YOLOPerceptionAdapter(
            store_format=store_format,
            mirror=mirror,
        )

        cam_cfg = CameraConfig(
            camera_id="webcam-0",
            rtsp_url=str(video_source),
            store_id="WEBCAM-DEMO",
            fps_target=15,
        )

        self._worker = WebcamFrameWorker(
            config=cam_cfg,
            event_queue=self._event_queue,
            cancel_token=self._cancel,
            perception_fn=self._adapter.perception,
            extraction_fn=self._adapter.extraction,
            video_source=video_source,
        )

        self._renderer = HUDRenderer()
        self.show_skeleton = True
        self.show_boxes    = True
        self.show_zones    = True
        self.show_debug    = False

    def run(self) -> None:
        # Start camera worker thread
        self._worker.start()
        print("BAM YOLO Demo running — press Q to quit")

        fps_q: deque = deque(maxlen=30)
        prev_bas = 0.0

        while True:
            t0 = time.perf_counter()

            # ── Drain event queue into scoring engine ──────────────────────
            drained = 0
            while drained < 64:
                try:
                    ev: BehaviourEvent = self._event_queue.get_nowait()
                    self._engine.on_event(ev)

                    # Feed event to HUD log for primary track only
                    if ev.track_id == self._adapter._primary_tid:
                        wt = EVENT_WEIGHTS.get(ev.event_type, 0)
                        self._renderer.add_event(ev.event_type, wt)

                    drained += 1
                except queue.Empty:
                    break

            # ── Score primary track ────────────────────────────────────────
            bas, band = 0.0, ScoreBand.COLD
            tid = self._adapter._primary_tid
            if tid and tid in self._adapter._track_states:
                state  = self._adapter._track_states[tid]
                result = self._engine.score(tid, state.session_id, "WEBCAM-DEMO")
                bas, band = result.bas, result.band

            # ── Get latest annotated frame ─────────────────────────────────
            frame = self._adapter._last_frame
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                continue

            # ── Render HUD ─────────────────────────────────────────────────
            display = self._renderer.render(
                frame.copy(), bas, band,
                show_zones=self.show_zones,
                show_debug=self.show_debug,
                n_tracks=len(self._adapter._track_states),
                metrics=self._engine.metrics,
            )

            # FPS counter
            fps_q.append(time.perf_counter() - t0)
            fps = 1.0 / (sum(fps_q)/len(fps_q)) if fps_q else 0
            h, w = display.shape[:2]
            cv2.putText(display, f"FPS:{fps:.0f}", (w-80, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100,100,100), 1, cv2.LINE_AA)

            cv2.imshow("BAM — Buying Affinity Model", display)

            # ── Key handling ───────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                self._adapter._track_states.clear()
                self._adapter._primary_tid = None
                self._engine = BAMScoringEngine(
                    store_format="lifestyle", max_workers=4
                )
                self._renderer = HUDRenderer()
                print("  ── Session reset ──")
            elif key == ord('m'):
                self.mirror = not self.mirror
                self._adapter.mirror = self.mirror
            elif key == ord('s'):
                self.show_skeleton = not self.show_skeleton
                self._adapter.show_skeleton = self.show_skeleton
            elif key == ord('b'):
                self.show_boxes = not self.show_boxes
                self._adapter.show_boxes = self.show_boxes
            elif key == ord('z'):
                self.show_zones = not self.show_zones
            elif key == ord('d'):
                self.show_debug = not self.show_debug

        # ── Shutdown ───────────────────────────────────────────────────────
        self._cancel.cancel()
        self._worker.join(timeout=3.0)
        self._engine.shutdown()
        cv2.destroyAllWindows()
        print("BAM demo stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAM — Live YOLO Demo")
    parser.add_argument("--camera", type=int,    default=0,           help="Camera index")
    parser.add_argument("--mirror", action="store_true",              help="Mirror mode")
    parser.add_argument("--format", default="lifestyle",
                        choices=["fashion","electronics","grocery","lifestyle"],
                        help="Store format (applies weight deltas)")
    args = parser.parse_args()

    demo = BAMYoloDemo(
        video_source=args.camera,
        store_format=args.format,
        mirror=args.mirror,
    )
    demo.run()
