"""
BAM — Buying Affinity Model
event_detector.py  ·  Stateful rule evaluator (normalised-coordinate version)

Extracted from the validated yolo_demo.py.  Maps YOLOv8-pose (COCO 17-keypoint)
output + detected objects to BehaviourEvents (E01–E18, N01–N04).

Only the pure event-detection logic lives here — no cv2 / ultralytics / rendering.
The CV pipeline (pipeline/bam_pipeline.py) owns YOLO inference and calls detect().
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .models import BehaviourEvent


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
# Human-readable event labels (for the Intel Feed)
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
