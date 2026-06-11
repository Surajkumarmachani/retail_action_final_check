# pipeline/cv_pipeline.py
# Full per-frame CV pipeline: YOLOv8+ByteTrack → MediaPipe → feature observation dicts.
#
# Thread-safety notes:
#   • yolo.track(persist=True) stores ByteTrack state INSIDE the YOLO object.
#     Each WebSocket connection therefore gets its own YOLO instance so track IDs
#     are scoped to the connection (PerConnectionTracker creates one per connection).
#   • MediaPipe Pose/FaceMesh/Hands are NOT thread-safe across calls; the same
#     instance must only be called from one thread at a time.  Because we await
#     run_in_executor() before the next frame arrives, a single connection's tracker
#     is never called concurrently.

import base64
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from constants import SAMPLE_EVERY

logger = logging.getLogger(__name__)

_mp_pose      = mp.solutions.pose
_mp_face_mesh = mp.solutions.face_mesh
_mp_hands     = mp.solutions.hands


# ── Per-connection state container ────────────────────────────────────────────

class PerConnectionTracker:
    """
    Owns one YOLO instance + MediaPipe solution instances per WebSocket connection.
    Call close() when the connection drops to release MediaPipe resources.
    """

    def __init__(self, yolo_path: str) -> None:
        from ultralytics import YOLO
        # Each connection gets its own YOLO object so ByteTrack state is isolated.
        self.yolo = YOLO(yolo_path)

        self.pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.4,
            refine_landmarks=False,
        )
        self.hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )
        self.frame_count: int = 0

    def close(self) -> None:
        self.pose.close()
        self.face_mesh.close()
        self.hands.close()


# ── Frame decode ──────────────────────────────────────────────────────────────

def decode_frame(b64_data: str) -> Optional[np.ndarray]:
    """Decode a Base64-encoded JPEG/PNG string into an OpenCV BGR frame."""
    try:
        data  = base64.b64decode(b64_data)
        arr   = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        logger.warning("Frame decode error: %s", exc)
        return None


# ── Main pipeline entry point (runs in ThreadPoolExecutor) ────────────────────

def process_frame(
    frame: np.ndarray,
    tracker: PerConnectionTracker,
) -> List[Dict[str, Any]]:
    """
    Run YOLOv8+ByteTrack on every frame; run MediaPipe every SAMPLE_EVERY frames.
    Returns one observation dict per detected+tracked person.

    This function is blocking and must be called via run_in_executor.
    """
    tracker.frame_count += 1
    do_mediapipe = (tracker.frame_count % SAMPLE_EVERY == 0)

    # ── YOLOv8 + ByteTrack ────────────────────────────────────────────────────
    results = tracker.yolo.track(
        frame,
        persist=True,
        classes=[0],               # person class only
        verbose=False,
        tracker="bytetrack.yaml",
    )

    observations: List[Dict[str, Any]] = []
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return observations

    boxes    = results[0].boxes
    track_ids: np.ndarray = boxes.id.cpu().numpy().astype(int)
    xyxy:     np.ndarray  = boxes.xyxy.cpu().numpy()
    h, w     = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Collect all bbox centres for social-proximity calculation
    centres: Dict[int, Tuple[float, float]] = {
        int(tid): (
            float((box[0] + box[2]) / 2),
            float((box[1] + box[3]) / 2),
        )
        for tid, box in zip(track_ids, xyxy)
    }

    for tid, box in zip(track_ids, xyxy):
        tid  = int(tid)
        x1   = max(0, int(box[0]))
        y1   = max(0, int(box[1]))
        x2   = min(w, int(box[2]))
        y2   = min(h, int(box[3]))

        obs: Dict[str, Any] = {
            "track_id": tid,
            "box":      [x1, y1, x2, y2],
            "position": centres[tid],
            "timestamp": time.monotonic(),
        }

        if do_mediapipe:
            crop = rgb[y1:y2, x1:x2]
            if crop.size > 0:
                obs.update(_analyse_pose(crop, tracker.pose))
                obs.update(_analyse_head_pose(crop, tracker.face_mesh))
                obs.update(_analyse_hands(crop, tracker.hands))
                # DeepFace emotion analysis (disabled by default for speed)
                # obs.update(_analyse_emotion(crop))

        obs.update(_compute_social(tid, centres))
        observations.append(obs)

    return observations


# ── Feature extractors ────────────────────────────────────────────────────────

def _analyse_pose(crop: np.ndarray, pose) -> Dict[str, Any]:
    result = pose.process(crop)
    if not result.pose_landmarks:
        return {"pose_detected": 0.0}

    lm = result.pose_landmarks.landmark
    ch, cw = crop.shape[:2]

    def pt(idx: int) -> np.ndarray:
        l = lm[idx]
        return np.array([l.x * cw, l.y * ch], dtype=float)

    ls, rs = pt(11), pt(12)   # shoulders
    lh, rh = pt(23), pt(24)   # hips
    lw, rw = pt(15), pt(16)   # wrists
    le, re = pt(13), pt(14)   # elbows

    shoulder_mid = (ls + rs) / 2
    hip_mid      = (lh + rh) / 2

    torso_vec  = hip_mid - shoulder_mid
    lean_angle = float(math.degrees(math.atan2(torso_vec[0], torso_vec[1])))

    shoulder_width_norm = float(np.linalg.norm(rs - ls)) / max(cw, 1)

    arm_cross        = bool(lw[0] > shoulder_mid[0] and rw[0] < shoulder_mid[0])
    step_back        = bool(hip_mid[1] > ch * 0.75)
    wrist_raised     = bool(lw[1] < ls[1] or rw[1] < rs[1])
    wrist_near_shelf = bool(lw[1] < ch / 3 or rw[1] < ch / 3)

    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba, bc = a - b, c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(math.degrees(math.acos(np.clip(cos, -1.0, 1.0))))

    elbow_angle_l  = _angle(ls, le, lw)
    elbow_angle_r  = _angle(rs, re, rw)
    body_turned_away = bool(shoulder_width_norm < 0.08)
    relaxed_posture  = bool(abs(lean_angle) < 15 and not arm_cross and not wrist_raised)

    chest_y = shoulder_mid[1] + 0.3 * (hip_mid[1] - shoulder_mid[1])
    holding_product = bool(
        (abs(lw[0] - shoulder_mid[0]) < cw * 0.15 and abs(lw[1] - chest_y) < ch * 0.15)
        or (abs(rw[0] - shoulder_mid[0]) < cw * 0.15 and abs(rw[1] - chest_y) < ch * 0.15)
    )

    return {
        "pose_detected":        1.0,
        "lean_angle":           lean_angle,
        "arm_cross":            arm_cross,
        "step_back":            step_back,
        "wrist_raised":         wrist_raised,
        "wrist_near_shelf":     wrist_near_shelf,
        "elbow_angle_l":        elbow_angle_l,
        "elbow_angle_r":        elbow_angle_r,
        "shoulder_width_norm":  shoulder_width_norm,
        "body_turned_away":     body_turned_away,
        "relaxed_posture":      relaxed_posture,
        "holding_product":      holding_product,
    }


def _analyse_head_pose(crop: np.ndarray, face_mesh) -> Dict[str, Any]:
    result = face_mesh.process(crop)
    if not result.multi_face_landmarks:
        return {
            "face_detected": 0.0, "head_yaw_proxy": 0.0,
            "head_pitch_proxy": 0.0, "looking_at_product": False,
        }

    lm   = result.multi_face_landmarks[0].landmark
    ch, cw = crop.shape[:2]

    def pt(idx: int) -> np.ndarray:
        l = lm[idx]
        return np.array([l.x * cw, l.y * ch, l.z], dtype=float)

    nose       = pt(1)
    left_ear   = pt(234)
    right_ear  = pt(454)
    chin       = pt(152)
    forehead   = pt(10)

    ear_mid     = (left_ear + right_ear) / 2
    ear_width   = np.linalg.norm(right_ear - left_ear) + 1e-6
    face_height = np.linalg.norm(chin - forehead) + 1e-6

    head_yaw_proxy   = float((nose[0] - ear_mid[0]) / ear_width * 90)
    head_pitch_proxy = float((nose[1] - ear_mid[1]) / face_height * 90)
    looking_at_product = bool(abs(head_yaw_proxy) < 20 and -10 < head_pitch_proxy < 30)

    return {
        "face_detected":        1.0,
        "head_yaw_proxy":       head_yaw_proxy,
        "head_pitch_proxy":     head_pitch_proxy,
        "looking_at_product":   looking_at_product,
    }


def _analyse_hands(crop: np.ndarray, hands) -> Dict[str, Any]:
    result = hands.process(crop)
    if not result.multi_hand_landmarks:
        return {"hands_detected": 0.0, "hand_open": False, "hand_pinch": False}

    lm   = result.multi_hand_landmarks[0].landmark
    tips = [lm[i] for i in [4, 8, 12, 16, 20]]
    pips = [lm[i] for i in [3, 6, 10, 14, 18]]

    hand_open  = bool(all(t.y < p.y for t, p in zip(tips, pips)))
    pinch_dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
    hand_pinch = bool(pinch_dist < 0.05)

    return {
        "hands_detected": float(len(result.multi_hand_landmarks)),
        "hand_open":      hand_open,
        "hand_pinch":     hand_pinch,
    }


def _analyse_emotion(crop: np.ndarray) -> Dict[str, Any]:
    """
    Optional DeepFace emotion analysis.  Disabled by default (GPU-speed sensitive).
    Enable by uncommenting the call in process_frame().
    """
    try:
        from deepface import DeepFace
        res    = DeepFace.analyze(crop, actions=["emotion"], enforce_detection=False)
        emo    = res[0]["emotion"]
        total  = sum(emo.values()) + 1e-9
        return {
            "emotion_detected":  1.0,
            "emo_happy":         emo.get("happy",     0) / total,
            "emo_interested":    emo.get("surprise",  0) / total,  # proxy
            "emo_neutral":       emo.get("neutral",   0) / total,
            "emo_disinterested": emo.get("sad",       0) / total,
            "emo_negative":      emo.get("angry",     0) / total,
            "emo_confused":      emo.get("fear",      0) / total,
        }
    except Exception:
        return {}


def _compute_social(
    tid: int,
    centres: Dict[int, Tuple[float, float]],
) -> Dict[str, Any]:
    others = {k: v for k, v in centres.items() if k != tid}
    if not others:
        return {"group_size": 1.0, "near_others": False, "avg_group_dist": 999.0}

    my = centres[tid]
    dists = [math.hypot(my[0] - c[0], my[1] - c[1]) for c in others.values()]
    avg_dist = sum(dists) / len(dists)

    return {
        "group_size":     float(1 + len(others)),
        "near_others":    any(d < 100 for d in dists),  # pixel threshold; camera-dependent
        "avg_group_dist": avg_dist,
    }
