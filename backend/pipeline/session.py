# pipeline/session.py
# LiveCustomerSession: two-level temporal buffer for one tracked person.
#
# Level-1 (inner) – per-frame observation dicts accumulated during the current window.
#                   Flushed every WINDOW_SECONDS of wall-clock time into one 41-d dict.
#
# Level-2 (outer) – ordered list of completed window dicts.  This is the raw sequence
#                   the Transformer consumes.  It grows until MAX_SEQ is reached, then
#                   the oldest window is dropped (sliding window).
#
# The key design difference from the notebook's for-loop:
#   Notebook:  CustomerSession lives inside a finite video loop; frame count drives
#              windowing (30 FPS × 5 s = 150 frames → 50 sampled frames per window).
#   Live:      Webcam FPS is variable and unknown.  We measure actual elapsed seconds
#              (time.monotonic()) so the 5-second window is always correct regardless
#              of whether the client sends 10 FPS or 30 FPS.

import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from constants import WINDOW_SECONDS


class LiveCustomerSession:
    def __init__(self, track_id: int):
        self.track_id      = track_id
        self.session_start = time.monotonic()   # for cumulative dwell_sec
        self.window_start  = time.monotonic()   # marks start of current level-1 window
        self.last_seen     = time.monotonic()

        # Level-2: completed feature-window dicts (input to the Transformer)
        self.windows: List[Dict[str, float]] = []

        # Cached inference result – only updated on flush to avoid redundant runs
        self.last_prob:    float = 0.0
        self.last_verdict: str   = "BROWSING"
        self.window_idx:   int   = 0

        self._reset_frame_buffer()

    # ── Public interface ───────────────────────────────────────────────────────

    def add_observation(self, obs: Dict[str, Any]) -> None:
        """Record one frame's extracted features into the current window buffer."""
        self.last_seen = time.monotonic()
        self.frame_obs.append(obs)

    def maybe_flush(self) -> bool:
        """
        If WINDOW_SECONDS have elapsed since the last flush, aggregate all buffered
        frame observations into one window dict and append it to self.windows.
        Returns True when a flush occurred (caller should re-run inference).
        """
        now = time.monotonic()
        if now - self.window_start < WINDOW_SECONDS:
            return False
        if not self.frame_obs:
            self.window_start = now
            return False

        window = self._aggregate()
        self.windows.append(window)
        self.window_idx += 1
        self._reset_frame_buffer()
        self.window_start = now
        return True

    # ── Private helpers ────────────────────────────────────────────────────────

    def _reset_frame_buffer(self) -> None:
        self.frame_obs: List[Dict[str, Any]] = []

    def _aggregate(self) -> Dict[str, float]:
        """
        Reduce the list of per-frame observation dicts into a single 41-feature
        window dict via averaging / fraction-counting.
        """
        obs = self.frame_obs
        n   = len(obs)
        now = time.monotonic()

        def mean(key: str, default: float = 0.0) -> float:
            vals = [o[key] for o in obs if key in o]
            return sum(vals) / len(vals) if vals else default

        def frac(key: str) -> float:
            """Fraction of frames where key is truthy."""
            return sum(1 for o in obs if o.get(key)) / n

        # ── Walk stats ────────────────────────────────────────────────────────
        dwell_sec = now - self.session_start           # cumulative total dwell time

        positions = [o["position"] for o in obs if "position" in o]
        speeds: List[float] = []
        direction_changes = 0
        prev_dir: Optional[bool] = None

        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            speed = math.hypot(dx, dy)
            speeds.append(speed)
            direction = dx > 0
            if prev_dir is not None and direction != prev_dir:
                direction_changes += 1
            prev_dir = direction

        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed  = max(speeds) if speeds else 0.0
        stop_frac  = sum(1 for s in speeds if s < 2.0) / len(speeds) if speeds else 0.0

        # ── MERL action fractions – zero at live inference time ───────────────
        # The MERL labels are dataset-specific annotations unavailable from webcam.
        # The model handles all-zero action fracs gracefully (they equal a standing
        # person with no known MERL activity).
        act_counts: defaultdict = defaultdict(int)
        for o in obs:
            a = o.get("action_id")
            if a is not None:
                act_counts[int(a)] += 1
        act_fracs = {f"act_{i}_frac": act_counts.get(i, 0) / n for i in range(6)}

        # ── Pose ──────────────────────────────────────────────────────────────
        yaw_history = [o.get("head_yaw_proxy", 0.0) for o in obs]

        return {
            "dwell_sec":            dwell_sec,
            "mean_speed":           mean_speed,
            "max_speed":            max_speed,
            "direction_changes":    float(direction_changes),
            "stop_frac":            stop_frac,
            **act_fracs,
            "pose_detected":        mean("pose_detected"),
            "lean_angle":           mean("lean_angle"),
            "arm_cross":            frac("arm_cross"),
            "step_back":            frac("step_back"),
            "wrist_raised":         frac("wrist_raised"),
            "wrist_near_shelf":     frac("wrist_near_shelf"),
            "elbow_angle_l":        mean("elbow_angle_l"),
            "elbow_angle_r":        mean("elbow_angle_r"),
            "shoulder_width_norm":  mean("shoulder_width_norm"),
            "body_turned_away":     frac("body_turned_away"),
            "relaxed_posture":      frac("relaxed_posture"),
            "holding_product":      frac("holding_product"),
            "face_detected":        mean("face_detected"),
            "head_yaw_proxy":       mean("head_yaw_proxy"),
            "head_pitch_proxy":     mean("head_pitch_proxy"),
            "looking_at_product":   frac("looking_at_product"),
            "repeated_view_rate":   self._repeated_view_rate(yaw_history),
            "hands_detected":       mean("hands_detected"),
            "hand_open":            frac("hand_open"),
            "hand_pinch":           frac("hand_pinch"),
            # Emotion features stay 0 when DeepFace is disabled
            "emotion_detected":     mean("emotion_detected"),
            "emo_happy":            mean("emo_happy"),
            "emo_interested":       mean("emo_interested"),
            "emo_neutral":          mean("emo_neutral"),
            "emo_disinterested":    mean("emo_disinterested"),
            "emo_negative":         mean("emo_negative"),
            "emo_confused":         mean("emo_confused"),
            "group_size":           mean("group_size", 1.0),
            "near_others":          frac("near_others"),
            "avg_group_dist":       mean("avg_group_dist", 999.0),
        }

    @staticmethod
    def _repeated_view_rate(yaw_history: List[float]) -> float:
        """Fraction of frames where head yaw is within ±15° of centre (product fixation)."""
        if not yaw_history:
            return 0.0
        return sum(1 for y in yaw_history if abs(y) < 15) / len(yaw_history)
