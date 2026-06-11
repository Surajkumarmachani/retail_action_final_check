# constants.py  –  single source of truth for all hyperparameters + feature layout

# ── 41-feature column order (MUST match training) ─────────────────────────────
FEATURE_COLS: list[str] = [
    # Walk stats (5)
    "dwell_sec", "mean_speed", "max_speed", "direction_changes", "stop_frac",
    # MERL action fractions (6) – zeroed in live inference (no MERL labels at runtime)
    "act_0_frac", "act_1_frac", "act_2_frac", "act_3_frac", "act_4_frac", "act_5_frac",
    # Pose (12)
    "pose_detected", "lean_angle", "arm_cross", "step_back", "wrist_raised",
    "wrist_near_shelf", "elbow_angle_l", "elbow_angle_r", "shoulder_width_norm",
    "body_turned_away", "relaxed_posture", "holding_product",
    # Head / gaze (5)
    "face_detected", "head_yaw_proxy", "head_pitch_proxy", "looking_at_product",
    "repeated_view_rate",
    # Hands (3)
    "hands_detected", "hand_open", "hand_pinch",
    # Emotion (7) – zeroed when DeepFace is disabled; vector stays length-41
    "emotion_detected", "emo_happy", "emo_interested", "emo_neutral",
    "emo_disinterested", "emo_negative", "emo_confused",
    # Social (3)
    "group_size", "near_others", "avg_group_dist",
]

assert len(FEATURE_COLS) == 41, "Feature column count must be exactly 41"

# ── BehaviourTransformer architecture ─────────────────────────────────────────
N_FEATS:  int   = 41
MAX_SEQ:  int   = 24
D_MODEL:  int   = 128
N_HEADS:  int   = 4
N_LAYERS: int   = 4
D_FF:     int   = 256
DROPOUT:  float = 0.15

# ── Inference thresholds ───────────────────────────────────────────────────────
PURCHASE_THRESHOLD: float = 0.725   # Youden's J optimum from training

# ── CV pipeline ───────────────────────────────────────────────────────────────
SAMPLE_EVERY: int   = 3          # run MediaPipe / DeepFace every Nth frame
WINDOW_SECONDS: float = 5.0      # wall-clock seconds per behavioural window
GC_TIMEOUT_SECONDS: float = 2.0  # garbage-collect tracks unseen this long

# ── Streaming ─────────────────────────────────────────────────────────────────
STREAM_FPS: int = 15             # frames-per-second the frontend targets
