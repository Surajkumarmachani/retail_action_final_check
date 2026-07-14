export interface PersonMetrics {
  pose_detected: boolean;
  lean_angle: number;
  arm_cross: boolean;
  wrist_raised: boolean;
  wrist_near_shelf: boolean;
  holding_product: boolean;
  relaxed_posture: boolean;
  body_turned_away: boolean;
  step_back: boolean;
  face_detected: boolean;
  looking_at_product: boolean;
  head_yaw_proxy: number;
  head_pitch_proxy: number;
  hands_detected: number;
  hand_open: boolean;
  hand_pinch: boolean;
  near_others: boolean;
  group_size: number;
}

export type BAMBand = "HOT" | "WARM" | "COLD";

/** One COCO keypoint: [x_norm, y_norm, confidence] in normalised (0-1) coords */
export type Keypoint = [number, number, number];

/** Posture/tilt angles (degrees) derived from pose keypoints on the backend */
export interface PostureAngles {
  shoulder_align?: number;
  torso_tilt?:     number;
  spine_angle?:    number;
  gaze_angle?:     number;
}

export interface Prediction {
  id:          number;
  box:         [number, number, number, number];
  prob:        number;                 // bas / 100 (backward-compat)
  verdict:     "LIKELY BUY" | "BROWSING";
  dwell_sec?:  number;
  metrics?:    PersonMetrics;
  // BAM fields
  bas?:        number;                 // Buying Affinity Score 0-100
  band?:       BAMBand;
  keypoints?:  Keypoint[];             // 17 COCO keypoints, normalised
  posture?:    PostureAngles;          // tilt/alignment angles (degrees)
  top_events?: string[];              // recent event codes (E08, E10, …)
  is_staff?:   boolean;
  windows?:    number;
}

export interface ProductDetection {
  box:      [number, number, number, number];
  class_id: number;
  label:    string;
  conf:     number;
}

export interface BAMEvent {
  track_id: number;
  type:     string;    // E01–E18, N01–N04
  label:    string;    // human-readable
  weight:   number;    // signed contribution
  conf:     number;
  ts:       number;
}

export interface ServerMessage {
  type:      "predictions" | "auth_ok" | "auth_error";
  data?:     Prediction[];
  products?: ProductDetection[];
  events?:   BAMEvent[];
  uid?:      string;
  detail?:   string;
}
