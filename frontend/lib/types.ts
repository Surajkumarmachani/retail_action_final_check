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

export interface Prediction {
  id:        number;
  box:       [number, number, number, number];
  prob:      number;
  verdict:   "LIKELY BUY" | "BROWSING";
  windows:   number;
  dwell_sec?: number;
  metrics?:  PersonMetrics;
}

export interface ProductDetection {
  box:      [number, number, number, number];
  class_id: number;
  label:    string;
  conf:     number;
}

export interface ServerMessage {
  type:      "predictions" | "auth_ok" | "auth_error";
  data?:     Prediction[];
  products?: ProductDetection[];
  uid?:      string;
  detail?:   string;
}
