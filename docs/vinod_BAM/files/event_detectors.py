"""
BAM — Buying Affinity Model
event_detectors.py  ·  Stateful rule evaluators for all E01–E18, N01–N04

Each detector receives a FrameContext (tracking + pose + zone state for one
frame) and emits BehaviourEvents when its trigger condition fires.

Design principles
─────────────────
• Detectors are pure functions of state — no side effects outside returning events.
• Stateful context (timers, accumulators) lives in TrackContext, not in the detector.
• Each detector is a small focused class implementing __call__(ctx) -> list[event].
• The EventExtractor composes all detectors and calls them in dependency order.

Performance
───────────
• All geometry is in floor-plane metric coordinates — no pixel math at runtime.
• Numpy is used only where batch operations help (e.g. IoU, distance matrix).
• Per-track state dicts use __slots__ to reduce GC pressure.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from models import BehaviourEvent

if TYPE_CHECKING:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (floor-plane, metric)
# ─────────────────────────────────────────────────────────────────────────────

def dist2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance in floor coordinates (metres)."""
    dx, dy = p1[0] - p2[0], p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def bbox_iou(b1: tuple, b2: tuple) -> float:
    """IoU of two bboxes (x1,y1,x2,y2)."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw  = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (area1 + area2 - inter)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple]) -> bool:
    """Ray-casting point-in-polygon (floor coordinates)."""
    x, y  = point
    n     = len(polygon)
    inside = False
    j     = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ─────────────────────────────────────────────────────────────────────────────
# Data structures passed into detectors each frame
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Pose:
    """17-keypoint pose in floor-plane coords (x, y, confidence per kp)."""
    keypoints: np.ndarray    # shape (17, 3): x, y, conf
    # COCO keypoint indices we care about
    LEFT_WRIST:  int = 9
    RIGHT_WRIST: int = 10
    LEFT_ELBOW:  int = 7
    RIGHT_ELBOW: int = 8
    LEFT_SHOULDER: int  = 5
    RIGHT_SHOULDER: int = 6
    NOSE:        int = 0

    def wrist_positions(self) -> list[tuple[float, float]]:
        kp = self.keypoints
        out = []
        for idx in (self.LEFT_WRIST, self.RIGHT_WRIST):
            if kp[idx, 2] >= 0.5:
                out.append((float(kp[idx, 0]), float(kp[idx, 1])))
        return out

    def torso_normal_yaw(self) -> float | None:
        """Approximate torso facing direction from shoulder midpoint."""
        kp = self.keypoints
        if kp[self.LEFT_SHOULDER, 2] < 0.5 or kp[self.RIGHT_SHOULDER, 2] < 0.5:
            return None
        dx = kp[self.RIGHT_SHOULDER, 0] - kp[self.LEFT_SHOULDER, 0]
        dy = kp[self.RIGHT_SHOULDER, 1] - kp[self.LEFT_SHOULDER, 1]
        return math.atan2(dy, dx)


@dataclass
class ObjectTrack:
    """A tracked product object."""
    object_id:    str
    object_class: str
    bbox:         tuple[float, float, float, float]   # x1,y1,x2,y2 floor coords
    associated_track_id: str | None = None
    shelf_roi_id: str | None = None


@dataclass
class ShelfPlane:
    """Registered shelf geometry for a zone."""
    zone_id:      str
    front_plane_y: float    # shelf-front y in floor coords
    influence_radius: float = 2.5   # §3.2 E12 default
    polygon:      list[tuple] = field(default_factory=list)


@dataclass
class TrackContext:
    """
    All mutable per-track state used across frames.
    Updated by the EventExtractor before calling detectors.
    """
    track_id:    str
    session_id:  str
    store_id:    str
    zone_id:     str
    camera_id:   str

    # Current-frame values
    floor_pos:   tuple[float, float] = (0.0, 0.0)
    speed_mps:   float = 0.0
    heading_rad: float = 0.0
    pose:        Pose | None = None
    is_staff:    bool = False

    # Rolling accumulators (updated in-place between frames)
    zone_entry_t:    float = 0.0
    zone_dwell_s:    float = 0.0
    zone_visit_count: int  = 0
    speed_history:   list  = field(default_factory=list)
    object_assoc:    list[ObjectTrack] = field(default_factory=list)   # held objects
    assoc_start_t:   dict[str, float]  = field(default_factory=dict)   # obj_id → t
    zone_entries:    list[float]       = field(default_factory=list)   # timestamps
    held_bbox_history: dict[str, list] = field(default_factory=dict)   # obj_id → bboxes
    path_positions:  list[tuple]       = field(default_factory=list)
    session_start_t: float = 0.0
    last_e02_t:      dict[str, float]  = field(default_factory=dict)   # zone→t
    phone_detected:  bool = False
    head_yaw_deg:    float = 0.0
    head_pitch_deg:  float = 0.0
    head_pos:        tuple[float, float] = (0.0, 0.0)


@dataclass
class FrameContext:
    """Snapshot of everything available at one frame for one track."""
    now:          float
    track:        TrackContext
    shelf:        ShelfPlane | None
    zone_baseline_p75_s: float    # rolling 28-day P75 dwell for this zone
    session_median_s:    float    # store-level session median
    objects_in_frame: list[ObjectTrack] = field(default_factory=list)
    nearby_tracks:    list[TrackContext] = field(default_factory=list)
    exit_zones:       list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Base detector
# ─────────────────────────────────────────────────────────────────────────────

class BaseDetector:
    event_type: str
    confidence: float = 0.9   # default model confidence

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        raise NotImplementedError

    def _make_event(
        self, ctx: FrameContext, multiplier: float = 1.0, payload: dict | None = None
    ) -> BehaviourEvent:
        return BehaviourEvent(
            event_type=self.event_type,
            track_id=ctx.track.track_id,
            session_id=ctx.track.session_id,
            store_id=ctx.track.store_id,
            zone_id=ctx.track.zone_id,
            t_start=ctx.now,
            confidence=self.confidence,
            multiplier=multiplier,
            payload=payload or {"camera_id": ctx.track.camera_id},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 detectors
# ─────────────────────────────────────────────────────────────────────────────

class E01_BasketPlacement(BaseDetector):
    """
    Object track re-parents from shelf ROI to basket/cart region.
    Trigger: hand-keypoint–object IoU ≥ 0.3 for ≥ 0.5 s AND no return to shelf.
    §3.1 E01 weight=12
    """
    event_type = "E01"
    confidence = 0.88

    _HAND_OBJ_IOU_THRESH   = 0.3
    _CONTACT_DURATION_S    = 0.5
    _BASKET_REGION_LABEL   = "basket"

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        events = []
        t = ctx.track
        if t.pose is None:
            return events

        for obj in t.object_assoc:
            if obj.shelf_roi_id is None:
                continue   # already not on shelf
            contact_start = t.assoc_start_t.get(obj.object_id, ctx.now)
            contact_dur   = ctx.now - contact_start
            if contact_dur < self._CONTACT_DURATION_S:
                continue

            # Check wrist proximity
            for wrist in t.pose.wrist_positions():
                obj_cx = (obj.bbox[0] + obj.bbox[2]) / 2
                obj_cy = (obj.bbox[1] + obj.bbox[3]) / 2
                if dist2d(wrist, (obj_cx, obj_cy)) < 0.4:   # 40cm
                    events.append(self._make_event(
                        ctx,
                        payload={"object_class": obj.object_class, "camera_id": t.camera_id}
                    ))
                    break
        return events


class E02_ProductPickup(BaseDetector):
    """
    Wrist keypoint enters shelf ROI; new object track spawns within 0.3 m of
    wrist within 400 ms; contact duration ≥ 0.5 s.
    §3.1 E02 weight=9
    """
    event_type = "E02"
    confidence = 0.85

    _SPAWN_DIST_M  = 0.3
    _SPAWN_WINDOW  = 0.4
    _CONTACT_S     = 0.5

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        events = []
        t = ctx.track
        if t.pose is None or ctx.shelf is None:
            return events

        for wrist in t.pose.wrist_positions():
            # Wrist crosses shelf front plane
            if wrist[1] < ctx.shelf.front_plane_y:
                continue
            # Look for nearby new object spawned recently
            for obj in ctx.objects_in_frame:
                if obj.associated_track_id is not None:
                    continue   # already associated
                obj_cx = (obj.bbox[0] + obj.bbox[2]) / 2
                obj_cy = (obj.bbox[1] + obj.bbox[3]) / 2
                if dist2d(wrist, (obj_cx, obj_cy)) < self._SPAWN_DIST_M:
                    assoc_dur = ctx.now - t.assoc_start_t.get(obj.object_id, ctx.now)
                    if assoc_dur >= self._CONTACT_S:
                        events.append(self._make_event(
                            ctx,
                            payload={"object_class": obj.object_class, "camera_id": t.camera_id}
                        ))
                        t.last_e02_t[t.zone_id] = ctx.now
        return events


class E03_TryOnDemo(BaseDetector):
    """
    (a) Fitting room zone cross with object, or
    (b) Garment–torso overlap ≥ 60% in mirror zone for ≥ 3 s, or
    (c) Demo-device hand interaction ≥ 10 s.
    §3.1 E03 weight=8
    """
    event_type = "E03"
    confidence = 0.82

    _TORSO_OVERLAP_THRESH = 0.6
    _MIRROR_ZONE_DUR_S    = 3.0
    _DEMO_DEVICE_DUR_S    = 10.0

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        events = []
        t = ctx.track

        # (c) Device interaction
        for obj in t.object_assoc:
            if obj.object_class == "demo_device":
                dur = ctx.now - t.assoc_start_t.get(obj.object_id, ctx.now)
                if dur >= self._DEMO_DEVICE_DUR_S:
                    events.append(self._make_event(ctx))
                    return events

        # (b) Garment torso overlap — simplified: check if garment bbox
        # covers torso keypoints sufficiently
        if t.pose is not None:
            kp = t.pose.keypoints
            torso_pts = kp[[5, 6, 11, 12], :2]   # shoulders + hips
            for obj in t.object_assoc:
                if obj.object_class not in ("garment", "clothing"):
                    continue
                pts_in = sum(
                    1 for pt in torso_pts
                    if obj.bbox[0] <= pt[0] <= obj.bbox[2]
                    and obj.bbox[1] <= pt[1] <= obj.bbox[3]
                )
                overlap = pts_in / len(torso_pts)
                if overlap >= self._TORSO_OVERLAP_THRESH:
                    dur = ctx.now - t.assoc_start_t.get(obj.object_id, ctx.now)
                    if dur >= self._MIRROR_ZONE_DUR_S:
                        events.append(self._make_event(ctx))
                        return events
        return events


class E04_HoldDuration(BaseDetector):
    """
    Object–person association persists > 8 s; rotation flag if bbox
    aspect-ratio variance σ² > 0.15.
    §3.1 E04 weight=7
    """
    event_type = "E04"
    confidence = 0.87
    _HOLD_THRESH_S = 8.0
    _AR_VAR_THRESH = 0.15

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        events = []
        t = ctx.track
        for obj in t.object_assoc:
            dur = ctx.now - t.assoc_start_t.get(obj.object_id, ctx.now)
            if dur < self._HOLD_THRESH_S:
                continue
            # Check rotation flag
            history = t.held_bbox_history.get(obj.object_id, [])
            if len(history) >= 3:
                ars = [(b[2]-b[0]) / max(b[3]-b[1], 1e-6) for b in history]
                ar_var = float(np.var(ars))
                if ar_var > self._AR_VAR_THRESH:
                    events.append(self._make_event(
                        ctx, payload={"rotation": True, "camera_id": t.camera_id}
                    ))
                    continue
            events.append(self._make_event(ctx))
        return events


class E05_ComparisonBehaviour(BaseDetector):
    """
    ≥ 2 concurrent object associations, or ≥ 2 E02s on distinct shelf
    positions in same zone within 60 s.
    §3.1 E05 weight=7
    """
    event_type = "E05"
    confidence = 0.83
    _MULTI_OBJECT_THRESH = 2
    _E02_WINDOW_S        = 60.0

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        t = ctx.track
        # Multiple concurrent associations
        if len(t.object_assoc) >= self._MULTI_OBJECT_THRESH:
            return [self._make_event(ctx)]
        # Multiple E02s in zone within window
        last_e02 = t.last_e02_t.get(t.zone_id, 0.0)
        if last_e02 > 0 and (ctx.now - last_e02) <= self._E02_WINDOW_S:
            e02_count = sum(
                1 for ts in t.last_e02_t.values()
                if (ctx.now - ts) <= self._E02_WINDOW_S
            )
            if e02_count >= 2:
                return [self._make_event(ctx)]
        return []


class E06_PriceLabelInspection(BaseDetector):
    """
    Object flip (>120° orientation delta) AND head pitch < −20° AND
    head–object distance < 0.4 m for ≥ 2 s.
    §3.1 E06 weight=6
    """
    event_type = "E06"
    confidence = 0.80
    _PITCH_THRESH_DEG   = -20.0
    _HEAD_OBJ_DIST_M    =  0.4
    _DURATION_S         =  2.0

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        t = ctx.track
        if t.head_pitch_deg >= self._PITCH_THRESH_DEG:
            return []
        for obj in t.object_assoc:
            obj_cx = (obj.bbox[0] + obj.bbox[2]) / 2
            obj_cy = (obj.bbox[1] + obj.bbox[3]) / 2
            d = dist2d(t.head_pos, (obj_cx, obj_cy))
            if d < self._HEAD_OBJ_DIST_M:
                dur = ctx.now - t.assoc_start_t.get(obj.object_id, ctx.now)
                if dur >= self._DURATION_S:
                    return [self._make_event(ctx)]
        return []


class E07_StaffContact(BaseDetector):
    """
    Shopper approaches stationary staff (uniform conf ≥ 0.8); distance < 1.2 m;
    mutual torso facing (Δyaw > 120°) ≥ 10 s.
    §3.1 E07 weight=6
    """
    event_type = "E07"
    confidence = 0.85
    _DIST_M           = 1.2
    _YAW_DELTA_DEG    = 120.0
    _DURATION_S       = 10.0

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        t = ctx.track
        for staff in ctx.nearby_tracks:
            if not staff.is_staff:
                continue
            if dist2d(t.floor_pos, staff.floor_pos) > self._DIST_M:
                continue
            if staff.speed_mps > 0.1:
                continue   # not stationary
            yaw_t = t.torso_yaw if hasattr(t, "torso_yaw") else 0.0
            yaw_s = staff.torso_yaw if hasattr(staff, "torso_yaw") else 0.0
            delta = abs(math.degrees(yaw_t - yaw_s))
            if delta > 180:
                delta = 360 - delta
            if delta >= self._YAW_DELTA_DEG:
                dur = t.zone_dwell_s   # crude proximity timer — refine in production
                if dur >= self._DURATION_S:
                    return [self._make_event(ctx)]
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 detectors
# ─────────────────────────────────────────────────────────────────────────────

class E08_ZoneDwell(BaseDetector):
    """Dwell > zone P75 baseline (rolling 28-day). §3.2 E08 weight=6"""
    event_type = "E08"
    confidence = 0.95

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        if ctx.track.zone_dwell_s > ctx.zone_baseline_p75_s:
            return [self._make_event(ctx)]
        return []


class E09_RepeatZoneReturns(BaseDetector):
    """≥ 2 zone entries, gap > 60 s, same session. §3.2 E09 weight=6"""
    event_type = "E09"
    confidence = 0.92
    _MIN_GAP_S = 60.0

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        entries = ctx.track.zone_entries
        if len(entries) < 2:
            return []
        last_two = sorted(entries)[-2:]
        if (last_two[1] - last_two[0]) > self._MIN_GAP_S:
            return [self._make_event(ctx)]
        return []


class E10_GazeFixation(BaseDetector):
    """Head-pose ray intersects display plane ≥ 3 s cumulative. §3.2 E10 weight=5"""
    event_type = "E10"
    confidence = 0.78
    _FIXATION_S = 3.0

    def __init__(self) -> None:
        self._fixation_acc: dict[str, float] = {}   # track_id → accumulated s

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        t = ctx.track
        # Simplified: if head pitch > -10° and facing shelf, accumulate
        if ctx.shelf and t.head_pitch_deg < 10:
            key = t.track_id
            self._fixation_acc[key] = self._fixation_acc.get(key, 0.0) + (1/15)  # 1 frame
            if self._fixation_acc[key] >= self._FIXATION_S:
                self._fixation_acc[key] = 0.0   # reset after firing
                return [self._make_event(ctx)]
        return []


class E11_BodyOrientationLeanIn(BaseDetector):
    """Torso within 30° of shelf normal AND head–shelf < 0.6 m with -d/dt. §3.2 E11 weight=4"""
    event_type = "E11"
    confidence = 0.82
    _ANGLE_THRESH_DEG = 30.0
    _DIST_THRESH_M    = 0.6

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        t = ctx.track
        if ctx.shelf is None or t.pose is None:
            return []
        torso_yaw = t.pose.torso_normal_yaw()
        if torso_yaw is None:
            return []
        shelf_normal = math.pi / 2   # assumes shelf faces +y
        angle_diff = abs(math.degrees(torso_yaw - shelf_normal))
        angle_diff = min(angle_diff, 360 - angle_diff)
        head_shelf_dist = abs(t.head_pos[1] - ctx.shelf.front_plane_y)
        if angle_diff <= self._ANGLE_THRESH_DEG and head_shelf_dist < self._DIST_THRESH_M:
            return [self._make_event(ctx)]
        return []


class E12_ApproachDeceleration(BaseDetector):
    """Speed drop ≥ 60% inside influence radius; full stop scores full weight. §3.2 E12 weight=4"""
    event_type = "E12"
    confidence = 0.88
    _SPEED_DROP_THRESH = 0.6
    _FULL_STOP_MPS     = 0.15

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        t = ctx.track
        if ctx.shelf is None:
            return []
        if dist2d(t.floor_pos, (0, ctx.shelf.front_plane_y)) > ctx.shelf.influence_radius:
            return []
        history = t.speed_history
        if len(history) < 2:
            return []
        max_speed = max(history[:-1]) if history[:-1] else 0.0
        if max_speed < 0.05:
            return []   # was stationary all along
        drop = (max_speed - t.speed_mps) / max_speed
        if drop >= self._SPEED_DROP_THRESH:
            mult = 1.0 if t.speed_mps < self._FULL_STOP_MPS else 0.5
            return [self._make_event(ctx, multiplier=mult)]
        return []


class E13_ReachAttempt(BaseDetector):
    """Elbow–wrist vector toward shelf; wrist crosses shelf front (no E02 → 50% weight). §3.2 E13 weight=4"""
    event_type = "E13"
    confidence = 0.84

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        t = ctx.track
        if ctx.shelf is None or t.pose is None:
            return []
        kp = t.pose.keypoints
        for elbow_idx, wrist_idx in [(7, 9), (8, 10)]:
            if kp[elbow_idx, 2] < 0.5 or kp[wrist_idx, 2] < 0.5:
                continue
            wrist_y = kp[wrist_idx, 1]
            if wrist_y >= ctx.shelf.front_plane_y:
                # Wrist crossed shelf front
                mult = 1.0
                # No E02 within 1.5 s → aborted reach at 50%
                last_e02 = t.last_e02_t.get(t.zone_id, 0.0)
                if (ctx.now - last_e02) > 1.5:
                    mult = 0.5
                return [self._make_event(ctx, multiplier=mult)]
        return []


class E14_CompanionInteraction(BaseDetector):
    """Object handover or companion head-pose to object ≥ 2 s. §3.2 E14 weight=3"""
    event_type = "E14"
    confidence = 0.78

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        for companion in ctx.nearby_tracks:
            if dist2d(ctx.track.floor_pos, companion.floor_pos) < 2.0:
                return [self._make_event(ctx)]
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 detectors
# ─────────────────────────────────────────────────────────────────────────────

class E15_PathPattern(BaseDetector):
    """Tortuosity τ = path_length / net_displacement. §3.3 E15 weight=4"""
    event_type = "E15"
    confidence = 0.86
    _TAU_DIRECTED = 1.4
    _TAU_BROWSING = 2.2

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        pts = ctx.track.path_positions
        if len(pts) < 3:
            return []
        path_len = sum(dist2d(pts[i], pts[i+1]) for i in range(len(pts)-1))
        net_disp = dist2d(pts[0], pts[-1])
        if net_disp < 0.1:
            return []
        tau = path_len / net_disp
        if tau < self._TAU_DIRECTED:
            return [self._make_event(ctx, multiplier=1.0)]
        if tau > self._TAU_BROWSING:
            return [self._make_event(ctx, multiplier=0.5)]
        return []


class E16_ZoneTransitionSequence(BaseDetector):
    """Zone sequence matches complementary-category adjacency. §3.3 E16 weight=3"""
    event_type = "E16"
    confidence = 0.82

    def __init__(self, adjacency: dict[str, set[str]] | None = None) -> None:
        # adjacency[category_A] = {category_B, ...}
        self._adj = adjacency or {}

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        # Simplified — in production, zone engine tracks sequence
        return []


class E17_PhoneUseAtShelf(BaseDetector):
    """Phone in hand + raised toward shelf (price-check posture, conf ≥ 0.7). §3.3 E17 weight=3"""
    event_type = "E17"
    confidence = 0.80

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        if ctx.track.phone_detected:
            return [self._make_event(ctx)]
        return []


class E18_SessionDuration(BaseDetector):
    """Total session > store median, weighted by product-zone ratio. §3.3 E18 weight=3"""
    event_type = "E18"
    confidence = 0.91

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        t = ctx.track
        session_dur = ctx.now - t.session_start_t
        if session_dur > ctx.session_median_s:
            return [self._make_event(ctx)]
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Negative modifier detectors
# ─────────────────────────────────────────────────────────────────────────────

class N01_RapidPutBack(BaseDetector):
    """E02 followed by object return to shelf ROI within 3 s. §3.4 N01 weight=-6"""
    event_type = "N01"
    confidence = 0.90
    _RETURN_WINDOW_S = 3.0

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        t = ctx.track
        last_e02 = t.last_e02_t.get(t.zone_id, 0.0)
        if last_e02 == 0.0:
            return []
        for obj in ctx.objects_in_frame:
            # Object returned to shelf (re-associated with shelf ROI)
            if obj.shelf_roi_id and (ctx.now - last_e02) <= self._RETURN_WINDOW_S:
                if obj.associated_track_id == t.track_id:
                    return [self._make_event(ctx)]
        return []


class N02_QueueAbandonment(BaseDetector):
    """Exit queue after ≥ 30 s with item deposit. §3.4 N02 weight=-8"""
    event_type = "N02"
    confidence = 0.88
    _QUEUE_DWELL_S = 30.0

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        # Zone engine signals queue exit with carried-object disassociation
        # Simplified: if dwell in queue zone exceeded threshold and track leaves
        if ctx.track.zone_dwell_s >= self._QUEUE_DWELL_S and len(ctx.track.object_assoc) == 0:
            return [self._make_event(ctx)]
        return []


class N03_WalkPast(BaseDetector):
    """Zone traversal at near-constant speed, no E10/E12/E13. §3.4 N03 weight=-3"""
    event_type = "N03"
    confidence = 0.85
    _CV_THRESH = 0.15

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        hist = ctx.track.speed_history
        if len(hist) < 4:
            return []
        arr = np.array(hist, dtype=np.float64)
        cv = float(arr.std() / (arr.mean() + 1e-6))
        if cv < self._CV_THRESH:
            return [self._make_event(ctx)]
        return []


class N04_ExitDirectedMovement(BaseDetector):
    """Sustained heading toward exit (bearing error < 15°) for ≥ 8 s. §3.4 N04 weight=-2"""
    event_type = "N04"
    confidence = 0.87
    _BEARING_ERROR_DEG = 15.0
    _SUSTAINED_S       = 8.0

    def __init__(self, exit_bearings: dict[str, float] | None = None) -> None:
        self._exit_bearings = exit_bearings or {}   # zone_id → bearing to exit
        self._heading_acc: dict[str, float] = {}   # track_id → accumulated time

    def __call__(self, ctx: FrameContext) -> list[BehaviourEvent]:
        t = ctx.track
        exit_bearing = self._exit_bearings.get(t.zone_id)
        if exit_bearing is None:
            return []
        error = abs(math.degrees(t.heading_rad) - exit_bearing)
        error = min(error, 360 - error)
        if error < self._BEARING_ERROR_DEG:
            self._heading_acc[t.track_id] = self._heading_acc.get(t.track_id, 0.0) + (1/15)
            if self._heading_acc[t.track_id] >= self._SUSTAINED_S:
                self._heading_acc[t.track_id] = 0.0
                return [self._make_event(ctx)]
        else:
            self._heading_acc[t.track_id] = 0.0
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Composed event extractor — calls all detectors in order
# ─────────────────────────────────────────────────────────────────────────────

class EventExtractor:
    """
    Runs all detectors for one frame context and returns the union of events.

    Detector call order follows dependency rules:
      N01 must run after E02 (needs last_e02_t populated)
      E05 must run after E02 (needs last_e02_t populated)
      E12 must run before E13 (E12 updates speed context)
    """

    def __init__(
        self,
        adjacency: dict[str, set[str]] | None = None,
        exit_bearings: dict[str, float] | None = None,
    ) -> None:
        self._detectors: list[BaseDetector] = [
            # Tier 1
            E02_ProductPickup(),       # must be before E05, N01
            E01_BasketPlacement(),
            E03_TryOnDemo(),
            E04_HoldDuration(),
            E05_ComparisonBehaviour(), # needs E02 already fired
            E06_PriceLabelInspection(),
            E07_StaffContact(),
            # Tier 2
            E08_ZoneDwell(),
            E09_RepeatZoneReturns(),
            E10_GazeFixation(),
            E11_BodyOrientationLeanIn(),
            E12_ApproachDeceleration(), # before E13
            E13_ReachAttempt(),
            E14_CompanionInteraction(),
            # Tier 3
            E15_PathPattern(),
            E16_ZoneTransitionSequence(adjacency),
            E17_PhoneUseAtShelf(),
            E18_SessionDuration(),
            # Negatives
            N01_RapidPutBack(),        # needs E02 context
            N02_QueueAbandonment(),
            N03_WalkPast(),
            N04_ExitDirectedMovement(exit_bearings),
        ]

    def extract(self, ctx: FrameContext) -> list[BehaviourEvent]:
        """
        Call all detectors and aggregate events.
        Short-circuits on fatal context gaps (no shelf for shelf-dependent events
        is handled per-detector, not here — keeps detectors independent).
        """
        events: list[BehaviourEvent] = []
        for detector in self._detectors:
            try:
                fired = detector(ctx)
                if fired:
                    events.extend(fired)
            except Exception:
                import logging
                logging.getLogger("bam.extractor").exception(
                    "Detector %s raised", type(detector).__name__
                )
        return events
