"""
BAM — Buying Affinity Model
models.py  ·  Canonical data structures

All hot-path objects are __slots__ classes to eliminate per-instance __dict__
overhead and improve cache locality at high throughput.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final, NamedTuple

# ─────────────────────────────────────────────────────────────────────────────
# Constants — never magic-numbered elsewhere
# ─────────────────────────────────────────────────────────────────────────────

DECAY_HALF_LIFE_S: Final[float] = 270.0          # §4.1 — configurable per store
CONFIDENCE_GATE: Final[float]   = 0.70           # §4.2 hard gate v1.0
ZONE_CAP_FACTOR: Final[float]   = 1.5            # §4.2 per-zone cap multiplier
LATE_ACCEPT_WINDOW_S: Final[float] = 10.0        # §4.2 back-date window
BAS_MAX: Final[float]           = 100.0
BAS_MIN: Final[float]           = 0.0

# Band thresholds (§4.4)
BAND_HOT_THRESHOLD: Final[float]  = 70.0
BAND_WARM_THRESHOLD: Final[float] = 40.0
BAND_HOT_EXPIRE: Final[float]     = 60.0         # alert auto-expires below 60

# ─────────────────────────────────────────────────────────────────────────────
# Event weights — single source of truth (§3)
# ─────────────────────────────────────────────────────────────────────────────

EVENT_WEIGHTS: Final[dict[str, float]] = {
    # Tier 1 — Direct Intent (Σ=55)
    "E01": 12.0,  # Basket/cart placement
    "E02":  9.0,  # Product pick-up
    "E03":  8.0,  # Try-on / demo engagement
    "E04":  7.0,  # Hold duration
    "E05":  7.0,  # Comparison behaviour
    "E06":  6.0,  # Price/label inspection
    "E07":  6.0,  # Shopper-initiated staff contact
    # Tier 2 — Engagement & Attention (Σ=32)
    "E08":  6.0,  # Zone dwell time
    "E09":  6.0,  # Repeat zone returns
    "E10":  5.0,  # Gaze fixation
    "E11":  4.0,  # Body orientation & lean-in
    "E12":  4.0,  # Approach deceleration
    "E13":  4.0,  # Reach attempts
    "E14":  3.0,  # Companion interaction
    # Tier 3 — Journey & Context (Σ=13)
    "E15":  4.0,  # Path pattern
    "E16":  3.0,  # Zone transition sequence
    "E17":  3.0,  # Phone use at shelf
    "E18":  3.0,  # Session duration vs baseline
    # Negative modifiers (§3.4)
    "N01": -6.0,  # Rapid put-back
    "N02": -8.0,  # Queue abandonment
    "N03": -3.0,  # Walk-past
    "N04": -2.0,  # Exit-directed movement
}

assert sum(w for w in EVENT_WEIGHTS.values() if w > 0) == 100.0, \
    "Positive event weights must sum to 100"

POSITIVE_EVENTS: Final[frozenset[str]] = frozenset(
    k for k, v in EVENT_WEIGHTS.items() if v > 0
)
NEGATIVE_EVENTS: Final[frozenset[str]] = frozenset(
    k for k, v in EVENT_WEIGHTS.items() if v < 0
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-format weight deltas (§6.2)
# ─────────────────────────────────────────────────────────────────────────────

FORMAT_DELTAS: Final[dict[str, dict[str, float]]] = {
    "fashion":     {"E03": +2, "E06": +1, "E17": -1, "E15": -2},
    "electronics": {"E03": +3, "E07": +2, "E08": -2, "E14": -3},
    "grocery":     {"E08": +2, "E15": +2, "E03": -3, "E06": +1, "E10": -2},  # net Δ=0
    "lifestyle":   {},  # tuned after first validation cycle
}


def build_weights(store_format: str) -> dict[str, float]:
    """Return final per-store weights with format deltas applied and validated."""
    weights = dict(EVENT_WEIGHTS)
    deltas = FORMAT_DELTAS.get(store_format, {})
    for event_id, delta in deltas.items():
        assert abs(delta) <= 3, f"Delta {delta} for {event_id} exceeds ±3 bound"
        weights[event_id] = weights[event_id] + delta
    pos_sum = sum(w for w in weights.values() if w > 0)
    assert abs(pos_sum - 100.0) < 1e-6, \
        f"Weight deltas broke Σ=100 invariant: got {pos_sum}"
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Score band classification
# ─────────────────────────────────────────────────────────────────────────────

class ScoreBand(IntEnum):
    COLD = 0
    WARM = 1
    HOT  = 2

    @classmethod
    def from_score(cls, score: float) -> "ScoreBand":
        if score >= BAND_HOT_THRESHOLD:
            return cls.HOT
        if score >= BAND_WARM_THRESHOLD:
            return cls.WARM
        return cls.COLD


# ─────────────────────────────────────────────────────────────────────────────
# Core event record — __slots__ for memory efficiency on hot path
# ─────────────────────────────────────────────────────────────────────────────

class BehaviourEvent:
    """
    Canonical event emitted by the event extraction layer (L4).

    __slots__ eliminates __dict__ overhead — critical when millions of events
    flow through per day across all cameras.
    """
    __slots__ = (
        "event_id", "event_type", "track_id", "session_id",
        "store_id", "zone_id", "t_start", "t_end",
        "confidence", "multiplier", "payload",
    )

    def __init__(
        self,
        event_type: str,
        track_id: str,
        session_id: str,
        store_id: str,
        zone_id: str,
        t_start: float,
        confidence: float,
        multiplier: float = 1.0,
        t_end: float | None = None,
        payload: dict | None = None,
    ) -> None:
        self.event_type  = event_type
        self.track_id    = track_id
        self.session_id  = session_id
        self.store_id    = store_id
        self.zone_id     = zone_id
        self.t_start     = t_start
        self.t_end       = t_end if t_end is not None else t_start
        self.confidence  = confidence
        self.multiplier  = multiplier
        self.payload     = payload or {}
        # Deterministic ID — §4.2 idempotency
        # Bucket to 1-second granularity so repeated fires in same second dedup
        t_bucket = int(t_start)
        raw = f"{track_id}:{event_type}:{zone_id}:{t_bucket}"
        self.event_id = hashlib.blake2b(
            raw.encode(), digest_size=16
        ).hexdigest()

    def __repr__(self) -> str:
        return (
            f"BehaviourEvent({self.event_type} track={self.track_id} "
            f"zone={self.zone_id} conf={self.confidence:.2f} "
            f"mult={self.multiplier:.1f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ledger entry — what the scoring engine stores per accepted event
# ─────────────────────────────────────────────────────────────────────────────

class LedgerEntry(NamedTuple):
    """Immutable entry appended to a track's decay ledger."""
    t_event:  float   # Unix timestamp of event
    contrib:  float   # weight × multiplier (already capped)
    event_id: str     # for dedup audit


# ─────────────────────────────────────────────────────────────────────────────
# Score result returned by the scoring engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ScoreResult:
    track_id:    str
    session_id:  str
    store_id:    str
    bas:         float
    band:        ScoreBand
    t_scored:    float
    top_events:  list[str] = field(default_factory=list)  # top-3 event types

    @property
    def band_name(self) -> str:
        return self.band.name

    def to_dict(self) -> dict:
        return {
            "track_id":   self.track_id,
            "session_id": self.session_id,
            "store_id":   self.store_id,
            "bas":        round(self.bas, 2),
            "band":       self.band_name,
            "t_scored":   self.t_scored,
            "top_events": self.top_events,
        }
