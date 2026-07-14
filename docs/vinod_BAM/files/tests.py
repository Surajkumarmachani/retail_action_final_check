"""
BAM — Buying Affinity Model
tests.py  ·  Correctness + performance test suite

Tests cover
───────────
  1. Weight invariants (Σ=100, format deltas bounded ±3)
  2. Scoring formula — decay, clamp, λ
  3. Confidence gate (events below 0.70 never scored)
  4. Idempotency (replay never double-scores)
  5. Zone cap (1.5× weight ceiling)
  6. Late-event rejection (> 10 s)
  7. Score bands (HOT/WARM/COLD thresholds)
  8. Session eviction (privacy compliance)
  9. Parallel throughput (score_all_tracks latency)
 10. Event extractor wiring (all 22 detector types instantiate)
"""

from __future__ import annotations

import math
import sys
import time
import unittest
from pathlib import Path

# Add the workspace root to path so the tests run from any checkout location.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (
    EVENT_WEIGHTS, BehaviourEvent, ScoreBand, LedgerEntry,
    build_weights, FORMAT_DELTAS, DECAY_HALF_LIFE_S,
    CONFIDENCE_GATE, ZONE_CAP_FACTOR,
)
from scoring_engine import BAMScoringEngine


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_event(
    event_type: str = "E02",
    track_id: str   = "trk_001",
    confidence: float = 0.90,
    multiplier: float = 1.0,
    zone_id: str    = "Z-TEST-01",
    t_offset_s: float = 0.0,   # offset from now (negative = in the past)
) -> BehaviourEvent:
    return BehaviourEvent(
        event_type=event_type,
        track_id=track_id,
        session_id="ses_001",
        store_id="RGL-TEST-01",
        zone_id=zone_id,
        t_start=time.time() + t_offset_s,
        confidence=confidence,
        multiplier=multiplier,
    )


def fresh_engine(store_format: str = "lifestyle") -> BAMScoringEngine:
    return BAMScoringEngine(store_format=store_format, max_workers=4)


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightInvariants(unittest.TestCase):

    def test_positive_weights_sum_100(self):
        pos_sum = sum(w for w in EVENT_WEIGHTS.values() if w > 0)
        self.assertAlmostEqual(pos_sum, 100.0, places=6)

    def test_event_count(self):
        positives = [k for k in EVENT_WEIGHTS if k.startswith("E")]
        negatives = [k for k in EVENT_WEIGHTS if k.startswith("N")]
        self.assertEqual(len(positives), 18, "Must have exactly 18 positive events")
        self.assertEqual(len(negatives),  4, "Must have exactly 4 negative modifiers")

    def test_format_deltas_bounded(self):
        for fmt, deltas in FORMAT_DELTAS.items():
            for eid, delta in deltas.items():
                self.assertLessEqual(abs(delta), 3,
                    f"{fmt}/{eid} delta {delta} exceeds ±3")

    def test_build_weights_fashion(self):
        w = build_weights("fashion")
        pos_sum = sum(v for v in w.values() if v > 0)
        self.assertAlmostEqual(pos_sum, 100.0, places=6)

    def test_build_weights_electronics(self):
        w = build_weights("electronics")
        pos_sum = sum(v for v in w.values() if v > 0)
        self.assertAlmostEqual(pos_sum, 100.0, places=6)

    def test_build_weights_grocery(self):
        w = build_weights("grocery")
        pos_sum = sum(v for v in w.values() if v > 0)
        self.assertAlmostEqual(pos_sum, 100.0, places=6)


class TestScoringFormula(unittest.TestCase):

    def test_single_event_no_decay(self):
        """E02 (weight=9) emitted now should produce ~9 BAS."""
        engine = fresh_engine()
        ev = make_event("E02", t_offset_s=0)
        engine.on_event(ev)
        result = engine.score("trk_001", "ses_001", "RGL-TEST-01")
        self.assertAlmostEqual(result.bas, 9.0, delta=0.2)

    def test_decay_half_life(self):
        """
        Directly validate λ: inject an event 5 s ago and verify the score
        matches weight × exp(-λ×5).  Cannot use 270 s because the 10 s
        late-acceptance window rejects events that old.
        """
        engine = fresh_engine()
        age_s = 5.0
        ev = make_event("E01", t_offset_s=-age_s)
        engine.on_event(ev)
        result = engine.score("trk_001", "ses_001", "RGL-TEST-01")
        lam = math.log(2) / DECAY_HALF_LIFE_S
        expected = EVENT_WEIGHTS["E01"] * math.exp(-lam * age_s)
        self.assertAlmostEqual(result.bas, expected, delta=0.3)

    def test_score_clamped_to_100(self):
        """Stacking many events should not exceed BAS=100."""
        engine = fresh_engine()
        for etype, weight in EVENT_WEIGHTS.items():
            if weight > 0:
                for i in range(5):
                    ev = make_event(
                        etype,
                        track_id="trk_clamp",
                        zone_id=f"Z-{i}",    # different zones → different caps
                        t_offset_s=0,
                        confidence=0.95,
                    )
                    engine.on_event(ev)
        result = engine.score("trk_clamp", "ses_001", "RGL-TEST-01")
        self.assertLessEqual(result.bas, 100.0)

    def test_score_never_negative(self):
        """BAS must be ≥ 0 even if negatives outweigh positives."""
        engine = fresh_engine()
        for ntype in ["N01", "N02", "N03", "N04"]:
            for _ in range(10):
                ev = make_event(ntype, track_id="trk_neg", t_offset_s=0)
                engine.on_event(ev)
        result = engine.score("trk_neg", "ses_001", "RGL-TEST-01")
        self.assertGreaterEqual(result.bas, 0.0)

    def test_multiplier_half(self):
        """E13 aborted reach at 0.5 multiplier should score half weight."""
        engine = fresh_engine()
        ev = make_event("E13", multiplier=0.5, t_offset_s=0)
        engine.on_event(ev)
        result = engine.score("trk_001", "ses_001", "RGL-TEST-01")
        expected = EVENT_WEIGHTS["E13"] * 0.5
        self.assertAlmostEqual(result.bas, expected, delta=0.2)

    def test_lambda_value(self):
        """λ = ln(2)/270 ≈ 0.002567."""
        lam = math.log(2) / DECAY_HALF_LIFE_S
        self.assertAlmostEqual(lam, 0.002567, places=5)


class TestConfidenceGate(unittest.TestCase):

    def test_low_confidence_not_scored(self):
        engine = fresh_engine()
        gated_calls = []
        engine._retrain_sink = lambda ev: gated_calls.append(ev)
        ev = make_event("E02", confidence=0.60)
        engine.on_event(ev)
        result = engine.score("trk_001", "ses_001", "RGL-TEST-01")
        self.assertEqual(result.bas, 0.0)
        self.assertEqual(len(gated_calls), 1)

    def test_exact_threshold_accepted(self):
        """Confidence exactly 0.70 should be accepted (gate is ≥ 0.70)."""
        engine = fresh_engine()
        ev = make_event("E02", confidence=0.70)
        engine.on_event(ev)
        result = engine.score("trk_001", "ses_001", "RGL-TEST-01")
        self.assertGreater(result.bas, 0.0)

    def test_just_below_threshold_rejected(self):
        engine = fresh_engine()
        ev = make_event("E02", confidence=0.699)
        engine.on_event(ev)
        result = engine.score("trk_001", "ses_001", "RGL-TEST-01")
        self.assertEqual(result.bas, 0.0)


class TestIdempotency(unittest.TestCase):

    def test_replay_does_not_double_score(self):
        """The same event sent twice must score only once."""
        engine = fresh_engine()
        ev = make_event("E01", t_offset_s=0)
        engine.on_event(ev)
        engine.on_event(ev)   # replay
        result = engine.score("trk_001", "ses_001", "RGL-TEST-01")
        self.assertAlmostEqual(result.bas, EVENT_WEIGHTS["E01"], delta=0.3)

    def test_unique_events_both_score(self):
        """Two distinct events in different zones must both contribute."""
        engine = fresh_engine()
        ev1 = make_event("E01", zone_id="Z-A", t_offset_s=0)
        ev2 = make_event("E02", zone_id="Z-B", t_offset_s=-1)
        engine.on_event(ev1)
        engine.on_event(ev2)
        result = engine.score("trk_001", "ses_001", "RGL-TEST-01")
        expected = EVENT_WEIGHTS["E01"] + EVENT_WEIGHTS["E02"] * math.exp(
            -math.log(2)/DECAY_HALF_LIFE_S * 1
        )
        self.assertAlmostEqual(result.bas, expected, delta=0.5)


class TestZoneCap(unittest.TestCase):

    def test_zone_cap_enforced(self):
        """
        Repeated E02 events in the same zone should be capped at 1.5×weight.
        E02 weight = 9 → cap = 13.5
        """
        engine = fresh_engine()
        for i in range(20):
            # Different t_start to get different event_ids (different t_bucket)
            ev = BehaviourEvent(
                event_type="E02",
                track_id="trk_cap",
                session_id="ses_cap",
                store_id="RGL-TEST-01",
                zone_id="Z-SAME",
                t_start=time.time() - i,   # unique per second
                confidence=0.95,
                multiplier=1.0,
            )
            engine.on_event(ev)

        result = engine.score("trk_cap", "ses_cap", "RGL-TEST-01")
        cap_limit = ZONE_CAP_FACTOR * EVENT_WEIGHTS["E02"]   # 13.5
        self.assertLessEqual(result.bas, cap_limit + 0.5)   # small decay tolerance

    def test_different_zones_not_capped_together(self):
        """E02 in zone A and zone B should each contribute up to their own cap."""
        engine = fresh_engine()
        # Fill cap in zone A
        for i in range(5):
            ev = BehaviourEvent(
                event_type="E02",
                track_id="trk_zones",
                session_id="ses_z",
                store_id="RGL-TEST-01",
                zone_id="Z-A",
                t_start=time.time() - i,
                confidence=0.95,
            )
            engine.on_event(ev)
        # One event in zone B
        ev_b = make_event("E02", track_id="trk_zones", zone_id="Z-B", t_offset_s=0)
        engine.on_event(ev_b)
        result = engine.score("trk_zones", "ses_z", "RGL-TEST-01")
        # Should exceed single-zone cap
        self.assertGreater(result.bas, ZONE_CAP_FACTOR * EVENT_WEIGHTS["E02"])


class TestScoreBands(unittest.TestCase):

    def test_hot_band(self):
        self.assertEqual(ScoreBand.from_score(70.0), ScoreBand.HOT)
        self.assertEqual(ScoreBand.from_score(85.0), ScoreBand.HOT)
        self.assertEqual(ScoreBand.from_score(100.0), ScoreBand.HOT)

    def test_warm_band(self):
        self.assertEqual(ScoreBand.from_score(40.0), ScoreBand.WARM)
        self.assertEqual(ScoreBand.from_score(55.0), ScoreBand.WARM)
        self.assertEqual(ScoreBand.from_score(69.9), ScoreBand.WARM)

    def test_cold_band(self):
        self.assertEqual(ScoreBand.from_score(0.0),  ScoreBand.COLD)
        self.assertEqual(ScoreBand.from_score(20.0), ScoreBand.COLD)
        self.assertEqual(ScoreBand.from_score(39.9), ScoreBand.COLD)


class TestSessionEviction(unittest.TestCase):

    def test_evict_removes_track(self):
        engine = fresh_engine()
        ev = make_event("E01")
        engine.on_event(ev)
        self.assertIn("trk_001", engine._tracks)
        engine.evict_session("trk_001")
        self.assertNotIn("trk_001", engine._tracks)

    def test_evict_idle_sessions(self):
        engine = fresh_engine()
        # Inject a valid (recent) event so the track is created
        ev = make_event("E01", track_id="trk_idle", t_offset_s=0)
        engine.on_event(ev)
        self.assertIn("trk_idle", engine._tracks)
        # Simulate the track being idle for 2000 s
        engine._tracks["trk_idle"].last_event_t = time.time() - 2000
        n_evicted = engine.evict_expired_sessions(session_timeout_s=1800)
        self.assertEqual(n_evicted, 1)
        self.assertNotIn("trk_idle", engine._tracks)


class TestParallelThroughput(unittest.TestCase):

    def test_score_all_tracks_parallel_latency(self):
        """
        score_all_tracks() with 100 active tracks should complete in < 200 ms.
        This validates the ThreadPoolExecutor parallelism.
        """
        engine = fresh_engine()
        N_TRACKS = 100
        N_EVENTS = 10

        # Populate N tracks with N events each
        for i in range(N_TRACKS):
            for j in range(N_EVENTS):
                ev = BehaviourEvent(
                    event_type="E02",
                    track_id=f"trk_{i:04d}",
                    session_id=f"ses_{i:04d}",
                    store_id="RGL-TEST-01",
                    zone_id=f"Z-{j}",
                    t_start=time.time() - j * 30,
                    confidence=0.90,
                )
                engine.on_event(ev)

        t0 = time.perf_counter()
        results = engine.score_all_tracks()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.assertEqual(len(results), N_TRACKS)
        self.assertLess(elapsed_ms, 200,
            f"score_all_tracks took {elapsed_ms:.1f} ms for {N_TRACKS} tracks")
        print(f"\n  score_all_tracks({N_TRACKS} tracks): {elapsed_ms:.1f} ms")

    def test_on_event_throughput(self):
        """
        on_event() should sustain > 10,000 events/sec on a single thread.
        Retail stores with 20 cameras × 15 fps × ~5 events/frame = 1,500 ev/s peak.
        """
        engine = fresh_engine()
        N_EVENTS = 5000
        events = []
        for i in range(N_EVENTS):
            events.append(BehaviourEvent(
                event_type="E02",
                track_id=f"trk_{i % 50:04d}",
                session_id=f"ses_{i % 50:04d}",
                store_id="RGL-TEST-01",
                zone_id=f"Z-{i % 5}",
                t_start=time.time() - (i * 2),  # stagger to avoid dedup
                confidence=0.90,
            ))

        t0 = time.perf_counter()
        for ev in events:
            engine.on_event(ev)
        elapsed_s = time.perf_counter() - t0

        rate = N_EVENTS / elapsed_s
        print(f"\n  on_event throughput: {rate:,.0f} events/sec")
        self.assertGreater(rate, 10_000,
            f"on_event too slow: {rate:.0f} ev/s (need > 10,000)")


class TestEventDetectorWiring(unittest.TestCase):

    def test_all_detectors_instantiate(self):
        """Verify EventExtractor composes all 22 detector types."""
        from event_detectors import EventExtractor
        extractor = EventExtractor()
        detector_types = {type(d).__name__ for d in extractor._detectors}
        expected = {
            "E01_BasketPlacement", "E02_ProductPickup", "E03_TryOnDemo",
            "E04_HoldDuration", "E05_ComparisonBehaviour", "E06_PriceLabelInspection",
            "E07_StaffContact", "E08_ZoneDwell", "E09_RepeatZoneReturns",
            "E10_GazeFixation", "E11_BodyOrientationLeanIn", "E12_ApproachDeceleration",
            "E13_ReachAttempt", "E14_CompanionInteraction", "E15_PathPattern",
            "E16_ZoneTransitionSequence", "E17_PhoneUseAtShelf", "E18_SessionDuration",
            "N01_RapidPutBack", "N02_QueueAbandonment", "N03_WalkPast",
            "N04_ExitDirectedMovement",
        }
        self.assertEqual(detector_types, expected)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
