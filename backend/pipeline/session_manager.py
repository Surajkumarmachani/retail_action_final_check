# pipeline/session_manager.py
# Manages the dict of active LiveCustomerSession objects per WebSocket connection.
# Garbage-collects tracks that have disappeared from the frame.

import logging
import time
from typing import Dict

from constants import GC_TIMEOUT_SECONDS
from pipeline.session import LiveCustomerSession

logger = logging.getLogger(__name__)


class SessionManager:
    """One SessionManager instance per WebSocket connection."""

    def __init__(self) -> None:
        self._sessions: Dict[int, LiveCustomerSession] = {}

    def get_or_create(self, track_id: int) -> LiveCustomerSession:
        if track_id not in self._sessions:
            self._sessions[track_id] = LiveCustomerSession(track_id)
            logger.debug("New session: track_id=%d  (total active=%d)",
                         track_id, len(self._sessions))
        return self._sessions[track_id]

    def garbage_collect(self) -> int:
        """
        Remove sessions for tracks not seen in GC_TIMEOUT_SECONDS.
        Returns the number of sessions removed.
        """
        now   = time.monotonic()
        stale = [tid for tid, s in self._sessions.items()
                 if now - s.last_seen > GC_TIMEOUT_SECONDS]
        for tid in stale:
            del self._sessions[tid]
            logger.debug("GC: removed track_id=%d", tid)
        return len(stale)

    @property
    def active_count(self) -> int:
        return len(self._sessions)
