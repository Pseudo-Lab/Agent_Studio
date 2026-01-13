"""Session store for HITL continuations."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from kiosk_agent.utils import get_logger

logger = get_logger(__name__)

Session = Dict[str, Any]


class SessionStore:
    """
    Thread-safe session store with LRU eviction.
    
    Features:
    - OrderedDict for LRU-like eviction
    - TTL-based cleanup
    - Max session limit enforcement
    """
    
    def __init__(self, *, max_sessions: int = 100, idle_ttl_sec: int = 3600):
        self.max_sessions = max_sessions
        self.idle_ttl_sec = idle_ttl_sec
        self._store: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()
    
    def count(self) -> int:
        """Get number of active sessions."""
        with self._lock:
            return len(self._store)
    
    def get(self, thread_id: str) -> Optional[Session]:
        """Get session by ID, updating access time."""
        with self._lock:
            session = self._store.get(thread_id)
            if session is None:
                return None
            self._touch_locked(thread_id, session)
            return session
    
    def set(self, thread_id: str, session: Session) -> None:
        """Create or update session."""
        with self._lock:
            self._store[thread_id] = session
            self._touch_locked(thread_id, session)
            self._cleanup_locked()
    
    def update(self, thread_id: str, **patch: Any) -> None:
        """Patch existing session."""
        with self._lock:
            session = self._store.get(thread_id)
            if session is None:
                return
            session.update(patch)
            self._touch_locked(thread_id, session)
    
    def touch(self, thread_id: str) -> None:
        """Update access time without modifying data."""
        with self._lock:
            session = self._store.get(thread_id)
            if session is None:
                return
            self._touch_locked(thread_id, session)
    
    def delete(self, thread_id: str) -> None:
        """Remove session."""
        with self._lock:
            self._store.pop(thread_id, None)
    
    def cleanup(self) -> None:
        """Remove stale sessions."""
        with self._lock:
            self._cleanup_locked()
    
    def _touch_locked(self, thread_id: str, session: Session) -> None:
        """Update access time (must hold lock)."""
        session["last_access"] = time.time()
        try:
            self._store.move_to_end(thread_id)
        except KeyError:
            pass
    
    def _cleanup_locked(self) -> None:
        """Clean up stale sessions (must hold lock)."""
        now = time.time()
        
        # Remove expired sessions
        stale = [
            tid for tid, session in self._store.items()
            if now - float(session.get("last_access", 0) or 0) > self.idle_ttl_sec
        ]
        for tid in stale:
            logger.debug(f"Removing stale session: {tid}")
            self._store.pop(tid, None)
        
        # Enforce max limit (LRU eviction)
        while len(self._store) > self.max_sessions:
            oldest = next(iter(self._store))
            logger.debug(f"Removing oldest session: {oldest}")
            self._store.pop(oldest, None)
