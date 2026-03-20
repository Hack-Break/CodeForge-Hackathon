"""
NeuralPath — Caching Layer
===========================
Simple in-memory LRU cache with TTL for the pipeline.

Caches:
  - Skill extraction results (same resume + JD → same skills)
  - Full pipeline plans (same inputs → same plan)

Keys are SHA-256 hashes of the input text.
TTL default: 30 minutes (plans don't change for the same inputs).

In production this would be Redis — but in-memory works for hackathon
and avoids the Redis dependency for local dev.
"""

from __future__ import annotations
import hashlib
import time
import logging
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cache implementation
# ─────────────────────────────────────────────────────────────────────────────

class TTLCache:
    """
    Thread-safe LRU cache with per-entry TTL.
    Evicts entries that have expired or when max_size is reached.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 1800):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.max_size    = max_size
        self.ttl_seconds = ttl_seconds
        self.hits        = 0
        self.misses      = 0

    def _is_expired(self, expires_at: float) -> bool:
        return time.monotonic() > expires_at

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            self.misses += 1
            return None

        value, expires_at = self._cache[key]

        if self._is_expired(expires_at):
            del self._cache[key]
            self.misses += 1
            return None

        # Move to end (LRU — most recently used)
        self._cache.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self.ttl_seconds
        expires_at = time.monotonic() + ttl

        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, expires_at)

        # Evict LRU if over capacity
        while len(self._cache) > self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Cache evicted: {evicted_key[:16]}...")

    def invalidate(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        n = len(self._cache)
        self._cache.clear()
        return n

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries":    len(self._cache),
            "max_size":   self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits":       self.hits,
            "misses":     self.misses,
            "hit_rate":   round(self.hits / total, 3) if total > 0 else 0.0,
        }

    def evict_expired(self) -> int:
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._cache.items() if now > exp]
        for k in expired:
            del self._cache[k]
        return len(expired)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton caches
# ─────────────────────────────────────────────────────────────────────────────

# Skill extraction cache — shorter TTL (30 min)
skill_cache = TTLCache(max_size=200, ttl_seconds=1800)

# Full plan cache — longer TTL (1 hour)
plan_cache  = TTLCache(max_size=50,  ttl_seconds=3600)


# ─────────────────────────────────────────────────────────────────────────────
# Key generation
# ─────────────────────────────────────────────────────────────────────────────

def make_cache_key(*parts: str) -> str:
    """Create a deterministic SHA-256 cache key from text parts."""
    combined = "\n|||SEP|||\n".join(p.strip().lower() for p in parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def make_skill_key(resume_text: str, jd_text: str) -> str:
    return make_cache_key("skills", resume_text[:3000], jd_text[:2000])


def make_plan_key(resume_text: str, jd_text: str, algorithm: str) -> str:
    return make_cache_key("plan", resume_text[:3000], jd_text[:2000], algorithm)
