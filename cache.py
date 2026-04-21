"""
cache.py — Thread-safe in-memory caching layer with TTL support.

Supports optional Redis backend. Falls back to in-memory dict.
Key strategy: hash(sorted(medicines) + sorted(current_medications))
TTL: 1 hour (configurable)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default TTL: 1 hour
DEFAULT_TTL_SECONDS = 3600


class CacheEntry:
    """Single cache entry with expiry tracking."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: int) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl

    @property
    def is_expired(self) -> bool:
        return time.monotonic() >= self.expires_at


class InMemoryCache:
    """
    Thread-safe in-memory cache with automatic TTL eviction.

    Used as the primary caching layer. Redis can be plugged in as an
    alternative by implementing the same interface.
    """

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS, max_size: int = 10_000) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._ttl = ttl
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    # ── Key Generation ────────────────────────────────────────────────────

    @staticmethod
    def build_key(medicines: list[str], current_medications: list[str] | None = None) -> str:
        """
        Build a deterministic cache key from sorted medicine + medication lists.

        Key = SHA-256( sorted(medicines) + "|" + sorted(current_medications) )
        """
        meds = sorted(m.strip().lower() for m in medicines if m.strip())
        current = sorted(m.strip().lower() for m in (current_medications or []) if m.strip())
        raw = json.dumps({"m": meds, "c": current}, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ── Core Operations ───────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value if it exists and hasn't expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with TTL. Evicts oldest entries if at capacity."""
        effective_ttl = ttl if ttl is not None else self._ttl
        with self._lock:
            # Evict expired entries if near capacity
            if len(self._store) >= self._max_size:
                self._evict_expired()
            # If still at capacity, remove oldest entries
            if len(self._store) >= self._max_size:
                to_remove = len(self._store) - self._max_size + 1
                keys_to_remove = list(self._store.keys())[:to_remove]
                for k in keys_to_remove:
                    del self._store[k]
            self._store[key] = CacheEntry(value, effective_ttl)

    def delete(self, key: str) -> bool:
        """Remove a specific key. Returns True if found."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Flush all cached entries."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    # ── Maintenance ───────────────────────────────────────────────────────

    def _evict_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        expired_keys = [k for k, v in self._store.items() if v.is_expired]
        for k in expired_keys:
            del self._store[k]
        return len(expired_keys)

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            return {
                "size": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (
                    self._hits / (self._hits + self._misses)
                    if (self._hits + self._misses) > 0
                    else 0.0
                ),
                "ttl_seconds": self._ttl,
                "max_size": self._max_size,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ─── Optional Redis Backend ──────────────────────────────────────────────────

class RedisCache:
    """
    Redis-backed cache. Requires `redis` package and a running Redis server.
    Falls back gracefully if Redis is unavailable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl: int = DEFAULT_TTL_SECONDS,
        password: str | None = None,
    ) -> None:
        self._ttl = ttl
        self._available = False
        try:
            import redis

            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            self._client.ping()
            self._available = True
            logger.info("Redis cache connected at %s:%d", host, port)
        except Exception as e:
            logger.warning("Redis unavailable (%s), falling back to in-memory cache", e)
            self._client = None

    @staticmethod
    def build_key(medicines: list[str], current_medications: list[str] | None = None) -> str:
        return InMemoryCache.build_key(medicines, current_medications)

    @property
    def is_available(self) -> bool:
        return self._available

    def get(self, key: str) -> Optional[Any]:
        if not self._available:
            return None
        try:
            data = self._client.get(f"drugsafety:{key}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error("Redis GET error: %s", e)
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if not self._available:
            return
        effective_ttl = ttl if ttl is not None else self._ttl
        try:
            self._client.setex(
                f"drugsafety:{key}",
                effective_ttl,
                json.dumps(value, default=str),
            )
        except Exception as e:
            logger.error("Redis SET error: %s", e)

    def delete(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            return bool(self._client.delete(f"drugsafety:{key}"))
        except Exception:
            return False

    def clear(self) -> None:
        if not self._available:
            return
        try:
            keys = self._client.keys("drugsafety:*")
            if keys:
                self._client.delete(*keys)
        except Exception as e:
            logger.error("Redis CLEAR error: %s", e)


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_cache(
    backend: str = "memory",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: str | None = None,
    ttl: int = DEFAULT_TTL_SECONDS,
) -> InMemoryCache | RedisCache:
    """
    Factory to create the appropriate cache backend.

    Args:
        backend: "memory" or "redis"
        redis_host: Redis server host
        redis_port: Redis server port
        redis_password: Redis password (optional)
        ttl: Time-to-live in seconds

    Returns:
        Cache instance with .get(), .set(), .delete(), .clear() interface
    """
    if backend == "redis":
        rc = RedisCache(host=redis_host, port=redis_port, password=redis_password, ttl=ttl)
        if rc.is_available:
            return rc
        logger.warning("Redis requested but unavailable. Using in-memory cache.")

    return InMemoryCache(ttl=ttl)
