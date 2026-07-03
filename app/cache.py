import json
import hashlib
import redis as redis_client
from typing import Any, Optional


_redis = None
VERSION_KEY = "vrag:version"


def get_redis(url: str):
    global _redis
    if _redis is None and url:
        try:
            _redis = redis_client.from_url(url, decode_responses=True, socket_timeout=3)
            _redis.ping()
            print("[cache] Redis connected")
        except Exception as e:
            print(f"[cache] Redis unavailable, running without cache: {e}")
            _redis = None
    return _redis


def get_version(r) -> str:
    """Get current cache version. All answer/plan keys include this version."""
    if r is None:
        return "0"
    try:
        v = r.get(VERSION_KEY)
        return v if v else "0"
    except Exception:
        return "0"


def bump_version(r) -> str:
    """Increment version — instantly orphans all existing answer/plan caches."""
    if r is None:
        return "0"
    try:
        new_version = r.incr(VERSION_KEY)
        print(f"[cache] Version bumped to {new_version} — all query caches invalidated")
        return str(new_version)
    except Exception:
        return "0"


def make_key(prefix: str, *parts: str) -> str:
    raw = "|".join(parts)
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"vrag:{prefix}:{digest}"


def make_versioned_key(prefix: str, version: str, *parts: str) -> str:
    """Cache key that includes version — invalidated when version bumps."""
    raw = "|".join(parts)
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"vrag:{prefix}:v{version}:{digest}"


def cache_get(r, key: str) -> Optional[Any]:
    if r is None:
        return None
    try:
        value = r.get(key)
        return json.loads(value) if value else None
    except Exception:
        return None


def cache_set(r, key: str, value: Any, ttl: int = 3600):
    if r is None:
        return
    try:
        r.setex(key, ttl, json.dumps(value))
    except Exception:
        pass
