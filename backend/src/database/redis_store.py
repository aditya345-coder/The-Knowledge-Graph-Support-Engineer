from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date
from typing import Any, TypedDict

import redis.asyncio as redis

from settings import settings

logger = logging.getLogger(__name__)


class SessionData(TypedDict, total=False):
    session_id: str
    repo_url: str
    stage: str
    completed_phases: list[str]
    current_phase: str
    percent: str
    message: str
    created_at: str
    updated_at: str


# ── Lua Scripts ──────────────────────────────────────────────────────

# Atomic phase completion: append phase to JSON array if not already present
_MARK_PHASE_LUA = """
local key = KEYS[1]
local phase = ARGV[1]
local timestamp = ARGV[2]
local phases_json = redis.call('HGET', key, 'completed_phases')
if not phases_json then
    phases_json = '[]'
end
local phases = cjson.decode(phases_json)
for _, p in ipairs(phases) do
    if p == phase then
        return 0
    end
end
table.insert(phases, phase)
redis.call('HSET', key, 'completed_phases', cjson.encode(phases), 'updated_at', timestamp)
return 1
"""

# Atomic rate limit: INCR + conditional EXPIRE
_RATE_LIMIT_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local count = redis.call('INCR', key)
if count == 1 then
    redis.call('EXPIRE', key, ttl)
end
return count
"""

# Atomic rate limit decrement: DECR with floor at 0
_RATE_LIMIT_DECR_LUA = """
local key = KEYS[1]
local val = redis.call('DECR', key)
if val < 0 then
    redis.call('SET', key, 0)
    return 0
end
return val
"""


class RedisStore:
    """Single source of truth for all Redis operations.

    Handles session state CRUD, rate limiting, and connection management.
    Uses redis.asyncio for non-blocking operations in FastAPI.
    """

    def __init__(self) -> None:
        self._client: redis.Redis | None = None
        self._sync_client: Any = None
        self._lock = threading.Lock()
        self._available = True
        self._last_error_time: float = 0.0

    async def _get_client(self) -> redis.Redis:
        """Lazy-initialize the async Redis client."""
        if self._client is not None:
            return self._client

        if not settings.REDIS_URL:
            raise ValueError("REDIS_URL is not configured")

        with self._lock:
            if self._client is not None:
                return self._client

            pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=20,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            self._client = redis.Redis(connection_pool=pool)
            return self._client

    def _check_available(self) -> bool:
        """Check if Redis should be used. Attempts reconnection after 30s cooldown."""
        if self._available:
            return True
        if time.monotonic() - self._last_error_time <= 30:
            return False
        try:
            if hasattr(self, "_sync_client") and self._sync_client is not None:
                self._sync_client.ping()
                self._available = True
                logger.info("Redis reconnection successful")
                return True
        except Exception:
            self._last_error_time = time.monotonic()
        return False

    # ── Session State ────────────────────────────────────────────────

    async def save_session(self, session_id: str, data: dict) -> None:
        """Write session fields to a Redis Hash with 24h TTL."""
        if not self._check_available():
            return

        try:
            client = await self._get_client()
            key = f"session:{session_id}"
            mapping = {k: json.dumps(v) if isinstance(v, (list, dict)) else str(v) for k, v in data.items()}
            await client.hset(key, mapping=mapping)
            await client.expire(key, 86400)
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for save_session: %s", e)
            self._last_error_time = time.monotonic()

    async def get_session(self, session_id: str) -> SessionData | None:
        """Retrieve session data. Returns None if expired or missing."""
        if not self._check_available():
            return None

        try:
            client = await self._get_client()
            key = f"session:{session_id}"
            data = await client.hgetall(key)
            if not data:
                return None

            result: SessionData = {}
            for k, v in data.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                if key == "completed_phases":
                    result["completed_phases"] = json.loads(val) if val else []
                elif key in ("session_id", "repo_url", "stage", "current_phase",
                             "percent", "message", "created_at", "updated_at"):
                    result[key] = val  # type: ignore[literal-required]
            return result
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for get_session: %s", e)
            self._last_error_time = time.monotonic()
            return None

    async def mark_phase_complete(self, session_id: str, phase: str) -> bool:
        """Atomically append a phase to completed_phases via Lua script."""
        if not self._check_available():
            return False

        try:
            client = await self._get_client()
            key = f"session:{session_id}"
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            result = await client.eval(_MARK_PHASE_LUA, 1, key, phase, timestamp)
            return result == 1
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for mark_phase_complete: %s", e)
            self._last_error_time = time.monotonic()
            return False

    async def update_session_stage(self, session_id: str, stage: str, **fields: str) -> None:
        """Update session stage and optional extra fields using HSET."""
        if not self._check_available():
            return

        try:
            client = await self._get_client()
            key = f"session:{session_id}"
            from datetime import datetime
            mapping = {"stage": stage, "updated_at": datetime.now().isoformat()}
            mapping.update({k: str(v) for k, v in fields.items()})
            await client.hset(key, mapping=mapping)  # type: ignore[arg-type]
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for update_session_stage: %s", e)
            self._last_error_time = time.monotonic()

    async def update_session_fields(self, session_id: str, **fields: str | list[str]) -> None:
        """Update arbitrary session fields using HSET (single roundtrip)."""
        if not self._check_available():
            return

        try:
            client = await self._get_client()
            key = f"session:{session_id}"
            from datetime import datetime
            mapping = {"updated_at": datetime.now().isoformat()}
            for k, v in fields.items():
                mapping[k] = json.dumps(v) if isinstance(v, list) else str(v)
            await client.hset(key, mapping=mapping)  # type: ignore[arg-type]
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for update_session_fields: %s", e)
            self._last_error_time = time.monotonic()

    async def delete_session(self, session_id: str) -> None:
        """Delete a session key."""
        if not self._check_available():
            return

        try:
            client = await self._get_client()
            await client.delete(f"session:{session_id}")
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for delete_session: %s", e)
            self._last_error_time = time.monotonic()

    async def get_running_sessions(self) -> list[dict]:
        """Find all sessions with stage=running via SCAN."""
        if not self._check_available():
            return []

        try:
            client = await self._get_client()
            result: list[dict] = []
            async for key in client.scan_iter(match="session:*", count=100):
                data = await client.hgetall(key)
                if data.get("stage") == "running":
                    session_id = key.removeprefix("session:")
                    result.append({"session_id": session_id, **data})
            return result
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for get_running_sessions: %s", e)
            self._last_error_time = time.monotonic()
            return []

    # ── Rate Limiting ────────────────────────────────────────────────

    def _rate_limit_key(self, user_id: str, namespace: str | None = None) -> str:
        today = date.today().isoformat()
        if namespace:
            return f"ratelimit:{user_id}:{today}:{namespace}"
        return f"ratelimit:{user_id}:{today}"

    async def check_rate_limit(self, user_id: str, limit: int) -> bool:
        """Increment counter and check against limit. Atomic via Lua."""
        if not self._check_available():
            return False

        try:
            client = await self._get_client()
            key = self._rate_limit_key(user_id)
            count = await client.eval(_RATE_LIMIT_LUA, 1, key, limit, 86400)
            return int(count) <= limit
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for check_rate_limit: %s", e)
            self._last_error_time = time.monotonic()
            return False

    async def decrement_rate_limit(self, user_id: str) -> None:
        """Decrement counter with floor at 0. Atomic via Lua."""
        if not self._check_available():
            return

        try:
            client = await self._get_client()
            key = self._rate_limit_key(user_id)
            await client.eval(_RATE_LIMIT_DECR_LUA, 1, key)
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for decrement_rate_limit: %s", e)
            self._last_error_time = time.monotonic()

    async def check_rate_limit_namespace(self, user_id: str, namespace: str, limit: int) -> bool:
        """Namespaced rate limit check. Atomic via Lua."""
        if not self._check_available():
            return False

        try:
            client = await self._get_client()
            key = self._rate_limit_key(user_id, namespace)
            count = await client.eval(_RATE_LIMIT_LUA, 1, key, limit, 86400)
            return int(count) <= limit
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for check_rate_limit_namespace: %s", e)
            self._last_error_time = time.monotonic()
            return False

    # ── Connection Health ────────────────────────────────────────────

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            client = await self._get_client()
            return await client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Sync Methods (for background tasks) ──────────────────────────

    def _get_sync_client(self):
        """Get or create a sync Redis client for background tasks."""
        import redis as sync_redis

        if not settings.REDIS_URL:
            raise ValueError("REDIS_URL is not configured")

        if not hasattr(self, "_sync_client") or self._sync_client is None:
            pool = sync_redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=20,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            self._sync_client = sync_redis.Redis(connection_pool=pool)
        return self._sync_client

    def save_session_sync(self, session_id: str, data: dict) -> None:
        """Sync version for background tasks."""
        if not self._check_available():
            return
        import redis as sync_redis
        try:
            client = self._get_sync_client()
            key = f"session:{session_id}"
            mapping = {k: json.dumps(v) if isinstance(v, (list, dict)) else str(v) for k, v in data.items()}
            client.hset(key, mapping=mapping)
            client.expire(key, 86400)
        except (sync_redis.ConnectionError, sync_redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for save_session_sync: %s", e)
            self._last_error_time = time.monotonic()

    def get_session_sync(self, session_id: str) -> SessionData | None:
        """Sync version for background tasks."""
        if not self._check_available():
            return None
        import redis as sync_redis
        try:
            client = self._get_sync_client()
            key = f"session:{session_id}"
            data = client.hgetall(key)
            if not data:
                return None
            result: SessionData = {}
            for k, v in data.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                if key == "completed_phases":
                    result["completed_phases"] = json.loads(val) if val else []
                elif key in ("session_id", "repo_url", "stage", "current_phase",
                             "percent", "message", "created_at", "updated_at"):
                    result[key] = val  # type: ignore[literal-required]
            return result
        except (sync_redis.ConnectionError, sync_redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for get_session_sync: %s", e)
            self._last_error_time = time.monotonic()
            return None

    def mark_phase_complete_sync(self, session_id: str, phase: str) -> bool:
        """Sync version for background tasks."""
        if not self._check_available():
            return False
        import redis as sync_redis
        try:
            client = self._get_sync_client()
            key = f"session:{session_id}"
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            result = client.eval(_MARK_PHASE_LUA, 1, key, phase, timestamp)
            return result == 1
        except (sync_redis.ConnectionError, sync_redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for mark_phase_complete_sync: %s", e)
            self._last_error_time = time.monotonic()
            return False

    def update_session_stage_sync(self, session_id: str, stage: str, **fields: str) -> None:
        """Sync version for background tasks."""
        if not self._check_available():
            return
        import redis as sync_redis
        try:
            client = self._get_sync_client()
            key = f"session:{session_id}"
            from datetime import datetime
            mapping = {"stage": stage, "updated_at": datetime.now().isoformat()}
            mapping.update({k: str(v) for k, v in fields.items()})
            client.hset(key, mapping=mapping)
        except (sync_redis.ConnectionError, sync_redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for update_session_stage_sync: %s", e)
            self._last_error_time = time.monotonic()

    def delete_session_sync(self, session_id: str) -> None:
        """Sync version for background tasks."""
        if not self._check_available():
            return
        import redis as sync_redis
        try:
            client = self._get_sync_client()
            client.delete(f"session:{session_id}")
        except (sync_redis.ConnectionError, sync_redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for delete_session_sync: %s", e)
            self._last_error_time = time.monotonic()

    def decrement_rate_limit_sync(self, user_id: str) -> None:
        """Sync version for background tasks."""
        if not self._check_available():
            return
        import redis as sync_redis
        try:
            client = self._get_sync_client()
            key = self._rate_limit_key(user_id)
            client.eval(_RATE_LIMIT_DECR_LUA, 1, key)
        except (sync_redis.ConnectionError, sync_redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for decrement_rate_limit_sync: %s", e)
            self._last_error_time = time.monotonic()

    def check_rate_limit_sync(self, user_id: str, limit: int) -> bool:
        """Sync version for background tasks."""
        if not self._check_available():
            return False
        import redis as sync_redis
        try:
            client = self._get_sync_client()
            key = self._rate_limit_key(user_id)
            count = client.eval(_RATE_LIMIT_LUA, 1, key, limit, 86400)
            return int(count) <= limit
        except (sync_redis.ConnectionError, sync_redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for check_rate_limit_sync: %s", e)
            self._last_error_time = time.monotonic()
            return False

    def check_rate_limit_namespace_sync(self, user_id: str, namespace: str, limit: int) -> bool:
        """Sync version for background tasks."""
        if not self._check_available():
            return False
        import redis as sync_redis
        try:
            client = self._get_sync_client()
            key = self._rate_limit_key(user_id, namespace)
            count = client.eval(_RATE_LIMIT_LUA, 1, key, limit, 86400)
            return int(count) <= limit
        except (sync_redis.ConnectionError, sync_redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for check_rate_limit_namespace_sync: %s", e)
            self._last_error_time = time.monotonic()
            return False


    # ── Repo List Persistence (30-day TTL) ──────────────────────────

    async def save_repo_list(self, user_id: str, repos: list[dict]) -> None:
        """Save repo list with 30-day TTL."""
        if not self._check_available():
            return
        try:
            client = await self._get_client()
            key = f"repo_list:{user_id}"
            await client.set(key, json.dumps(repos), ex=2592000)
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for save_repo_list: %s", e)
            self._last_error_time = time.monotonic()

    async def get_repo_list(self, user_id: str) -> list[dict]:
        """Retrieve repo list."""
        if not self._check_available():
            return []
        try:
            client = await self._get_client()
            key = f"repo_list:{user_id}"
            data = await client.get(key)
            return json.loads(data) if data else []
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for get_repo_list: %s", e)
            self._last_error_time = time.monotonic()
            return []

    def save_repo_list_sync(self, user_id: str, repos: list[dict]) -> None:
        """Synchronous version of save_repo_list for use in background threads."""
        if not self._check_available():
            return
        try:
            client = self._get_sync_client()
            key = f"repo_list:{user_id}"
            client.set(key, json.dumps(repos), ex=2592000)
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for save_repo_list_sync: %s", e)
            self._last_error_time = time.monotonic()

    def get_repo_list_sync(self, user_id: str) -> list[dict]:
        """Synchronous version of get_repo_list for use in background threads."""
        if not self._check_available():
            return []
        try:
            client = self._get_sync_client()
            key = f"repo_list:{user_id}"
            data = client.get(key)
            return json.loads(data) if data else []
        except (redis.ConnectionError, redis.TimeoutError, ValueError) as e:
            logger.warning("Redis unavailable for get_repo_list_sync: %s", e)
            self._last_error_time = time.monotonic()
            return []


# ── Module-level singleton ───────────────────────────────────────────

_redis_store: RedisStore | None = None
_singleton_lock = threading.Lock()


def get_redis_store() -> RedisStore:
    """Get or create the singleton RedisStore instance."""
    global _redis_store
    if _redis_store is None:
        with _singleton_lock:
            if _redis_store is None:
                _redis_store = RedisStore()
    return _redis_store
