"""
NovaDev Cloud — Queue Manager
Redis-backed FIFO queue for free users, priority skip for paid users.

Layout:
  novadev:queue:free  — sorted set, score = timestamp (FIFO)
  novadev:queue:paid  — sorted set, score = timestamp (FIFO within paid)
  novadev:session:{id} — hash of session metadata
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Optional

import aioredis

from core.config import settings
from core.websocket_manager import ws_manager
from docker.pool import DockerPool, SlotType

log = logging.getLogger("queue_manager")

FREE_QUEUE  = "novadev:queue:free"
PAID_QUEUE  = "novadev:queue:paid"
SESSION_KEY = "novadev:session:{}"


class QueueManager:
    def __init__(self, docker_pool: DockerPool):
        self.pool = docker_pool
        self.redis: Optional[aioredis.Redis] = None

    async def _connect_redis(self):
        self.redis = await aioredis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        log.info("Queue manager connected to Redis.")

    # ── Public API ──────────────────────────────────────────────────────────

    async def enqueue(self, user_id: str, tier: str) -> dict:
        """
        Add a user to the queue.
        Paid users skip the free queue and get assigned immediately if a slot is free.
        """
        session_id = str(uuid.uuid4())
        slot_type  = SlotType.PAID if tier == "paid" else SlotType.FREE

        # Try immediate assignment first
        slot = await self.pool.acquire_slot(slot_type, user_id, session_id)
        if slot:
            await self._persist_session(session_id, user_id, tier, slot.slot_id)
            await ws_manager.send(user_id, {
                "event": "session_started",
                "session_id": session_id,
                "slot_id": slot.slot_id,
                "port": slot.port,
                "tier": tier,
                "expires_at": slot.expires_at.isoformat(),
            })
            log.info("Immediate slot assigned: user=%s tier=%s slot=%d",
                     user_id, tier, slot.slot_id)
            return {"status": "started", "session_id": session_id, "slot_id": slot.slot_id}

        # No slot available — add to queue
        queue_key = PAID_QUEUE if tier == "paid" else FREE_QUEUE
        score = datetime.utcnow().timestamp()
        payload = json.dumps({"user_id": user_id, "tier": tier, "session_id": session_id})

        queue_len = await self.redis.zcard(queue_key)
        if queue_len >= settings.MAX_QUEUE_LENGTH:
            return {"status": "queue_full", "message": "Queue is at capacity. Try again later."}

        await self.redis.zadd(queue_key, {payload: score})
        pos = await self._queue_position(user_id, tier)
        est_wait = pos * 2  # rough estimate: 2 min per position

        await ws_manager.send(user_id, {
            "event": "queued",
            "position": pos,
            "estimated_wait_min": est_wait,
            "tier": tier,
        })
        log.info("User queued: user=%s tier=%s position=%d", user_id, tier, pos)
        return {"status": "queued", "position": pos, "estimated_wait_min": est_wait}

    async def dequeue_user(self, user_id: str, tier: str):
        """Remove a user from the queue (cancelled / timed out)."""
        queue_key = PAID_QUEUE if tier == "paid" else FREE_QUEUE
        members = await self.redis.zrange(queue_key, 0, -1)
        for m in members:
            data = json.loads(m)
            if data["user_id"] == user_id:
                await self.redis.zrem(queue_key, m)
                log.info("User %s removed from queue.", user_id)
                return

    async def on_slot_released(self, slot_type: SlotType):
        """Called after a slot is recycled — try to assign next queued user."""
        queue_key = PAID_QUEUE if slot_type == SlotType.PAID else FREE_QUEUE
        # For freed paid slots, also check paid queue first
        next_entry = await self.redis.zrange(queue_key, 0, 0)
        if not next_entry:
            return

        payload = json.loads(next_entry[0])
        user_id    = payload["user_id"]
        tier       = payload["tier"]
        session_id = payload["session_id"]

        slot = await self.pool.acquire_slot(slot_type, user_id, session_id)
        if slot:
            await self.redis.zrem(queue_key, next_entry[0])
            await self._persist_session(session_id, user_id, tier, slot.slot_id)
            await ws_manager.send(user_id, {
                "event": "session_started",
                "session_id": session_id,
                "slot_id": slot.slot_id,
                "port": slot.port,
                "tier": tier,
                "expires_at": slot.expires_at.isoformat(),
            })
            log.info("Queued user %s got slot %d", user_id, slot.slot_id)

    # ── Background loop ──────────────────────────────────────────────────────

    async def run(self):
        """Main background task: expire sessions + drain queues."""
        await self._connect_redis()
        log.info("Queue manager loop started.")
        while True:
            try:
                # 1. Expire sessions
                released = await self.pool.expire_sessions()
                for slot_id in released:
                    slot = self.pool.slots[slot_id]
                    asyncio.create_task(self.on_slot_released(slot.slot_type))

                # 2. Try to drain free queue into any idle free slot
                free_idle = sum(
                    1 for s in self.pool.slots.values()
                    if s.slot_type == SlotType.FREE and s.state.value == "idle"
                )
                for _ in range(free_idle):
                    await self.on_slot_released(SlotType.FREE)

                # 3. Try to drain paid queue
                paid_idle = sum(
                    1 for s in self.pool.slots.values()
                    if s.slot_type == SlotType.PAID and s.state.value == "idle"
                )
                for _ in range(paid_idle):
                    await self.on_slot_released(SlotType.PAID)

            except Exception as e:
                log.error("Queue loop error: %s", e)

            await asyncio.sleep(10)

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _queue_position(self, user_id: str, tier: str) -> int:
        queue_key = PAID_QUEUE if tier == "paid" else FREE_QUEUE
        members = await self.redis.zrange(queue_key, 0, -1)
        for i, m in enumerate(members):
            if json.loads(m)["user_id"] == user_id:
                return i + 1
        return -1

    async def _persist_session(self, session_id, user_id, tier, slot_id):
        key = SESSION_KEY.format(session_id)
        await self.redis.hset(key, mapping={
            "session_id": session_id,
            "user_id": user_id,
            "tier": tier,
            "slot_id": slot_id,
            "started_at": datetime.utcnow().isoformat(),
        })
        ttl = settings.PAID_SESSION_MINUTES * 60 if tier == "paid" else settings.FREE_SESSION_MINUTES * 60
        await self.redis.expire(key, ttl + 60)

    async def queue_stats(self) -> dict:
        free_len = await self.redis.zcard(FREE_QUEUE)
        paid_len = await self.redis.zcard(PAID_QUEUE)
        return {
            "free_queue_length": free_len,
            "paid_queue_length": paid_len,
            "total_queued": free_len + paid_len,
        }
