"""
NovaDev Cloud — GPU Quota Manager
Single source of truth for the 30h/week Kaggle GPU budget.

KEY INSIGHT: The 30h quota is consumed by notebook RUNTIME, not by user sessions.
Every second the Kaggle notebook is alive costs quota — regardless of how many
users are on it. So the strategy is:

  Start notebook  → when first user gets a slot
  Stop notebook   → 10 min after last session ends (idle timeout)
  Never run it    → unless someone needs it

This alone can extend 30h across a full 7-day week even with heavy usage.

Quota gates (enforced before any session is granted):
  > 6h remaining   → normal operations
  3h – 6h          → warn all users + admin, throttle free queue
  1h – 3h          → paid users only, free sessions blocked
  < 1h             → all new sessions blocked, existing ones finish
  0h (exhausted)   → total block, show reset countdown to users
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import aioredis

log = logging.getLogger("quota_manager")

QUOTA_TOTAL_SECONDS  = 30 * 3600          # 30h in seconds
WARN_THRESHOLD_S     = 6 * 3600           # Warn at < 6h
PAID_ONLY_THRESHOLD_S = 3 * 3600          # Paid-only at < 3h
BLOCK_ALL_THRESHOLD_S = 1 * 3600          # Block all at < 1h
IDLE_SHUTDOWN_SECS   = 600                # Stop notebook 10min after last session

# Redis keys
KEY_QUOTA_USED    = "novadev:quota:used_seconds"   # float, accrued this week
KEY_QUOTA_START   = "novadev:quota:week_start_ts"  # ISO timestamp of current week start
KEY_NB_STARTED    = "novadev:quota:nb_started_ts"  # When current notebook started
KEY_NB_RUNNING    = "novadev:quota:nb_running"     # "1" or "0"
KEY_LAST_SESSION  = "novadev:quota:last_session_ts"# When last session ended
KEY_GATE          = "novadev:quota:gate"           # "normal" | "warn" | "paid_only" | "blocked"


class QuotaManager:
    """
    Tracks GPU quota in Redis with second-level precision.
    All other components call check_gate() before starting a session.
    """

    def __init__(self, redis: aioredis.Redis):
        self.redis = redis
        self._notebook_start_ts: Optional[datetime] = None
        self._accrual_task: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None

    # ── Startup ──────────────────────────────────────────────────────────────

    async def start(self):
        await self._ensure_week_reset()
        asyncio.create_task(self._quota_loop())
        log.info("Quota manager started. Remaining: %.1fh",
                 await self.remaining_hours())

    # ── Gate check (called before every session grant) ───────────────────────

    async def check_gate(self, tier: str) -> dict:
        """
        Returns {"allowed": True/False, "gate": str, "remaining_h": float, "message": str}
        Call this before granting any session slot.
        """
        remaining_s = await self._remaining_seconds()
        remaining_h = remaining_s / 3600

        if remaining_s <= 0:
            return {
                "allowed": False,
                "gate": "blocked",
                "remaining_h": 0,
                "message": f"Weekly GPU quota exhausted. Resets {await self._reset_countdown()}.",
            }

        if remaining_s < BLOCK_ALL_THRESHOLD_S:
            return {
                "allowed": False,
                "gate": "blocked",
                "remaining_h": round(remaining_h, 2),
                "message": f"Only {remaining_h:.1f}h GPU left this week. New sessions blocked to protect active users. Resets {await self._reset_countdown()}.",
            }

        if remaining_s < PAID_ONLY_THRESHOLD_S:
            if tier == "free":
                return {
                    "allowed": False,
                    "gate": "paid_only",
                    "remaining_h": round(remaining_h, 2),
                    "message": f"Under 3h GPU quota remaining — free sessions paused. Upgrade to paid to continue.",
                }
            return {
                "allowed": True,
                "gate": "paid_only",
                "remaining_h": round(remaining_h, 2),
                "message": f"Low quota ({remaining_h:.1f}h). Paid priority active.",
            }

        if remaining_s < WARN_THRESHOLD_S:
            return {
                "allowed": True,
                "gate": "warn",
                "remaining_h": round(remaining_h, 2),
                "message": f"GPU quota running low: {remaining_h:.1f}h remaining this week.",
            }

        return {
            "allowed": True,
            "gate": "normal",
            "remaining_h": round(remaining_h, 2),
            "message": "",
        }

    # ── Notebook lifecycle (tied to quota accrual) ────────────────────────────

    async def on_notebook_started(self):
        """Call this when the Kaggle notebook spins up."""
        now = datetime.now(timezone.utc)
        self._notebook_start_ts = now
        await self.redis.set(KEY_NB_STARTED, now.isoformat())
        await self.redis.set(KEY_NB_RUNNING, "1")

        # Cancel any pending idle shutdown
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()

        log.info("Notebook started — quota accrual begins.")

    async def on_notebook_stopped(self):
        """Call this when the Kaggle notebook stops (planned or unplanned)."""
        if self._notebook_start_ts:
            elapsed = (datetime.now(timezone.utc) - self._notebook_start_ts).total_seconds()
            await self._add_to_quota(elapsed)
            self._notebook_start_ts = None

        await self.redis.set(KEY_NB_RUNNING, "0")
        await self.redis.delete(KEY_NB_STARTED)
        log.info("Notebook stopped — quota accrual paused.")

    async def on_last_session_ended(self):
        """
        Call this when the pool has zero active sessions.
        Starts a 10-minute idle timer before stopping the notebook.
        """
        await self.redis.set(KEY_LAST_SESSION,
                             datetime.now(timezone.utc).isoformat())

        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()

        self._idle_task = asyncio.create_task(self._idle_shutdown())
        log.info("Last session ended. Notebook will stop in %ds if no new sessions.",
                 IDLE_SHUTDOWN_SECS)

    async def on_session_started(self):
        """Cancel any pending idle shutdown when a new session arrives."""
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            log.info("Idle shutdown cancelled — new session started.")

    # ── Accrual ───────────────────────────────────────────────────────────────

    async def _quota_loop(self):
        """Every 60s, flush accrued seconds to Redis and update the gate state."""
        while True:
            await asyncio.sleep(60)
            try:
                if self._notebook_start_ts and await self.redis.get(KEY_NB_RUNNING) == "1":
                    await self._add_to_quota(60)
                await self._update_gate()
                remaining = await self.remaining_hours()
                log.debug("Quota tick: %.2fh remaining", remaining)
            except Exception as e:
                log.error("Quota loop error: %s", e)

    async def _add_to_quota(self, seconds: float):
        """Atomically add seconds to the weekly accrual counter."""
        await self.redis.incrbyfloat(KEY_QUOTA_USED, seconds)

    async def _update_gate(self):
        """Recompute and cache the current gate state."""
        remaining_s = await self._remaining_seconds()
        if remaining_s <= 0:
            gate = "blocked"
        elif remaining_s < BLOCK_ALL_THRESHOLD_S:
            gate = "blocked"
        elif remaining_s < PAID_ONLY_THRESHOLD_S:
            gate = "paid_only"
        elif remaining_s < WARN_THRESHOLD_S:
            gate = "warn"
        else:
            gate = "normal"
        await self.redis.set(KEY_GATE, gate, ex=120)

    # ── Idle shutdown ─────────────────────────────────────────────────────────

    async def _idle_shutdown(self):
        """Stop the notebook after IDLE_SHUTDOWN_SECS of no activity."""
        await asyncio.sleep(IDLE_SHUTDOWN_SECS)
        log.info("Idle timeout reached — stopping notebook.")
        await self.on_notebook_stopped()
        # Signal lifecycle manager to stop the Kaggle kernel
        await self.redis.publish("novadev:broadcast",
                                 "STOP_NOTEBOOK_IDLE")

    # ── Weekly reset ──────────────────────────────────────────────────────────

    async def _ensure_week_reset(self):
        """Reset quota counter every Monday 00:00 UTC."""
        week_start = await self.redis.get(KEY_QUOTA_START)
        now = datetime.now(timezone.utc)

        if week_start:
            ws = datetime.fromisoformat(week_start)
            # Reset if we've crossed into a new week
            if now >= ws + timedelta(days=7):
                await self._do_reset(now)
        else:
            # First run — find this Monday
            days_since_monday = now.weekday()
            monday = (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            await self.redis.set(KEY_QUOTA_START, monday.isoformat())
            log.info("Quota week started: %s", monday.isoformat())

    async def _do_reset(self, now: datetime):
        """Perform the weekly reset."""
        await self.redis.set(KEY_QUOTA_USED, "0")
        monday = now.replace(hour=0, minute=0, second=0, microsecond=0)
        monday -= timedelta(days=monday.weekday())
        await self.redis.set(KEY_QUOTA_START, monday.isoformat())
        log.info("Weekly GPU quota reset. New week starts %s", monday.isoformat())

        # Notify all connected users
        await self.redis.publish("novadev:broadcast",
                                 '{"event":"quota_reset","message":"Weekly GPU quota has reset. 30h available."}')

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _remaining_seconds(self) -> float:
        used_str = await self.redis.get(KEY_QUOTA_USED)
        used_s = float(used_str) if used_str else 0.0

        # Also add live accrual for current notebook session
        if self._notebook_start_ts and await self.redis.get(KEY_NB_RUNNING) == "1":
            live_s = (datetime.now(timezone.utc) - self._notebook_start_ts).total_seconds()
            # live_s is already partially flushed by the 60s loop; subtract what's been flushed
            used_s_redis = float(await self.redis.get(KEY_QUOTA_USED) or 0)
            used_s = used_s_redis + (live_s % 60)  # only unflushed portion

        return max(0.0, QUOTA_TOTAL_SECONDS - used_s)

    async def remaining_hours(self) -> float:
        return round(await self._remaining_seconds() / 3600, 2)

    async def _reset_countdown(self) -> str:
        week_start = await self.redis.get(KEY_QUOTA_START)
        if not week_start:
            return "Monday 00:00 UTC"
        next_reset = datetime.fromisoformat(week_start) + timedelta(days=7)
        delta = next_reset - datetime.now(timezone.utc)
        h = int(delta.total_seconds() // 3600)
        m = int((delta.total_seconds() % 3600) // 60)
        return f"in {h}h {m}m (Monday 00:00 UTC)"

    async def status(self) -> dict:
        """Full quota status — used by admin API and frontend."""
        remaining_s = await self._remaining_seconds()
        used_s = QUOTA_TOTAL_SECONDS - remaining_s
        gate = await self.redis.get(KEY_GATE) or "normal"
        nb_running = await self.redis.get(KEY_NB_RUNNING) == "1"
        nb_age_s = 0.0
        if self._notebook_start_ts:
            nb_age_s = (datetime.now(timezone.utc) - self._notebook_start_ts).total_seconds()

        return {
            "quota_total_h":     30,
            "quota_used_h":      round(used_s / 3600, 2),
            "quota_remaining_h": round(remaining_s / 3600, 2),
            "quota_pct_used":    round(used_s / QUOTA_TOTAL_SECONDS * 100, 1),
            "gate":              gate,                        # normal/warn/paid_only/blocked
            "notebook_running":  nb_running,
            "notebook_age_min":  round(nb_age_s / 60, 1),
            "reset_countdown":   await self._reset_countdown(),
            "thresholds": {
                "warn_h":      WARN_THRESHOLD_S / 3600,
                "paid_only_h": PAID_ONLY_THRESHOLD_S / 3600,
                "block_all_h": BLOCK_ALL_THRESHOLD_S / 3600,
            },
        }
