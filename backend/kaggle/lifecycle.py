"""
NovaDev Cloud — Kaggle Notebook Lifecycle Manager

Kaggle hard limits:
  - 12h  max per notebook session
  - 30h  GPU per week (resets Monday UTC)

Strategy to fit within limits without disturbing users:

  At 11h 00m → spin up "standby" notebook (NB-2)
  At 11h 30m → auto-save all workspaces to cloud storage
  At 11h 45m → migrate all active sessions to NB-2 (zero-downtime)
  At 12h 00m → old notebook (NB-1) closes, NB-2 becomes primary
  At 12h 00m → NB-3 starts warming (for the next rotation)

Weekly 30h quota:
  - Track usage per-notebook in Redis
  - Warn admin + users when < 6h remains
  - If quota exhausted: block new sessions, keep existing alive,
    show countdown to weekly reset (Monday 00:00 UTC)
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import aioredis
import httpx

from core.config import settings
from core.websocket_manager import ws_manager

log = logging.getLogger("kaggle_lifecycle")

# ── Constants ─────────────────────────────────────────────────────────────────
NOTEBOOK_MAX_HOURS   = 12
WARMUP_AT_HOURS      = 11.0        # Start standby notebook
SAVE_AT_HOURS        = 11.5        # Auto-save all workspaces
MIGRATE_AT_HOURS     = 11.75       # Migrate sessions to standby
WARN_AT_HOURS        = 10.5        # Warn users (1.5h before limit)

WEEKLY_GPU_LIMIT_H   = 30
QUOTA_WARN_REMAINING = 6           # Warn when < 6h left this week
QUOTA_KEY            = "novadev:gpu_quota_used_seconds"
QUOTA_RESET_KEY      = "novadev:gpu_quota_reset_ts"

KAGGLE_BASE          = "https://www.kaggle.com/api/v1"


class NotebookState(str, Enum):
    OFFLINE  = "offline"
    STARTING = "starting"
    ACTIVE   = "active"      # Serving users
    STANDBY  = "standby"     # Warm, not yet primary
    DRAINING = "draining"    # Being migrated away from
    CLOSED   = "closed"


@dataclass
class KaggleNotebook:
    index: int                          # 1, 2, 3... rolling
    kernel_slug: str = ""
    state: NotebookState = NotebookState.OFFLINE
    started_at: Optional[datetime] = None
    gpu_seconds_used: float = 0.0

    @property
    def age_hours(self) -> float:
        if not self.started_at:
            return 0.0
        delta = datetime.now(timezone.utc) - self.started_at
        return delta.total_seconds() / 3600

    @property
    def hours_remaining(self) -> float:
        return max(0.0, NOTEBOOK_MAX_HOURS - self.age_hours)


class KaggleLifecycleManager:
    """
    Manages the rolling notebook strategy to stay within Kaggle's limits.
    Runs as a background asyncio task.
    """

    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.auth = (settings.KAGGLE_USERNAME, settings.KAGGLE_KEY)

        # Rolling notebook pool — we keep at most 2 alive at once
        self.primary: Optional[KaggleNotebook] = None
        self.standby: Optional[KaggleNotebook] = None
        self._nb_index = 0

        # Listeners that get called on migration
        self._on_migrate_callbacks = []

    # ── Startup ──────────────────────────────────────────────────────────────

    async def start(self):
        self.redis = await aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await self._ensure_quota_reset()
        self.primary = await self._launch_notebook()
        log.info("Primary notebook online: %s", self.primary.kernel_slug)
        asyncio.create_task(self._lifecycle_loop())
        asyncio.create_task(self._quota_tracker())

    # ── Main lifecycle loop ──────────────────────────────────────────────────

    async def _lifecycle_loop(self):
        """Checks notebook age every 60s and drives the rotation state machine."""
        while True:
            await asyncio.sleep(60)
            try:
                await self._tick()
            except Exception as e:
                log.error("Lifecycle tick error: %s", e)

    async def _tick(self):
        if not self.primary:
            return

        age = self.primary.age_hours
        remaining = self.primary.hours_remaining

        # ── Phase 1: warm up standby ─────────────────────────────────────────
        if age >= WARMUP_AT_HOURS and self.standby is None:
            log.info("Notebook age %.1fh — launching standby notebook", age)
            self.standby = await self._launch_notebook(state=NotebookState.STANDBY)
            await self._broadcast_all({
                "event":   "notebook_rotation_warning",
                "message": "A fresh GPU notebook is warming up. Your session will migrate in ~45 minutes with zero interruption.",
                "minutes_until_migrate": int((MIGRATE_AT_HOURS - age) * 60),
                "hours_remaining_primary": round(remaining, 2),
            })

        # ── Phase 2: auto-save all workspaces ────────────────────────────────
        if age >= SAVE_AT_HOURS and self.standby:
            log.info("Auto-saving all workspaces before migration...")
            await self._auto_save_all()
            await self._broadcast_all({
                "event":   "notebook_autosave",
                "message": "Auto-save complete. Migration to fresh notebook starting soon.",
            })

        # ── Phase 3: migrate sessions ─────────────────────────────────────────
        if age >= MIGRATE_AT_HOURS and self.standby and self.standby.state == NotebookState.STANDBY:
            log.info("Migrating sessions to standby notebook...")
            await self._migrate_to_standby()

        # ── Phase 4: retire old notebook ──────────────────────────────────────
        if age >= NOTEBOOK_MAX_HOURS - 0.05:  # within 3 min of limit
            log.info("Primary notebook reached limit. Retiring.")
            await self._retire_primary()

    # ── Notebook operations ──────────────────────────────────────────────────

    async def _launch_notebook(self, state=NotebookState.ACTIVE) -> KaggleNotebook:
        self._nb_index += 1
        slug = f"novadev-pool-{self._nb_index:04d}"

        nb = KaggleNotebook(
            index=self._nb_index,
            kernel_slug=slug,
            state=NotebookState.STARTING,
            started_at=datetime.now(timezone.utc),
        )

        payload = {
            "id":              f"{settings.KAGGLE_USERNAME}/{slug}",
            "title":           f"NovaDev Pool {self._nb_index:04d}",
            "code_file":       self._pool_kernel_source(),
            "language":        "python",
            "kernel_type":     "notebook",
            "is_private":      True,
            "enable_gpu":      True,
            "enable_internet": False,
            "accelerator":     settings.KAGGLE_ACCELERATOR,
        }

        try:
            async with httpx.AsyncClient(auth=self.auth, timeout=30) as client:
                resp = await client.post(f"{KAGGLE_BASE}/kernels", json=payload)
                resp.raise_for_status()
            nb.state = state
            log.info("Notebook %s launched (state=%s)", slug, state)
        except Exception as e:
            log.error("Failed to launch notebook %s: %s", slug, e)
            nb.state = NotebookState.OFFLINE

        await self.redis.set(f"novadev:notebook:{slug}:started_at",
                             nb.started_at.isoformat(), ex=86400)
        return nb

    async def _migrate_to_standby(self):
        """Promote standby → primary, drain old primary."""
        old = self.primary
        new = self.standby

        new.state = NotebookState.ACTIVE
        old.state = NotebookState.DRAINING
        self.primary = new
        self.standby = None

        # Fire migration callbacks (Docker pool will re-point containers)
        for cb in self._on_migrate_callbacks:
            try:
                await cb(old_slug=old.kernel_slug, new_slug=new.kernel_slug)
            except Exception as e:
                log.error("Migration callback error: %s", e)

        await self._broadcast_all({
            "event":   "notebook_migrated",
            "message": "Your session has been seamlessly moved to a fresh GPU notebook. No work was lost.",
            "new_notebook": new.kernel_slug,
        })
        log.info("Migration complete: %s → %s", old.kernel_slug, new.kernel_slug)

        # Stop old notebook after a short grace period
        asyncio.create_task(self._stop_notebook_after(old, delay_min=5))

    async def _retire_primary(self):
        if self.primary:
            await self._stop_notebook(self.primary)
            self.primary = None

    async def _stop_notebook_after(self, nb: KaggleNotebook, delay_min: int):
        await asyncio.sleep(delay_min * 60)
        await self._stop_notebook(nb)

    async def _stop_notebook(self, nb: KaggleNotebook):
        slug = f"{settings.KAGGLE_USERNAME}/{nb.kernel_slug}"
        try:
            async with httpx.AsyncClient(auth=self.auth, timeout=15) as client:
                await client.post(f"{KAGGLE_BASE}/kernels/{slug}/stop")
            nb.state = NotebookState.CLOSED
            log.info("Notebook %s stopped.", nb.kernel_slug)
        except Exception as e:
            log.warning("Could not stop notebook %s: %s", nb.kernel_slug, e)

    # ── Auto-save ─────────────────────────────────────────────────────────────

    async def _auto_save_all(self):
        """
        Signals all active containers to checkpoint their workspace to S3/GCS.
        Each container runs a sidecar that listens for this signal via Redis pub/sub.
        """
        await self.redis.publish("novadev:broadcast", "AUTOSAVE_ALL")
        log.info("AUTOSAVE_ALL broadcast sent.")

    # ── Weekly quota tracker ──────────────────────────────────────────────────

    async def _quota_tracker(self):
        """Every 5 min, fetch Kaggle quota and warn when running low."""
        while True:
            await asyncio.sleep(300)
            try:
                used_h = await self._fetch_kaggle_gpu_used()
                remaining_h = WEEKLY_GPU_LIMIT_H - used_h
                await self.redis.set("novadev:gpu_quota_remaining_h",
                                     str(round(remaining_h, 2)), ex=600)

                if remaining_h <= 0:
                    log.warning("GPU quota EXHAUSTED. Blocking new sessions.")
                    await self.redis.set("novadev:quota_exhausted", "1", ex=3600)
                    await self._broadcast_all({
                        "event":   "quota_exhausted",
                        "message": "Weekly GPU quota reached. New sessions blocked until Monday 00:00 UTC. Existing sessions continue.",
                        "resets_at": self._next_monday_utc(),
                    })
                elif remaining_h <= QUOTA_WARN_REMAINING:
                    await self.redis.delete("novadev:quota_exhausted")
                    log.warning("GPU quota low: %.1fh remaining", remaining_h)
                    await self._broadcast_all({
                        "event":   "quota_low",
                        "message": f"Only {remaining_h:.1f}h of GPU time left this week. Resets Monday 00:00 UTC.",
                        "remaining_hours": round(remaining_h, 1),
                    })
                else:
                    await self.redis.delete("novadev:quota_exhausted")
            except Exception as e:
                log.error("Quota tracker error: %s", e)

    async def _fetch_kaggle_gpu_used(self) -> float:
        """Fetch used GPU hours from Kaggle API."""
        try:
            async with httpx.AsyncClient(auth=self.auth, timeout=15) as client:
                resp = await client.get(f"{KAGGLE_BASE}/users/{settings.KAGGLE_USERNAME}")
                data = resp.json()
                return float(data.get("gpuQuotaUsed", 0))
        except Exception:
            return 0.0

    async def _ensure_quota_reset(self):
        """On startup, check if we need to reset the weekly quota tracking."""
        reset_ts = await self.redis.get(QUOTA_RESET_KEY)
        now = datetime.now(timezone.utc)
        if reset_ts:
            reset_dt = datetime.fromisoformat(reset_ts)
            if now >= reset_dt:
                await self.redis.delete(QUOTA_KEY)
                await self.redis.set(QUOTA_RESET_KEY, self._next_monday_utc())
        else:
            await self.redis.set(QUOTA_RESET_KEY, self._next_monday_utc())

    def _next_monday_utc(self) -> str:
        now = datetime.now(timezone.utc)
        days_ahead = (7 - now.weekday()) % 7 or 7
        next_monday = (now + timedelta(days=days_ahead)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return next_monday.isoformat()

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _broadcast_all(self, message: dict):
        await ws_manager.broadcast(message)

    def on_migrate(self, callback):
        """Register a callback for when a notebook migration happens."""
        self._on_migrate_callbacks.append(callback)

    def _pool_kernel_source(self) -> str:
        """The long-running kernel that keeps the Kaggle notebook alive."""
        return """
import time, os, subprocess, signal

# Start Jupyter kernel gateway for container communication
subprocess.Popen(["jupyter", "kernelgateway", "--port=8888",
                  "--KernelGatewayApp.allow_origin=*"])

# Keep notebook alive (Kaggle kills idle notebooks after 20min)
print("NovaDev pool kernel alive.")
while True:
    time.sleep(60)
    print(f"Heartbeat: {time.strftime('%H:%M:%S')}")
"""

    def status(self) -> dict:
        return {
            "primary": {
                "slug": self.primary.kernel_slug if self.primary else None,
                "state": self.primary.state if self.primary else None,
                "age_hours": round(self.primary.age_hours, 2) if self.primary else 0,
                "hours_remaining": round(self.primary.hours_remaining, 2) if self.primary else 0,
            },
            "standby": {
                "slug": self.standby.kernel_slug if self.standby else None,
                "state": self.standby.state if self.standby else None,
            },
            "warmup_at_hours": WARMUP_AT_HOURS,
            "migrate_at_hours": MIGRATE_AT_HOURS,
            "notebook_limit_hours": NOTEBOOK_MAX_HOURS,
            "weekly_limit_hours": WEEKLY_GPU_LIMIT_H,
        }


lifecycle_manager = KaggleLifecycleManager()
