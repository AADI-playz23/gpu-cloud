"""
NovaDev Cloud — Docker Pool Manager
Manages 40 pre-warmed containers: 30 free + 10 paid.
Each container maps to a VRAM partition on T4×2.

Free  slot: ~0.53 GB VRAM  (--memory / cgroup limit)
Paid  slot: ~3.20 GB VRAM  (6× free — double session power)
"""
import asyncio
import docker
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from core.config import settings

log = logging.getLogger("docker_pool")


class SlotType(str, Enum):
    FREE = "free"
    PAID = "paid"


class SlotState(str, Enum):
    IDLE     = "idle"       # Container warm, waiting for user
    ACTIVE   = "active"     # Running user session
    QUEUED   = "queued"     # Assigned to queued user, starting up
    DRAINING = "draining"   # Session ended, container recycling


@dataclass
class ContainerSlot:
    slot_id: int
    slot_type: SlotType
    state: SlotState = SlotState.IDLE
    container_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    started_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    port: int = 0
    vram_gb: float = 0.0


class DockerPool:
    def __init__(self, free_slots: int = 30, paid_slots: int = 10):
        self.free_slots = free_slots
        self.paid_slots = paid_slots
        self.slots: Dict[int, ContainerSlot] = {}
        self._client = docker.from_env()
        self._lock = asyncio.Lock()

    async def initialise(self):
        """Pre-warm all containers at startup."""
        log.info("Pre-warming %d free + %d paid containers...",
                 self.free_slots, self.paid_slots)

        # Slot layout:
        #   0–29  → free  (30 slots)
        #  30–39  → paid  (10 slots)
        for i in range(self.free_slots):
            self.slots[i] = ContainerSlot(
                slot_id=i,
                slot_type=SlotType.FREE,
                port=settings.DOCKER_BASE_PORT + i,
                vram_gb=settings.FREE_SLOT_VRAM_GB,
            )
            await self._start_idle_container(self.slots[i])

        for i in range(self.paid_slots):
            idx = self.free_slots + i
            self.slots[idx] = ContainerSlot(
                slot_id=idx,
                slot_type=SlotType.PAID,
                port=settings.DOCKER_BASE_PORT + idx,
                vram_gb=settings.PAID_SLOT_VRAM_GB,
            )
            await self._start_idle_container(self.slots[idx])

        log.info("Docker pool ready: %d slots total", len(self.slots))

    async def _start_idle_container(self, slot: ContainerSlot):
        """Launch a warm container for the given slot."""
        image = (settings.DOCKER_IMAGE_PAID
                 if slot.slot_type == SlotType.PAID
                 else settings.DOCKER_IMAGE_FREE)

        # VRAM limit via NVIDIA cgroup (bytes)
        vram_bytes = int(slot.vram_gb * 1024 ** 3)

        try:
            container = self._client.containers.run(
                image,
                detach=True,
                name=f"novadev-slot-{slot.slot_id}",
                network=settings.DOCKER_NETWORK,
                ports={"8888/tcp": slot.port},
                environment={
                    "SLOT_ID": str(slot.slot_id),
                    "SLOT_TYPE": slot.slot_type,
                    "VRAM_LIMIT_GB": str(slot.vram_gb),
                },
                device_requests=[
                    docker.types.DeviceRequest(
                        count=-1,
                        capabilities=[["gpu"]],
                        options={"memory": str(vram_bytes)},
                    )
                ],
                mem_limit="8g",
                labels={"novadev": "true", "slot": str(slot.slot_id)},
            )
            slot.container_id = container.id
            log.debug("Slot %d container started: %s", slot.slot_id, container.short_id)
        except Exception as e:
            log.error("Failed to start container for slot %d: %s", slot.slot_id, e)

    # ── Slot acquisition ────────────────────────────────────────────────────

    async def acquire_slot(self, slot_type: SlotType,
                           user_id: str, session_id: str) -> Optional[ContainerSlot]:
        """Claim an idle slot for a user. Returns None if no slot available."""
        async with self._lock:
            for slot in self.slots.values():
                if slot.slot_type == slot_type and slot.state == SlotState.IDLE:
                    timeout_min = (settings.PAID_SESSION_MINUTES
                                   if slot_type == SlotType.PAID
                                   else settings.FREE_SESSION_MINUTES)
                    slot.state     = SlotState.ACTIVE
                    slot.user_id   = user_id
                    slot.session_id = session_id
                    slot.started_at = datetime.utcnow()
                    slot.expires_at = slot.started_at + timedelta(minutes=timeout_min)
                    log.info("Slot %d assigned → user %s (session %s)",
                             slot.slot_id, user_id, session_id)
                    return slot
        return None

    async def release_slot(self, slot_id: int):
        """Release a slot back to idle after session ends."""
        async with self._lock:
            slot = self.slots.get(slot_id)
            if not slot:
                return
            slot.state      = SlotState.DRAINING
            slot.user_id    = None
            slot.session_id = None
            slot.started_at = None
            slot.expires_at = None

        # Recycle container outside the lock
        await self._recycle_container(slot)

    async def _recycle_container(self, slot: ContainerSlot):
        """Stop + restart container to get a clean environment."""
        try:
            container = self._client.containers.get(slot.container_id)
            container.stop(timeout=5)
            container.remove()
        except Exception:
            pass

        await self._start_idle_container(slot)
        async with self._lock:
            slot.state = SlotState.IDLE
        log.info("Slot %d recycled and ready.", slot.slot_id)

    # ── Watchdog ─────────────────────────────────────────────────────────────

    async def expire_sessions(self):
        """Called by queue manager to kill expired sessions."""
        now = datetime.utcnow()
        to_release = []
        for slot in self.slots.values():
            if slot.state == SlotState.ACTIVE and slot.expires_at and now >= slot.expires_at:
                log.info("Session expired on slot %d (user %s)", slot.slot_id, slot.user_id)
                to_release.append(slot.slot_id)

        for slot_id in to_release:
            await self.release_slot(slot_id)
        return to_release

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = len(self.slots)
        active = sum(1 for s in self.slots.values() if s.state == SlotState.ACTIVE)
        idle   = sum(1 for s in self.slots.values() if s.state == SlotState.IDLE)
        vram_used = sum(s.vram_gb for s in self.slots.values() if s.state == SlotState.ACTIVE)
        return {
            "total_slots": total,
            "active": active,
            "idle": idle,
            "vram_used_gb": round(vram_used, 2),
            "vram_total_gb": settings.TOTAL_VRAM_GB,
            "vram_pct": round(vram_used / settings.TOTAL_VRAM_GB * 100, 1),
            "free_active": sum(1 for s in self.slots.values()
                               if s.slot_type == SlotType.FREE and s.state == SlotState.ACTIVE),
            "paid_active": sum(1 for s in self.slots.values()
                               if s.slot_type == SlotType.PAID and s.state == SlotState.ACTIVE),
        }

    async def shutdown_all(self):
        for slot in self.slots.values():
            try:
                c = self._client.containers.get(slot.container_id)
                c.stop(timeout=3)
                c.remove()
            except Exception:
                pass
