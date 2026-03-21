"""
NovaDev Cloud — Relay Webhook Handler
Receives all events from the GitHub Actions relay controller.
This is the bridge between GitHub Actions and your running backend.

Events handled:
  notebook_started        → mark cluster as live, open session gate
  notebook_idle_stopped   → mark cluster as offline, queue users
  notebook_start_failed   → alert admin, keep queue open
  quota_report            → update quota display for all users
  quota_gate_update       → enforce new gate (block/warn/paid_only/normal)
  quota_exhausted         → block all new sessions
  quota_weekly_reset      → reset all per-user counters
  rotation_starting       → warn users, trigger auto-save
  rotation_ready          → migrate sessions to new notebook
  rotation_complete       → confirm migration done
  rotation_failed         → alert admin
"""

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.config import settings
from core.websocket_manager import ws_manager

log = logging.getLogger("relay_webhook")
router = APIRouter()

# Per-user weekly quota limits (enforced by backend, tracked in Redis)
FREE_USER_QUOTA_H = 1.0
PAID_USER_QUOTA_H = 2.0


def verify_signature(body: bytes, sig_header: str) -> bool:
    expected = hmac.new(
        settings.BACKEND_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig_header or "")


@router.post("/api/internal/relay-webhook")
async def relay_webhook(request: Request):
    body = await request.body()
    sig  = request.headers.get("X-NovaDev-Sig", "")

    if not verify_signature(body, sig):
        log.warning("Invalid webhook signature — rejected.")
        raise HTTPException(401, "Invalid signature")

    event_data = json.loads(body)
    event = event_data.get("event", "unknown")
    log.info("Relay webhook: %s", event)

    handler = EVENT_HANDLERS.get(event)
    if handler:
        await handler(request.app.state, event_data)
    else:
        log.warning("Unhandled relay event: %s", event)

    return {"ok": True, "event": event}


# ── Event handlers ────────────────────────────────────────────────────────────

async def on_notebook_started(app_state, data: dict):
    slug = data.get("slug", "")
    log.info("Notebook online: %s", slug)

    # Mark cluster as live in Redis
    redis = app_state.redis
    await redis.set("novadev:cluster:online", "1")
    await redis.set("novadev:cluster:slug", slug)
    await redis.set("novadev:cluster:started_at", data.get("started_at", ""))

    # Broadcast to all connected users
    await ws_manager.broadcast({
        "event":   "cluster_online",
        "message": "GPU cluster is online. Your session is ready.",
        "slug":    slug,
    })

    # Drain any queued users now that cluster is live
    docker_pool  = app_state.docker_pool
    queue_manager = app_state.queue_manager
    idle_slots = sum(
        1 for s in docker_pool.slots.values()
        if s.state.value == "idle"
    )
    for _ in range(idle_slots):
        from docker.pool import SlotType
        await queue_manager.on_slot_released(SlotType.FREE)
        await queue_manager.on_slot_released(SlotType.PAID)


async def on_notebook_idle_stopped(app_state, data: dict):
    log.info("Notebook went idle and was stopped.")
    redis = app_state.redis
    await redis.set("novadev:cluster:online", "0")
    await redis.delete("novadev:cluster:slug")

    await ws_manager.broadcast({
        "event":   "cluster_offline",
        "message": "GPU cluster is offline (no active users). It will restart when someone joins.",
    })


async def on_quota_gate_update(app_state, data: dict):
    gate        = data.get("gate", "normal")
    remaining_h = data.get("remaining_h", 0)
    redis       = app_state.redis
    await redis.set("novadev:quota:gate", gate, ex=600)
    await redis.set("novadev:quota:remaining_h", str(remaining_h), ex=600)

    msg_map = {
        "warn":      f"GPU quota running low: {remaining_h:.1f}h remaining this week.",
        "paid_only": f"Only {remaining_h:.1f}h GPU quota left — free sessions paused.",
        "blocked":   f"Only {remaining_h:.1f}h GPU quota left — all new sessions blocked.",
    }
    if gate in msg_map:
        await ws_manager.broadcast({
            "event":       "quota_gate_change",
            "gate":        gate,
            "remaining_h": remaining_h,
            "message":     msg_map[gate],
        })


async def on_quota_exhausted(app_state, data: dict):
    log.warning("GPU QUOTA EXHAUSTED — blocking all new sessions.")
    redis = app_state.redis
    await redis.set("novadev:quota:gate", "exhausted", ex=86400)

    reset_at = data.get("resets_at", "Monday 00:00 UTC")
    await ws_manager.broadcast({
        "event":    "quota_exhausted",
        "message":  f"Weekly GPU quota exhausted. New sessions blocked until {reset_at}.",
        "resets_at": reset_at,
    })


async def on_quota_weekly_reset(app_state, data: dict):
    log.info("Weekly quota reset!")
    redis = app_state.redis

    # Reset system quota gate
    await redis.set("novadev:quota:gate", "normal")
    await redis.set("novadev:quota:remaining_h", str(data.get("new_quota_h", 30)))
    await redis.delete("novadev:quota:exhausted")

    # Reset all per-user quota counters
    user_keys = await redis.keys("novadev:user_quota:*")
    if user_keys:
        await redis.delete(*user_keys)
        log.info("Reset %d user quota counters.", len(user_keys))

    await ws_manager.broadcast({
        "event":   "quota_reset",
        "message": "Weekly GPU quota has reset! 30h available. Free: 1h/user · Paid: 2h/user.",
        "free_h":  data.get("free_user_quota_h", FREE_USER_QUOTA_H),
        "paid_h":  data.get("paid_user_quota_h", PAID_USER_QUOTA_H),
    })


async def on_rotation_starting(app_state, data: dict):
    log.info("Notebook rotation starting: %s → %s",
             data.get("old_slug"), data.get("new_slug"))

    # Trigger auto-save for all active sessions
    redis = app_state.redis
    await redis.publish("novadev:broadcast", "AUTOSAVE_ALL")

    await ws_manager.broadcast({
        "event":   "rotation_starting",
        "message": "GPU notebook is rotating (12h limit approaching). Your work is being auto-saved. Migration in ~2 minutes.",
    })


async def on_rotation_ready(app_state, data: dict):
    """New notebook is running — migrate sessions."""
    new_slug = data.get("new_slug", "")
    log.info("New notebook ready: %s — migrating sessions.", new_slug)

    redis = app_state.redis
    await redis.set("novadev:cluster:slug", new_slug)
    await redis.set("novadev:cluster:started_at", data.get("ts", ""))

    # Signal Docker pool to re-point kernel connections to new notebook
    await redis.publish("novadev:broadcast", json.dumps({
        "cmd":      "REROUTE_KERNEL",
        "new_slug": new_slug,
    }))

    await ws_manager.broadcast({
        "event":   "rotation_migrating",
        "message": "Migrating to fresh GPU notebook. Your session will resume in seconds.",
    })


async def on_rotation_complete(app_state, data: dict):
    log.info("Rotation complete. New primary: %s", data.get("new_slug"))
    await ws_manager.broadcast({
        "event":   "rotation_complete",
        "message": "Migration complete. GPU session restored with no work lost.",
    })


async def on_rotation_failed(app_state, data: dict):
    log.error("Rotation FAILED: %s", data.get("reason", ""))
    await ws_manager.broadcast({
        "event":   "rotation_failed",
        "message": "GPU notebook rotation encountered an issue. Admin has been alerted. Your session may need to restart.",
    })


async def on_notebook_start_failed(app_state, data: dict):
    log.error("Notebook start failed: %s", data.get("error", ""))
    await ws_manager.broadcast({
        "event":   "cluster_start_failed",
        "message": "GPU cluster failed to start. Retrying in 2 minutes. You remain in queue.",
    })


# ── Demand endpoint (queried by relay controller every 30s) ───────────────────

@router.get("/api/internal/demand")
async def check_demand(request: Request):
    """
    Called by GitHub Actions relay controller to check if the notebook
    should be running. Returns True if any user needs GPU.
    """
    docker_pool   = request.app.state.docker_pool
    queue_manager = request.app.state.queue_manager

    active_sessions = sum(
        1 for s in docker_pool.slots.values()
        if s.state.value == "active"
    )
    queue_stats = await queue_manager.queue_stats()
    queue_len   = queue_stats.get("total_queued", 0)

    has_demand = active_sessions > 0 or queue_len > 0

    return {
        "has_demand":      has_demand,
        "active_sessions": active_sessions,
        "queue_length":    queue_len,
    }


# ── Per-user quota enforcement ────────────────────────────────────────────────

async def check_user_quota(redis, user_id: str, tier: str) -> dict:
    """
    Enforce per-user weekly quota: free=1h, paid=2h.
    Called before granting any session slot.
    """
    key = f"novadev:user_quota:{user_id}"
    used_s = float(await redis.get(key) or 0)
    limit_h = PAID_USER_QUOTA_H if tier == "paid" else FREE_USER_QUOTA_H
    limit_s = limit_h * 3600
    remaining_s = max(0, limit_s - used_s)

    if remaining_s <= 0:
        return {
            "allowed": False,
            "reason":  f"Your weekly {tier} quota ({limit_h}h) is exhausted. Resets Monday 00:00 UTC.",
            "used_h":  round(used_s / 3600, 2),
            "limit_h": limit_h,
        }

    return {
        "allowed":     True,
        "used_h":      round(used_s / 3600, 2),
        "remaining_h": round(remaining_s / 3600, 2),
        "limit_h":     limit_h,
    }


async def accrue_user_quota(redis, user_id: str, session_seconds: float):
    """Add consumed GPU seconds to user's weekly counter."""
    key = f"novadev:user_quota:{user_id}"
    await redis.incrbyfloat(key, session_seconds)
    # Expire after 8 days (covers full week + buffer)
    await redis.expire(key, 8 * 86400)


# ── Event dispatch table ──────────────────────────────────────────────────────

EVENT_HANDLERS = {
    "notebook_started":     on_notebook_started,
    "notebook_idle_stopped":on_notebook_idle_stopped,
    "notebook_start_failed":on_notebook_start_failed,
    "quota_report":         on_quota_gate_update,
    "quota_gate_update":    on_quota_gate_update,
    "quota_exhausted":      on_quota_exhausted,
    "quota_weekly_reset":   on_quota_weekly_reset,
    "quota_low":            on_quota_gate_update,
    "quota_critical":       on_quota_exhausted,
    "rotation_starting":    on_rotation_starting,
    "rotation_ready":       on_rotation_ready,
    "rotation_complete":    on_rotation_complete,
    "rotation_failed":      on_rotation_failed,
}
