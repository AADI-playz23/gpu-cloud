#!/usr/bin/env python3
"""
NovaDev Cloud — Quota Tracker
Runs every hour via GitHub Actions.

Responsibilities:
  - Fetch used GPU hours from Kaggle API
  - Track per-user quota (free: 1h/week, paid: 2h/week)
  - Detect weekly reset (Monday 00:00 UTC)
  - Send full quota report to backend
  - Alert admin if quota is critical
"""

import hashlib
import hmac
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("quota_tracker")

KAGGLE_USERNAME     = os.environ["KAGGLE_USERNAME"]
KAGGLE_KEY          = os.environ["KAGGLE_KEY"]
BACKEND_URL         = os.environ["BACKEND_WEBHOOK_URL"]
BACKEND_SECRET      = os.environ["BACKEND_SECRET"]
KAGGLE_BASE         = "https://www.kaggle.com/api/v1"
KAGGLE_AUTH         = (KAGGLE_USERNAME, KAGGLE_KEY)

# Quota config
SYSTEM_QUOTA_H      = 30.0   # Total system GPU hours per week
FREE_USER_QUOTA_H   = 1.0    # Each free user gets 1h/week
PAID_USER_QUOTA_H   = 2.0    # Each paid user gets 2h/week

# Gate thresholds
GATE_WARN_H         = 6.0
GATE_PAID_ONLY_H    = 3.0
GATE_BLOCK_ALL_H    = 1.0


def sign(payload: str) -> str:
    return hmac.new(BACKEND_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def next_monday_utc() -> datetime:
    now = datetime.now(timezone.utc)
    days_ahead = (7 - now.weekday()) % 7 or 7
    return (now + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def is_monday_reset() -> bool:
    """True if this is the Monday 00:00–01:00 UTC window (weekly reset)."""
    now = datetime.now(timezone.utc)
    return now.weekday() == 0 and now.hour == 0


def compute_gate(remaining_h: float) -> str:
    if remaining_h <= 0:
        return "exhausted"
    if remaining_h <= GATE_BLOCK_ALL_H:
        return "blocked"
    if remaining_h <= GATE_PAID_ONLY_H:
        return "paid_only"
    if remaining_h <= GATE_WARN_H:
        return "warn"
    return "normal"


def user_quota_report(
    free_users_active: int,
    paid_users_active: int,
    system_used_h: float,
) -> dict:
    """
    Calculate per-user quota allocations given current system usage.

    The system quota (30h) is shared by ALL users.
    Per-user limits (1h free / 2h paid) are enforced separately by the backend.
    The system quota just determines whether the notebook is running at all.
    """
    system_remaining_h = max(0, SYSTEM_QUOTA_H - system_used_h)

    # Max sessions possible in remaining quota
    free_session_min    = 30      # minutes
    paid_session_min    = 120     # minutes
    remaining_min       = system_remaining_h * 60

    max_free_sessions   = int(remaining_min / free_session_min)
    max_paid_sessions   = int(remaining_min / paid_session_min)

    return {
        "system": {
            "total_h":     SYSTEM_QUOTA_H,
            "used_h":      round(system_used_h, 2),
            "remaining_h": round(system_remaining_h, 2),
            "pct_used":    round(system_used_h / SYSTEM_QUOTA_H * 100, 1),
        },
        "per_user_limits": {
            "free_h":      FREE_USER_QUOTA_H,
            "paid_h":      PAID_USER_QUOTA_H,
            "free_session_min": free_session_min,
            "paid_session_min": paid_session_min,
        },
        "capacity": {
            "max_free_sessions_remaining": max_free_sessions,
            "max_paid_sessions_remaining": max_paid_sessions,
        },
        "gate":            compute_gate(system_remaining_h),
        "resets_at":       next_monday_utc().isoformat(),
        "is_weekly_reset": is_monday_reset(),
    }


async def run():
    async with httpx.AsyncClient(timeout=20) as client:

        # ── 1. Fetch Kaggle quota ──────────────────────────────────────────────
        try:
            resp = await client.get(
                f"{KAGGLE_BASE}/users/{KAGGLE_USERNAME}",
                auth=KAGGLE_AUTH,
            )
            resp.raise_for_status()
            kaggle_data = resp.json()
            used_h = float(kaggle_data.get("gpuQuotaUsed", 0))
            log.info("Kaggle GPU used this week: %.2fh / %.0fh", used_h, SYSTEM_QUOTA_H)
        except Exception as e:
            log.error("Failed to fetch Kaggle quota: %s", e)
            used_h = 0.0

        # ── 2. Fetch active user counts from backend ───────────────────────────
        free_users  = 0
        paid_users  = 0
        try:
            stats_resp = await client.get(
                f"{BACKEND_URL}/api/sessions/stats",
                headers={"X-NovaDev-Secret": BACKEND_SECRET},
            )
            stats = stats_resp.json()
            free_users = stats.get("free_active", 0)
            paid_users = stats.get("paid_active", 0)
        except Exception as e:
            log.warning("Could not fetch session stats: %s", e)

        # ── 3. Build report ───────────────────────────────────────────────────
        report = user_quota_report(free_users, paid_users, used_h)
        gate   = report["gate"]
        remaining_h = report["system"]["remaining_h"]

        log.info("Gate: %s | Remaining: %.2fh | Free active: %d | Paid active: %d",
                 gate, remaining_h, free_users, paid_users)

        # ── 4. Weekly reset detection ──────────────────────────────────────────
        if is_monday_reset():
            log.info("WEEKLY RESET DETECTED — notifying backend.")
            reset_payload = json.dumps({
                "event": "quota_weekly_reset",
                "new_quota_h": SYSTEM_QUOTA_H,
                "free_user_quota_h": FREE_USER_QUOTA_H,
                "paid_user_quota_h": PAID_USER_QUOTA_H,
                "resets_at": next_monday_utc().isoformat(),
            })
            await client.post(
                f"{BACKEND_URL}/api/internal/relay-webhook",
                content=reset_payload,
                headers={
                    "Content-Type": "application/json",
                    "X-NovaDev-Sig": sign(reset_payload),
                },
            )

        # ── 5. Send quota report webhook ───────────────────────────────────────
        event = (
            "quota_exhausted"  if gate == "exhausted" else
            "quota_critical"   if gate == "blocked"   else
            "quota_low"        if gate in ("warn", "paid_only") else
            "quota_report"
        )
        payload = json.dumps({"event": event, **report})
        await client.post(
            f"{BACKEND_URL}/api/internal/relay-webhook",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-NovaDev-Sig": sign(payload),
            },
        )
        log.info("Quota report sent. Event: %s", event)

        # ── 6. Print summary ───────────────────────────────────────────────────
        print("\n" + "="*50)
        print(f"  NovaDev GPU Quota Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print("="*50)
        print(f"  System: {used_h:.2f}h used / {SYSTEM_QUOTA_H:.0f}h total ({report['system']['pct_used']}%)")
        print(f"  Remaining: {remaining_h:.2f}h  |  Gate: {gate.upper()}")
        print(f"  Per user — Free: {FREE_USER_QUOTA_H}h/week  |  Paid: {PAID_USER_QUOTA_H}h/week")
        print(f"  Resets: {next_monday_utc().strftime('%A %Y-%m-%d %H:%M UTC')}")
        print(f"  Max free sessions remaining: {report['capacity']['max_free_sessions_remaining']}")
        print(f"  Max paid sessions remaining: {report['capacity']['max_paid_sessions_remaining']}")
        print("="*50 + "\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
