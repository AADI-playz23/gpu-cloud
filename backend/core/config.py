"""
NovaDev Cloud — Configuration
All settings driven by environment variables.
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "NovaDev Cloud"
    SECRET_KEY: str = "change-me-in-production"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "https://yourdomain.com"]

    # ── Tier limits ──────────────────────────────────────
    FREE_CONTAINER_SLOTS: int = 30        # Slots for free users
    PAID_CONTAINER_SLOTS: int = 10        # Slots for paid users
    FREE_SESSION_MINUTES: int = 30        # Session timeout free
    PAID_SESSION_MINUTES: int = 120       # Session timeout paid (2h)
    FREE_CONTAINERS_PER_SESSION: int = 1
    PAID_CONTAINERS_PER_SESSION: int = 2
    MAX_QUEUE_LENGTH: int = 50
    IDLE_EXPIRE_MINUTES: int = 5          # Kill idle sessions early

    # ── GPU / VRAM partition ─────────────────────────────
    # 2x T4 = 32 GB total
    # 40 partitions: 30 free @ ~0.53 GB each, 10 paid @ 3.2 GB each
    TOTAL_VRAM_GB: float = 32.0
    FREE_SLOT_VRAM_GB: float = 0.53      # ~16 GB / 30
    PAID_SLOT_VRAM_GB: float = 3.2       # ~16 GB / 10 (double power)

    # ── Kaggle ───────────────────────────────────────────
    KAGGLE_USERNAME: str = ""
    KAGGLE_KEY: str = ""
    KAGGLE_KERNEL_SLUG: str = "novadev-gpu-runner"
    KAGGLE_ACCELERATOR: str = "nvidiaTeslaT4"   # or "gpu" for T4x2
    KAGGLE_POLL_INTERVAL_SEC: int = 3
    KAGGLE_GPU_QUOTA_HOURS: int = 30

    # ── Docker ───────────────────────────────────────────
    DOCKER_IMAGE_FREE: str = "novadev/notebook-free:latest"
    DOCKER_IMAGE_PAID: str = "novadev/notebook-paid:latest"
    DOCKER_NETWORK: str = "novadev-net"
    DOCKER_BASE_PORT: int = 8800          # Containers get 8800-8839

    # ── Redis (queue backend) ─────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://novadev:novadev@localhost/novadev"

    class Config:
        env_file = ".env"


settings = Settings()
