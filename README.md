# AADI-playz23/gpu-cloud

NovaDev Cloud — GPU notebook platform backend + GitHub Actions automation.

## GitHub repo
https://github.com/AADI-playz23/gpu-cloud

## Secrets to add
Go to: github.com/AADI-playz23/gpu-cloud → Settings → Secrets and variables → Actions

| Secret                | Value                                      |
|-----------------------|--------------------------------------------|
| KAGGLE_USERNAME       | your Kaggle username                       |
| KAGGLE_KEY            | your Kaggle API key (from kaggle.com/account) |
| KAGGLE_KERNEL_SLUG    | novadev-gpu-pool                           |
| BACKEND_WEBHOOK_URL   | https://your-server-domain.com             |
| BACKEND_SECRET        | any 32-char random string                  |

Generate BACKEND_SECRET:
  python -c "import secrets; print(secrets.token_hex(32))"

## Workflow schedule
  gpu_relay.yml        — runs at 00:00, 06:00, 12:00, 18:00 UTC (24/7 relay)
  quota_tracker.yml    — runs every hour at :05
  notebook_rotation.yml — triggered by relay at 11h notebook age

## Structure
  backend/             FastAPI server
  scripts/             GitHub Actions Python scripts
  .github/workflows/   3 GHA workflow files
  kaggle_notebooks/    Kaggle GPU pool notebook
  docker/              docker-compose + Dockerfiles
  requirements.txt     Python dependencies
  .env.example         Copy to .env and fill values

## Deploy backend
  cp .env.example .env
  # fill .env with your values
  docker compose -f docker/docker-compose.yml up -d

## Quota
  System:   30h/week GPU (Kaggle hard limit)
  Free:     1h/week per user, 30min sessions
  Paid:     2h/week per user, 2h sessions
  Reset:    Monday 00:00 UTC (automatic)
