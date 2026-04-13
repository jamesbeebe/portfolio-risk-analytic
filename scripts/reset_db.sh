#!/usr/bin/env bash
set -euo pipefail

if [ -f "portfolio_risk.db" ]; then
  rm "portfolio_risk.db"
fi

.venv/bin/alembic upgrade head

echo "Database reset complete"
