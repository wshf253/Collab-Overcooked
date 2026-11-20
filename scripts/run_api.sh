#!/usr/bin/env bash
# Run coop.days / coop.decision with a hosted-API ego (docs/coop-setting.md, "API backends").
#
#   bash scripts/run_api.sh configs/coop/decision/urgent.yaml --stage llm --ego-backend opus \
#       --out outputs/decision_urgent_opus --workers 8 --max-cost 20
#
# Keys: exported in the environment, or in a file outside the repo (default ~/.coop_api_env,
# override with COOP_API_ENV) holding lines such as `export ANTHROPIC_API_KEY=...`. The file must
# be readable by you only (chmod 600); the script refuses it otherwise and never prints it.
# COOP_MODULE picks the module (default coop.days). QWEN=1: the partner is the Qwen LLM partner,
# so this serves Qwen through scripts/run_qwen38_days.sh (needs a GPU job) and the ego stays on
# the API; without it no model server is started (scripted partner: CPU job, --gpus 0).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEYFILE="${COOP_API_ENV:-$HOME/.coop_api_env}"
if [ -f "$KEYFILE" ]; then
  perms="$(stat -c '%a' "$KEYFILE")"
  if [ "${perms: -2}" != "00" ]; then
    echo "refusing $KEYFILE: readable by others (mode $perms); run chmod 600 $KEYFILE" >&2
    exit 1
  fi
  set +x
  # shellcheck disable=SC1090
  source "$KEYFILE"
fi
cd "$REPO"
export PYTHONPATH="src:lib/overcooked_ai${PYTHONPATH:+:$PYTHONPATH}"
if [ "${QWEN:-0}" = "1" ]; then
  exec bash scripts/run_qwen38_days.sh "$@"
fi
exec .venv/bin/python -m "${COOP_MODULE:-coop.days}" "$@"
