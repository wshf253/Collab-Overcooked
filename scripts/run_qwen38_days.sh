#!/usr/bin/env bash
# Serve Qwen3.8-27B with vLLM inside this job, check that thinking is off, run a
# coop.days config with the LLM ego, then stop the server.
#
#   bash scripts/run_qwen38_days.sh configs/coop/days_llm.yaml --out outputs/days_llm --workers 16
#
# Everything after the config is passed to `python -m coop.days`.
# With ICRL_LLM_BASE_URL already set, no server is started and that endpoint is used.
# Adapted from ICRL4AHT scripts/run_qwen38_coop.sh (same variables and defaults).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: run_qwen38_days.sh <config.yaml> --out <dir> [coop.days options]}"
shift
SERVE_ENV="${LLM_SERVE_ENV:-$HOME/work/llm-serving/.venv-qwen35}"
MODEL_PATH="${LLM_MODEL_PATH:-/lab/ckpt/jsj412/Qwen3.8-27B}"
export ICRL_LLM_MODEL="${ICRL_LLM_MODEL:-qwen3.8-27b}"
LOG_DIR="$REPO/outputs/serve_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}_$$"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"                  # parallel runs (--workers) only help if vLLM batches them
GPU_MEM="${GPU_MEM:-0.85}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"      # empty = no parser (thinking then appears inline)
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"     # skip the vision tower of multimodal checkpoints
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"                 # 1 = slower but avoids CUDA-graph capture issues
PY="$REPO/.venv/bin/python"

cd "$REPO"
test -x "$PY" || { echo 'Missing .venv: run uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements-coop.txt' >&2; exit 1; }
mkdir -p "$LOG_DIR"
export PYTHONPATH="$REPO/src:$REPO/lib/overcooked_ai" PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
SERVER_PID=''
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -z "${ICRL_LLM_BASE_URL:-}" ]]; then
    test -x "$SERVE_ENV/bin/vllm" || { echo "Missing $SERVE_ENV/bin/vllm; set LLM_SERVE_ENV." >&2; exit 1; }
    test -f "$MODEL_PATH/config.json" || { echo "Missing $MODEL_PATH/config.json; set LLM_MODEL_PATH." >&2; exit 1; }
    PORT="$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
    export ICRL_LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
    ARGS=(--served-model-name "$ICRL_LLM_MODEL" --host 127.0.0.1 --port "$PORT"
          --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 16384
          --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization "$GPU_MEM" --enable-prefix-caching)
    [[ -n "$REASONING_PARSER" ]] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
    [[ "$LANGUAGE_MODEL_ONLY" == 1 ]] && ARGS+=(--language-model-only)
    [[ "$ENFORCE_EAGER" == 1 ]] && ARGS+=(--enforce-eager)
    "$SERVE_ENV/bin/vllm" serve "$MODEL_PATH" "${ARGS[@]}" > "$LOG_DIR/vllm.log" 2>&1 &
    SERVER_PID=$!
    echo "Starting vLLM ($MODEL_PATH). Log: $LOG_DIR/vllm.log"
    READY=0
    for ((i=0; i<240; i++)); do
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 80 "$LOG_DIR/vllm.log"; exit 1; }
        if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then READY=1; break; fi
        sleep 5
    done
    [[ "$READY" == 1 ]] || { echo 'vLLM startup timed out.' >&2; tail -n 80 "$LOG_DIR/vllm.log"; exit 1; }
else
    echo "Using the existing endpoint $ICRL_LLM_BASE_URL; it will not be stopped by this script."
fi

# Thinking must be off (the agent expects short JSON replies). The first passing
# chat_template_kwargs is used unless COOP_CHAT_TEMPLATE_KWARGS is set.
if ! PICKED="$("$PY" -m coop.check --base-url "$ICRL_LLM_BASE_URL" --model "$ICRL_LLM_MODEL" \
        --pick --report "$LOG_DIR/thinking_check.jsonl")"; then
    cat "$LOG_DIR/thinking_check.jsonl" 2>/dev/null || true
    exit 1
fi
cat "$LOG_DIR/thinking_check.jsonl"
export COOP_CHAT_TEMPLATE_KWARGS="${COOP_CHAT_TEMPLATE_KWARGS:-$PICKED}"
echo "chat_template_kwargs: $COOP_CHAT_TEMPLATE_KWARGS"

"$PY" -m coop.days "$CONFIG" "$@" 2>&1 | tee "$LOG_DIR/run.log"
