#!/usr/bin/env bash
# Runs on the vast.ai machine every time it starts (see BOOTSTRAP in vast/launch.py).
#
# The pipeline keeps its progress in /workspace/work. If the credit runs out, vast.ai
# stops the machine; when credit is back the supervisor restarts it (or moves the data
# to a new machine) and this script continues where it stopped.
#
# When the pipeline ends (finished or a code error) the machine stops itself, so an
# idle GPU never keeps billing. The label tells the supervisor which case it was.
#
# Log lines the supervisor parses: "[run] start", "[state] {...}", "[run] exit <code>".

set -uo pipefail
cd "$(dirname "$0")/.."
export CANDID_RUN="${CANDID_RUN:-v0}"
export CANDID_WORK_DIR="/workspace/work/runs/$CANDID_RUN"
export PYTHONUNBUFFERED=1
mkdir -p "$CANDID_WORK_DIR"
exec > >(tee -a /workspace/run.log) 2>&1

PY=$(command -v python3 || command -v python)

vast_self() {  # vast_self '<json>': update this instance with its own scoped API key
  [[ -n "${CONTAINER_ID:-}" && -n "${CONTAINER_API_KEY:-}" ]] || return 0
  curl -fsS -X PUT "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/" \
    -H "Authorization: Bearer ${CONTAINER_API_KEY}" -H "Content-Type: application/json" -d "$1" >/dev/null \
    && echo "[run] vast update $1" || echo "[run] vast update failed: $1"
}

finish() {
  code=$?
  echo "[run] exit $code"
  [[ "${CANDID_KEEP_ALIVE:-0}" == "1" ]] && return
  if [[ $code -eq 0 ]]; then vast_self "{\"label\": \"candid-$CANDID_RUN-done\"}"
  elif [[ $code -ne 124 ]]; then vast_self "{\"label\": \"candid-$CANDID_RUN-failed\"}"
  fi
  vast_self '{"state": "stopped"}'
}
trap finish EXIT

echo "[run] start $(date -u +%FT%TZ) instance=${CONTAINER_ID:-?} gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
command -v git >/dev/null && command -v cmake >/dev/null || \
  (apt-get update -qq && apt-get install -y -qq git cmake build-essential >/dev/null)

# Never let our extra packages move the versions vLLM was built against.
"$PY" -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|transformers|vllm|xformers|flashinfer[a-z_-]*|numpy|pillow|tokenizers|safetensors|huggingface[_-]hub|opentelemetry[a-z_-]*)==' \
  > /workspace/constraints.txt
"$PY" -m pip install -q -c /workspace/constraints.txt "trl==1.18.2" "peft==0.22.0" "datasets>=5.2,<6" \
  "flash-linear-attention==0.6.0" "ddgs==9.16.1" "trafilatura==2.3.1" || exit 3

# Exit code 124 = this start's time limit; the supervisor simply starts the machine again.
timeout "${CANDID_MAX_HOURS:-12}h" "$PY" -m candid.pipeline
