#!/usr/bin/env bash
# Runs on the vast.ai machine every time it starts (see BOOTSTRAP in vast/launch.py).
#
# The pipeline keeps its progress in /workspace/work on this machine's disk. If the
# credit runs out, vast.ai stops the machine; the disk stays, and when credit is back
# the supervisor starts the same machine and this script continues where it stopped.
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

vast_self() {  # vast_self <command...>; manages this instance with its own scoped key
  [[ -n "${CONTAINER_ID:-}" ]] && vastai "$@" --api-key "${CONTAINER_API_KEY:-}" || true
}

finish() {
  code=$?
  echo "[run] exit $code"
  [[ "${CANDID_KEEP_ALIVE:-0}" == "1" ]] && return
  if [[ $code -eq 0 ]]; then vast_self label instance "$CONTAINER_ID" "candid-$CANDID_RUN-done"
  elif [[ $code -ne 124 ]]; then vast_self label instance "$CONTAINER_ID" "candid-$CANDID_RUN-failed"
  fi
  vast_self stop instance "$CONTAINER_ID"
}
trap finish EXIT

echo "[run] start $(date -u +%FT%TZ) instance=${CONTAINER_ID:-?} gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
command -v git >/dev/null && command -v cmake >/dev/null || \
  (apt-get update -qq && apt-get install -y -qq git cmake build-essential >/dev/null)
pip install -q "trl==1.18.2" "peft==0.22.0" "datasets>=5.2,<6" "flash-linear-attention==0.6.0" \
  "huggingface_hub[hf_transfer]" "vastai==1.11.2" "ddgs==9.16.1" "trafilatura==2.3.1" || exit 3

# Exit code 124 = this start's time limit; the supervisor simply starts the machine again.
timeout "${CANDID_MAX_HOURS:-12}h" python -m candid.pipeline
