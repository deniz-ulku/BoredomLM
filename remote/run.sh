#!/usr/bin/env bash
# Runs on the vast.ai machine (started by vast/launch.py through the instance's onstart).
#
# Stages: generate -> eval base -> train -> eval tuned -> gate -> export GGUF -> upload.
# Whatever happens, reports are uploaded and the machine destroys itself at the end,
# so a crashed or finished run never keeps billing.
#
# Required env (set by the launcher):   CANDID_HUB_REPO, CANDID_MAX_HOURS
# Required env (vast account settings): HF_TOKEN
# Optional: CANDID_STAGES (default "all"), CANDID_KEEP_ALIVE=1 (do not self-destroy; for debugging)

set -uo pipefail
export CANDID_WORK_DIR="${CANDID_WORK_DIR:-/workspace/work}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$CANDID_WORK_DIR"
LOG="$CANDID_WORK_DIR/run.log"
exec > >(tee -a "$LOG") 2>&1

cd "$(dirname "$0")/.."
st() { python -m remote.hub status "$@"; }

finish() {
  code=$?
  st "finished" "exit=$code"
  python -m remote.hub upload reports || true
  if [[ "${CANDID_KEEP_ALIVE:-0}" != "1" && -n "${CONTAINER_ID:-}" ]]; then
    echo "[run] destroying instance $CONTAINER_ID"
    vastai destroy instance "$CONTAINER_ID" --api-key "${CONTAINER_API_KEY:-}" -y \
      || vastai stop instance "$CONTAINER_ID" --api-key "${CONTAINER_API_KEY:-}" -y || true
  fi
}
trap finish EXIT

if [[ -z "${HF_TOKEN:-}" && -n "${CANDID_HUB_REPO:-}" ]]; then
  echo "[run] HF_TOKEN is not set. Add it in vast.ai: Account -> Environment Variables."
  exit 2
fi

st "setup"
command -v git >/dev/null && command -v cmake >/dev/null || \
  (apt-get update -qq && apt-get install -y -qq git cmake build-essential >/dev/null)
pip install -q "trl==1.18.2" "peft==0.22.0" "datasets>=5.2,<6" "flash-linear-attention==0.6.0" \
  "huggingface_hub[hf_transfer]" "vastai==1.11.2" "ddgs==9.16.1" "trafilatura==2.3.1" || exit 3

MODEL_BASE="$(python -c 'from candid.config import Config; print(Config.from_env().base_model)')"
STAGES="${CANDID_STAGES:-all}"
want() { [[ "$STAGES" == "all" || ",$STAGES," == *",$1,"* ]]; }

run_stage() {  # name, command...
  local name=$1; shift
  st "$name"
  local t0=$SECONDS
  "$@"
  local rc=$?
  echo "[run] $name finished in $(( (SECONDS - t0) / 60 )) min, exit $rc"
  [[ $rc -eq 0 ]] || { st "$name" "failed exit=$rc"; exit $rc; }
}

main() {
  want generate   && run_stage generate   python -m candid.generate
  python -m remote.hub upload reports || true
  want eval_base  && run_stage eval_base  python -m candid.evaluate base "$MODEL_BASE"
  want train      && run_stage train      python -m candid.train
  python -m remote.hub upload adapter || true
  want eval_tuned && run_stage eval_tuned python -m candid.evaluate tuned "$CANDID_WORK_DIR/merged"

  st "gate"
  if python -m candid.evaluate gate; then
    want export && run_stage export bash remote/export_gguf.sh
    python -m remote.hub upload merged || true
    python -m remote.hub upload gguf || true
    st "done" "gate passed"
  else
    st "done" "gate FAILED: tuned model lost capability or behaviour did not improve; see run/report.md"
  fi
}

# Hard wall-clock limit on the whole pipeline, independent of the launcher.
export -f main run_stage st want
timeout "${CANDID_MAX_HOURS:-10}h" bash -c main
