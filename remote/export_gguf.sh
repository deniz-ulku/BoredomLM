#!/usr/bin/env bash
# Merged HF checkpoint -> GGUF (language model + vision projector) for llama.cpp on iOS and Android.
set -euo pipefail
W="${CANDID_WORK_DIR:-/workspace/work}"
QUANTS="${CANDID_GGUF_QUANTS:-Q4_K_M,Q8_0}"
OUT="$W/gguf"; mkdir -p "$OUT"

[[ -d /workspace/llama.cpp ]] || git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
cd /workspace/llama.cpp
PY=$(command -v python3 || command -v python)
"$PY" -m pip install -q -c /workspace/constraints.txt ./gguf-py
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build build --target llama-quantize -j "$(nproc)" >/dev/null

"$PY" convert_hf_to_gguf.py "$W/merged" --outtype f16 --outfile "$OUT/candid-4b-f16.gguf"
"$PY" convert_hf_to_gguf.py "$W/merged" --mmproj --outtype f16 --outfile "$OUT/candid-4b-mmproj-f16.gguf"

IFS=',' read -ra QS <<< "$QUANTS"
for q in "${QS[@]}"; do
  ./build/bin/llama-quantize "$OUT/candid-4b-f16.gguf" "$OUT/candid-4b-$q.gguf" "$q"
done
rm -f "$OUT/candid-4b-f16.gguf"   # large and not needed on phones
ls -lh "$OUT"
