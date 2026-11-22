#!/usr/bin/env bash
# Merged HF checkpoint -> GGUF (language model + vision projector) for llama.cpp on iOS and Android.
set -euo pipefail
W="${CANDID_WORK_DIR:-/workspace/work}"
QUANTS="${CANDID_GGUF_QUANTS:-Q4_K_M,Q8_0}"
OUT="$W/gguf"; mkdir -p "$OUT"

# The converter must match the llama.cpp the app ships: llama.rn 0.12.9 bundles build b10256
# (app/node_modules/llama.rn/cpp/common/build-info.cpp). Newer converters fuse the DeltaNet
# projections (attn_qkv_gate, ssm_beta_alpha), which that build cannot load. Bump together with
# llama.rn; tests/test_pipeline.py checks the two agree.
LLAMA_CPP_REF="${CANDID_LLAMA_CPP_REF:-b10256}"
LC="/workspace/llama.cpp-$LLAMA_CPP_REF"
[[ -d "$LC" ]] || git clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp "$LC"
cd "$LC"
PY=$(command -v python3 || command -v python)
"$PY" -m pip install -q -c /workspace/constraints.txt ./gguf-py
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null
cmake --build build --target llama-quantize -j "$(nproc)" >/dev/null

"$PY" convert_hf_to_gguf.py "$W/merged" --outtype f16 --outfile "$OUT/candid-4b-f16.gguf"
"$PY" convert_hf_to_gguf.py "$W/merged" --mmproj --outtype f16 --outfile "$OUT/candid-4b-mmproj-f16.gguf"

# The first quant is the phone format and must succeed; later ones are extras that may fail
# (e.g. on a full disk) without failing the export.
IFS=',' read -ra QS <<< "$QUANTS"
for i in "${!QS[@]}"; do
  q="${QS[$i]}"
  if ./build/bin/llama-quantize "$OUT/candid-4b-f16.gguf" "$OUT/candid-4b-$q.gguf.part" "$q"; then
    mv "$OUT/candid-4b-$q.gguf.part" "$OUT/candid-4b-$q.gguf"
  else
    rm -f "$OUT/candid-4b-$q.gguf.part"
    if [[ $i -eq 0 ]]; then echo "[export] $q failed"; exit 1; fi
    echo "[export] optional $q failed (free disk: $(df -h --output=avail "$OUT" | tail -1)); skipping"
  fi
done
rm -f "$OUT/candid-4b-f16.gguf"   # large and not needed on phones
ls -lh "$OUT"
echo "[export] converted with llama.cpp $LLAMA_CPP_REF"
