#!/usr/bin/env bash
# Serve OpenJev-FP8 with vLLM on :8000 and the vendor's decision helper (shim.py) on :3000, both on loopback.
# Flags follow the vendor's serve/SERVE.md, except --cpu-offload-gb and --gpu-memory-utilization: the 28.3 GiB of
# FP8 weights don't fit next to what Windows already holds on the 32 GB card, so OFFLOAD_GB of them stay in RAM.
# Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"

MODEL=openjev/openjev-FP8
OFFLOAD_GB=${OFFLOAD_GB:-6}
GPU_UTIL=${GPU_UTIL:-0.88}
MAX_LEN=${MAX_LEN:-16384}

# vLLM's GPU worker needs pinned (UVA) buffers, which it disables on WSL2 unless asked, and the offloaded weights are
# pinned too. Windows caps pinned memory at about half the host RAM (~31 GB here), and going near the cap can freeze it.
# FlashInfer's sampler is JIT-compiled and needs nvcc; the one-token letter readout doesn't need it.
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 uv run vllm serve "$MODEL" --host 127.0.0.1 --port 8000 --served-model-name qwen \
  --enable-prefix-caching --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt '{"image":1}' --trust-remote-code --max-num-seqs 16 --max-logprobs 64 \
  --gdn-prefill-backend triton --quantization fp8 --cpu-offload-gb "$OFFLOAD_GB" &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

until curl -sf http://127.0.0.1:8000/health > /dev/null; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited before it was ready" >&2; exit 1; }
  sleep 5
done

# The helper lives in the main repo, unmodified (sha256 81a22f1b...); the calibration values are the vendor's measured ones.
SHIM=$(uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('openjev/openjev', 'helper/shim.py'))")
VLLM=http://127.0.0.1:8000/v1 TOKENIZER="$MODEL" \
READOUT_T=0.85 READOUT_NOUL_T=1.829074 READOUT_NOUL_BIAS=0 \
READOUT_TARGETED=1 READOUT_INSTR_STYLE=pyrepr SHIM_STAGGER=1 \
uv run python "$SHIM" --host 127.0.0.1 --port 3000
