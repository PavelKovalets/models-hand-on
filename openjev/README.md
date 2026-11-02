# openjev

LoopAI's open-weight decision model (CC BY-NC 4.0, non-commercial; helper code Apache 2.0), released 2026-09-20. It's a fine-tune of Qwen3.8-27B (HF architecture `qwen3_5`, vision included) tuned so that one forward pass answers a typed question. You send a state (text, JSON, a DOM or a screenshot) plus questions of type `choice` (up to 52 options), `noul` (yes/no) or `score` (ordered levels). The answer is read from the option-letter scores at the first output position and calibrated into probabilities. There's no generation and nothing to parse. It is served by vLLM plus the vendor's `helper/shim.py`, which exposes `POST /v1/systemone`.

## Model Overview

The backbone is Alibaba's Qwen3.8-27B, a dense vision-language model with 64 layers in a hybrid layout: each block of four has three Gated DeltaNet (linear attention) layers and one gated full-attention layer, each followed by an FFN, with 262k tokens of native context. OpenJev adds no new head. It gives each option a letter, asks the question and reads the base LM head's scores for exactly those letters at the first output position, so one forward pass produces one score per option for up to 52 options. The helper then calibrates those scores into probabilities you can threshold. Loop fine-tuned the base with LoRA adapters and merged them into the released weights. The training data was public web and desktop agent steps (each a screenshot, the page text and the action a person took) plus public classification tasks. The correct answer was moved around the option list during training, so the model picks by what an option says, not where it sits, and the answer-flip rate under shuffling fell from 18.5% to 2.3% (vendor numbers). Loop trained seven versions in five days and released the best one.

- [Building OpenJev in five days](https://blog.tryloop.ai/building-openjev-in-five-days/): blog post by Loop, 2026-10-01, on the readout, training data and LoRA recipe.
- [openjev/openjev model card](https://huggingface.co/openjev/openjev): official page, with the "How it works" paragraph and the base model's attribution in `NOTICE`.
- [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B): official page for the backbone architecture. Qwen3.8-27B has no arxiv paper of its own.

## Checkpoints

| Checkpoint | Size | Used here | Text accuracy, 10k questions (vendor) |
|---|---|---|---|
| [`openjev/openjev`](https://huggingface.co/openjev/openjev) | bf16, 51 GiB | no, too big for 32 GB VRAM and the 48 GB RAM load | 84.0% (served with online FP8) |
| [`openjev/openjev-FP8`](https://huggingface.co/openjev/openjev-FP8) | FP8 e4m3, 28.3 GiB | **yes** | 84.2% |
| [`openjev/openjev-GGUF`](https://huggingface.co/openjev/openjev-GGUF) | Q4_K_M 15.4 to Q8_0 26.6 GiB | no: text only, and the helper has no llama.cpp backend yet | Q4_K_M 82.8%, Q8_0 82.6% on a separate 1,789-question set |

Other vendor numbers, all measured on one H100 with FP8: hosted Jev API 85.4% on the same 10k questions; next web action on unseen sites 87.4%; next desktop action from a screenshot 88.0%; answer flips under option shuffling 2.3%; about 80 ms for a short text decision and about 210 ms for a web step.

**Fit on the 5090.** Windows already holds about 3 GiB of the card, so the 28.3 GiB of FP8 weights don't fit by themselves. `serve.sh` keeps `OFFLOAD_GB` (default 6) of them in RAM with vLLM's `--cpu-offload-gb`. That leaves 22.4 GiB of weights on the GPU and a 2.45 GiB KV cache (about 31k tokens, or 1.9 full 16k requests at a time). Every forward pass streams the offloaded weights over PCIe, so latencies are well above the vendor's.

**WSL workarounds in `serve.sh`.**
- `VLLM_WSL2_ENABLE_PIN_MEMORY=1`: vLLM's GPU worker needs pinned (UVA) memory, which it turns off on WSL2 by default (the error is "UVA is not available"). The offloaded weights are pinned too. Windows caps pinned memory at about half the host RAM (about 31 GB on this 61.6 GB machine), and vLLM warns that nearing the cap can freeze Windows. Keep `OFFLOAD_GB` well below it.
- `VLLM_USE_FLASHINFER_SAMPLER=0`: FlashInfer's sampler is JIT-compiled and needs `nvcc`, which isn't installed. The one-token readout doesn't need it.

## Run

```bash
uv sync
./serve.sh                          # vLLM on :8000 and the helper on :3000, loopback only; Ctrl-C stops both
OFFLOAD_GB=4 GPU_UTIL=0.9 ./serve.sh  # less offload if it fits, which is faster
```

With the servers running, open the notebook in VS Code:

1. Open [try.ipynb](try.ipynb) from the VS Code window that's connected to WSL (the status bar at bottom left says `WSL: Ubuntu`). The first time, install the **Jupyter** and **Python** extensions when VS Code asks. They have to be installed in WSL, not only on Windows.
2. Click **Select Kernel** at the top right, then **Python Environments**, and pick `openjev/.venv/bin/python`. If it isn't listed, choose **Enter interpreter path** and enter `~/models-hand-on/openjev/.venv/bin/python`. `uv sync` already installed `ipykernel` into that venv.
3. Click **Run All**. Then edit `CASES` and rerun that cell and the one after it. The first two cells only set things up. `CASES` ends with the same inputs as GLiNER2.5-Decide's `try.py`, so you can compare the two models. To classify your own .md files by medical document type, put them in `DOCS` in the cell after that (GLiNER's `--docs`).

Without VS Code, run it headless. The outputs are written back into the notebook:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace try.ipynb
```

Or call the API directly:

```bash
curl -s localhost:3000/v1/systemone -H 'Content-Type: application/json' -d '{"model":"openjev","state":"Customer: charged twice!","questions":{"route":{"type":"choice","instructions":"Which team?","criteria":{"billing":null,"shipping":null}}}}'
```

Versions are pinned to the vendor's locked set (`vllm 0.29.0`, `torch 2.13.0+cu130`, `transformers 5.17.0`, `openai 3.16.2`), and the helper's calibration values in `serve.sh` are theirs. The helper file is downloaded into the HF cache unmodified.

## Findings

_Not tried yet._
