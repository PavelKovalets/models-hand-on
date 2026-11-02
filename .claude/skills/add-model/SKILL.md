---
name: add-model
description: Set up a new open model in models-hand-on so it runs on the local RTX 5090. Takes an HF repo ID, URL or model name, checks size, license and fit, then scaffolds a uv project folder with a notebook and README, installs it, smoke-runs it on the GPU and adds the root README row. Use whenever the user wants to try, add, set up, play with or test-drive a model here ("let's try X", "add <org/repo>", "can we run X on the 5090?"), including a new variant of a model family that already has a folder.
---

# Add a model

Take a model from an HF repo ID to a folder that runs on the 5090. Facts about the machine and the folder conventions are in CLAUDE.md. This skill holds the setup steps.

## 1. Facts and fit

```bash
python3 .claude/skills/add-model/scripts/hf_facts.py <org/repo> [<org/repo> ...]
```

It prints license, gating, library, custom code, quantization, parameter count and weight size per format. Pass sibling repos too (the smaller sizes, the official AWQ/FP8/GGUF quants) when they're likely to matter. Then read the model card (`https://huggingface.co/<org/repo>/raw/main/README.md`) for the recommended package, minimum versions, usage snippet and vendor benchmark numbers.

Decide three things:

- **Folder.** Folder name = the main repo's name without the org, exactly (`fastino/GLiNER2.5-Decide` → `GLiNER2.5-Decide`). If a folder for the family already exists and the new model uses the same package and API, it's a variant: add it there as another `MODEL` option and a row in its README table. Don't make a new folder.
- **Fit.** Weights load through 48 GB RAM into 32 GB VRAM, and inference needs headroom for activations and KV cache. Count the dtype you'll load at, not the stored one: F32 weights loaded as bf16 take half. Windows keeps about 3 GB of the card for itself (`nvidia-smi --query-gpu=memory.used --format=csv`), so only about 29 GB is really free. Below about 25 GB, go ahead. Above that, or near 48 GB on disk, **stop and propose options**: an official quantized repo (FP8 runs natively on Blackwell, then AWQ/GPTQ, then GGUF with llama.cpp), bitsandbytes 4-bit, or a smaller sibling, with sizes for each. Wait for the user to choose.
- **Stack.** Use what the card recommends: its own package, transformers, sentence-transformers or vLLM. Note any `trust_remote_code`, and any flash-attn or other compiled kernels (those need the CUDA toolkit, see CLAUDE.md).

If the repo is gated, check that `hf auth whoami` works and that the user has accepted the terms. Otherwise ask them to do both before the download step.

### Literature

Search the web for write-ups of the model's architecture and training (model name, org, "arxiv", "architecture"). The card's links and the project's GitHub are a good start. Rank sources in this order:

1. An arxiv paper (or tech report) on the model's architecture.
2. A technical blog post that explains the architecture: Medium, the HF blog, Substack, an engineering blog. A launch announcement that doesn't explain how the model works doesn't count.
3. The official page: the project's GitHub README or the company's site.

Prefer sources about this exact model. When it is a fine-tune of a base model that has its own paper (Qwen, Llama, Gemma), cite the model's own best source first and add the base model's paper as a second source for the backbone. Use 1–3 sources, and read them, not only the abstracts or snippets. If nothing beyond the model card exists, say so in the section and cite the card.

## 2. Scaffold

From the repo root, generate `pyproject.toml` and `.python-version`:

```bash
python3 .claude/skills/add-model/scripts/new_project.py <Folder> --description "<one sentence>" [--python 3.12] [--cuda cu130] [--pytorch REQ ...] [--no-jupyter]
```

`--pytorch` is repeatable and defaults to `torch>=2.14`. When the stack pins PyTorch packages, pass every pinned one, so they all come from the CUDA index and don't conflict. vLLM is the usual case: see its pins with `curl -s https://pypi.org/pypi/vllm/<ver>/json`, under `requires_dist`. For example: `--pytorch torch==2.13.0 --pytorch torchvision==0.28.0 --pytorch torchaudio==2.11.0`.

By default the template adds a `dev` group with ipykernel (the VS Code kernel) and nbconvert (for headless runs), so `uv sync` alone gives a folder that runs notebooks. Pass `--no-jupyter` only for a script-only folder. The template also keeps `package = false`, limits environments to Linux x86_64 and makes the PyTorch packages direct dependencies on the explicit PyTorch index. A `tool.uv.sources` entry doesn't apply to transitive dependencies, so don't remove them. Pick `--python` from what the package supports. Change `--cuda` only if the stack needs a torch built for a different CUDA.

Then, in `<Folder>/`, write:

- **`try.ipynb`**: the default entry point (see below).
- **`README.md`**: same shape as [GLiNER2.5-Decide/README.md](../../../GLiNER2.5-Decide/README.md). One paragraph on what the model is, who released it, when, the license and the package. Right after it, a `## Model Overview` section from the literature search: 5–7 sentences on the architecture first (backbone, heads, how inference reads the output, size, context), then the training recipe (base model, data, objectives, tuning stages), followed by a bulleted list of the 1–3 sources used, each with a link and a few words on what it is (`arxiv paper`, `blog post`, `official page`). State only what the sources say. A `## Checkpoints` section with a checkpoint table with size and vendor scores, labelled as vendor numbers. A **Run** section. For a notebook it covers opening it in VS Code over WSL, picking `<Folder>/.venv/bin/python` under **Select Kernel → Python Environments** and running all cells, plus the headless `nbconvert` command (copy the steps from [openjev/README.md](../../../openjev/README.md)). `## Findings` with `_Not tried yet._`.

Then `uv add <deps>` from the folder (it writes `uv.lock`).

### The notebook

Keep it short. Its job is to let the user poke at the model, not to be a benchmark harness. Cells:

1. Markdown: model name, link, one line on what to edit.
2. `MODEL = "<default repo>"`, with the variants in a comment, plus imports.
3. Load: time it, print the GPU name, `torch.cuda.reset_peak_memory_stats()`.
4. `CASES`: 3–4 realistic inputs that show what the model is for (for an embedder, pairs to compare; for a classifier, text and labels; for an LLM, a few prompts including one that's hard for it).
5. Run: a warm-up call first, then each case with its ms timing.
6. `print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")`.

Write it as JSON with `nbformat` 4.5, an `id` on every cell and empty outputs.

When the model runs behind a server (vLLM, llama.cpp), put the server in a `serve.sh` and make the notebook its client. The model's memory then lives in another process, so read VRAM with `nvidia-smi` instead of `torch.cuda`.

Use a script (`try.py` with argparse, like GLiNER's) instead only when a notebook gets clumsy: batch runs over files, CLI flags you'll keep changing, long-running servers such as `vllm serve`, or anything you'd pipe. Say why when you pick a script.

## 3. Install and smoke-run

From the folder:

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.get_device_name(), torch.version.cuda, torch.cuda.get_arch_list())"
```

`sm_120` must be in the arch list. If it's missing, the torch index or a pin is wrong. Fix that before going further.

vLLM on WSL needs two things: `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (otherwise "UVA is not available"), and, with no CUDA toolkit installed, `VLLM_USE_FLASHINFER_SAMPLER=0` (otherwise "Could not find nvcc"). `--cpu-offload-gb` pins the offloaded weights. Windows caps pinned memory at about half the host's 61.6 GB, and nearing the cap can freeze Windows, so ask the user before offloading more than a few GB.

Then run the notebook once:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace try.ipynb
```

This downloads the weights into the HF cache, so for big ones run it in the background. If it fails, fix the cause (missing dep, dtype, attention implementation, remote code) and rerun until the cells finish. A folder that doesn't run isn't done. Read the outputs. Report load time, per-case latency and peak VRAM, and whether the answers look sane.

## 4. Root README and wrap-up

Add a row to the Models table in the root README.md: `| [<Folder>](<Folder>/) | <one-line what> | – | – |`. Tried and Verdict stay `–` until the user records findings. For a variant, update the existing row's "What" cell.

Finish with a short summary: what was set up, the fit numbers, the smoke-run numbers and anything odd. Then offer to commit, with a message like `Add the <Folder> folder` (or `Add <variant> to <Folder>`). Commit only if the user says yes.
