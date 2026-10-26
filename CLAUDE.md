# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A sandbox for trying new open models by hand on a local RTX 5090. It is not a product and has no tests, lint config or CI. To verify a change, run the folder's script or notebook on the GPU.

## Environment

- Everything runs in WSL2 Ubuntu 26.04, with the repo at `~/models-hand-on` on ext4. The README covers machine setup. Don't work from a `/mnt/c` checkout, because venvs and weight loads are several times slower there.
- The GPU is an RTX 5090: 32 GB VRAM, Blackwell `sm_120`. It needs torch built for CUDA 12.8 or newer, and older wheels or kernels fail with "no kernel image". The Windows driver supports CUDA 13.4.
- Never install an NVIDIA driver inside WSL. Install a CUDA toolkit only when a model has to compile kernels (flash-attn and similar).
- WSL has 48 GB of RAM (`.wslconfig` on the Windows side). Checkpoint loads go through RAM, so a checkpoint near that size has to be quantized or streamed.
- Weights live in the HF cache (`~/.cache/huggingface`), never in the repo. `.gitignore` blocks weight file types.

## Layout conventions

- **One folder per model family**, named after the main HF repo exactly (`GLiNER2.5-Decide`). Variants that share a package and API live in the same folder and are chosen with a `--model` flag. They don't get folders of their own.
- **Each folder is an independent uv project**: its own `pyproject.toml`, `uv.lock`, `.python-version` and `.venv`. There is no root workspace and nothing is shared between folders, because models pin conflicting torch, transformers and vLLM versions. Set `package = false`.
- **pyproject pattern**: copy [GLiNER2.5-Decide/pyproject.toml](GLiNER2.5-Decide/pyproject.toml). `environments` is limited to Linux x86_64, and torch comes from the explicit `pytorch-cu130` index. torch has to be a direct dependency, because a `tool.uv.sources` entry doesn't apply to transitive ones. Change the CUDA index only if a model's stack needs a different one.
- **Contents**: a notebook or scripts plus a README, whichever suits the model. Notebook folders need `uv add --dev ipykernel`. The folder README holds the model facts, how to run it and a **Findings** section. Mark vendor benchmark numbers as vendor numbers, separate from anything measured here.
- **Root README**: every folder has a row in the Models table. Update its Tried and Verdict cells when findings land.

## Commands

Run these from inside a model folder:

```bash
uv sync                      # create .venv from uv.lock
uv run try.py                # or whatever the folder's entry point is
uv add <pkg>                 # add a dependency and relock
uv lock --upgrade-package <pkg>
uv run python -c "import torch; print(torch.cuda.get_device_name(), torch.version.cuda, torch.cuda.get_arch_list())"
```
