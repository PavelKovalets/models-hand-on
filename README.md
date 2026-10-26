# models-hand-on

Trying new open models hands-on, locally on an RTX 5090 (32 GB, Blackwell sm_120) under WSL2 Ubuntu 26.04. Each folder is one model family.

## Models

| Folder | What | Tried | Verdict |
|---|---|---|---|
| [GLiNER2.5-Decide](GLiNER2.5-Decide/) | Fastino schema classifiers, Decide-1B and 340M | – | – |

## Machine setup (once)

1. **Ubuntu.** In Windows, run `wsl --install -d Ubuntu-26.04` and create the Linux user when prompted. `C:\Users\<you>\.wslconfig` gives WSL 48 GB of RAM and 16 GB of swap. Run `wsl --shutdown` after changing it.
2. **GPU.** Don't install an NVIDIA driver inside Ubuntu, because WSL uses the Windows one: `nvidia-smi` should already work. Install a CUDA toolkit (`cuda-toolkit` from NVIDIA's `wsl-ubuntu` apt repo) only when a model has to compile kernels.
3. **uv.** `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. **Repo.** Run `ssh-keygen -t ed25519`, add the public key to GitHub and set `git config --global user.name` and `user.email`. Then `git clone git@github.com:PavelKovalets/models-hand-on.git ~/models-hand-on`.
5. **Editor.** From Ubuntu, `code ~/models-hand-on` opens it in VS Code over Remote-WSL. Run Claude Code there.

Keep the repo and the Hugging Face cache (`~/.cache/huggingface`) on the Linux filesystem. Anything under `/mnt/c` is several times slower.
