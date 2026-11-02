"""Print the facts that decide how to set up an HF model here: license, gating, library, weight sizes, fit.

Stdlib only, so it runs before any venv exists: `python3 hf_facts.py <org/repo> [<org/repo> ...]`.
Reads HF_TOKEN (or the token saved by `hf auth login`) for gated repos.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

VRAM_GIB = 32
RAM_GIB = 48
WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx")


def token() -> str | None:
    if t := os.environ.get("HF_TOKEN"):
        return t
    f = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    return f.read_text().strip() if f.exists() else None


def get(url: str) -> dict:
    req = urllib.request.Request(url)
    if t := token():
        req.add_header("Authorization", f"Bearer {t}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def facts(repo: str) -> None:
    try:
        info = get(f"https://huggingface.co/api/models/{repo}?blobs=true")
    except urllib.error.HTTPError as e:
        print(f"## {repo}\n  HTTP {e.code}: {'not found or private' if e.code in (401, 404) else e.reason}\n")
        return
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"## {repo}\n  network error, retry: {e}\n")
        return

    card = info.get("cardData") or {}
    config = info.get("config") or {}
    st = info.get("safetensors") or {}
    print(f"## {info['id']}")
    print(f"  license       {card.get('license', '?')}  {card.get('license_name', '')}".rstrip())
    print(f"  gated         {info.get('gated', False)}")
    print(f"  library       {info.get('library_name', '?')}   pipeline {info.get('pipeline_tag', '?')}")
    print(f"  architectures {config.get('architectures', '?')}   model_type {config.get('model_type', '?')}")
    if config.get("auto_map"):
        print("  custom code   yes (auto_map), needs trust_remote_code=True")
    if q := config.get("quantization_config"):
        print(f"  quantized     {q.get('quant_method', q)}")
    if st:
        print(f"  parameters    {st.get('total', 0) / 1e9:.2f}B  by dtype {st.get('parameters')}")
    print(f"  base_model    {card.get('base_model', '-')}")
    print(f"  last modified {info.get('lastModified', '?')[:10]}   downloads {info.get('downloads', '?')}")

    # Group weight files by format; a repo often ships several (safetensors + gguf + onnx).
    groups: dict[str, int] = {}
    for s in info.get("siblings", []):
        name, size = s["rfilename"], s.get("size") or 0
        if name.endswith(WEIGHT_EXTS):
            key = f"{Path(name).parent}/*{Path(name).suffix}" if "/" in name else f"*{Path(name).suffix}"
            groups[key] = groups.get(key, 0) + size
    for key, size in sorted(groups.items(), key=lambda kv: -kv[1]):
        b = size / 2**30
        fit = "fits VRAM" if b < VRAM_GIB * 0.8 else "too big for VRAM as is"
        ram = "" if b < RAM_GIB * 0.8 else ", near/over the 48 GB RAM load limit"
        print(f"  weights       {key:<28} {gib(size):>10}  ({fit}{ram})")
    if not groups:
        print("  weights       none found (custom package may download them, check the card)")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for repo in sys.argv[1:]:
        facts(repo.removeprefix("https://huggingface.co/").strip("/"))
