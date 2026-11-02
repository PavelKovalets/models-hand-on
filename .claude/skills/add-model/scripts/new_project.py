"""Write pyproject.toml and .python-version for a new model folder.

`python3 new_project.py <Folder> --description "..." [--python 3.12] [--cuda cu130] [--no-jupyter]`
Run from the repo root. Refuses to overwrite an existing pyproject.toml. Add deps afterwards with `uv add`.
"""

import argparse
import re
from pathlib import Path

TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
description = "{description}"
requires-python = "{requires_python}"
dependencies = [
    "torch>=2.14",  # direct dependency so the CUDA index below applies
]
{dev}
[tool.uv]
package = false
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]  # runs in WSL only

[tool.uv.sources]
torch = {{ index = "pytorch-{cuda}" }}

[[tool.uv.index]]
name = "pytorch-{cuda}"
url = "https://download.pytorch.org/whl/{cuda}"
explicit = true
"""

# Kernel for VS Code notebooks, and nbconvert for running them headless: `uv run jupyter nbconvert --execute`.
JUPYTER_DEV = """
[dependency-groups]
dev = ["ipykernel>=7.4", "nbconvert>=7.17"]
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="named after the main HF repo, e.g. GLiNER2.5-Decide")
    parser.add_argument("--description", required=True, help="one sentence on what the folder tries")
    parser.add_argument("--python", default="3.12", help="pinned minor version; requires-python allows only it")
    parser.add_argument("--cuda", default="cu130", help="PyTorch wheel index, change only if the stack needs it")
    parser.add_argument("--no-jupyter", action="store_true", help="script-only folder, skip the ipykernel/nbconvert dev group")
    args = parser.parse_args()

    pyproject = args.folder / "pyproject.toml"
    if pyproject.exists():
        raise SystemExit(f"{pyproject} exists; this is a variant, edit that folder instead")
    major, minor = args.python.split(".")[:2]
    args.folder.mkdir(parents=True, exist_ok=True)
    pyproject.write_text(TEMPLATE.format(
        name=re.sub(r"[^a-z0-9]+", "-", args.folder.name.lower()).strip("-"),
        description=args.description.replace('"', '\\"'),
        requires_python=f">={major}.{minor},<{major}.{int(minor) + 1}",
        cuda=args.cuda,
        dev="" if args.no_jupyter else JUPYTER_DEV,
    ))
    (args.folder / ".python-version").write_text(f"{major}.{minor}\n")
    print(f"wrote {pyproject} and .python-version ({major}.{minor}, {args.cuda})")


if __name__ == "__main__":
    main()
