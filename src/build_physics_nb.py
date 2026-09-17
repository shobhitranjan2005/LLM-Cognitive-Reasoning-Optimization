"""Turn physics_orchestrator.py into the notebook Kaggle actually runs.

The worker is embedded rather than shipped beside the notebook: Kaggle runs a
single file, and a second file would have to become its own dataset just to be
importable.

    python build_physics_nb.py
"""
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "physics_orchestrator.py"
WORKER = HERE / "physics_worker.py"
NB = HERE / "physics_nb" / "physics_eval.ipynb"

src = io.open(SRC, encoding="utf-8").read()
worker = io.open(WORKER, encoding="utf-8").read()

# The worker goes inside an r'''...''' literal, so a lone ''' or a trailing
# backslash in it would end the literal early and break the notebook.
assert "'''" not in worker, "worker contains a triple quote"
assert not worker.rstrip().endswith("\\"), "worker ends with a backslash"
src = src.replace("__WORKER_SOURCE__", worker)

# Shell magics have to stay commented in the .py so it still parses as Python;
# they must be live in the notebook. Dropping this step once already cost a run:
# bitsandbytes was never installed and every arm died on the import.
lines = src.split("\n")
for i, line in enumerate(lines):
    if line.startswith("# !"):
        lines[i] = line[2:]
src = "\n".join(lines)
assert "\n!pip" in src, "the install cell did not survive uncommenting"

# Kaggle's log viewer mangles anything outside ASCII, and a mangled traceback is
# a debugging session wasted.
bad = {c for c in src if ord(c) > 127}
assert not bad, f"non-ascii in the notebook: {sorted(bad)}"

cells, buf, kind = [], [], "code"


def flush():
    text = "".join(buf).strip("\n")
    if not text:
        return
    if kind == "markdown":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#")
                         for l in text.split("\n"))
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": text.splitlines(keepends=True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                      "execution_count": None,
                      "source": text.splitlines(keepends=True)})


for line in src.split("\n"):
    if line.startswith("# %%"):
        flush()
        buf, kind = [], "markdown" if "[markdown]" in line else "code"
        continue
    buf.append(line + "\n")
flush()

NB.parent.mkdir(parents=True, exist_ok=True)
io.open(NB, "w", encoding="utf-8").write(json.dumps({
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5}, indent=1))

meta = NB.parent / "kernel-metadata.json"
io.open(meta, "w", encoding="utf-8").write(json.dumps({
    "id": "shobhitranjan2005/loom-physics-eval",
    "title": "LOOM physics eval",
    "code_file": NB.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["shobhitranjan2005/loom-a-v1-control-adapter",
                        "shobhitranjan2005/loom-eval-bundle",
                        "shobhitranjan2005/loom-train",
                        "shobhitranjan2005/deep-reason"],
    "competition_sources": [],
    "kernel_sources": [],
}, indent=2))

print(f"{NB} -- {len(cells)} cells, {NB.stat().st_size:,} bytes")
print(f"{meta}")
