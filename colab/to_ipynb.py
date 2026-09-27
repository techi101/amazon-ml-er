"""Convert er_notebook.py (cells marked with '# %%') into a Colab .ipynb."""
import json
import re
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("er_notebook.py"))
dst = src.with_name(sys.argv[2] if len(sys.argv) > 2 else "amazon_er_colab.ipynb")
text = src.read_text(encoding="utf-8-sig")
cells = []
for block in re.split(r"^# %%", text, flags=re.M)[1:]:
    is_md = block.startswith(" [markdown]")
    body = block.split("\n", 1)[1] if "\n" in block else ""
    body = body.strip("\n")
    if is_md:
        lines = [re.sub(r"^# ?", "", l) for l in body.splitlines()]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
    else:
        body = re.sub(r"^#!", "!", body, flags=re.M)   # shell lines are commented in the .py
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{len(cells)} cells -> {dst}")
