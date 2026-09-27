"""Build the Colab-only edition of v2 (data from the user's Google Drive, no Kaggle / gdown)."""
import ast
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
block = Path(sys.argv[1]).read_text(encoding="utf-8").rstrip("\n").split("\n")
L = (here / "er_v2.py").read_text(encoding="utf-8-sig").split("\n")
a = next(i for i, l in enumerate(L) if l.startswith("# ---- data location"))
b = next(i for i, l in enumerate(L) if l.startswith("OUT_DIR = os.environ.get("))
L = L[:a] + block + L[b + 1:]
s = "\n".join(L)
hdr_old = s[:s.index("\n# %%\n#!pip")]
hdr_new = """# %% [markdown]
# # Business Entity Resolution v2 (Colab GPU): Amazon ML Challenge 2026
#
# **Before running:**
# 1. The dataset folder must be in your Google Drive (for a shared folder: open it in Drive → **Add shortcut to Drive** → My Drive).
#    Its path goes in `DRIVE_DATA` in cell 3 (default `/content/drive/MyDrive/dataset`).
# 2. **Runtime → Change runtime type → T4 GPU** (High-RAM too if your plan has it).
# 3. **Runtime → Run all**, and allow the Google Drive permission when asked. Keep the tab open.
#
# Outputs: `/content/output`, copied to `MyDrive/amazon_er_output` at the end
# (`matching_results.tsv`, `candidate_pairs.tsv`, `validation.json`, models).
#
# Pipeline: cleaning (9 Indian scripts, aliases, website names, digit swaps, street numbers, states) → 14 kinds of lookup keys
# → LightGBM stage 1 → multilingual MiniLM cross-encoder on uncertain pairs (GPU) → LightGBM stage 2 with competition and
# S2↔S3 support features → one-owner rule + threshold / expected-F0.5 decision picked on validation."""
s = hdr_new + s[len(hdr_old):]
s = s.replace("#!pip -q install rapidfuzz lightgbm psutil gdown transformers", "#!pip -q install rapidfuzz lightgbm psutil transformers")
assert "kaggle" not in s.lower() and "gdown" not in s, [l for l in s.split("\n") if "kaggle" in l.lower() or "gdown" in l]
ast.parse(s)
(here / "er_v2_colab.py").write_text(s, encoding="utf-8")
subprocess.run([sys.executable, str(here / "to_ipynb.py"), str(here / "er_v2_colab.py"), "amazon_er_v2_colab.ipynb"], check=True)
