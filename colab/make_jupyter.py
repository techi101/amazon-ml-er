"""Build the local-Jupyter edition of v2 (Windows laptop, 16 GB RAM, NVIDIA GPU optional)."""
import ast
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
L = (here / "er_v2.py").read_text(encoding="utf-8-sig").split("\n")
a = next(i for i, l in enumerate(L) if l.startswith("# ---- data location"))
b = next(i for i, l in enumerate(L) if l.startswith("OUT_DIR = os.environ.get("))
block = '''# ---- data location: folder on this computer ------------------------------------------
# Folder that contains train\\ and test\\ with the 7 .tsv files. Change this to where your dataset is.
DATA_DIR = os.environ.get("ER_DATA_DIR", r"C:\\Users\\Lenovo\\Downloads\\dataset")
IN_COLAB = False
for f in ["train/train_source1.tsv", "train/train_source2.tsv", "train/train_source3.tsv",
          "train/train_ground_truth.tsv", "test/test_source1.tsv", "test/test_source2.tsv", "test/test_source3.tsv"]:
    assert os.path.exists(os.path.join(DATA_DIR, f)), f"missing {os.path.join(DATA_DIR, f)}: fix DATA_DIR above"
print("DATA_DIR =", DATA_DIR)
OUT_DIR = os.environ.get("ER_OUT_DIR", os.path.join(os.getcwd(), "output"))'''.split("\n")
L = L[:a] + block + L[b + 1:]
s = "\n".join(L)
hdr_old = s[:s.index("\n# %%\n#!pip")]
hdr_new = """# %% [markdown]
# # Business Entity Resolution v2 (local Jupyter): Amazon ML Challenge 2026
#
# Built for a Windows laptop with 16 GB RAM and an NVIDIA GPU (e.g. i5-12450H + RTX 2050 4 GB). Also runs without a GPU (the cross-encoder step is then skipped).
#
# **One-time setup** (in a terminal / Anaconda Prompt):
# ```
# pip install rapidfuzz lightgbm pyarrow pandas scikit-learn psutil transformers jupyter
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# ```
# The second line installs the GPU build of PyTorch. Check it with `python -c "import torch; print(torch.cuda.is_available())"` (should print `True`).
#
# **Run:**
# 1. Set `DATA_DIR` in cell 3 to your dataset folder (the one with `train\\` and `test\\`).
# 2. Close other heavy programs (browser tabs, games): the run uses up to ~12 GB of RAM.
# 3. Kernel → Restart & Run All. Plug the laptop in and stop it sleeping; the full run takes a few hours.
#
# Outputs go to `output\\` next to this notebook: `matching_results.tsv`, `candidate_pairs.tsv`, `validation.json`, models.
#
# Pipeline: cleaning (9 Indian scripts, aliases, website names, digit swaps, street numbers, states) → 14 kinds of lookup keys
# → LightGBM stage 1 → multilingual MiniLM cross-encoder on uncertain pairs (GPU) → LightGBM stage 2 with competition and
# S2↔S3 support features → one-owner rule + threshold / expected-F0.5 decision picked on validation."""
s = hdr_new + s[len(hdr_old):]
s = s.replace("#!pip -q install rapidfuzz lightgbm psutil gdown transformers\n", "")
s = s.replace('TRAIN_S1_SAMPLE = int(os.environ.get("ER_TRAIN_S1", 300_000))', 'TRAIN_S1_SAMPLE = int(os.environ.get("ER_TRAIN_S1", 120_000))')
s = s.replace('VAL_S1_SAMPLE = int(os.environ.get("ER_VAL_S1", 60_000))', 'VAL_S1_SAMPLE = int(os.environ.get("ER_VAL_S1", 30_000))')
s = s.replace("CE_TRAIN_MAX = int(os.environ.get(\"ER_CE_TRAIN_MAX\", 600_000))", "CE_TRAIN_MAX = int(os.environ.get(\"ER_CE_TRAIN_MAX\", 400_000))")
s = s.replace("def ce_train(a, b, y, epochs=1, bs=256, lr=4e-5):", "def ce_train(a, b, y, epochs=1, bs=128, lr=4e-5):   # batch 128 fits a 4 GB GPU")
s = s.replace("def ce_predict(tok, model, a, b, bs=1024):", "def ce_predict(tok, model, a, b, bs=512):")
s = s[:s.index("# %% [markdown]\n# ## 13. Save outputs")].rstrip("\n") + "\n"
for bad in ("kaggle", "gdown", "google.colab"):
    assert bad not in s.lower(), bad
ast.parse(s)
(here / "er_v2_jupyter.py").write_text(s, encoding="utf-8")
subprocess.run([sys.executable, str(here / "to_ipynb.py"), str(here / "er_v2_jupyter.py"), "amazon_er_v2_jupyter.ipynb"], check=True)
