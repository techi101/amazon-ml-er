"""Build the Kaggle edition of the v2 notebook (+ kernel-metadata.json for `kaggle kernels push`)."""
import json
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
src = (here / "er_v2.py").read_text(encoding="utf-8-sig")
header_end = src.index("\n# %%\n#!pip")
kaggle_header = """# %% [markdown]
# # Business Entity Resolution v2: Kaggle GPU edition (Amazon ML Challenge 2026)
#
# **Before you run (right-hand panel → Session options):**
# 1. **Accelerator → GPU T4 x2** (or P100).
# 2. **Internet → On** (needed once, to install `rapidfuzz` and download the cross-encoder model; requires a phone-verified Kaggle account).
# 3. **+ Add Input** → attach the challenge dataset (the one with `train/` and `test/` .tsv files). The notebook finds the files by itself.
#
# Then **Run All** (or **Save Version → Save & Run All** to run in the background, up to 12 h, so you can close the tab).
# Outputs land in `/kaggle/working/output/`: `matching_results.tsv`, `candidate_pairs.tsv`, `validation.json`, the models.
# Download them from the **Output** tab of the saved version.
#
# Pipeline: cleaning (9 Indian scripts, aliases, website names, digit swaps, street numbers, states) → 14 kinds of lookup keys
# → LightGBM stage 1 → multilingual MiniLM cross-encoder on uncertain pairs (GPU) → LightGBM stage 2 with competition and
# S2↔S3 support features → one-owner rule + threshold / expected-F0.5 decision picked on validation."""
body = src[header_end:].replace('TRAIN_S1_SAMPLE = int(os.environ.get("ER_TRAIN_S1", 300_000))', 'TRAIN_S1_SAMPLE = int(os.environ.get("ER_TRAIN_S1", 160_000))').replace('VAL_S1_SAMPLE = int(os.environ.get("ER_VAL_S1", 60_000))', 'VAL_S1_SAMPLE = int(os.environ.get("ER_VAL_S1", 40_000))').replace("#!pip -q install rapidfuzz lightgbm psutil gdown transformers",
                                "#!pip -q install rapidfuzz")
(here / "er_v2_kaggle.py").write_text(kaggle_header + body, encoding="utf-8")
subprocess.run([sys.executable, str(here / "to_ipynb.py"), str(here / "er_v2_kaggle.py"), "amazon_er_v2_kaggle.ipynb"], check=True)
meta = {"id": "USERNAME/amazon-er-v2", "title": "amazon-er-v2", "code_file": "amazon_er_v2_kaggle.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": ["satwiksps/amazon-ml-challenge-2026"],
        "competition_sources": [], "kernel_sources": []}
(here / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("kaggle edition written")
