# CONTEXT: read this first (written for a human *and* for an AI assistant)

Snapshot pushed 27 Sep 2026. Goal of whoever runs this: **run the pipeline on the full dataset
and send back `output/matching_results.tsv` (and `output/candidate_pairs.tsv`)**.

---

## 1. Quick start (for the human)

1. Install Python 3.11 or 3.12, then in a terminal (from the repo root):
   ```
   pip install -r requirements.txt
   # NVIDIA GPU (optional, enables the cross-encoder step, better score):
   pip uninstall -y torch; pip install torch --index-url https://download.pytorch.org/whl/cu121
   # (requirements.txt already installs a CPU torch; the line above swaps in the GPU build)
   ```
2. Get the dataset (NOT in this repo, 2.5 GB). You need a folder laid out as:
   ```
   <DATA>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
   <DATA>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
   ```
   Sources: the team leader's Unstop download, or the team Google Drive folder
   https://drive.google.com/drive/folders/1AAnZuLWXNj6dRALnz1KlL9mPMWAA8y6b ,
   or the Kaggle dataset `satwiksps/amazon-ml-challenge-2026`.
3. Run (from the repo root). Windows PowerShell:
   ```
   $env:ER_DATA_DIR = "D:\path\to\dataset"
   $env:ER_OUT_DIR  = "$PWD\output"
   python colab\er_v2_jupyter.py
   ```
   Linux/macOS: `ER_DATA_DIR=/path/to/dataset ER_OUT_DIR=$PWD/output python colab/er_v2_jupyter.py`
   (Or open `colab/amazon_er_v2_jupyter.ipynb` in Jupyter, set `DATA_DIR` in cell 3, Run All.)
4. When it ends it prints `['PASS']` (format check). Send back:
   - `output/matching_results.tsv`  ← the file uploaded to the leaderboard
   - `output/candidate_pairs.tsv`
   - `output/validation.json` and the console log (copy/paste or `python ... > run.log 2>&1`)

Expected: several hours, peak RAM ~12 GB on 16 GB machines (more RAM = safer), disk ~5 GB free.
Keep the PC plugged in and stop it sleeping.

---

## 2. The challenge (so an AI can reason about it)

Amazon ML Challenge 2026, "Business Entity Resolution" (72-hour hackathon on Unstop).
We are running it as a friendly mock between college groups.

- 3 sources of business records, each row: `entity_id, business_name, business_address, country`.
  All files are **TSV** (read with `sep="\t"`, `quoting=3`, `keep_default_na=False`).
- Source 1 (S1) is a clean, deduplicated reference list. For **every S1 record** output all
  matching S2/S3 records (zero, one or many).
- Train: S1 2.21M, S2 5.03M, S3 5.29M; countries US and India.
  Test: S1 1.73M, S2 4.89M, S3 5.08M; countries India 47%, US 38%, **France 15% (not in train)**.
- Ground truth (train only): one row per S1, comma-separated matched ids (empty = singleton).

**Metric**: F0.5 computed per S1 record, then averaged (macro). Precision weighs 2× recall.
A singleton (no true match) scores 1.0 for an empty prediction and 0.0 for any prediction.

**Output files** (tab-separated, exact headers):
```
source1_entity_id	matched_entity_ids        <- matching_results.tsv (scored)
source1_entity_id	candidate_entity_ids      <- candidate_pairs.tsv (audited, not scored)
```
One row per test S1 id, empty list allowed, no duplicate ids, only S2/S3 ids that exist in test,
matches must be a subset of candidates.

**Hard rules**: no external data / APIs / geocoding / internet lookups of businesses. Final model
must be MIT or Apache-2.0 licensed and ≤ 8B parameters. Final zip = `output/` + runnable code +
README + requirements + filled `Documentation_template.md`.

## 3. Facts found in the data (verified on the full train set)

- Singletons only **5.6%**; mean 3.46 matches per S1 (max 11) → recall matters a lot.
- **One-owner rule is exact**: of 7.64M matched S2/S3 records, none belongs to 2 S1 records.
- 26% of S2/S3 records match nothing; 27% of those have the *same cleaned name* as some S1
  record at a different address (hard look-alikes).
- ~10% of S2/S3 names are in Indian scripts (Devanagari, Telugu, Kannada, Tamil, Bengali,
  Gujarati, Malayalam, Oriya, Gurmukhi); state names also appear in native script.
- Alias names: `Korwexnyla trading as Gold Consulting`, `X aka/DBA/formerly Y`; some pool names
  are pure random words at the correct address (only the address can match those).
- Noise: case, accents, leetspeak (5ervices, lndia), junk prefixes (#, @, --, >>), legal suffix
  add/drop, address reorder, state full/abbr, city typos, house numbers with leading zeros or
  off by a few, ~4.5% empty addresses. France: R./Rue, Av., Bd, Nº, BIS, SARL/SAS/EURL/EI.
- True pairs share a name word 85%, an address word 95%, house number ±10 79%; none of the three 0.01%.

More detail: `docs/PLAN.md` and `dev/reports/noise_audit.md`.

## 4. The pipeline (current = v2, `colab/er_v2.py`)

`colab/er_v2.py` is the master script; `colab/make_{colab,kaggle,jupyter}.py` generate the
editions (`er_v2_colab.py`, `er_v2_kaggle.py`, `er_v2_jupyter.py` + matching `.ipynb`).
**Edit `er_v2.py`, then regenerate editions** — don't hand-edit the editions.

1. Clean names/addresses: Indic transliteration (rule table + 156-entry dictionary learned from
   training pairs), accents, junk, leetspeak, website names split, alias split, legal forms,
   honorifics, street numbers, street names, states (US/IN/FR).
2. Candidates: 14 kinds of lookup keys per country (whole-name keys for main name and alias,
   rare single words typo-tolerant, house number + street word, street name without number),
   then trimming (best by combined, name-only and address-only quick scores).
3. Stage 1 LightGBM (~70 pair features) on half A of sampled train S1.
4. Optional cross-encoder (multilingual MiniLM, Apache-2.0, 118M params) on uncertain pairs, GPU only.
5. Stage 2 LightGBM on half B with stage-1/cross-encoder scores, competition features on both
   sides, S2↔S3 support.
6. Decision: one-owner rule + threshold or expected-F0.5 set choice, picked on validation.
7. Test predicted per country, written streaming, then format-checked.

Env vars: `ER_DATA_DIR`, `ER_OUT_DIR`, `ER_TRAIN_S1` (S1 records used for training; default
120000 in the local `er_v2_jupyter.py`, safe on 16 GB, raise to 200000-300000 on 32 GB+),
`ER_VAL_S1` (held out for scoring; 30000 local), `ER_RUN_TEST` (1), `ER_CE` (1 if CUDA),
`ER_CE_TRAIN_MAX`, `ER_CLEAN_CACHE` (optional folder to cache cleaned text between runs).

Update 27 Sep ~18:40 (pushed): state-scoped lookup keys; missing states inferred from city words
learned from S1 addresses; native-script state spellings; memory fix (competing claims only for
validation targets' top 3; stage-2 competition context on the S1 side only). Full-India check on
a 16 GB laptop: peak RAM ~4 GB. The same code runs on Kaggle (techie1011/amazon-er-v2 v3 GPU and
techie1011/amazon-er-v2-cpu).

`code/business_entity_resolution/` is the **older v1** package (TF-IDF blocking, two-stage
LightGBM). Keep it for reference; v2 scores higher.

## 5. Results so far (regional slice only: Ohio + Kerala + Gironde, ~9% of test)

| run | val macro F0.5 | candidate recall | ceiling |
|---|---:|---:|---:|
| v1 | 0.9763 | 0.974 | — |
| v2 (CPU, no cross-encoder) | **0.9816** (US 0.984, India 0.976) | 0.989 | 0.996 |

The slice outputs (`dev/real_subset/...`) are NOT submittable (they cover only 160k of 1.73M S1).
**No full-test run has finished yet: that is the job.**

Full-data candidate recall (share of true links the candidate step keeps; the slice was optimistic):
| | before 27 Sep fixes | after state keys + state inference |
|---|---:|---:|
| India (883k S1 × 4.1M targets) | ~91% | 94.0% generated, **93.8% after trimming** |
| US | 96.9% | not re-measured yet |
State inference filled 116k targets (India records with no state 9.0% -> 3.2%).
The other session is diagnosing the remaining ~6% India misses; if a fix lands it will be a new
commit, so `git pull` before starting a run.

Known risks / what to look at if you have time:
- France predicts ~2.17 matches/S1 and ~18.5% empty lists vs 3.3 and 5.6% for US/India;
  France may be under-matched (no French training data). Check France rows by eye.
- On the full data, candidate recall (India 93.8%) is now the biggest loss, then decisions.
- A reported leaderboard score at rank 9 was 0.9988 (unconfirmed), so there is headroom.

## 6. Instructions for an AI assistant helping the runner

- Priority 1: get a **full** `matching_results.tsv` that prints PASS. Don't refactor first.
- If RAM runs out: lower `ER_TRAIN_S1` (e.g. 150000) before touching code; the test phase
  already runs per country and streams output.
- Don't add external data, APIs, geocoders, or non-MIT/Apache models (disqualification).
- Never upload the slice outputs. Never commit the dataset or outputs to git (see `.gitignore`).
- Report back: final validation F0.5 per country, candidate recall, runtime, peak RAM, and any
  code changes made (as a diff or a commit on a branch).
