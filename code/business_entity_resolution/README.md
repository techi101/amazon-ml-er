# Business Entity Resolution: reproduction

CPU only, no internet access needed at run time, no external data.

## Setup
```
pip install -r requirements.txt
```

## Run end to end
Point `--root` at the `student_resource/` directory (the one holding `dataset/train`, `dataset/test`, `utils/`):
```
cd code/business_entity_resolution
python -m src.run --root <path/to/student_resource> --mode full
```
It writes:
- `<root>/output/matching_results.tsv`: final matches (the leaderboard file)
- `<root>/output/candidate_pairs.tsv`: every candidate the model scored
- `<root>/output/reports/validation_report.json`: data profile, blocking recall, out-of-fold F0.5, country-transfer scores, feature importance

It then runs our own format check and, when present, the official `utils/validate_submission.py`.

`--mode cv` stops after validation. `--skip-transfer` skips the leave-one-country-out runs (saves time).

## Pipeline
1. `normalize.py`: country-agnostic cleaning (accents, punctuation, EN/IN/FR abbreviations, legal suffixes, DBA aliases, postcode/house number, landmarks).
2. `blocking.py`: TF-IDF views fitted per split; union of 6 retrievers per target source; capped per Source-1 record.
3. `features.py`: 85 similarity, evidence and context features; stage-2 context from stage-1 probabilities.
4. `model.py` / `pipeline.py`: two-stage LightGBM, grouped 5-fold out-of-fold, isotonic calibration.
5. `decide.py`: one-owner rule + expected-F0.5 set choice per Source-1 record (the empty list is chosen when nothing is convincing).
6. The decision variant is picked by out-of-fold macro F0.5 on train.
