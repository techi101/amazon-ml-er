# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

> Draft. Copy these sections into the official `Documentation_template.md` if its headings differ.
> Full-data numbers come from `output/validation.json` of the submitted run (Kaggle, v3 GPU, 3.75 h).

## 1. Problem and approach in one paragraph

For every Source 1 (S1) record we must return all Source 2/3 records describing the same business.
The score is F0.5 per S1 record, averaged, so a wrong link costs about twice as much as a missed one,
and an S1 record with no match scores 1.0 only for an empty list. Our pipeline:
**clean → generate candidates with many cheap lookup keys → score pairs with a two-stage
LightGBM model (plus an optional multilingual cross-encoder) → choose each S1 record's final set
with the one-owner rule and an expected-F0.5 rule.**
Only the provided data is used: no external databases, APIs, geocoding or internet lookups.
All models are MIT or Apache-2.0 licensed and far below 8B parameters.

## 2. Data analysis that shaped the design

| Finding (full training set) | Consequence |
|---|---|
| 2.21M S1, 10.3M S2+S3 records (test 1.73M / 9.97M) | Blocking must be key-based and run per country, in compact memory |
| Only 5.6% of S1 are singletons; mean 3.46 matches (max 11) | Recall matters; the empty list is not a safe default |
| Of 7.64M matched S2/S3 records, **0** belong to more than one S1 | One-owner rule is exact and used as a hard constraint |
| 26% of S2/S3 match nothing; 27% of those share the cleaned name of an S1 record at another address | The address must confirm the name (hard look-alikes) |
| ~10% of S2/S3 names in 9 Indian scripts | Rule-based transliteration + dictionary learned from training pairs |
| Alias names (`X trading as Y`, `aka`, `DBA`, `formerly`); some names are random words at the right address | Split aliases; add address-only keys and features |
| France appears only in test (15% of S1) | Country treated as an open label; French abbreviations and legal forms handled by rule; leave-one-country-out check |
| True pairs share a name word 85%, an address word 95%, house number ±10 79%, none of these 0.01% | A union of name keys and address keys can reach a very high recall |

## 3. Cleaning (normalisation)

- Unicode folding, accent removal, lower-casing, `&` → `and`, junk prefixes/suffixes (`#`, `@`, `--`,
  `>>`, `(ID: 123)`, `| ...`), phone-number tails, `null` / `N/A` literals.
- Digit-for-letter repair (`5ervices`, `lndia`, `c0m`), website/handle names split into words
  using the training vocabulary (`#empiremolecular` → `empire molecular`).
- **Indian scripts**: one relative table covers all 9 Indic Unicode blocks (they share one layout),
  handling the inherent vowel, vowel signs and virama. A word dictionary is then learned from
  training pairs whose target name is in a native script, by aligning words position by position
  (e.g. `praaivet → private`, `phaundeshan → foundation`; 669 entries on the full training set).
- Aliases: `aka / dba / fka / formerly / nee / t/a / doing business as / trading as / known as`
  split into main name and alias; both are kept.
- Legal forms (EN, IN, FR: Inc/LLC/Corp/Ltd/Pvt/LLP/SARL/SAS/SASU/EURL/EI/SA…), honorifics
  (Mr, Smt, Shri, Dr…) and filler words (Services, Center, Partners…) separated from the core name.
- Addresses: street types canonicalised (St/Street, Rd, Ave/Av, Bd/Blvd, R./Rue, Imp.…), house
  number extracted with leading zeros stripped (unit/floor/plot numbers skipped), street name,
  PO box, postcode, and state normalised across full name / code / native script
  (US states, Indian states incl. alternative codes, French regions and departments).
- A phonetic "skeleton" of each word for typo-tolerant keys.

## 4. Candidate generation (blocking)

Within each country, S1 and S2/S3 records are joined on **14 kinds of lookup key**; a pair is a
candidate if any key matches:

| Key family | Keys |
|---|---|
| Whole name | normalised core name (`n_key`), its phonetic skeleton (`n_skel`), name with spaces removed (`n_nospace`), for main name **and** alias |
| Rare name words | two rarest words (`n_rare2`), rarest word + house number (`n_rare1_num`), single rare words (`n_tok`, `n_tok_skel`, typo-tolerant) |
| Address | house number + first/second street word (`a_num_w1`, `a_num_w2`), street words with/without number (`a_w1w2`, `a_w1w2_n`), street name (`a_street`), street + number (`a_street_num`), rare address words (`a_tok`) |

Each key has frequency caps (e.g. a key shared by more than 300 records is skipped) so common
words do not explode the pair count. Candidates are then trimmed: for every S2/S3 record keep its
best 8 S1 records by a quick combined score plus the best 3 by name-only and by address-only score,
and at most 50 candidates per S1 record. The trimmed set is exactly what the model scores and is
written to `candidate_pairs.tsv`.

Blocking quality (validation, held-out S1 records): candidate recall **0.989** on the regional
development slice and **0.9595** on the full data (US 0.9726). Pairs per S1 record: ~14–27.

## 5. Pair features (~74)

- **Name**: ratio, partial ratio, token-set / token-sort ratios (rapidfuzz), Jaro-Winkler on the
  space-free name, word Jaccard, containment, skeleton equality and ratios, best alias-to-name
  score, count of "unexplained" words on each side, token counts.
- **Address**: token-set / partial / sorted ratios, weighted word Jaccard, containment,
  unexplained words, empty-address flags, street-name similarity.
- **Structured agreement** (agree / conflict / missing): house number, postcode, state, street,
  legal form; number containment and symmetric difference.
- **Record flags**: target in native script, has alias, joined words, unknown-word share,
  source (S2 vs S3), which of the 14 keys produced the pair.
- **Competition features (stage 2)**: the pair's rank and margin among all candidates of the same
  S1 record and among all S1 records claiming the same S2/S3 record; S2↔S3 support (an S2 and an
  S3 candidate of the same S1 record that match each other support each other).

## 6. Model

- Sampled training S1 records (with every competing claim on their S2/S3 records, so competition
  features are complete) are split into **half A**, **half B** and a **validation** set.
- **Stage 1**: LightGBM binary classifier (MIT) on half A
  (`num_leaves=127, learning_rate=0.05, feature_fraction=0.8, bagging 0.8, λ2=1`, early stopping).
- **Cross-encoder (optional, GPU)**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (Apache-2.0, 118M parameters) fine-tuned on half-A pairs to read both records together; it
  scores only the uncertain pairs.
- **Stage 2**: LightGBM on half B using stage-1 and cross-encoder scores plus the competition
  features.
- Probabilities calibrated with isotonic regression.

## 7. Decision: from probabilities to final lists

1. **One-owner rule**: each S2/S3 record is given only to the S1 record with the highest
   probability (exact in the training labels).
2. Per S1 record, either a probability threshold or the **expected-F0.5 set choice**: candidates
   sorted by probability, and the top-k (k = 0…n) with the highest expected F0.5 is kept; k = 0
   (empty list) is chosen when nothing is convincing, which handles singletons.
3. The rule (and threshold) is picked on half A and reported on the untouched validation half.

## 8. Validation

- Metric re-implemented exactly (per-S1 F0.5, macro average, singletons included).
- Split by S1 record, so a business never appears in both training and validation.
- Leave-one-country-out check as a proxy for unseen France.
- Results:

| Run | Validation macro F0.5 | US | India | Candidate recall | Ceiling* |
|---|---:|---:|---:|---:|---:|
| Regional slice (Ohio, Kerala, Gironde) | 0.9816 | 0.984 | 0.976 | 0.989 | 0.996 |
| Full data (submitted run: 80k/80k training S1, 40k validation, GPU cross-encoder) | **0.97272** | 0.97825 | 0.96461 | 0.9595 | 0.98465 |

\*Ceiling = score with perfect decisions on our candidates.

Test output: 1,732,544 S1 rows, 5,652,360 matches; US 3.32 matches/S1 (5.7% empty),
France 3.32 (5.0% empty), India 3.20 (6.5% empty). Passes the official `validate_submission.py`.

## 9. Reproducibility and compliance

- `pip install -r requirements.txt`, set `ER_DATA_DIR`, run `python colab/er_v2_jupyter.py`
  (or the Colab/Kaggle notebook). It writes `output/matching_results.tsv`,
  `output/candidate_pairs.tsv`, `output/validation.json` and runs the format check.
- Fixed seed (42). Peak RAM about 12 GB for the local edition.
- External resources: none at run time except downloading the pretrained Apache-2.0 MiniLM
  weights when a GPU is used. Abbreviation, state and transliteration tables are hand-written or
  learned from the provided training data.

## 10. What we would do with more time

- Raise candidate recall further with state-scoped keys (in progress) and more address keys.
- A larger cross-encoder on all mid-probability pairs.
- French-specific error analysis on the test distribution (no French labels available).
