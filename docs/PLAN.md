# Amazon ML Challenge 2026: data analysis and plan (27 Sep 2026)

## 1. The dataset in numbers
| | train | test |
|---|---:|---:|
| Source 1 records | 2,206,821 | 1,732,544 |
| Source 2 records | 5,034,616 | 4,887,273 |
| Source 3 records | 5,285,603 | 5,082,316 |
| Countries (S1) | US 60%, India 40% | India 47%, US 38%, **France 15%** |

Ground truth:
- Singletons: 5.6% of Source 1 records have no match.
- Matches per Source 1 record: mean 3.46, max 11. Most records have 2 to 5 matches.
- **One-owner rule holds exactly.** 7.64M matched S2/S3 records, 0 with more than one Source 1 owner.
- 26% of S2/S3 records (2.68M) match nothing. These are the look-alikes. 27% of them have exactly the same cleaned name as some Source 1 record but a different address (for example "Youth Services", "Creative Builders").

## 2. Noise found (share of true pairs)
- Names in Indic scripts: about 7% of pairs, about 20% of India pool records. Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi. State names also appear in native script (for example ಕರ್ನಾಟಕ, महाराष्ट्र).
- Alias names: a random word joined to the real name by "DBA", "t/a", "a/k/a", "formerly", "doing business as" (for example "Korwexnyla trading as Gold Consulting"). Some pool names are **only** a random word ("Hsssoo") with the same address. There, only the address can match.
- Legal suffixes added, dropped or changed; extra words ("Services", "Partners", "Authority"); upper/lower case; added accents; leetspeak (5ervices, lndia); junk prefixes (#, @, --, >>, ***); bracketed tails.
- Addresses:
  - components reordered;
  - state written in full vs abbreviated vs native script;
  - city typos;
  - house number with leading zeros (011265) or off by a few (1786 vs 1791);
  - "##", "N/A" and "null" literals;
  - about 4.5% empty.
- France (test only): R./Rue, Av./Avenue, Bd, Imp., Nº, BIS; department ("Nord", "Gironde") vs region ("Hauts-de-France"); French legal forms SARL, SAS, SASU, EURL, EI, SA.

What true pairs share (207k sampled pairs):
- a name word: 85%;
- an address word, usually city or street: 95%;
- house numbers within ±10: 79%;
- **none of the three: 0.01%.**

## 3. On the "99.99% precision" goal
The score is F0.5 per Source 1 record, not precision. We can raise precision by predicting less, but each dropped true match costs recall. The data also contains true matches that look impossible, such as a random-word name at the same address, or "Family Association" vs "Family Service".
- Across about 6M test matches, 99.99% precision means at most 600 wrong links. The label noise alone makes that unreachable without collapsing recall.
- **Target: maximise F0.5.** Current real-data validation is 0.976. A realistic finish is 0.985 to 0.99.

## 4. Where the remaining points are
1. **Blocking recall ceiling: 97.4%** on the real-data subset. It's the biggest single loss. The missed pairs are mostly alias names (name shares nothing, address identical) and Indic-script names.
2. Precision on look-alikes: same name at a different address, and different businesses at the same address.
3. France: no training data, so French abbreviations and legal forms must be canonicalised by rule.

## 5. Pipeline to build next
1. **Normalise**: strip junk, undo leetspeak, split aliases (keep both halves), transliterate Indic scripts (rules plus a dictionary learned from training pairs), canonicalise street types, states and legal forms (EN, IN, FR), strip leading zeros from numbers.
2. **Blocking (union)**, within country:
   - (a) rare name tokens;
   - (b) (city or postcode, house number);
   - (c) (street word, house number);
   - (d) TF-IDF char n-grams on the transliterated name.
   Target recall ceiling of at least 99.5% with about 15 to 20 candidates per Source 1 record.
3. **Features**: the current 54, plus alias-half similarity, house-number distance, native-script flag and address-only match strength.
4. **LightGBM**, grouped by Source 1 record, with a 2-stage cascade (a cheap model prunes, a full model scores). Calibrate the probabilities.
5. **Decision**: one-owner rule, then the threshold or expected-F0.5 set choice tuned on held-out Source 1 records. Check leave-one-country-out.
6. **Scale**: process per country and per state partition to fit in 16 GB of RAM. The full test run is 1.7M Source 1 records × about 10M pool records.
