# %% [markdown]
# # Business Entity Resolution v2 (GPU): Amazon ML Challenge 2026
#
# Hybrid pipeline, end to end on the full dataset:
# 1. **Clean** names and addresses: 9 Indian scripts transliterated (+ a dictionary learned from training pairs), accents, junk, tags, digit-for-letter swaps, website names split into words, aliases (`X aka / DBA / formerly Y`), legal forms, honorifics, street numbers (unit/floor numbers skipped), street names, states (US / India / France).
# 2. **Candidates** from 14 kinds of lookup key per country: whole-name keys (main name *and* alias), single rare words (typo-tolerant), house number + street word, street name without number. Trimming keeps the best matches by combined, name-only and address-only score.
# 3. **Stage 1**: LightGBM on ~70 pair features, trained on half A of the sampled S1 records.
# 4. **Cross-encoder (GPU)**: multilingual MiniLM (Apache-2.0, 118M parameters) fine-tuned on half A to read both records together; it scores the uncertain pairs.
# 5. **Stage 2**: LightGBM on half B with stage-1 and cross-encoder scores, each pair's standing against competing candidates on both sides, and S2 ↔ S3 support.
# 6. **Decision**: one-owner rule (each S2/S3 record goes to at most one S1 record; exact in the training labels) + a threshold or expected-F0.5 set choice, picked on validation.
#
# **Runtime:** Runtime → Change runtime type → **T4 GPU** (High-RAM if available). Then Runtime → Run all.
# Without a GPU it still runs; the cross-encoder step is skipped.
# **Data:** copied automatically from the team's shared Drive folder (cell 3). Outputs go to `/content/output`
# and are copied to `MyDrive/amazon_er_output` at the end.

# %%
#!pip -q install rapidfuzz lightgbm psutil gdown transformers

# %%
import gc
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist
from sklearn.isotonic import IsotonicRegression

# ---- data location -------------------------------------------------------------------
# If automatic detection fails, put the folder that holds the .tsv files here, e.g.
# "/kaggle/input/amazon-ml-challenge-2026" or "/content/drive/MyDrive/dataset".
DATA_DIR_MANUAL = ""
# Team's shared Drive folder (Colab only). If the automatic download fails, open the folder link,
# click "Add shortcut to Drive" (into My Drive) and rerun: the files are then found through Drive.
DRIVE_FOLDER_ID = "1AAnZuLWXNj6dRALnz1KlL9mPMWAA8y6b"
IN_KAGGLE = os.path.exists("/kaggle/input")
IN_COLAB = os.path.exists("/content") and not IN_KAGGLE
WORK = "/kaggle/working" if IN_KAGGLE else ("/content" if IN_COLAB else os.getcwd())
NEEDED = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv",
          "test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]


def find_data_dir(root):
    """Find the 7 challenge files anywhere under root (any folder layout, zips extracted) and
    return a directory laid out as <dir>/train/*.tsv and <dir>/test/*.tsv."""
    if not root or not os.path.exists(root):
        return None
    found = {}
    for dirpath, _, files in os.walk(root, followlinks=True):
        for f in files:
            if f.lower() in NEEDED and f.lower() not in found:
                found[f.lower()] = os.path.join(dirpath, f)
            elif f.lower().endswith(".zip") and len(found) < len(NEEDED):
                import zipfile
                dst = os.path.join(WORK, "dataset_unzipped", os.path.splitext(f)[0])
                if not os.path.exists(dst):
                    print("extracting", os.path.join(dirpath, f))
                    zipfile.ZipFile(os.path.join(dirpath, f)).extractall(dst)
                for d2, _, f2 in os.walk(dst):
                    for g in f2:
                        if g.lower() in NEEDED and g.lower() not in found:
                            found[g.lower()] = os.path.join(d2, g)
        if len(found) == len(NEEDED):
            break
    if len(found) < len(NEEDED):
        if found:
            print(f"under {root}: found {sorted(found)}; missing {sorted(set(NEEDED) - set(found))}")
        return None
    t1 = os.path.dirname(found["train_source1.tsv"])
    if (os.path.basename(t1) == "train" and os.path.dirname(t1) == os.path.dirname(os.path.dirname(found["test_source1.tsv"]))
            and all(os.path.dirname(found[f]) == t1 for f in NEEDED if f.startswith("train"))):
        return os.path.dirname(t1)                     # already <dir>/train, <dir>/test
    arranged = os.path.join(WORK, "dataset_arranged")  # any other layout: link files into place
    for f, path in found.items():
        split = "train" if f.startswith("train") else "test"
        os.makedirs(os.path.join(arranged, split), exist_ok=True)
        link = os.path.join(arranged, split, f)
        if not os.path.exists(link):
            try:
                os.symlink(path, link)
            except OSError:
                import shutil
                shutil.copy(path, link)
    return arranged


DATA_DIR = os.environ.get("ER_DATA_DIR") or (find_data_dir(DATA_DIR_MANUAL) if DATA_DIR_MANUAL else None)
if DATA_DIR is None and IN_KAGGLE:          # Kaggle: attach the dataset with "+ Add Input"
    DATA_DIR = find_data_dir("/kaggle/input")
if DATA_DIR is None and IN_COLAB:
    DATA_DIR = find_data_dir("/content/dataset")
    if DATA_DIR is None:
        try:
            import gdown
            gdown.download_folder(id=DRIVE_FOLDER_ID, output="/content/dataset", quiet=False, use_cookies=False)
        except Exception as e:
            print("gdown download failed:", e)
        DATA_DIR = find_data_dir("/content/dataset")
    if DATA_DIR is None:
        from google.colab import drive
        drive.mount("/content/drive")
        DATA_DIR = find_data_dir("/content/drive/MyDrive") or find_data_dir("/content/drive/Shareddrives")
if DATA_DIR is None:
    DATA_DIR = find_data_dir(os.getcwd())
if DATA_DIR is None:
    for r in ("/kaggle/input", "/content", "/content/drive/MyDrive", os.getcwd()):
        if os.path.exists(r):
            print(f"contents of {r}:", sorted(os.listdir(r))[:30])
    raise FileNotFoundError("Dataset not found. Set DATA_DIR_MANUAL (top of this cell) to the folder that "
                            "contains the 7 .tsv files, or on Kaggle attach the dataset with '+ Add Input'.")
print("DATA_DIR =", DATA_DIR)
for f in NEEDED:
    split = "train" if f.startswith("train") else "test"
    assert os.path.exists(f"{DATA_DIR}/{split}/{f}"), f"missing {DATA_DIR}/{split}/{f}"
if IN_KAGGLE:
    os.environ.setdefault("ER_OUT_DIR", "/kaggle/working/output")
OUT_DIR = os.environ.get("ER_OUT_DIR", os.path.join(WORK, "output"))
TRAIN_S1_SAMPLE = int(os.environ.get("ER_TRAIN_S1", 300_000))   # S1 records for training (half A + half B)
VAL_S1_SAMPLE = int(os.environ.get("ER_VAL_S1", 60_000))       # held-out S1 records for scoring
RUN_TEST = os.environ.get("ER_RUN_TEST", "1") == "1"
SEED = 42

# candidate generation
CAP_S1 = 25        # a key shared by more S1 records than this is too generic -> skipped
CAP_T = 300        # same for S2/S3 records
TOP_PER_T = 8      # keep the best N S1 candidates per S2/S3 record (by quick score)
TOP_PER_S1 = 50    # and at most N candidates per S1 record
TOP_BY_ONE = 3     # plus the best N by name alone and by address alone
os.makedirs(OUT_DIR, exist_ok=True)
T0 = time.time()


def log(msg):
    try:
        import psutil
        mem = f" | RAM {psutil.Process().memory_info().rss / 2**30:.1f} GB"
    except Exception:
        mem = ""
    print(f"[{time.time() - T0:7.0f}s] {msg}{mem}", flush=True)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def compact(df):
    """Store text columns as Arrow strings (a fraction of the memory of Python string objects)."""
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string[pyarrow]")
    return df


def sel(s, idx):
    """Rows idx of a column as a numpy array; text comes back as Python strings (object array)."""
    if isinstance(s.dtype, pd.StringDtype):
        return np.asarray(s.array.take(np.asarray(idx)), dtype=object)
    return s.to_numpy()[idx]

# %% [markdown]
# ## 1. Vocabulary: hand-written domain knowledge (no external data)

# %%
def _inv(groups):
    return {v: k for k, vs in groups.items() for v in vs + [k]}

LEGAL = _inv({
    "llc": ["l.l.c"], "inc": ["incorporated", "lnc", "incorp"], "corp": ["corporation", "corpn"],
    "co": ["company", "cie", "compagnie"], "ltd": ["limited", "ltda", "lmtd"], "pvt": ["private", "pvte", "prvt"],
    "plc": [], "llp": [], "lp": [], "pllc": [], "pc": [], "pa": [], "sarl": [], "sas": [], "sasu": [],
    "sa": [], "eurl": [], "sci": [], "snc": [], "scop": [], "gmbh": [], "opc": [], "selarl": [],
})
HONOR = {"mr", "mrs", "ms", "smt", "shri", "sri", "shree", "dr", "sh", "kumari", "km", "late", "mx", "mme", "mlle", "m"}
STOP = {"the", "of", "and", "a", "an", "de", "du", "des", "la", "le", "les", "l", "d", "et", "en", "aux", "au"}
# words the noise generator adds/drops (measured on train pairs); ignored for keys, down-weighted in features
FILLER = {"center", "centre", "services", "service", "partners", "labs", "lab", "sys", "one", "commission",
          "district", "council", "federation", "authority", "society", "trust", "foundation", "group",
          "associates", "india", "france", "usa", "enterprises", "solutions", "trading", "global",
          "international", "intl", "holdings", "ventures"}
ALIAS_WORDS = {"dba", "aka", "fka", "formerly", "nee", "ta"}
ALIAS_PHRASE = re.compile(r"\b(doing business as|also known as|formerly known as|known as|trading as)\b")

ADDR_CANON = _inv({
    "st": ["street", "str", "strt", "saint", "sainte", "ste"], "rd": ["road", "raod"],
    "ave": ["avenue", "av", "avn", "aven"], "blvd": ["boulevard", "bd", "bld", "boul", "bvd"],
    "dr": ["drive", "drv"], "ln": ["lane"], "ct": ["court", "crt"], "cir": ["circle", "circ"],
    "pl": ["place"], "ter": ["terrace", "terr"], "trl": ["trail"], "pkwy": ["parkway", "pky"],
    "hwy": ["highway"], "sq": ["square"], "pt": ["point"], "mt": ["mount"], "ft": ["fort"],
    "_n": ["north", "nord", "n"], "_s": ["south", "sud", "s"], "_e": ["east", "est", "e"],
    "_w": ["west", "ouest", "w"], "_ne": ["northeast", "ne"], "_nw": ["northwest", "nw"],
    "_se": ["southeast", "se"], "_sw": ["southwest", "sw"],
    "rue": ["r"], "allee": ["all", "al"], "rte": ["route"], "quai": ["q"], "imp": ["impasse"],
    "ch": ["chemin", "che"], "cours": [], "fbg": ["faubourg"],
    "ngr": ["nagar", "nagara", "nagr"], "mg": ["marg", "maarg"], "sec": ["sector", "sect"],
    "ph": ["phase"], "col": ["colony"], "mkt": ["market"], "chowk": ["chauk", "chk"],
    "bazar": ["bazaar", "bzr"], "apt": ["apartments", "apartment", "apts", "appt"],
    "bldg": ["building"], "ext": ["extension", "extn"], "opp": ["opposite"], "near": ["nr"],
    "bhd": ["behind"],
})
ADDR_DROP = {"no", "number", "num", "h", "hno", "house", "door", "plot", "flat", "shop", "unit", "suite",
             "fl", "floor", "flr", "room", "rm", "office", "po", "box", "pmb", "pob", "city", "of", "village",
             "town", "township", "twp", "townshp", "cdp", "hq", "region", "the", "and", "de", "du", "des",
             "la", "le", "les", "bis", "null", "none", "na", "c", "o", "nd", "th"}

US_STATES = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky",
    "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc"}
IN_STATES = {"andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "andaman and nicobar islands": "an",
    "lakshadweep": "ld", "dadra and nagar haveli and daman and diu": "dn"}
FR_REGIONS = {"hauts de france": "hdf", "nouvelle aquitaine": "naq", "pays de la loire": "pdl",
    "ile de france": "idf", "auvergne rhone alpes": "ara", "provence alpes cote d azur": "paca",
    "occitanie": "occ", "grand est": "ges", "bretagne": "bre", "normandie": "nor",
    "bourgogne franche comte": "bfc", "centre val de loire": "cvl", "corse": "cor",
    "nord": "hdf", "pas de calais": "hdf", "somme": "hdf", "aisne": "hdf", "oise": "hdf",
    "gironde": "naq", "landes": "naq", "pyrenees atlantiques": "naq", "charente maritime": "naq",
    "dordogne": "naq", "lot et garonne": "naq", "vienne": "naq", "haute vienne": "naq",
    "loire atlantique": "pdl", "vendee": "pdl", "maine et loire": "pdl", "sarthe": "pdl", "mayenne": "pdl",
    "seine saint denis": "idf", "hauts de seine": "idf", "val de marne": "idf", "essonne": "idf",
    "yvelines": "idf", "val d oise": "idf", "seine et marne": "idf", "rhone": "ara", "isere": "ara",
    "haute savoie": "ara", "savoie": "ara", "puy de dome": "ara", "bouches du rhone": "paca",
    "alpes maritimes": "paca", "vaucluse": "paca", "haute garonne": "occ", "herault": "occ", "gard": "occ",
    "bas rhin": "ges", "haut rhin": "ges", "moselle": "ges", "ille et vilaine": "bre", "finistere": "bre",
    "morbihan": "bre", "cotes d armor": "bre", "seine maritime": "nor", "calvados": "nor",
    "cote d or": "bfc", "doubs": "bfc", "loiret": "cvl", "indre et loire": "cvl"}
CODE_ALIAS = {"ts": "tg", "ct": "cg", "ut": "uk", "or": "od"}   # India's alternative codes


def _state_lookup():
    raw = {}
    for d in (US_STATES, IN_STATES, FR_REGIONS):
        for name, code in d.items():
            raw[name] = code
            raw[code] = code
    for a, b in CODE_ALIAS.items():
        raw[a] = b
    return {k: CODE_ALIAS.get(v, v) for k, v in raw.items()}

STATE_LOOKUP = _state_lookup()

# %% [markdown]
# ## 2. Transliteration of Indian scripts
# All 9 Indic Unicode blocks share one layout, so one table (written relative to the block start) covers
# Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam.
# Consonants carry an inherent "a" that is removed before a vowel sign, a virama, or at the end of a word.

# %%
_CONS = "k kh g gh n ch chh j jh n t th d dh n t th d dh n n p ph b bh m y r r l l l v sh sh s h".split()
_REL = {}
for off, v in {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri", 0x0C: "li",
               0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
               0x60: "ri", 0x61: "li"}.items():
    _REL[off] = v
for off, v in zip(range(0x15, 0x3A), _CONS):
    _REL[off] = v + "\x01"
for off, v in zip(range(0x58, 0x60), "q kh g z r rh f y".split()):
    _REL[off] = v + "\x01"
for off, v in {0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "ri", 0x45: "e",
               0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x56: "ai",
               0x57: "au", 0x62: "li", 0x63: "li"}.items():
    _REL[off] = "\x03" + v
_REL.update({0x4D: "\x02", 0x01: "n", 0x02: "n", 0x03: "h", 0x3C: "", 0x3D: "", 0x55: "", 0x70: "n", 0x71: ""})
for d in range(10):
    _REL[0x66 + d] = str(d)
INDIC = {}
for base in (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00):
    for off, v in _REL.items():
        if unicodedata.name(chr(base + off), None):
            INDIC[base + off] = v
INDIC.update({0x09CE: "t", 0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k",
              0x0D54: "m", 0x0D55: "y", 0x0D56: "l", 0x0964: " ", 0x0965: " ", 0x200C: "", 0x200D: ""})
SPECIAL = {ord("’"): "'", ord("‘"): "'", ord("ß"): "ss", ord("æ"): "ae", ord("œ"): "oe", ord("ø"): "o",
           ord("đ"): "d", ord("ł"): "l", ord("ı"): "i", ord("–"): "-", ord("—"): "-"}
INDIC_PAT = "[ऀ-ൿ]"


def translit_indic(s):
    s = s.str.translate(INDIC)
    s = s.str.replace("\x01\x03", "", regex=False).str.replace("\x01\x02", "", regex=False)
    s = s.str.replace(r"\x01(?=[\s,.\-/()]|$)", "", regex=True)      # word-final inherent vowel
    s = s.str.replace("\x01", "a", regex=False)
    return s.str.replace(r"[\x02\x03]", "", regex=True)


def fold_text(s):
    """Indic -> Latin, accents stripped, lowercase."""
    s = s.fillna("").astype(str)
    m = s.str.contains(INDIC_PAT, regex=True)
    if m.any():
        s = s.copy()
        s.loc[m] = translit_indic(s.loc[m])
    s = s.str.translate(SPECIAL).str.normalize("NFKD").str.replace(r"[̀-ͯ]", "", regex=True)
    return s.str.lower(), m.to_numpy()


_CONFUSE = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e"})
_PHON = [("tion", "sn"), ("sion", "sn"), ("ph", "f"), ("sh", "s"), ("ch", "\x04"), ("kh", "k"), ("gh", "g"),
         ("th", "t"), ("dh", "d"), ("bh", "b"), ("ck", "k"), ("qu", "k"), ("q", "k"), ("x", "ks"), ("z", "s"),
         ("w", "v"), ("ce", "se"), ("ci", "si"), ("cy", "sy"), ("c", "k"), ("\x04", "c"), ("i", "l"), ("h", "")]


@lru_cache(maxsize=4_000_000)
def skel(w):
    """Sound-alike key: English 'foundation' and Hindi-transliterated 'phaundeshan' both -> 'fndsn'."""
    if any(c.isalpha() for c in w):
        w = w.translate(_CONFUSE)
    for a, b in _PHON:
        w = w.replace(a, b)
    if not w:
        return w
    out = w[0] + re.sub(r"[aeouy]", "", w[1:])
    return re.sub(r"(.)\1+", r"\1", out)

# %% [markdown]
# ## 3. Name and address cleaning

# %%
RE_ID = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)")
RE_PIPE = re.compile(r"\|.*$")
RE_PHONE = re.compile(r"\s-\s*\d{6,}")
RE_HASHNUM = re.compile(r"#\s*\d+")
RE_LETTER = re.compile(r"[a-z]")
RE_DOMAIN = re.compile(r"(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|c0m|co\.in|in|net|org|co|fr|biz|io|info)\b")
RE_HANDLE = re.compile(r"(?:^|\s)[#@]([a-z0-9]{3,})\b")
RE_NONWORD = re.compile(r"[^a-z0-9 ]+")
RE_SINGLE_RUN = re.compile(r"\b[a-z](?: [a-z])+\b")
RE_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
RE_DIGIT = re.compile(r"\d")
_LEET = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "4": "a"})


def _leet(tok):
    if RE_DIGIT.search(tok) and not RE_ORDINAL.match(tok) and sum(c.isalpha() for c in tok) >= 3:
        return tok.translate(_LEET)
    return tok


def name_tokens(t):
    """Folded name string -> (main tokens, alias tokens, joined website/handle string)."""
    joined = ""
    t = RE_PIPE.sub(" ", RE_ID.sub(" ", t))
    t = RE_PHONE.sub(" ", t)
    t2 = RE_HASHNUM.sub(" ", t)
    if RE_LETTER.search(t2):
        t = t2
    m = RE_DOMAIN.search(t) or RE_HANDLE.search(t)
    if m:
        joined = _leet(m.group(1).replace("-", ""))
        t = t[:m.start()] + " " + t[m.end():]
    t = ALIAS_PHRASE.sub(" dba ", t.replace("&", " and ").replace("+", " and ").replace("'", ""))
    t = RE_NONWORD.sub(" ", t)
    t = RE_SINGLE_RUN.sub(lambda mm: mm.group(0).replace(" ", ""), t)
    toks, prev = [], None
    for x in t.split():
        x = _leet(x)
        if x != prev:
            toks.append(x)
        prev = x
    alias = []
    for i, x in enumerate(toks):
        if x in ALIAS_WORDS and i > 0:
            alias = [y for y in toks[i + 1:] if y not in ALIAS_WORDS]
            toks = toks[:i]
            break
    return toks, alias, joined


def segment(s, vocab, max_len=24):
    """Split 'pclmedicalcentre' into known words (vocab: word -> log-frequency)."""
    n = len(s)
    best = [0.0] + [-1e18] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            w = s[j:i]
            sc = vocab.get(w)
            if sc is None:
                sc = -12.0 * len(w) if len(w) <= 2 else None
            if sc is not None and best[j] + sc > best[i]:
                best[i], back[i] = best[j] + sc, j
    out, i = [], n
    while i > 0:
        out.append(s[back[i]:i])
        i = back[i]
    return out[::-1]


def split_name(toks, translit=None):
    if translit:
        toks = [translit.get(x, x) for x in toks]
    legal = sorted({LEGAL[x] for x in toks if x in LEGAL})
    core = [x for x in toks if x not in LEGAL and x not in HONOR and x not in STOP] or [x for x in toks if x not in STOP] or toks
    core2 = [x for x in core if x not in FILLER] or core
    return toks, core, core2, legal


def name_frame(raw, vocab=None, translit=None):
    folded, native = fold_text(raw)
    rows = []
    for t, nat in zip(folded.tolist(), native):
        toks, alias, joined = name_tokens(t)
        if joined:
            seg = segment(joined, vocab) if vocab else [joined]
            toks = toks + seg
        toks, core, core2, legal = split_name(toks, translit if nat else None)
        _, acore, acore2, _ = split_name(alias, translit if nat else None) if alias else ([], [], [], [])
        rows.append((" ".join(toks), " ".join(core), " ".join(core2), " ".join(sorted(set(core2))),
                     " ".join(sorted({skel(x) for x in core2})), "".join(core2), " ".join(acore2),
                     " ".join(legal), len(core2), bool(joined)))
    df = pd.DataFrame(rows, columns=["nm_full", "nm_core", "nm_core2", "nm_key", "nm_skel", "nm_nospace",
                                     "nm_alias", "nm_legal", "nm_ntok", "nm_joined"])
    df["nm_native"] = native
    return df


RE_POBOX = re.compile(r"\b(p\s?o\s?box|post box|pmb|pob)\s*#?\s*\d+")
RE_NULL = re.compile(r"\b(null|none|n/a)\b")
RE_ORD_ADDR = re.compile(r"\b(\d+)(st|nd|rd|th|er|eme|ere|e)\b")
RE_ALNUM = re.compile(r"[a-z]+|\d+")


def _state_of(comp):
    key = " ".join(RE_NONWORD.sub(" ", comp.replace("'", " ")).split())
    if not key or RE_DIGIT.search(key) or key.count(" ") > 6:
        return ""
    return STATE_LOOKUP.get(key) or STATE_SKEL.get(" ".join(skel(w) for w in key.split()), "")

STATE_SKEL = {}
for _k, _v in STATE_LOOKUP.items():
    if len(_k) > 3:
        STATE_SKEL.setdefault(" ".join(skel(w) for w in _k.split()), _v)


def addr_frame(raw):
    folded, native = fold_text(raw)
    rows = []
    for t in folded.tolist():
        t = RE_NULL.sub(" ", RE_POBOX.sub(" ", t))
        state, rest = "", []
        for c in t.split(","):
            c = c.strip()
            if not c:
                continue
            st = "" if state else _state_of(c)
            if st:
                state = st
            else:
                rest.append(c)
        nums, words, street = [], [], set()
        street_num = first_free = ""
        for comp in rest:
            text = RE_ORD_ADDR.sub(r"\1", comp).replace("/", " ").replace("-", " ").replace("'", " ")
            toks = RE_ALNUM.findall(RE_NONWORD.sub(" ", text))
            comp_words = []
            for j, x in enumerate(toks):
                if x.isdigit():
                    n = x.lstrip("0") or "0"
                    nums.append(n)
                    is_unit = j > 0 and toks[j - 1] in UNIT_RAW
                    if not is_unit:
                        first_free = first_free or n
                        # a street number is followed by a word in the same component ("1090 parsons ave")
                        if not street_num and any(not y.isdigit() and y not in ADDR_DROP for y in toks[j + 1:j + 3]):
                            street_num = n
                    continue
                w = ADDR_CANON.get(x, x)
                if len(w) > 1 and w not in ADDR_DROP and x not in ADDR_DROP:
                    words.append(w)
                    comp_words.append(w)
            if any(w in STREET_TYPES for w in comp_words):
                street |= {skel(w) for w in comp_words if w not in STREET_TYPES and not w.startswith("_")}
        num1 = street_num or first_free or (nums[0] if nums else "")
        pc = nums[-1] if (len(nums) > 1 and len(nums[-1]) in (5, 6) and nums[-1] != num1) else ""
        rows.append((" ".join(words), " ".join(sorted(set(words))), " ".join(dict.fromkeys(nums)),
                     num1, pc, state, " ".join(sorted(street)), not words and not nums))
    df = pd.DataFrame(rows, columns=["ad_words", "ad_wset", "ad_nums", "ad_num1", "ad_pc", "ad_state",
                                     "ad_street", "ad_empty"])
    return df


UNIT_RAW = {"fl", "floor", "flr", "unit", "suite", "ste", "room", "rm", "apt", "apartment", "office", "shop",
            "flat", "block", "blk", "wing", "level", "lvl"}
STREET_TYPES = {"st", "rd", "ave", "blvd", "dr", "ln", "ct", "cir", "pl", "ter", "trl", "pkwy", "hwy", "sq",
                "rue", "allee", "rte", "quai", "imp", "ch", "cours", "fbg", "mg", "way", "pt", "cross", "main"}

# %% [markdown]
# ## 4. Learn a transliteration dictionary from the training pairs
# About 7% of true matches spell the name in an Indian script (`सिल्वर फाउंडेशन प्राइवेट लिमिटेड`).
# Those names are word-for-word transliterations of the S1 name, so aligning words position by position
# on the training pairs gives a dictionary like `praaivet -> private`, `phaundeshan -> foundation`.

# %%
def learn_translit(n_pairs=400_000):
    gt = read_tsv(f"{DATA_DIR}/train/train_ground_truth.tsv")
    s1 = read_tsv(f"{DATA_DIR}/train/train_source1.tsv")
    s1 = s1[s1.country == "India"].set_index("entity_id")
    gt = gt[gt.source1_entity_id.isin(s1.index)]
    links = gt.assign(t=gt.matched_entity_ids.str.split(",")).explode("t")
    links = links[links.t.fillna("") != ""]
    links = links.sample(min(n_pairs, len(links)), random_state=SEED)
    need = set(links.t)
    tg = []
    for s in (2, 3):
        d = read_tsv(f"{DATA_DIR}/train/train_source{s}.tsv")
        d = d[d.entity_id.isin(need) & d.business_name.str.contains(INDIC_PAT, regex=True)]
        tg.append(d[["entity_id", "business_name"]])
        del d
        gc.collect()
    tg = pd.concat(tg).set_index("entity_id")
    links = links[links.t.isin(tg.index)]
    a_f, _ = fold_text(s1.loc[links.source1_entity_id, "business_name"].reset_index(drop=True))
    b_f, _ = fold_text(tg.loc[links.t, "business_name"].reset_index(drop=True))
    cnt, tot = Counter(), Counter()
    for a, b in zip(a_f.tolist(), b_f.tolist()):
        ta, tb = name_tokens(a)[0], name_tokens(b)[0]
        if len(ta) == len(tb):
            for x, y in zip(tb, ta):
                tot[x] += 1
                if x != y:
                    cnt[(x, y)] += 1
    best = {}
    for (x, y), c in cnt.most_common():
        if x not in best and c >= 2 and c / tot[x] >= 0.5:
            best[x] = y
    log(f"transliteration dictionary: {len(best)} entries from {len(links)} native-script pairs")
    return best

TRANSLIT = learn_translit()
print(list(TRANSLIT.items())[:25])

# %% [markdown]
# ## 5. Per-country processing: clean → keys → candidates → quick score → context

# %%
def df_counts(series):
    return series[series != ""].str.split().explode().value_counts()


def rare_tokens(texts, df, k=2, max_df=None):
    out = []
    for t in texts:
        ws = sorted(set(t.split()), key=lambda w: (df.get(w, 0), w))
        if max_df is not None:
            ws = [w for w in ws if df.get(w, 0) <= max_df]
        out.append(ws[:k])
    return out


def hkey(parts, ok):
    h = pd.util.hash_array(np.array(parts, dtype=object)).astype(np.int64)
    h[~np.asarray(ok)] = 0
    return h


def build_keys(F, name_df, addr_df, country):
    """kind -> list of (row index array, int64 key array). A row can emit several keys per kind
    (its alias as well as its main name; one key per rare word for the token kinds)."""
    c = country + "|"
    n = len(F)
    K = defaultdict(list)

    def add(kind, strings, ok, rows=None):
        rows = np.arange(n) if rows is None else np.asarray(rows)
        if len(rows):
            K[kind].append((rows, hkey(strings, ok)))

    num1 = F.ad_num1.tolist()
    views = [F.nm_core2.tolist()]
    alias = F.nm_alias.tolist()
    if any(alias):
        views.append(alias)
    for core2 in views:
        tok = [sorted(set(t.split())) for t in core2]
        rn = [sorted(t, key=lambda w: (name_df.get(w, 0), w))[:2] for t in tok]
        add("n_key", [c + "k" + " ".join(t) for t in tok], [bool(t) for t in tok])
        sk = [" ".join(sorted({skel(w) for w in t})) for t in tok]
        add("n_skel", [c + "s" + x for x in sk], [len(x) >= 3 for x in sk])
        ns = ["".join(t.split()) for t in core2]
        add("n_nospace", [c + "j" + x for x in ns], [len(x) >= 6 for x in ns])
        add("n_rare2", [c + "r" + " ".join(sorted(r)) for r in rn], [len(r) == 2 for r in rn])
        add("n_rare1_num", [c + "q" + (r[0] if r else "") + "|" + m for r, m in zip(rn, num1)],
            [bool(r) and bool(m) for r, m in zip(rn, num1)])
        # token blocking: every informative name word on its own (typos elsewhere in the name survive)
        rows, words = [], []
        for i, t in enumerate(tok):
            for w in t:
                if len(w) >= 3 and w not in FILLER:
                    rows.append(i)
                    words.append(w)
        add("n_tok", [c + "t" + w for w in words], [True] * len(words), rows)
        add("n_tok_skel", [c + "u" + skel(w) for w in words], [len(skel(w)) >= 3 for w in words], rows)

    ra = rare_tokens(F.ad_wset.tolist(), addr_df)
    add("a_num_w1", [c + "a" + m + "|" + (r[0] if r else "") for r, m in zip(ra, num1)],
        [bool(r) and bool(m) for r, m in zip(ra, num1)])
    add("a_num_w2", [c + "b" + m + "|" + (r[1] if len(r) > 1 else "") for r, m in zip(ra, num1)],
        [len(r) > 1 and bool(m) for r, m in zip(ra, num1)])
    add("a_w1w2", [c + "v" + " ".join(sorted(r)) for r in ra], [len(r) > 1 for r in ra])
    rn_main = rare_tokens(F.nm_core2.tolist(), name_df, k=1)
    add("a_w1w2_n", [c + "w" + " ".join(sorted(r)) + "|" + (q[0] if q else "") for r, q in zip(ra, rn_main)],
        [len(r) > 1 and bool(q) for r, q in zip(ra, rn_main)])
    rows, words = [], []
    for i, t in enumerate(F.ad_wset.tolist()):
        for w in t.split():
            if len(w) >= 4 and w not in STREET_TYPES and not w.startswith("_"):
                rows.append(i)
                words.append(w)
    add("a_tok", [c + "x" + w for w in words], [True] * len(words), rows)
    st, stt = F.ad_street.tolist(), F.ad_state.tolist()
    add("a_street", [c + "y" + s + "|" + x for x, s in zip(st, stt)], [bool(x) for x in st])
    add("a_street_num", [c + "z" + s + "|" + x + "|" + m for x, s, m in zip(st, stt, num1)],
        [bool(x) and bool(m) for x, m in zip(st, num1)])
    return K

# kind -> (max S1 records sharing a key, max S2/S3 records sharing it). Keys above the cap are too
# generic to be useful and are skipped. Word-level keys get tighter caps because they are looser.
KEY_CAPS = {"n_key": (25, 300), "n_skel": (25, 300), "n_nospace": (25, 300), "n_rare2": (25, 300),
            "n_rare1_num": (25, 300), "a_num_w1": (25, 300), "a_num_w2": (25, 300), "a_w1w2": (25, 300),
            "a_w1w2_n": (25, 300), "n_tok": (6, 150), "n_tok_skel": (6, 150), "a_tok": (6, 150),
            "a_street": (25, 300), "a_street_num": (25, 300)}
KEY_NAMES = list(KEY_CAPS)


def join_key(parts1, partst, cap1, capt):
    def frame(parts, col):
        if not parts:
            return pd.DataFrame({"k": np.array([], np.int64), col: np.array([], np.int32)})
        d = pd.DataFrame({"k": np.concatenate([k for _, k in parts]),
                          col: np.concatenate([r for r, _ in parts]).astype(np.int32)})
        return d[d.k != 0].drop_duplicates()
    a, b = frame(parts1, "i1"), frame(partst, "it")
    ca, cb = a.k.value_counts(), b.k.value_counts()
    good = ca.index[ca <= cap1].intersection(cb.index[cb <= capt])
    a, b = a[a.k.isin(good)], b[b.k.isin(good)]
    m = a.merge(b, on="k")
    return m.i1.to_numpy(np.int32), m.it.to_numpy(np.int32)


def tset(a, b):
    return cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100


def group_context(g, v):
    """rank within group (0 = best), margin over best other member, group size."""
    order = np.lexsort((-v, g))
    gs, vs = g[order], v[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(gs)), 0))
    rank = np.arange(len(gs)) - start
    nxt = np.minimum(start + 1, len(gs) - 1)
    second = np.where((start + 1 < len(gs)) & (gs[nxt] == gs), vs[nxt], np.nan)
    other = np.where(rank == 0, second, vs[start])
    margin = np.where(np.isnan(other), 1.0, vs - other)
    size = np.bincount(gs)[gs] if len(gs) else gs
    r, m, s = np.empty_like(rank), np.empty(len(v), np.float32), np.empty(len(v), np.float32)
    r[order], m[order], s[order] = rank, margin, size
    return r.astype(np.float32), m, s


CLEAN_CHUNK = 500_000


def clean_frame(raw, vocab):
    """Cleaned views of one side, built in chunks and stored compactly."""
    parts = []
    for s in range(0, len(raw), CLEAN_CHUNK):
        r = raw.iloc[s:s + CLEAN_CHUNK]
        f = pd.concat([name_frame(r.business_name, vocab, TRANSLIT), addr_frame(r.business_address)], axis=1)
        f["entity_id"] = r.entity_id.to_numpy(dtype=object)
        f["txt"] = (fold_text(r.business_name)[0] + " | " + fold_text(r.business_address)[0]).str.slice(0, 160).to_numpy(dtype=object)
        parts.append(compact(f))
    return pd.concat(parts, ignore_index=True)


def prepare_country(s1_raw, t_raw, country, keep_raw=False):
    """Clean both sides, generate candidates, quick-score and prune them."""
    vocab_c = Counter()
    for s in range(0, len(s1_raw), CLEAN_CHUNK):
        for t in name_frame(s1_raw.business_name.iloc[s:s + CLEAN_CHUNK]).nm_full.tolist():
            vocab_c.update(t.split())
    tot = sum(vocab_c.values()) or 1
    vocab = {w: math.log(c / tot) for w, c in vocab_c.items() if len(w) > 1}
    A, B = clean_frame(s1_raw, vocab), clean_frame(t_raw, vocab)
    B["is_s3"] = B.entity_id.str.startswith("S3").to_numpy(dtype=bool)
    log(f"  {country}: cleaned {len(A):,} S1 + {len(B):,} S2/S3")

    name_df = df_counts(pd.concat([A.nm_core2, B.nm_core2])).to_dict()
    addr_df = df_counts(pd.concat([A.ad_wset, B.ad_wset])).to_dict()
    s1_vocab = set(df_counts(A.nm_core2).index)
    B["unk_frac"] = [np.mean([w not in s1_vocab for w in t.split()]) if t else 1.0 for t in B.nm_core2.tolist()]
    A["unk_frac"] = 0.0
    KA, KB = build_keys(A, name_df, addr_df, country), build_keys(B, name_df, addr_df, country)
    codes, masks = [], []
    for bit, kn in enumerate(KEY_NAMES):
        i1, it = join_key(KA.get(kn, []), KB.get(kn, []), *KEY_CAPS[kn])
        codes.append(i1.astype(np.int64) * len(B) + it)
        masks.append(np.full(len(i1), 1 << bit, np.int32))
    del KA, KB
    codes, masks = np.concatenate(codes), np.concatenate(masks)
    order = np.argsort(codes, kind="stable")
    codes, masks = codes[order], masks[order]
    first = np.r_[True, codes[1:] != codes[:-1]] if len(codes) else np.array([], bool)
    starts = np.flatnonzero(first)
    keymask = np.bitwise_or.reduceat(masks, starts) if len(starts) else masks
    codes = codes[starts]
    del order, masks
    P = pd.DataFrame({"i1": (codes // len(B)).astype(np.int32), "it": (codes % len(B)).astype(np.int32),
                      "keymask": keymask})
    del codes
    log(f"  {country}: {len(P):,} raw candidate pairs")

    # quick scores: name (best of main name and alias on either side), address, street number
    qn = np.zeros(len(P), np.float32)
    qa = np.zeros(len(P), np.float32)
    num_eq = np.zeros(len(P), bool)
    step = 3_000_000
    for s in range(0, len(P), step):
        i1, it = P.i1.to_numpy()[s:s + step], P.it.to_numpy()[s:s + step]
        ac, bc = sel(A.nm_core, i1), sel(B.nm_core, it)
        q = tset(ac.tolist(), bc.tolist())
        for x, y in ((ac, sel(B.nm_alias, it)), (sel(A.nm_alias, i1), bc)):
            m = (x != "") & (y != "")
            if m.any():
                q[m] = np.maximum(q[m], tset(x[m].tolist(), y[m].tolist()))
        qn[s:s + step] = q
        qa[s:s + step] = tset(sel(A.ad_words, i1).tolist(), sel(B.ad_words, it).tolist())
        a1 = sel(A.ad_num1, i1)
        num_eq[s:s + step] = (a1 == sel(B.ad_num1, it)) & (a1 != "")
    P["qn"], P["qa"] = qn, qa
    P["qs"] = 0.55 * qn + 0.35 * qa + 0.10 * num_eq
    P_raw = P.copy() if keep_raw else None
    # keep, per S2/S3 record: the best by combined score, and also the best by name alone and by
    # address alone (an invented brand name at the exact address must survive)
    it_all = P.it.to_numpy()
    r_q = group_context(it_all, P.qs.to_numpy())[0]
    r_n = group_context(it_all, qn)[0]
    r_a = group_context(it_all, (qa + 0.5 * num_eq).astype(np.float32))[0]
    P = P[(r_q < TOP_PER_T) | (r_n < TOP_BY_ONE) | (r_a < TOP_BY_ONE)].reset_index(drop=True)
    r_1, _, _ = group_context(P.i1.to_numpy(), P.qs.to_numpy())
    P = P[r_1 < TOP_PER_S1].reset_index(drop=True)
    for g, nm in (("i1", "s1"), ("it", "tg")):
        r, m, sz = group_context(P[g].to_numpy(), P.qs.to_numpy())
        P[f"{nm}_qs_rank"], P[f"{nm}_qs_margin"], P[f"{nm}_ncand"] = r, m, sz
        r, m, _ = group_context(P[g].to_numpy(), P.qn.to_numpy())
        P[f"{nm}_qn_rank"], P[f"{nm}_qn_margin"] = r, m
        r, m, _ = group_context(P[g].to_numpy(), P.qa.to_numpy())
        P[f"{nm}_qa_rank"], P[f"{nm}_qa_margin"] = r, m
    log(f"  {country}: {len(P):,} pairs after pruning ({len(P) / max(1, len(A)):.1f} per S1)")
    idf_n = {w: math.log((len(A) + len(B)) / c) for w, c in name_df.items()}
    idf_a = {w: math.log((len(A) + len(B)) / c) for w, c in addr_df.items()}
    if keep_raw:
        return A, B, P, idf_n, idf_a, P_raw
    return A, B, P, idf_n, idf_a

# %% [markdown]
# ## 6. Pair features

# %%
def _set_feats(xs, ys, idf, dflt):
    out = np.zeros((len(xs), 4), np.float32)
    for k, (x, y) in enumerate(zip(xs, ys)):
        if not x or not y:
            out[k] = (-1, -1, 0, 0)
            continue
        X, Y = set(x.split()), set(y.split())
        I = X & Y
        wi = sum(idf.get(t, dflt) for t in I)
        wx = sum(idf.get(t, dflt) for t in X - I)
        wy = sum(idf.get(t, dflt) for t in Y - I)
        out[k] = (len(I) / min(len(X), len(Y)), wi / (wi + wx + wy + 1e-9), wx, wy)
    return out


def _num_feats(xs, ys):
    out = np.zeros((len(xs), 2), np.float32)
    for k, (x, y) in enumerate(zip(xs, ys)):
        if not x or not y:
            out[k] = (-1, 0)
            continue
        X, Y = set(x.split()), set(y.split())
        out[k] = (len(X & Y) / min(len(X), len(Y)), len(X ^ Y))
    return out


def _state3(a, b):
    return np.where((a == "") | (b == ""), 0, np.where(a == b, 1, -1)).astype(np.float32)


def pair_features(P, A, B, idf_n, idf_a):
    i1, it = P.i1.to_numpy(), P.it.to_numpy()
    col = lambda D, c, idx: sel(D[c], idx)
    L = lambda D, c, idx: col(D, c, idx).tolist()
    F = {}
    for b in range(len(KEY_NAMES)):
        F[f"key_{KEY_NAMES[b]}"] = ((P.keymask.to_numpy() >> b) & 1).astype(np.float32)
    for c in ("qs", "qn", "qa", "s1_ncand", "tg_ncand") + tuple(
            f"{g}_{q}_{k}" for g in ("s1", "tg") for q in ("qs", "qn", "qa") for k in ("rank", "margin")):
        F[c] = P[c].to_numpy(np.float32)
    an, bn = L(A, "nm_core", i1), L(B, "nm_core", it)
    F["n_ratio"] = cpdist(an, bn, scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100
    F["n_tset"] = tset(an, bn)
    F["n_tsort"] = cpdist(an, bn, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100
    F["n_partial"] = cpdist(an, bn, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32) / 100
    a2, b2 = L(A, "nm_core2", i1), L(B, "nm_core2", it)
    F["n2_tset"] = tset(a2, b2)
    F["n2_tsort"] = cpdist(a2, b2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100
    F["n_full_ratio"] = cpdist(L(A, "nm_full", i1), L(B, "nm_full", it), scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100
    F["n_jw_nospace"] = cpdist(L(A, "nm_nospace", i1), L(B, "nm_nospace", it),
                               scorer=distance.JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
    F["n_skel_ratio"] = cpdist(L(A, "nm_skel", i1), L(B, "nm_skel", it), scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100
    F["n_skel_tset"] = tset(L(A, "nm_skel", i1), L(B, "nm_skel", it))
    aa, ba = col(A, "nm_alias", i1), col(B, "nm_alias", it)
    alias1 = tset(np.where(aa != "", aa, "\x00").tolist(), bn)
    alias2 = tset(an, np.where(ba != "", ba, "\x00").tolist())
    F["n_alias_best"] = np.maximum(alias1, alias2)
    F["t_has_alias"] = (ba != "").astype(np.float32)
    F["s1_has_alias"] = (aa != "").astype(np.float32)
    F["n_best_tset"] = np.maximum(F["n2_tset"], F["n_alias_best"])
    F["n_key_eq"] = (col(A, "nm_key", i1) == col(B, "nm_key", it)).astype(np.float32)
    F["n_skel_eq"] = (col(A, "nm_skel", i1) == col(B, "nm_skel", it)).astype(np.float32)
    sf = _set_feats(a2, b2, idf_n, 12.0)
    F["n_contain"], F["n_wjac"], F["n_unexpl_s1"], F["n_unexpl_t"] = sf.T
    F["legal_state"] = _state3(col(A, "nm_legal", i1), col(B, "nm_legal", it))
    F["n_ntok_s1"] = col(A, "nm_ntok", i1).astype(np.float32)
    F["n_ntok_t"] = col(B, "nm_ntok", it).astype(np.float32)
    F["t_native"] = col(B, "nm_native", it).astype(np.float32)
    F["t_joined"] = col(B, "nm_joined", it).astype(np.float32)
    F["t_unk_frac"] = col(B, "unk_frac", it).astype(np.float32)
    F["t_is_s3"] = col(B, "is_s3", it).astype(np.float32)

    aw, bw = L(A, "ad_words", i1), L(B, "ad_words", it)
    F["a_tset"] = tset(aw, bw)
    F["a_ratio_sorted"] = cpdist(L(A, "ad_wset", i1), L(B, "ad_wset", it), scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100
    F["a_partial"] = cpdist(aw, bw, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32) / 100
    sa = _set_feats(L(A, "ad_wset", i1), L(B, "ad_wset", it), idf_a, 12.0)
    F["a_contain"], F["a_wjac"], F["a_unexpl_s1"], F["a_unexpl_t"] = sa.T
    nf = _num_feats(L(A, "ad_nums", i1), L(B, "ad_nums", it))
    F["num_contain"], F["num_symdiff"] = nf.T
    F["num1_state"] = _state3(col(A, "ad_num1", i1), col(B, "ad_num1", it))
    F["pc_state"] = _state3(col(A, "ad_pc", i1), col(B, "ad_pc", it))
    F["state_state"] = _state3(col(A, "ad_state", i1), col(B, "ad_state", it))
    F["street_state"] = _state3(col(A, "ad_street", i1), col(B, "ad_street", it))
    F["street_tset"] = tset(L(A, "ad_street", i1), L(B, "ad_street", it))
    F["a_empty_t"] = col(B, "ad_empty", it).astype(np.float32)
    F["a_empty_s1"] = col(A, "ad_empty", i1).astype(np.float32)
    return pd.DataFrame(F)


def features_chunked(P, A, B, idf_n, idf_a, chunk=2_000_000):
    parts = [pair_features(P.iloc[s:s + chunk], A, B, idf_n, idf_a) for s in range(0, len(P), chunk)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

# %% [markdown]
# ## 7. Decision: one-owner rule + expected-F0.5 set per S1 record

# %%
def one_owner(it, p):
    best = pd.Series(p).groupby(it).transform("max").to_numpy()
    return np.where(p >= best - 1e-9, p, 0.0)


def choose_ef(i1, p, K=12, S=256, seed=0):
    """Returns a boolean mask over pairs: selected as matches."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((-p, i1))
    s, pp = i1[order], p[order]
    first = np.r_[True, s[1:] != s[:-1]] if len(s) else np.array([], bool)
    start = np.maximum.accumulate(np.where(first, np.arange(len(s)), 0))
    rank = np.arange(len(s)) - start
    keep = (rank < K) & (pp > 1e-4)
    s, pp, rank, idx = s[keep], pp[keep], rank[keep], order[keep]
    picked = np.zeros(len(p), bool)
    if not len(s):
        return picked
    uniq, inv = np.unique(s, return_inverse=True)
    M = np.zeros((len(uniq), K), np.float32)
    M[inv, rank] = pp
    best_k = np.zeros(len(uniq), np.int64)
    k = np.arange(1, K + 1, dtype=np.float32)
    for c in range(0, len(uniq), 20000):
        Pm = M[c:c + 20000]
        D = rng.random((Pm.shape[0], S, K), dtype=np.float32) < Pm[:, None, :]
        T = D.sum(-1, dtype=np.float32)
        TP = np.cumsum(D, -1, dtype=np.float32)
        f = np.where(TP > 0, 1.25 * TP / np.maximum(1.25 * TP + 0.25 * (T[..., None] - TP) + (k - TP), 1e-9), 0)
        ef = np.concatenate([(T == 0).mean(1, keepdims=True), f.mean(1)], axis=1)
        best_k[c:c + 20000] = ef.argmax(1)
    picked[idx[rank < best_k[inv]]] = True
    return picked


def macro_f05(pred_sets, true_sets, ids):
    tot = 0.0
    for s in ids:
        P, T = pred_sets.get(s, ()), true_sets.get(s, ())
        if not T and not P:
            tot += 1.0
            continue
        tp = len(set(P) & set(T))
        if tp == 0:
            continue
        pr, rc = tp / len(P), tp / len(T)
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / max(1, len(ids))


def to_sets(s1_ids, cand_ids, mask):
    out = defaultdict(list)
    for a, b in zip(s1_ids[mask], cand_ids[mask]):
        out[a].append(b)
    return out

# %% [markdown]
# ## 8. Stage-2 context, support and cross-encoder helpers

# %%
def p_context(s1_codes, t_codes, p):
    """How a pair's stage-1 probability compares with its competitors on both sides."""
    out = {}
    for g, nm in ((s1_codes, "s1"), (t_codes, "tg")):
        r, m, _ = group_context(g, p.astype(np.float32))
        out[f"{nm}_p1_rank"], out[f"{nm}_p1_margin"] = r, m
        out[f"{nm}_p1_sum"] = pd.Series(p).groupby(g).transform("sum").to_numpy(np.float32)
        out[f"{nm}_p1_n50"] = pd.Series(p > 0.5).groupby(g).transform("sum").to_numpy(np.float32)
    return out


def support(s1_codes, t_core, t_addr, t_src, p, top=6):
    """Another strong candidate of the same S1 record that looks like this record supports it
    (S2 <-> S3 agreement); computed on each S1 record's top candidates only."""
    df = pd.DataFrame({"g": s1_codes, "p": p, "row": np.arange(len(p))})
    sub = df.sort_values(["g", "p"], ascending=[True, False]).groupby("g").head(top)
    m = sub.merge(sub, on="g", suffixes=("", "_o"))
    m = m[m.row != m.row_o]
    same = np.zeros(len(p), np.float32)
    cross = np.zeros(len(p), np.float32)
    n_strong = np.zeros(len(p), np.float32)
    if len(m):
        r, ro = m.row.to_numpy(), m.row_o.to_numpy()
        sim = 0.6 * tset(t_core[r].tolist(), t_core[ro].tolist()) + 0.4 * tset(t_addr[r].tolist(), t_addr[ro].tolist())
        sup = np.minimum(sim, m.p_o.to_numpy())
        is_cross = t_src[r] != t_src[ro]
        d = pd.DataFrame({"row": r, "sup": sup, "cross": is_cross, "strong": (sim > 0.8) & (m.p_o.to_numpy() > 0.5)})
        g = d.groupby(["row", "cross"]).sup.max().unstack(fill_value=0.0)
        if False in g.columns:
            same[g.index.to_numpy()] = g[False].to_numpy()
        if True in g.columns:
            cross[g.index.to_numpy()] = g[True].to_numpy()
        s = d.groupby("row").strong.sum()
        n_strong[s.index.to_numpy()] = s.to_numpy()
    return {"sup_same_src": same, "sup_cross_src": cross, "sup_n_strong": n_strong}


def stage2_extra(s1_codes, t_codes, t_core, t_addr, t_src, p1, ce):
    """Stage-2 columns on top of the stage-1 features (computed over all pairs of a country)."""
    E = {"p1": p1.astype(np.float32)}
    E.update(p_context(s1_codes, t_codes, p1))
    E.update(support(s1_codes, t_core, t_addr, t_src, p1))
    E["ce"] = ce.astype(np.float32)          # -1 = not scored (outside the uncertain band)
    return E


def stage2_matrix(X, *args):
    """Adds the stage-2 columns to X in place (no copy of the feature table)."""
    for k, v in stage2_extra(*args).items():
        X[k] = v
    return X


import torch
USE_CE = os.environ.get("ER_CE", "1" if torch.cuda.is_available() else "0") == "1"
CE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # Apache-2.0, 118M params
CE_MAXLEN, CE_BAND = 64, (0.003, 0.997)
CE_TRAIN_MAX = int(os.environ.get("ER_CE_TRAIN_MAX", 600_000))
CE_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ce_train(a, b, y, epochs=1, bs=256, lr=4e-5):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL, num_labels=1).to(CE_DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(y) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps), pct_start=0.06)
    lossf = torch.nn.BCEWithLogitsLoss()
    use_amp = CE_DEV == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    model.train()
    step = 0
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(y))
        for s in range(0, len(y), bs):
            idx = perm[s:s + bs]
            enc = tok([a[i] for i in idx], [b[i] for i in idx], truncation=True, max_length=CE_MAXLEN,
                      padding=True, return_tensors="pt").to(CE_DEV)
            with torch.autocast(device_type=CE_DEV, dtype=torch.float16, enabled=use_amp):
                logit = model(**enc).logits.squeeze(-1)
                loss = lossf(logit.float(), torch.tensor(y[idx], dtype=torch.float32, device=CE_DEV))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                log(f"    cross-encoder step {step}/{steps} loss {loss.item():.4f}")
    model.eval()
    return tok, model


@torch.no_grad()
def ce_predict(tok, model, a, b, bs=1024):
    out = np.zeros(len(a), np.float32)
    use_amp = CE_DEV == "cuda"
    for s in range(0, len(a), bs):
        enc = tok(a[s:s + bs], b[s:s + bs], truncation=True, max_length=CE_MAXLEN, padding=True,
                  return_tensors="pt").to(CE_DEV)
        with torch.autocast(device_type=CE_DEV, dtype=torch.float16, enabled=use_amp):
            out[s:s + bs] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


def ce_scores(p1, a_txt, b_txt):
    ce = np.full(len(p1), -1.0, np.float32)
    band = (p1 > CE_BAND[0]) & (p1 < CE_BAND[1])
    if USE_CE and band.any():
        idx = np.flatnonzero(band)
        cap = int(os.environ.get("ER_CE_INFER_MAX", 0))   # dev only: limit CPU test runs
        if cap:
            idx = idx[:cap]
        ce[idx] = ce_predict(CE_TOK, CE_NET, [str(a_txt[i]) for i in idx], [str(b_txt[i]) for i in idx])
    return ce

# %% [markdown]
# ## 9. Training split: candidates for every country; features for the sampled S1 records
# Sampled S1 records are split into **half A** (stage 1 + cross-encoder), **half B** (stage 2) and
# **validation**. Every competing claim on their S2/S3 records is included too ("closure"), so the
# one-owner rule and the competition features see the full picture.

# %%
def load_split(split):
    d = {s: compact(read_tsv(f"{DATA_DIR}/{split}/{split}_source{s}.tsv")) for s in (1, 2, 3)}
    t = pd.concat([d[2], d[3]], ignore_index=True)
    del d[2], d[3]
    gc.collect()
    return d[1], t


gt = read_tsv(f"{DATA_DIR}/train/train_ground_truth.tsv")
TRUE = {r.source1_entity_id: [x for x in r.matched_entity_ids.split(",") if x] for r in gt.itertuples()}
OWNER = {x: s for s, lst in TRUE.items() for x in lst}
del gt
rng = np.random.default_rng(SEED)
all_s1 = np.array(sorted(TRUE))
perm = rng.permutation(len(all_s1))
nA = nB = TRAIN_S1_SAMPLE // 2
GROUP = {}
for i in perm[:nA]:
    GROUP[all_s1[i]] = 1
for i in perm[nA:nA + nB]:
    GROUP[all_s1[i]] = 2
for i in perm[nA + nB:nA + nB + VAL_S1_SAMPLE]:
    GROUP[all_s1[i]] = 3
log(f"half A {nA:,}, half B {nB:,}, validation {VAL_S1_SAMPLE:,} S1 records")

s1_all, t_all = load_split("train")
feat_parts, meta_parts, recall = [], [], Counter()
for country in s1_all.country.unique():
    s1c = s1_all[s1_all.country == country].reset_index(drop=True)
    tc = t_all[t_all.country == country].reset_index(drop=True)
    if len(tc) == 0:
        continue
    A, B, P, idf_n, idf_a = prepare_country(s1c, tc, country)
    s1_id, t_id = A.entity_id.to_numpy(dtype=object), B.entity_id.to_numpy(dtype=object)
    owner = np.array([OWNER.get(x, "") for x in t_id], dtype=object)
    i1a, ita = P.i1.to_numpy(), P.it.to_numpy()
    y = (owner[ita] == s1_id[i1a]).astype(np.int8)
    grp = pd.Series(s1_id).map(GROUP).fillna(0).astype(np.int8).to_numpy()[i1a]
    closure = np.isin(ita, np.unique(ita[grp > 0]))
    take = (grp > 0) | closure
    va_s1 = [s for s in s1_id if GROUP.get(s) == 3]
    recall["true_links"] += sum(len(TRUE[s]) for s in va_s1)
    recall["found_links"] += int(y[grp == 3].sum())
    Psub = P[take].reset_index(drop=True)
    feat_parts.append(features_chunked(Psub, A, B, idf_n, idf_a))
    ii, tt = Psub.i1.to_numpy(), Psub.it.to_numpy()
    S = lambda arr: pd.array(arr, dtype="string[pyarrow]")     # compact text storage
    meta_parts.append(pd.DataFrame({
        "s1": S(s1_id[ii]), "t": S(t_id[tt]), "y": y[take], "grp": grp[take], "country": country,
        "a_txt": S(sel(A.txt, ii)), "b_txt": S(sel(B.txt, tt)),
        "t_core": S(sel(B.nm_core, tt)), "t_addr": S(sel(B.ad_words, tt)),
        "t_src": B.is_s3.to_numpy()[tt]}))
    log(f"  {country}: {take.sum():,} feature rows; validation candidate recall so far "
        f"{recall['found_links'] / max(1, recall['true_links']):.4f}")
    del A, B, P, Psub, s1c, tc
    gc.collect()
del s1_all, t_all
gc.collect()
X = pd.concat(feat_parts, ignore_index=True)
M = pd.concat(meta_parts, ignore_index=True)
del feat_parts, meta_parts
gc.collect()
FEATURE_COLS = list(X.columns)
CAND_RECALL = recall["found_links"] / max(1, recall["true_links"])
log(f"feature matrix {X.shape}; candidate recall on validation {CAND_RECALL:.4f}")

# %% [markdown]
# ## 10. Stage 1 (half A) → cross-encoder (half A, GPU) → stage 2 (half B) → calibration and rule (validation)

# %%
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=os.cpu_count(), verbose=-1, seed=SEED)


def fit_lgb(Xm, ym, groups, rounds=3000):
    """Train with early stopping on 10% of the S1 records of this half."""
    hold = pd.util.hash_array(groups.astype(object)) % 10 == 0
    dtr = lgb.Dataset(Xm[~hold], ym[~hold])
    dva = lgb.Dataset(Xm[hold], ym[hold], reference=dtr)
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])


grp = M.grp.to_numpy()
yv = M.y.to_numpy()
mA = grp == 1
m1 = fit_lgb(X[mA], yv[mA], M.s1.to_numpy()[mA])
p1 = m1.predict(X, num_iteration=m1.best_iteration)
log(f"stage 1 trained ({m1.best_iteration} trees)")

if USE_CE:
    band = mA & (p1 > 0.01) & (p1 < 0.99)
    easy = mA & ~band
    idx = np.flatnonzero(band)
    idx = np.concatenate([idx, rng.choice(np.flatnonzero(easy), min(easy.sum(), len(idx) // 3), replace=False)])
    if len(idx) > CE_TRAIN_MAX:
        idx = rng.choice(idx, CE_TRAIN_MAX, replace=False)
    log(f"cross-encoder: training on {len(idx):,} half-A pairs ({band.sum():,} uncertain)")
    CE_TOK, CE_NET = ce_train(M.a_txt.to_numpy()[idx].tolist(), M.b_txt.to_numpy()[idx].tolist(), yv[idx].astype(np.float32))
ce = np.full(len(M), -1.0, np.float32)
rest = ~mA
if USE_CE:
    ce[rest] = ce_scores(p1[rest], M.a_txt.to_numpy()[rest], M.b_txt.to_numpy()[rest])
    log(f"cross-encoder scored {(ce >= 0).sum():,} uncertain pairs")

s1_codes = pd.factorize(M.s1)[0]
t_codes = pd.factorize(M.t)[0]
G = stage2_matrix(X, s1_codes, t_codes, M.t_core.to_numpy(), M.t_addr.to_numpy(), M.t_src.to_numpy(), p1, ce)
STAGE2_COLS = list(G.columns)
mB = grp == 2
m2 = fit_lgb(G[mB], yv[mB], M.s1.to_numpy()[mB])
raw2 = m2.predict(G, num_iteration=m2.best_iteration)
log(f"stage 2 trained ({m2.best_iteration} trees)")
imp = pd.Series(m2.feature_importance("gain"), index=STAGE2_COLS).sort_values(ascending=False)
print((imp / imp.sum() * 100).round(2).head(25))

va_ids = np.array(sorted(s for s, g in GROUP.items() if g == 3))
half_a = set(va_ids[: len(va_ids) // 2])
va_rows = M.s1.isin(set(va_ids)).to_numpy()
cal_rows = M.s1.isin(half_a).to_numpy()
cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw2[cal_rows], yv[cal_rows])
M["p"] = cal.predict(raw2)

# one-owner over every claim we scored on validation targets
V = M[va_rows | M.t.isin(M.t[va_rows])][["s1", "t", "y", "country", "p"]].copy()
V["p_own"] = one_owner(V.t.to_numpy(), V.p.to_numpy())
V = V[V.s1.isin(set(va_ids))].reset_index(drop=True)
RULES = ["ef_owner", "ef_no_owner"] + [f"thr{t}_{o}" for o in ("owner", "no_owner")
                                       for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]


def apply_rule(sub, rule):
    pcol = sub.p_own.to_numpy() if rule.endswith("_owner") and not rule.endswith("no_owner") else sub.p.to_numpy()
    if rule.startswith("ef"):
        return choose_ef(pd.factorize(sub.s1)[0], pcol)
    return pcol >= float(rule[3:].split("_")[0])


def evaluate(sub, ids, rules=RULES):
    return {r: macro_f05(to_sets(sub.s1.to_numpy(), sub.t.to_numpy(), apply_rule(sub, r)), TRUE, ids) for r in rules}

ids_a, ids_b = sorted(half_a), sorted(set(va_ids) - half_a)
Va, Vb = V[V.s1.isin(half_a)], V[~V.s1.isin(half_a)]
res_a = evaluate(Va, ids_a)
BEST_RULE = max(res_a, key=res_a.get)
res_b = evaluate(Vb, ids_b)
oracle = macro_f05(to_sets(Vb.s1.to_numpy(), Vb.t.to_numpy(), Vb.y.to_numpy() == 1), TRUE, ids_b)
print("tuning half:", {k: round(v, 5) for k, v in sorted(res_a.items(), key=lambda kv: -kv[1])[:6]})
print(f"chosen rule: {BEST_RULE}")
print(f"VALIDATION macro F0.5 (untouched half): {res_b[BEST_RULE]:.5f}   "
      f"| ceiling with perfect decisions on our candidates: {oracle:.5f} | candidate recall {CAND_RECALL:.4f}")
country_of = dict(zip(M.s1, M.country))
for c in sorted(set(country_of[s] for s in ids_b if s in country_of)):
    ids_c = [s for s in ids_b if country_of.get(s) == c]
    print(f"  {c}: {evaluate(Vb[Vb.country == c], ids_c, [BEST_RULE])[BEST_RULE]:.5f} on {len(ids_c):,} S1")
json.dump({"tuning": res_a, "validation": res_b, "rule": BEST_RULE, "oracle": oracle,
           "candidate_recall": CAND_RECALL}, open(f"{OUT_DIR}/validation.json", "w"), indent=2)
m1.save_model(f"{OUT_DIR}/stage1.txt")
m2.save_model(f"{OUT_DIR}/stage2.txt")

# %% [markdown]
# ## 11. Test split: candidates → stage 1 → cross-encoder → stage 2 → decision → submission files

# %%
def decide(s1_codes, t_codes, p, rule):
    pcol = one_owner(t_codes, p) if rule.endswith("_owner") and not rule.endswith("no_owner") else p
    if rule.startswith("ef"):
        return choose_ef(s1_codes, pcol)
    return pcol >= float(rule[3:].split("_")[0])


if RUN_TEST:
    del X, G, M, V
    gc.collect()
    s1_all, t_all = load_split("test")
    cand_path, match_path = f"{OUT_DIR}/candidate_pairs.tsv", f"{OUT_DIR}/matching_results.tsv"
    fc = open(cand_path, "w", encoding="utf-8", newline="\n")
    fm = open(match_path, "w", encoding="utf-8", newline="\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    n_rows = n_match = 0
    for country in s1_all.country.unique():
        s1c = s1_all[s1_all.country == country].reset_index(drop=True)
        tc = t_all[t_all.country == country].reset_index(drop=True)
        s1_id = s1c.entity_id.to_numpy()
        if len(tc) == 0:
            for s in s1_id:
                fc.write(f"{s}\t\n")
                fm.write(f"{s}\t\n")
            n_rows += len(s1_id)
            continue
        A, B, P, idf_n, idf_a = prepare_country(s1c, tc, country)
        t_id = B.entity_id.to_numpy(dtype=object)
        # two passes over 2M-pair chunks so a country's full feature table never sits in memory
        ii, tt = P.i1.to_numpy(), P.it.to_numpy()
        CH = 2_000_000
        q1 = np.concatenate([m1.predict(pair_features(P.iloc[s:s + CH], A, B, idf_n, idf_a)[FEATURE_COLS],
                                        num_iteration=m1.best_iteration) for s in range(0, len(P), CH)])
        cec = ce_scores(q1, sel(A.txt, ii), sel(B.txt, tt))
        E = stage2_extra(ii, tt, sel(B.nm_core, tt), sel(B.ad_words, tt), B.is_s3.to_numpy()[tt], q1, cec)
        p = np.empty(len(P))
        for s in range(0, len(P), CH):
            Fx = pair_features(P.iloc[s:s + CH], A, B, idf_n, idf_a)[FEATURE_COLS]
            for k, v in E.items():
                Fx[k] = v[s:s + CH]
            p[s:s + CH] = cal.predict(m2.predict(Fx[STAGE2_COLS], num_iteration=m2.best_iteration))
            del Fx
        del E, q1, cec
        chosen = decide(ii, tt, p, BEST_RULE)
        cands, matches = defaultdict(list), defaultdict(list)
        for a, b, m in zip(ii, tt, chosen):
            cands[a].append(t_id[b])
            if m:
                matches[a].append(t_id[b])
        for i, s in enumerate(s1_id):
            fc.write(f"{s}\t{','.join(cands.get(i, []))}\n")
            fm.write(f"{s}\t{','.join(matches.get(i, []))}\n")
        n_rows += len(s1_id)
        n_match += int(chosen.sum())
        log(f"  {country}: {len(s1_id):,} S1 rows, {int(chosen.sum()) / len(s1_id):.2f} matches per S1, "
            f"{sum(1 for i in range(len(s1_id)) if i not in matches) / len(s1_id):.3f} empty")
        del A, B, P, p, chosen, cands, matches
        gc.collect()
    fc.close()
    fm.close()
    log(f"wrote {n_rows:,} rows, {n_match:,} matches -> {match_path}")

# %% [markdown]
# ## 12. Format check (same rules as `utils/validate_submission.py`)

# %%
def check_submission(match_path, cand_path, test_dir):
    ids = lambda f: set(read_tsv(f).entity_id)
    s1 = ids(f"{test_dir}/test_source1.tsv")
    tg = ids(f"{test_dir}/test_source2.tsv") | ids(f"{test_dir}/test_source3.tsv")
    problems = []
    parsed = {}
    for path, col in ((match_path, "matched_entity_ids"), (cand_path, "candidate_entity_ids")):
        d = read_tsv(path)
        if list(d.columns) != ["source1_entity_id", col]:
            problems.append(f"{path}: header {list(d.columns)}")
        if d.source1_entity_id.duplicated().any():
            problems.append(f"{path}: duplicate S1 rows")
        if set(d.source1_entity_id) != s1:
            problems.append(f"{path}: S1 ids differ from test_source1")
        lists = d[col].str.split(",")
        for s, lst in zip(d.source1_entity_id, lists):
            lst = [x for x in lst if x]
            if len(lst) != len(set(lst)) or any(x not in tg for x in lst):
                problems.append(f"{path}: bad list for {s}")
                break
        parsed[col] = dict(zip(d.source1_entity_id, lists))
    for s, lst in parsed["matched_entity_ids"].items():
        cset = set(parsed["candidate_entity_ids"].get(s, []))
        if any(x and x not in cset for x in lst):
            problems.append(f"match not in candidates for {s}")
            break
    return problems or ["PASS"]

if RUN_TEST:
    print(check_submission(f"{OUT_DIR}/matching_results.tsv", f"{OUT_DIR}/candidate_pairs.tsv", f"{DATA_DIR}/test"))

# %% [markdown]
# ## 13. Save outputs to Google Drive

# %%
if IN_COLAB and os.environ.get("ER_OUT_DIR") is None:
    import shutil
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        dst = "/content/drive/MyDrive/amazon_er_output"
        shutil.copytree(OUT_DIR, dst, dirs_exist_ok=True)
        print("copied outputs to", dst)
    except Exception as e:
        print("Drive copy skipped:", e, "- download from the Files panel instead")
