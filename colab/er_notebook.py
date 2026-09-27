# %% [markdown]
# # Business Entity Resolution: Amazon ML Challenge 2026
#
# End to end on the full dataset:
# 1. **Clean** every name and address: transliterate the 9 Indian scripts, strip accents, junk prefixes, tags like `(ID: 123)` / `#123` / `| www…`, undo digit-for-letter swaps (`Materia1s`), split website names back into words (`pclmedicalcentre.com`), separate legal forms, honorifics and filler words, normalise street words, state names and codes (US, India, France).
# 2. **Candidates** from exact lookup keys, per country: sorted core name, sound-alike name, rare-word pairs, space-free name, house number + rare street word.
# 3. **Score** each candidate pair with LightGBM (~55 features).
# 4. **Decide**: one-owner rule (each S2/S3 record goes to at most one S1 record; exact in the training labels), then each S1 record's match set is chosen to maximise expected F0.5.
# 5. **Validate** on held-out S1 records, then write `matching_results.tsv` + `candidate_pairs.tsv` for the test set.
#
# **Runtime:** High-RAM runtime (Runtime → Change runtime type). No GPU needed. Then Runtime → Run all.
# **Data:** copied automatically from the team's shared Drive folder (cell 2). Outputs go to `/content/output`
# and are copied to `MyDrive/amazon_er_output` at the end.

# %%
#!pip -q install rapidfuzz lightgbm psutil gdown

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
TRAIN_S1_SAMPLE = int(os.environ.get("ER_TRAIN_S1", 300_000))   # S1 records whose pairs train the model
VAL_S1_SAMPLE = int(os.environ.get("ER_VAL_S1", 100_000))       # held-out S1 records for scoring
RUN_TEST = os.environ.get("ER_RUN_TEST", "1") == "1"
SEED = 42

# candidate generation
CAP_S1 = 25        # a key shared by more S1 records than this is too generic -> skipped
CAP_T = 300        # same for S2/S3 records
TOP_PER_T = 8      # keep the best N S1 candidates per S2/S3 record (by quick score)
TOP_PER_S1 = 40    # and at most N candidates per S1 record
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
             "la", "le", "les", "bis", "null", "none", "na", "c", "o"}

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
        text = RE_ORD_ADDR.sub(r"\1", " ".join(rest)).replace("/", " ").replace("-", " ").replace("'", " ")
        text = RE_NONWORD.sub(" ", text)
        nums, words = [], []
        for x in RE_ALNUM.findall(text):
            if x.isdigit():
                nums.append(x.lstrip("0") or "0")
            else:
                w = ADDR_CANON.get(x, x)
                if len(w) > 1 and w not in ADDR_DROP and x not in ADDR_DROP:
                    words.append(w)
        num1 = nums[0] if nums else ""
        pc = nums[-1] if (len(nums) > 1 and len(nums[-1]) in (5, 6)) else ""
        rows.append((" ".join(words), " ".join(sorted(set(words))), " ".join(dict.fromkeys(nums)),
                     num1, pc, state, not words and not nums))
    df = pd.DataFrame(rows, columns=["ad_words", "ad_wset", "ad_nums", "ad_num1", "ad_pc", "ad_state", "ad_empty"])
    return df

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
    c = country + "|"
    core2 = F.nm_core2.tolist()
    nkey, nsk, nns = F.nm_key.tolist(), F.nm_skel.tolist(), F.nm_nospace.tolist()
    rn = rare_tokens(core2, name_df)
    ra = rare_tokens(F.ad_wset.tolist(), addr_df)
    num1 = F.ad_num1.tolist()
    K = {}
    K["n_key"] = hkey([c + "k" + x for x in nkey], [bool(x) for x in nkey])
    K["n_skel"] = hkey([c + "s" + x for x in nsk], [len(x) >= 3 for x in nsk])
    K["n_nospace"] = hkey([c + "j" + x for x in nns], [len(x) >= 6 for x in nns])
    K["n_rare2"] = hkey([c + "r" + " ".join(sorted(r)) for r in rn], [len(r) == 2 for r in rn])
    K["n_rare1_num"] = hkey([c + "q" + (r[0] if r else "") + "|" + n for r, n in zip(rn, num1)],
                            [bool(r) and bool(n) for r, n in zip(rn, num1)])
    K["a_num_w1"] = hkey([c + "a" + n + "|" + (r[0] if r else "") for r, n in zip(ra, num1)],
                         [bool(r) and bool(n) for r, n in zip(ra, num1)])
    K["a_num_w2"] = hkey([c + "b" + n + "|" + (r[1] if len(r) > 1 else "") for r, n in zip(ra, num1)],
                         [len(r) > 1 and bool(n) for r, n in zip(ra, num1)])
    K["a_w1w2"] = hkey([c + "v" + " ".join(sorted(r)) for r in ra], [len(r) > 1 for r in ra])
    K["a_w1w2_n"] = hkey([c + "w" + " ".join(sorted(r)) + "|" + (rn[i][0] if rn[i] else "") for i, r in enumerate(ra)],
                         [len(r) > 1 and bool(rn[i]) for i, r in enumerate(ra)])
    return K

KEY_NAMES = ["n_key", "n_skel", "n_nospace", "n_rare2", "n_rare1_num", "a_num_w1", "a_num_w2", "a_w1w2", "a_w1w2_n"]


def join_key(k1, kt):
    a = pd.DataFrame({"k": k1, "i1": np.arange(len(k1), dtype=np.int32)})
    b = pd.DataFrame({"k": kt, "it": np.arange(len(kt), dtype=np.int32)})
    a, b = a[a.k != 0], b[b.k != 0]
    ca, cb = a.k.value_counts(), b.k.value_counts()
    good = ca.index[ca <= CAP_S1].intersection(cb.index[cb <= CAP_T])
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


def prepare_country(s1_raw, t_raw, country, keep_raw=False):
    """Clean both sides, generate candidates, quick-score and prune them."""
    s1n = name_frame(s1_raw.business_name)
    vocab_c = Counter(w for t in s1n.nm_full.tolist() for w in t.split())
    tot = sum(vocab_c.values()) or 1
    vocab = {w: math.log(c / tot) for w, c in vocab_c.items() if len(w) > 1}
    s1n = name_frame(s1_raw.business_name, vocab, TRANSLIT)
    tn = name_frame(t_raw.business_name, vocab, TRANSLIT)
    A = pd.concat([s1n, addr_frame(s1_raw.business_address)], axis=1)
    B = pd.concat([tn, addr_frame(t_raw.business_address)], axis=1)
    A["entity_id"] = s1_raw.entity_id.to_numpy()
    B["entity_id"] = t_raw.entity_id.to_numpy()
    B["is_s3"] = B.entity_id.str.startswith("S3").to_numpy()
    log(f"  {country}: cleaned {len(A):,} S1 + {len(B):,} S2/S3")

    name_df = df_counts(pd.concat([A.nm_core2, B.nm_core2])).to_dict()
    addr_df = df_counts(pd.concat([A.ad_wset, B.ad_wset])).to_dict()
    s1_vocab = set(df_counts(A.nm_core2).index)
    B["unk_frac"] = [np.mean([w not in s1_vocab for w in t.split()]) if t else 1.0 for t in B.nm_core2.tolist()]
    A["unk_frac"] = 0.0
    KA, KB = build_keys(A, name_df, addr_df, country), build_keys(B, name_df, addr_df, country)
    codes, masks = [], []
    for bit, kn in enumerate(KEY_NAMES):
        i1, it = join_key(KA[kn], KB[kn])
        codes.append(i1.astype(np.int64) * len(B) + it)
        masks.append(np.full(len(i1), 1 << bit, np.int16))
    del KA, KB
    codes, masks = np.concatenate(codes), np.concatenate(masks)
    order = np.argsort(codes, kind="stable")
    codes, masks = codes[order], masks[order]
    first = np.r_[True, codes[1:] != codes[:-1]] if len(codes) else np.array([], bool)
    starts = np.flatnonzero(first)
    keymask = np.bitwise_or.reduceat(masks, starts) if len(starts) else masks
    codes = codes[starts]
    P = pd.DataFrame({"i1": (codes // len(B)).astype(np.int32), "it": (codes % len(B)).astype(np.int32),
                      "keymask": keymask})
    log(f"  {country}: {len(P):,} raw candidate pairs")

    # quick score and pruning
    qs = np.zeros(len(P), np.float32)
    for s in range(0, len(P), 5_000_000):
        i1, it = P.i1.to_numpy()[s:s + 5_000_000], P.it.to_numpy()[s:s + 5_000_000]
        qn = tset(A.nm_core.to_numpy()[i1].tolist(), B.nm_core.to_numpy()[it].tolist())
        qa = tset(A.ad_words.to_numpy()[i1].tolist(), B.ad_words.to_numpy()[it].tolist())
        num_eq = (A.ad_num1.to_numpy()[i1] == B.ad_num1.to_numpy()[it]) & (A.ad_num1.to_numpy()[i1] != "")
        qs[s:s + 5_000_000] = 0.55 * qn + 0.35 * qa + 0.10 * num_eq
    P["qs"] = qs
    P_raw = P.copy() if keep_raw else None
    r_t, _, _ = group_context(P.it.to_numpy(), qs)
    P = P[r_t < TOP_PER_T].reset_index(drop=True)
    r_1, _, _ = group_context(P.i1.to_numpy(), P.qs.to_numpy())
    P = P[r_1 < TOP_PER_S1].reset_index(drop=True)
    for g, nm in (("i1", "s1"), ("it", "tg")):
        r, m, sz = group_context(P[g].to_numpy(), P.qs.to_numpy())
        P[f"{nm}_qs_rank"], P[f"{nm}_qs_margin"], P[f"{nm}_ncand"] = r, m, sz
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
    col = lambda D, c, idx: D[c].to_numpy()[idx]
    L = lambda D, c, idx: col(D, c, idx).tolist()
    F = {}
    for b in range(len(KEY_NAMES)):
        F[f"key_{KEY_NAMES[b]}"] = ((P.keymask.to_numpy() >> b) & 1).astype(np.float32)
    for c in ("qs", "s1_qs_rank", "s1_qs_margin", "s1_ncand", "tg_qs_rank", "tg_qs_margin", "tg_ncand"):
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
    sel = np.zeros(len(p), bool)
    if not len(s):
        return sel
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
    sel[idx[rank < best_k[inv]]] = True
    return sel


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
# ## 8. Training split: build candidates for every country, features for the sampled S1 records

# %%
def load_split(split):
    d = {s: read_tsv(f"{DATA_DIR}/{split}/{split}_source{s}.tsv") for s in (1, 2, 3)}
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
TRAIN_IDS = set(all_s1[perm[:TRAIN_S1_SAMPLE]])
VAL_IDS = set(all_s1[perm[TRAIN_S1_SAMPLE:TRAIN_S1_SAMPLE + VAL_S1_SAMPLE]])
log(f"train S1 sample {len(TRAIN_IDS):,}, validation S1 sample {len(VAL_IDS):,}")

s1_all, t_all = load_split("train")
feat_parts, meta_parts, recall = [], [], Counter()
for country in s1_all.country.unique():
    s1c = s1_all[s1_all.country == country].reset_index(drop=True)
    tc = t_all[t_all.country == country].reset_index(drop=True)
    A, B, P, idf_n, idf_a = prepare_country(s1c, tc, country)
    s1_id, t_id = A.entity_id.to_numpy(), B.entity_id.to_numpy()
    owner = np.array([OWNER.get(x, "") for x in t_id], dtype=object)
    y = (owner[P.it.to_numpy()] == s1_id[P.i1.to_numpy()]).astype(np.int8)
    in_tr = pd.Series(s1_id).isin(TRAIN_IDS).to_numpy()[P.i1.to_numpy()]
    in_va = pd.Series(s1_id).isin(VAL_IDS).to_numpy()[P.i1.to_numpy()]
    va_targets = np.unique(P.it.to_numpy()[in_va])
    closure = np.isin(P.it.to_numpy(), va_targets)          # every competing claim on validation targets
    take = in_tr | closure
    # blocking recall on the validation sample
    va_s1 = [s for s in s1_id if s in VAL_IDS]
    recall["true_links"] += sum(len(TRUE[s]) for s in va_s1)
    recall["found_links"] += int(y[in_va].sum())
    Psub = P[take].reset_index(drop=True)
    Fx = features_chunked(Psub, A, B, idf_n, idf_a)
    meta = pd.DataFrame({"s1": s1_id[Psub.i1.to_numpy()], "t": t_id[Psub.it.to_numpy()], "y": y[take],
                         "in_tr": in_tr[take], "in_va": in_va[take], "country": country})
    feat_parts.append(Fx)
    meta_parts.append(meta)
    log(f"  {country}: {len(Fx):,} feature rows; val blocking recall so far "
        f"{recall['found_links'] / max(1, recall['true_links']):.4f}")
    del A, B, P, Psub, s1c, tc
    gc.collect()
del s1_all, t_all
gc.collect()
X = pd.concat(feat_parts, ignore_index=True)
M = pd.concat(meta_parts, ignore_index=True)
del feat_parts, meta_parts
log(f"feature matrix {X.shape}; positives {M.y.mean():.3f}; blocking recall on val "
    f"{recall['found_links'] / max(1, recall['true_links']):.4f}")

# %% [markdown]
# ## 9. Train LightGBM, calibrate, choose the decision rule on validation half A, report on half B

# %%
va_ids = np.array(sorted(VAL_IDS))
half_a = set(va_ids[: len(va_ids) // 2])
M["va_a"] = M.s1.isin(half_a).to_numpy() & M.in_va.to_numpy()
M["va_b"] = M.in_va.to_numpy() & ~M.va_a.to_numpy()
tr_m, es_m = M.in_tr.to_numpy() & ~M.in_va.to_numpy(), M.va_a.to_numpy()
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=os.cpu_count(), verbose=-1, seed=SEED)
dtr = lgb.Dataset(X[tr_m], M.y[tr_m])
dva = lgb.Dataset(X[es_m], M.y[es_m], reference=dtr)
model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
raw = model.predict(X, num_iteration=model.best_iteration)
cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw[es_m], M.y[es_m])
M["p"] = cal.predict(raw)
imp = pd.Series(model.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print((imp / imp.sum() * 100).round(2).head(25))

# one-owner over all claims we scored for validation targets
val_rows = M[M.in_va | M.t.isin(M.t[M.in_va])].copy()
val_rows["p_own"] = one_owner(val_rows.t.to_numpy(), val_rows.p.to_numpy())
val_rows = val_rows[val_rows.in_va].reset_index(drop=True)


def evaluate(mask_rows, ids):
    sub = val_rows[mask_rows].reset_index(drop=True)
    codes = pd.factorize(sub.s1)[0]
    res = {}
    for name, pcol in (("no_owner", "p"), ("owner", "p_own")):
        sel = choose_ef(codes, sub[pcol].to_numpy())
        res[f"ef_{name}"] = macro_f05(to_sets(sub.s1.to_numpy(), sub.t.to_numpy(), sel), TRUE, ids)
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            sel = sub[pcol].to_numpy() >= thr
            res[f"thr{thr}_{name}"] = macro_f05(to_sets(sub.s1.to_numpy(), sub.t.to_numpy(), sel), TRUE, ids)
    return res

ids_a = sorted(half_a)
ids_b = sorted(set(va_ids) - half_a)
res_a = evaluate(val_rows.s1.isin(half_a).to_numpy(), ids_a)
BEST_RULE = max(res_a, key=res_a.get)
res_b = evaluate(~val_rows.s1.isin(half_a).to_numpy(), ids_b)
print("half A (tuning):", {k: round(v, 4) for k, v in res_a.items()})
print(f"chosen rule: {BEST_RULE}")
print(f"VALIDATION macro F0.5 on half B (untouched): {res_b[BEST_RULE]:.4f}")
country_of = dict(zip(M.s1, M.country))
for c in M.country.unique():
    ids_c = [s for s in ids_b if country_of.get(s) == c]
    m = (~val_rows.s1.isin(half_a).to_numpy()) & (val_rows.country.to_numpy() == c)
    print(f"  {c}: {evaluate(m, ids_c)[BEST_RULE]:.4f} on {len(ids_c):,} S1")
json.dump({"half_a": res_a, "half_b": res_b, "rule": BEST_RULE,
           "blocking_recall_val": recall["found_links"] / max(1, recall["true_links"])},
          open(f"{OUT_DIR}/validation.json", "w"), indent=2)
model.save_model(f"{OUT_DIR}/lgb_model.txt")
FEATURE_COLS = list(X.columns)

# %% [markdown]
# ## 10. Test split: candidates → features → probabilities → decision → submission files

# %%
def decide(P_s1, P_t, p, rule):
    pcol = one_owner(P_t, p) if rule.endswith("_owner") and not rule.endswith("no_owner") else p
    if rule.startswith("ef"):
        return choose_ef(P_s1, pcol)
    return pcol >= float(rule[3:].split("_")[0])


if RUN_TEST:
    del X, M, val_rows
    gc.collect()
    s1_all, t_all = load_split("test")
    cand_path, match_path = f"{OUT_DIR}/candidate_pairs.tsv", f"{OUT_DIR}/matching_results.tsv"
    fc, fm = open(cand_path, "w", encoding="utf-8", newline="\n"), open(match_path, "w", encoding="utf-8", newline="\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    n_rows = n_match = 0
    for country in s1_all.country.unique():
        s1c = s1_all[s1_all.country == country].reset_index(drop=True)
        tc = t_all[t_all.country == country].reset_index(drop=True)
        s1_id = s1c.entity_id.to_numpy()
        if len(tc) == 0:
            for s in s1_id:
                fc.write(f"{s}\t\n"); fm.write(f"{s}\t\n")
            n_rows += len(s1_id)
            continue
        A, B, P, idf_n, idf_a = prepare_country(s1c, tc, country)
        t_id = B.entity_id.to_numpy()
        p = np.zeros(len(P))
        for s in range(0, len(P), 2_000_000):
            Fx = pair_features(P.iloc[s:s + 2_000_000], A, B, idf_n, idf_a)
            p[s:s + 2_000_000] = cal.predict(model.predict(Fx[FEATURE_COLS], num_iteration=model.best_iteration))
            del Fx
        sel = decide(P.i1.to_numpy(), P.it.to_numpy(), p, BEST_RULE)
        cands, matches = defaultdict(list), defaultdict(list)
        for a, b, m in zip(P.i1.to_numpy(), P.it.to_numpy(), sel):
            cands[a].append(t_id[b])
            if m:
                matches[a].append(t_id[b])
        for i, s in enumerate(s1_id):
            fc.write(f"{s}\t{','.join(cands.get(i, []))}\n")
            fm.write(f"{s}\t{','.join(matches.get(i, []))}\n")
        n_rows += len(s1_id)
        n_match += int(sel.sum())
        log(f"  {country}: {len(s1_id):,} S1 rows, {int(sel.sum()):,} matches, "
            f"{sum(1 for i in range(len(s1_id)) if i not in matches) / len(s1_id):.3f} empty")
        del A, B, P, p, sel, cands, matches
        gc.collect()
    fc.close(); fm.close()
    log(f"wrote {n_rows:,} rows, {n_match:,} matches -> {match_path}")

# %% [markdown]
# ## 11. Format check (same rules as `utils/validate_submission.py`)

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
# ## 12. Save outputs to Google Drive

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
