"""Stage 1: turn raw name/address strings into clean, comparable views.

Everything here is country-agnostic: the same rules run on every record, so a
country the model never saw in training (France) goes through the same path.
Abbreviation lists cover English, Indian-English and French usage; they are
hand-written domain knowledge, not looked up from any external source.
"""
import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------- vocab maps
def _groups_to_map(groups):
    m = {}
    for canon, variants in groups.items():
        for v in variants:
            m[v] = canon
        m[canon] = canon
    return m

# Legal forms: canonicalised, and removed from the "core" name.
LEGAL = _groups_to_map({
    "corp": ["corporation", "corpn", "corpo"],
    "inc": ["incorporated", "incorp"],
    "co": ["company", "cie", "compagnie", "comp"],
    "ltd": ["limited", "ltda", "lmtd", "ltd"],
    "pvt": ["private", "pvte", "prvt", "pte"],
    "llc": [], "llp": [], "lp": [], "plc": [], "pllc": [], "gmbh": [],
    "sarl": [], "sas": [], "sasu": [], "sa": [], "eurl": [], "sci": [],
    "snc": [], "scop": [], "selarl": [], "scp": [], "ag": [], "bv": [], "nv": [],
    "opc": [],
})
# Frequent descriptive words: canonicalised but kept in the core name.
NAME_WORDS = _groups_to_map({
    "and": ["&", "et", "n", "und"],
    "group": ["grp", "groupe", "grupo"],
    "intl": ["international", "internationale", "int", "intnl"],
    "svc": ["services", "service", "svcs", "srvcs", "servs"],
    "tech": ["technologies", "technology", "technologie", "technologies", "techno"],
    "ind": ["industries", "industry", "industrie", "inds", "indus"],
    "ent": ["enterprises", "enterprise", "entreprise", "entp", "entps"],
    "assoc": ["associates", "associes", "associate", "assocs"],
    "mfg": ["manufacturing", "manufacturers", "mfrs"],
    "bros": ["brothers", "bro"],
    "sys": ["systems", "system", "systemes", "systeme"],
    "sol": ["solutions", "solution", "soln", "solns"],
    "mgmt": ["management", "mgt"],
    "dev": ["development", "developers", "developpement"],
    "hosp": ["hospital", "hopital"],
    "rest": ["restaurant", "restaurants", "resto"],
    "pharma": ["pharmaceuticals", "pharmaceutical", "pharmacie", "pharmacy"],
    "dr": ["doctor", "docteur"],
    "mfrs": [],
    "natl": ["national", "nationale"],
    "assn": ["association"],
    "univ": ["university", "universite"],
    "inst": ["institute", "institut"],
    "ctr": ["center", "centre", "cntr"],
    "st": ["saint"], "ste": ["sainte"],
    "mt": ["mount"],
})
NAME_STOP = {"the", "of", "a", "an", "and", "le", "la", "les", "l", "de", "du",
             "des", "d", "aux", "au"}

ADDR_WORDS = _groups_to_map({
    "st": ["street", "str", "saint", "strt"],
    "ste": ["suite", "sainte"],
    "rd": ["road", "raod"],
    "ave": ["avenue", "av", "avn", "aven"],
    "blvd": ["boulevard", "bd", "bld", "boul", "bvd"],
    "dr": ["drive", "drv"],
    "ln": ["lane"],
    "pl": ["place"],
    "ct": ["court", "crt"],
    "sq": ["square"],
    "hwy": ["highway", "hiway"],
    "pkwy": ["parkway", "pky"],
    "ter": ["terrace"],
    "cir": ["circle"],
    "fl": ["floor", "flr", "etage"],
    "apt": ["apartment", "appt", "appartement", "flat"],
    "bldg": ["building", "bat", "batiment", "bldng"],
    "twr": ["tower", "tour"],
    "n": ["north", "nord"], "s": ["south", "sud"], "e": ["east", "est"],
    "w": ["west", "ouest"],
    "ne": ["northeast"], "nw": ["northwest"], "se": ["southeast"], "sw": ["southwest"],
    "near": ["nr", "nearby", "pres", "beside", "besides", "adj", "adjacent"],
    "opp": ["opposite", "opps", "oppo", "facing"],
    "bhd": ["behind"],
    "sec": ["sector", "sect", "sctr"],
    "ph": ["phase"],
    "ngr": ["nagar", "nagr", "nager"],
    "mg": ["marg", "maarg"],
    "col": ["colony"],
    "mkt": ["market", "mkts", "marche"],
    "xrd": ["cross"],
    "main": ["mn"],
    "indl": ["industrial", "industriel", "ind"],
    "estt": ["estate"],
    "cplx": ["complex", "cmplx"],
    "jn": ["junction", "jct", "jnc"],
    "po": ["post"],
    "dist": ["district", "distt", "dt"],
    "vill": ["village", "vlg", "vil"],
    "ch": ["chemin", "che", "chem"],
    "rte": ["route"],
    "imp": ["impasse"],
    "all": ["allee"],
    "fbg": ["faubourg"],
    "qu": ["quai"],
    "mt": ["mount"],
    "ft": ["fort"],
    "gali": ["galli", "gully"],
    "chowk": ["chauk", "chok"],
    "bazar": ["bazaar", "bzr"],
    "mohalla": ["mohala", "mohallah"],
    "hsg": ["housing"],
    "soc": ["society"],
    "ctr": ["center", "centre", "cntr"],
    "univ": ["university", "universite"],
    "hosp": ["hospital", "hopital"],
    "sta": ["station", "stn", "gare"],
    "rly": ["railway"],
    "pkg": ["parking"],
    "plz": ["plaza"],
    "mall": [],
})
ADDR_DROP = {"no", "number", "num", "nr", "house", "h", "plot", "shop", "unit",
             "door", "the", "of", "and", "de", "du", "des", "la", "le", "les", "l", "d"}
ADDR_DROP.discard("nr")  # "nr" means "near" (mapped above)

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "maharashtra": "mh", "karnataka": "ka", "tamil nadu": "tn",
    "tamilnadu": "tn", "uttar pradesh": "up", "west bengal": "wb",
    "gujarat": "gj", "rajasthan": "rj", "telangana": "tg", "andhra pradesh": "ap",
    "kerala": "kl", "madhya pradesh": "mp", "punjab": "pb", "haryana": "hr",
    "bihar": "br", "odisha": "od", "orissa": "od", "jharkhand": "jh",
    "chhattisgarh": "cg", "uttarakhand": "uk", "himachal pradesh": "hp",
    "jammu and kashmir": "jk",
}
# State names are rewritten to their code (both sides get the same treatment,
# so the only effect is that "California" and "CA" become equal).
STATE_PHRASES = sorted({**US_STATES, **IN_STATES}.items(), key=lambda kv: -len(kv[0]))
_STATE_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k, _ in STATE_PHRASES) + r")\b")
_STATE_MAP = dict(STATE_PHRASES)

ORDINALS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
            "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
            "premier": "1", "premiere": "1", "deuxieme": "2", "troisieme": "3"}
_ORD_RE = re.compile(r"\b(\d+)(st|nd|rd|th|er|ere|e|eme|ème)\b")
_ZIP4_RE = re.compile(r"\b(\d{5})-\d{4}\b")
_PIN_SPLIT_RE = re.compile(r"\b(\d{3})\s(\d{3})\b")
_LANDMARK_RE = re.compile(
    r"^\s*(near|nr|opp|opposite|behind|beside|besides|next to|adjacent to|adj|"
    r"close to|facing|in front of|pres de|pres du|pres des|en face de|en face du|"
    r"a cote de|derriere)\b")
_DBA_RE = re.compile(r"\b(dba|d b a|aka|a k a|t a|trading as|doing business as|"
                     r"formerly|fka|f k a)\b")

# ------------------------------------------------------------------ helpers
def strip_accents(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def basic_clean(s):
    """Lowercase, remove accents and punctuation, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = strip_accents(s).lower()
    s = s.replace("&", " and ").replace("@", " at ").replace("n°", " no ")
    s = re.sub(r"(\w)'s\b", r"\1s", s)
    s = re.sub(r"['’`]", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def skeleton(tok):
    """Rough transliteration-invariant form of a token (Hindi/French spellings)."""
    t = tok
    for a, b in (("ph", "f"), ("sh", "s"), ("kh", "k"), ("gh", "g"), ("th", "t"),
                 ("bh", "b"), ("dh", "d"), ("ch", "c"), ("ck", "k"), ("qu", "k"),
                 ("w", "v"), ("z", "s"), ("y", "i"), ("ee", "i"), ("oo", "u"),
                 ("ou", "u"), ("aa", "a"), ("q", "k"), ("x", "ks")):
        t = t.replace(a, b)
    t = re.sub(r"(.)\1+", r"\1", t)          # collapse doubled letters
    if len(t) > 3:
        t = t[0] + re.sub(r"[aeiou]", "", t[1:])  # drop inner vowels
    return t


def _tokens(s):
    return s.split() if s else []


# ------------------------------------------------------------------ names
def normalize_name(raw):
    clean = basic_clean(raw)
    parts = [p.strip() for p in _DBA_RE.split(clean)]
    # _DBA_RE.split keeps the separators at odd positions; drop them
    aliases = [p for i, p in enumerate(parts) if i % 2 == 0 and p]
    if not aliases:
        aliases = [clean]

    def canon(text):
        toks, legal = [], []
        for t in _tokens(text):
            t = ORDINALS.get(t, t)
            if t in LEGAL:
                legal.append(LEGAL[t])
                continue
            t = NAME_WORDS.get(t, t)
            toks.append(t)
        core = [t for t in toks if t not in NAME_STOP] or toks
        return toks, core, legal

    alias_cores, legal_all = [], []
    main_toks, main_core, main_legal = canon(aliases[0])
    for a in aliases:
        _, core, legal = canon(a)
        alias_cores.append(" ".join(core))
        legal_all += legal
    acronym = "".join(t[0] for t in main_core if t and not t.isdigit()) if len(main_core) > 1 else ""
    return {
        "name_clean": clean,
        "name_canon": " ".join(main_toks),
        "name_core": " ".join(main_core),
        "name_aliases": "|".join(dict.fromkeys(alias_cores)),
        "name_legal": " ".join(sorted(set(legal_all))),
        "name_acronym": acronym,
        "name_nospace": "".join(main_core),
        "name_skel": " ".join(skeleton(t) for t in main_core),
    }


# ------------------------------------------------------------------ addresses
def normalize_address(raw):
    if not isinstance(raw, str):
        raw = ""
    s = unicodedata.normalize("NFKC", raw)
    s = strip_accents(s).lower()
    s = _ZIP4_RE.sub(r"\1", s)
    # landmarks: comma/semicolon separated components introduced by near/opp/...
    comps = [c for c in re.split(r"[,;()\n]", s) if c.strip()]
    landmark, core_comps = [], []
    for c in comps:
        cc = basic_clean(c)
        (landmark if _LANDMARK_RE.match(cc) else core_comps).append(cc)
    core_text = " ".join(core_comps)
    lm_text = " ".join(landmark)

    def canon(text):
        text = _PIN_SPLIT_RE.sub(r"\1\2", text)
        text = _ORD_RE.sub(r"\1", text)
        text = _STATE_RE.sub(lambda m: _STATE_MAP[m.group(1)], text)
        out = []
        for t in _tokens(text):
            t = ORDINALS.get(t, t)
            if t in ADDR_DROP:
                continue
            # split "12a" / "b12" into number + letter so numbers compare cleanly
            m = re.fullmatch(r"(\d+)([a-z]{1,2})", t) or re.fullmatch(r"([a-z]{1,2})(\d+)", t)
            if m:
                out += list(m.groups())
                continue
            out.append(ADDR_WORDS.get(t, t))
        return out

    core_toks = canon(core_text)
    lm_toks = canon(lm_text)
    nums = [t for t in core_toks if t.isdigit()]
    postcode = ""
    for t in reversed(nums):
        if len(t) in (5, 6):
            postcode = t
            break
    house = ""
    for t in nums:
        if t != postcode and len(t) <= 5:
            house = t
            break
    words = [t for t in core_toks if not t.isdigit()]
    return {
        "addr_clean": basic_clean(raw),
        "addr_core": " ".join(core_toks),
        "addr_words": " ".join(words),
        "addr_landmark": " ".join(lm_toks),
        "addr_nums": " ".join(nums),
        "addr_postcode": postcode,
        "addr_house": house,
        "addr_skel": " ".join(skeleton(t) for t in words),
    }


def normalize_country(raw):
    c = basic_clean(raw)
    return {"usa": "us", "united states": "us", "united states of america": "us",
            "u s": "us", "u s a": "us", "america": "us", "in": "india",
            "bharat": "india", "fr": "france", "republique francaise": "france"}.get(c, c)


def normalize_frame(df):
    names = pd.DataFrame([normalize_name(x) for x in df["business_name"]])
    addrs = pd.DataFrame([normalize_address(x) for x in df["business_address"]])
    out = pd.concat([df[["entity_id"]].reset_index(drop=True), names, addrs], axis=1)
    out["country_norm"] = [normalize_country(x) for x in df["country"]]
    out["full"] = (out["name_core"] + " " + out["addr_core"]).str.strip()
    out["source"] = out["entity_id"].str.slice(0, 2)
    return out
