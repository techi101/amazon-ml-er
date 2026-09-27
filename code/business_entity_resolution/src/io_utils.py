"""Reading and writing the challenge's tab-separated files."""
from pathlib import Path

import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[COLS].copy()
    for c in COLS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df.reset_index(drop=True)


def read_split(split_dir, split):
    """Load the three sources of one split ('train' or 'test')."""
    d = Path(split_dir)
    return {s: read_source(d / f"{split}_source{s}.tsv") for s in (1, 2, 3)}


def parse_id_list(s):
    if not isinstance(s, str):
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def read_ground_truth(path):
    """Return {source1_entity_id: set(matched ids)}."""
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {r.source1_entity_id.strip(): set(parse_id_list(r.matched_entity_ids))
            for r in gt.itertuples(index=False)}


def write_id_lists(path, s1_ids, lists, col):
    """One row per Source-1 id; lists maps s1 id -> ordered iterable of ids.

    Written by hand (not pandas) so ID lists are never quoted."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for sid in s1_ids:
            seen, out = set(), []
            for x in lists.get(sid, ()):
                if x not in seen and not x.startswith("S1-"):
                    seen.add(x)
                    out.append(x)
            f.write(f"{sid}\t{','.join(out)}\n")
