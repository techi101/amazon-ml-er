"""Stages 2-3: vector representations and candidate generation (blocking).

For every Source-1 record we take the union of several cheap retrievers, each
run separately against Source 2 and Source 3:
  A. character TF-IDF on the core name
  B. character TF-IDF on name + address
  C. word TF-IDF on name + address (BM25-like: rare words dominate)
  D. same postcode, ranked by name similarity
  E. identical space-free core name
  R. reverse: for each S2/S3 record, its best Source-1 records
The union (capped per Source-1 record) is exactly what the matcher scores and
what goes into candidate_pairs.tsv.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

VIEWS = {
    # name -> (column, vectorizer kwargs)
    "name_char": ("name_core", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)),
    "full_char": ("full", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
    "addr_char": ("addr_core", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
    "full_word": ("full", dict(analyzer="word", token_pattern=r"\S+", min_df=1, sublinear_tf=True)),
    "name_word": ("name_core", dict(analyzer="word", token_pattern=r"\S+", min_df=1, sublinear_tf=True)),
    "skel_char": ("name_skel", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)),
}


def build_views(n1, nt):
    """Fit each TF-IDF view on all records of this split (S1 + S2 + S3).

    Fitting on the split itself means word rarity is measured on the data being
    matched: on the test split that includes the French records."""
    mats = {}
    for view, (col, kw) in VIEWS.items():
        vec = TfidfVectorizer(dtype=np.float32, **kw)
        vec.fit(pd.concat([n1[col], nt[col]]).replace("", " "))
        mats[view] = (vec.transform(n1[col]).tocsr(), vec.transform(nt[col]).tocsr())
    return mats


def rowwise_cos(A, B, ia, ib, batch=500_000):
    """Cosine of A[ia[i]] and B[ib[i]] for every i (rows are L2-normalised)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), batch):
        a, b = A[ia[s:s + batch]], B[ib[s:s + batch]]
        out[s:s + batch] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def topk(Q, D, k, budget=40_000_000):
    """Top-k columns of Q @ D.T per row. Returns (q_idx, d_idx, score)."""
    if Q.shape[0] == 0 or D.shape[0] == 0 or k <= 0:
        return (np.array([], int),) * 2 + (np.array([], np.float32),)
    k = min(k, D.shape[0])
    chunk = max(16, budget // max(1, D.shape[0]))
    DT = D.T.tocsc()
    qs, ds, ss = [], [], []
    for s in range(0, Q.shape[0], chunk):
        sim = (Q[s:s + chunk] @ DT).toarray()
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        val = np.take_along_axis(sim, idx, axis=1)
        rows = np.repeat(np.arange(s, s + sim.shape[0]), k)
        keep = val.ravel() > 0
        qs.append(rows[keep]); ds.append(idx.ravel()[keep]); ss.append(val.ravel()[keep])
    return np.concatenate(qs), np.concatenate(ds), np.concatenate(ss)


def _key_block(keys1, keyst, t_rows, max_group=500):
    """Pairs sharing an exact, non-empty key (skip keys that are too common)."""
    t = pd.DataFrame({"key": keyst[t_rows], "t": t_rows})
    t = t[t.key != ""]
    sizes = t.groupby("key").size()
    t = t[t.key.map(sizes) <= max_group]
    q = pd.DataFrame({"key": keys1, "q": np.arange(len(keys1))})
    q = q[q.key != ""]
    m = q.merge(t, on="key")
    return m.q.to_numpy(), m.t.to_numpy()


def generate_candidates(n1, nt, mats, cfg):
    """Return a DataFrame of (i1, it) index pairs plus blocker flags."""
    b = cfg["blocking"]
    parts = []

    def add(q, t, name):
        parts.append(pd.DataFrame({"i1": q, "it": t, "blk": name}))

    for src in ("S2", "S3"):
        rows = np.flatnonzero(nt["source"].to_numpy() == src)
        if len(rows) == 0:
            continue
        for view, k, name in (("name_char", b["k_name_char"], "A"),
                              ("full_char", b["k_full_char"], "B"),
                              ("full_word", b["k_full_word"], "C")):
            Q, T = mats[view]
            q, d, _ = topk(Q, T[rows], k)
            add(q, rows[d], name)
        # reverse direction: each target record proposes its best S1 records
        Q, T = mats["full_char"]
        d, q, _ = topk(T[rows], Q, b["k_reverse"])
        add(q, rows[d], "R")
        # postcode block, ranked by name similarity
        q, t = _key_block(n1["addr_postcode"].to_numpy(), nt["addr_postcode"].to_numpy(), rows)
        if len(q):
            s = rowwise_cos(*mats["name_char"], q, t)
            df = pd.DataFrame({"i1": q, "it": t, "s": s}).sort_values(["i1", "s"], ascending=[True, False])
            df = df.groupby("i1").head(b["k_postcode"])
            add(df.i1.to_numpy(), df.it.to_numpy(), "D")
        # identical space-free core name
        q, t = _key_block(n1["name_nospace"].to_numpy(), nt["name_nospace"].to_numpy(), rows, max_group=50)
        add(q, t, "E")

    allp = pd.concat(parts, ignore_index=True)
    flags = pd.crosstab([allp.i1, allp.it], allp.blk).clip(upper=1)
    flags.columns = [f"blk_{c}" for c in flags.columns]
    for c in ("A", "B", "C", "D", "E", "R"):
        if f"blk_{c}" not in flags.columns:
            flags[f"blk_{c}"] = 0
    pairs = flags.reset_index()
    pairs["n_blockers"] = pairs.filter(like="blk_").sum(axis=1)

    # cap per S1 record, best-first by the stronger of name / full similarity
    i1, it = pairs.i1.to_numpy(), pairs.it.to_numpy()
    pairs["cos_name_char"] = rowwise_cos(*mats["name_char"], i1, it)
    pairs["cos_full_char"] = rowwise_cos(*mats["full_char"], i1, it)
    pairs["_rank_key"] = np.maximum(pairs.cos_name_char, pairs.cos_full_char) + 0.01 * pairs.n_blockers
    pairs = (pairs.sort_values(["i1", "_rank_key"], ascending=[True, False])
                  .groupby("i1").head(b["max_candidates"])
                  .drop(columns="_rank_key")
                  .reset_index(drop=True))
    return pairs
