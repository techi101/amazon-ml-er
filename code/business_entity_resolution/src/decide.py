"""Stage 6: turn pair probabilities into one match list per Source-1 record.

1. one-owner rule: Source 1 is deduplicated, so an S2/S3 record belongs to at
   most one Source-1 record. Every competing claim except the strongest is
   scaled down.
2. expected-F0.5 set choice: for each Source-1 record, sort its candidates by
   probability and output the top-k (k = 0..max_k) that maximises the expected
   per-record F0.5. k = 0 (empty list) wins when no candidate is convincing,
   which is what earns the singleton credit.
"""
import numpy as np
import pandas as pd


def one_owner(df, strength=1.0):
    """df: columns i1, it, p. Down-weight all but the best S1 claim per target.

    strength=1 zeroes the losers, 0 leaves them unchanged."""
    if strength <= 0:
        return df.p.to_numpy()
    best = df.groupby("it").p.transform("max").to_numpy()
    p = df.p.to_numpy()
    is_best = p >= best - 1e-12
    # ties: keep all tied claims
    return np.where(is_best, p, p * (1.0 - strength))


def _best_k(probs, rng, n_samples, max_k):
    """probs sorted descending. Returns k maximising E[F0.5] under independence."""
    n = min(len(probs), max_k)
    if n == 0:
        return 0
    all_p = probs  # matches outside the top-n still count toward recall
    draws = rng.random((n_samples, len(all_p))) < all_p
    T = draws.sum(axis=1)                              # true matches in this draw
    tp = np.cumsum(draws[:, :n], axis=1)               # TP when predicting top-k
    k = np.arange(1, n + 1)
    denom = 1.25 * tp + 0.25 * (T[:, None] - tp) + (k - tp)
    f = np.where(tp > 0, 1.25 * tp / np.maximum(denom, 1e-9), 0.0)
    ef = np.r_[np.mean(T == 0), f.mean(axis=0)]        # index 0 = predict nothing
    return int(np.argmax(ef))


def choose_sets(df, cfg, prob_col="p", seed=0):
    """df: columns s1_id, cand_id, prob_col. Returns {s1_id: [cand ids]}."""
    d = cfg["decision"]
    rng = np.random.default_rng(seed)
    out = {}
    df = df.sort_values(["s1_id", prob_col], ascending=[True, False])
    for sid, g in df.groupby("s1_id", sort=False):
        p = g[prob_col].to_numpy()
        p = p[p > 1e-4] if (p > 1e-4).any() else p[:0]
        k = _best_k(p, rng, d["mc_samples"], d["max_k"])
        out[sid] = g.cand_id.to_numpy()[:k].tolist()
    return out


def threshold_sets(df, thr, prob_col="p"):
    """Simple alternative: every candidate with probability >= thr."""
    keep = df[df[prob_col] >= thr]
    return keep.groupby("s1_id").cand_id.apply(list).to_dict()
