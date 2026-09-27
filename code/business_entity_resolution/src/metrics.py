"""The official score: F0.5 per Source-1 record, macro-averaged, singletons included."""
import numpy as np


def f05_one(pred, true):
    pred, true = set(pred), set(true)
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred, truth, s1_ids):
    """pred, truth: {s1_id: iterable of ids}. Averaged over s1_ids."""
    return float(np.mean([f05_one(pred.get(s, ()), truth.get(s, ())) for s in s1_ids]))


def breakdown(pred, truth, s1_ids, country_of=None):
    """Score split by singleton / non-singleton (and by country if given)."""
    rows = {}
    for s in s1_ids:
        t = truth.get(s, ())
        key = "singleton" if not t else "has_match"
        f = f05_one(pred.get(s, ()), t)
        rows.setdefault(key, []).append(f)
        if country_of is not None:
            rows.setdefault(f"country={country_of.get(s, '?')}", []).append(f)
    return {k: (round(float(np.mean(v)), 4), len(v)) for k, v in sorted(rows.items())}


def blocking_report(cands, truth, s1_ids, n_targets):
    """Recall ceiling and reduction ratio of the candidate set."""
    tp = tot = n_c = 0
    for s in s1_ids:
        t, c = set(truth.get(s, ())), set(cands.get(s, ()))
        tp += len(t & c); tot += len(t); n_c += len(c)
    ceiling = macro_f05({s: set(cands.get(s, ())) & set(truth.get(s, ())) for s in s1_ids}, truth, s1_ids)
    return {"pair_recall": tp / max(1, tot), "n_true_pairs": tot, "n_candidates": n_c,
            "avg_cand_per_s1": n_c / max(1, len(s1_ids)),
            "reduction_ratio": 1 - n_c / max(1, len(s1_ids) * n_targets),
            "oracle_macro_f05": ceiling}
