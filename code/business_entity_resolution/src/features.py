"""Stage 4: pair features.

Three kinds:
  * similarity  - how alike the two records' names/addresses are
  * evidence    - rare shared words, agreeing/conflicting numbers and postcodes
  * context     - how this candidate compares with the *other* candidates of the
                  same Source-1 record, and with the other Source-1 records that
                  want the same S2/S3 record (the main false-merge guard)
No feature encodes which country a record is from, only whether two records'
country labels agree, so the model transfers to countries unseen in training.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

from .blocking import rowwise_cos


def _fz(scorer, a, b):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / 100.0


def _norm_sim(metric, a, b):
    return cpdist(a, b, scorer=metric.normalized_similarity, workers=-1, dtype=np.float32)


def _idf(series_list):
    df = Counter()
    n = 0
    for s in series_list:
        for text in s:
            n += 1
            df.update(set(text.split()))
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}, math.log(n + 1) + 1.0


def _token_evidence(a_list, b_list, idf, default_idf):
    """IDF-weighted overlap statistics for aligned lists of strings."""
    n = len(a_list)
    out = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        A, B = set(a_list[i].split()), set(b_list[i].split())
        if not A or not B:
            out[i] = (-1, 0, 0, 0, 0, 0, float(not A and not B))
            continue
        inter, only_a, only_b = A & B, A - B, B - A
        w = lambda S: sum(idf.get(t, default_idf) for t in S)
        wi, wa, wb = w(inter), w(only_a), w(only_b)
        out[i] = (wi / (wi + wa + wb),                         # weighted Jaccard
                  max((idf.get(t, default_idf) for t in inter), default=0.0),  # rarest shared word
                  len(inter),
                  min(wa, wb), max(wa, wb),                    # unexplained mass per side
                  len(inter) / min(len(A), len(B)),            # overlap coefficient
                  0.0)
    return out


def _alias_best(a_alias, b_alias, base):
    """Best token-set ratio over DBA/trade-name aliases (only where aliases exist)."""
    out = base.copy()
    for i in np.flatnonzero([("|" in x) or ("|" in y) for x, y in zip(a_alias, b_alias)]):
        out[i] = max(fuzz.token_set_ratio(x, y) / 100.0
                     for x in a_alias[i].split("|") for y in b_alias[i].split("|"))
    return out


def _num_features(a_nums, b_nums, a_pc, b_pc, a_h, b_h):
    n = len(a_nums)
    out = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        A, B = set(a_nums[i].split()), set(b_nums[i].split())
        pc = 0.0 if (not a_pc[i] or not b_pc[i]) else (1.0 if a_pc[i] == b_pc[i] else -1.0)
        pc3 = 0.0 if (not a_pc[i] or not b_pc[i]) else float(a_pc[i][:3] == b_pc[i][:3])
        hs = 0.0 if (not a_h[i] or not b_h[i]) else (1.0 if a_h[i] == b_h[i] else -1.0)
        jac = len(A & B) / len(A | B) if (A and B) else -1.0
        conflict = len(A ^ B) if (A and B) else 0
        # a number from one side appears anywhere in the other side's numbers
        out[i] = (pc, pc3, hs, jac, conflict, float(bool(A) and bool(B)))
    return out


def _group_context(df, col, group, prefix):
    """rank within group, gap to group max, margin over the best *other* member."""
    v = df[col].to_numpy()
    g = df[group].to_numpy()
    order = np.lexsort((-v, g))
    gs, vs = g[order], v[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(gs)), 0))
    rank = np.arange(len(gs)) - start
    top1 = vs[start]
    # second best per group: value at start+1 if it exists in the same group
    second = np.where((start + 1 < len(gs)) & (gs[np.minimum(start + 1, len(gs) - 1)] == gs),
                      vs[np.minimum(start + 1, len(vs) - 1)], np.nan)
    other_best = np.where(rank == 0, second, top1)
    res = pd.DataFrame(index=df.index[order])
    res[f"{prefix}_rank"] = rank.astype(np.float32)
    res[f"{prefix}_gap_top"] = (top1 - vs).astype(np.float32)
    res[f"{prefix}_margin"] = np.where(np.isnan(other_best), 1.0, vs - other_best).astype(np.float32)
    return res.loc[df.index]


def build_features(pairs, n1, nt, mats):
    """pairs has i1 (row in n1) and it (row in nt). Returns a feature DataFrame."""
    i1, it = pairs.i1.to_numpy(), pairs.it.to_numpy()
    a, b = n1.iloc[i1].reset_index(drop=True), nt.iloc[it].reset_index(drop=True)
    F = pd.DataFrame(index=pairs.index)

    for c in ("blk_A", "blk_B", "blk_C", "blk_D", "blk_E", "blk_R", "n_blockers",
              "cos_name_char", "cos_full_char"):
        F[c] = pairs[c].to_numpy(np.float32)
    for view in ("addr_char", "full_word", "name_word", "skel_char"):
        F[f"cos_{view}"] = rowwise_cos(*mats[view], i1, it)

    L = lambda s: s.tolist()
    an, bn = L(a.name_core), L(b.name_core)
    F["nm_ratio"] = _fz(fuzz.ratio, an, bn)
    F["nm_partial"] = _fz(fuzz.partial_ratio, an, bn)
    F["nm_tsort"] = _fz(fuzz.token_sort_ratio, an, bn)
    F["nm_tset"] = _fz(fuzz.token_set_ratio, an, bn)
    F["nm_jw"] = _norm_sim(distance.JaroWinkler, an, bn)
    F["nm_lev"] = _norm_sim(distance.Levenshtein, an, bn)
    F["nm_clean_ratio"] = _fz(fuzz.ratio, L(a.name_clean), L(b.name_clean))
    F["nm_nospace_ratio"] = _fz(fuzz.ratio, L(a.name_nospace), L(b.name_nospace))
    F["nm_skel_tset"] = _fz(fuzz.token_set_ratio, L(a.name_skel), L(b.name_skel))
    F["nm_alias_best"] = _alias_best(L(a.name_aliases), L(b.name_aliases), F["nm_tset"].to_numpy())
    F["nm_exact"] = (a.name_core.to_numpy() == b.name_core.to_numpy()).astype(np.float32)
    ta, tb = a.name_core.str.split().str[0].fillna(""), b.name_core.str.split().str[0].fillna("")
    F["nm_first_tok_eq"] = ((ta == tb) & (ta != "")).to_numpy(np.float32)
    acr_a, acr_b = a.name_acronym.to_numpy(), b.name_acronym.to_numpy()
    F["nm_acronym"] = (((acr_a != "") & (acr_a == b.name_nospace.to_numpy())) |
                       ((acr_b != "") & (acr_b == a.name_nospace.to_numpy()))).astype(np.float32)
    la, lb = a.name_legal.to_numpy(), b.name_legal.to_numpy()
    F["legal_state"] = np.select([(la == "") | (lb == ""), la == lb], [0.0, 1.0], -1.0).astype(np.float32)
    na, nb = a.name_core.str.split().str.len(), b.name_core.str.split().str.len()
    F["nm_ntok_min"] = np.minimum(na, nb).to_numpy(np.float32)
    F["nm_ntok_diff"] = (na - nb).abs().to_numpy(np.float32)
    F["nm_len_ratio"] = (np.minimum(a.name_core.str.len(), b.name_core.str.len()) /
                         np.maximum(1, np.maximum(a.name_core.str.len(), b.name_core.str.len()))).to_numpy(np.float32)

    aa, ba = L(a.addr_core), L(b.addr_core)
    F["ad_ratio"] = _fz(fuzz.ratio, aa, ba)
    F["ad_partial"] = _fz(fuzz.partial_ratio, aa, ba)
    F["ad_tset"] = _fz(fuzz.token_set_ratio, aa, ba)
    F["ad_tsort"] = _fz(fuzz.token_sort_ratio, aa, ba)
    F["ad_words_tset"] = _fz(fuzz.token_set_ratio, L(a.addr_words), L(b.addr_words))
    F["ad_skel_tset"] = _fz(fuzz.token_set_ratio, L(a.addr_skel), L(b.addr_skel))
    F["ad_empty"] = ((a.addr_core == "").astype(int) + (b.addr_core == "").astype(int)).to_numpy(np.float32)
    lma, lmb = a.addr_landmark.to_numpy(), b.addr_landmark.to_numpy()
    F["lm_tset"] = np.where((lma != "") & (lmb != ""), _fz(fuzz.token_set_ratio, L(a.addr_landmark), L(b.addr_landmark)), -1.0).astype(np.float32)
    # landmark of one record appearing in the other's address
    F["lm_cross"] = np.maximum(
        np.where(lma != "", _fz(fuzz.token_set_ratio, L(a.addr_landmark), ba), -1.0),
        np.where(lmb != "", _fz(fuzz.token_set_ratio, L(b.addr_landmark), aa), -1.0)).astype(np.float32)

    idf_n, d_n = _idf([n1.name_core, nt.name_core])
    idf_a, d_a = _idf([n1.addr_words, nt.addr_words])
    ev_n = _token_evidence(an, bn, idf_n, d_n)
    ev_a = _token_evidence(L(a.addr_words), L(b.addr_words), idf_a, d_a)
    for j, nm in enumerate(("wjac", "max_idf", "n_shared", "unexpl_min", "unexpl_max", "overlap", "both_empty")):
        F[f"nm_{nm}"] = ev_n[:, j]
        F[f"ad_{nm}"] = ev_a[:, j]
    F["nm_s1_max_idf"] = [max((idf_n.get(t, d_n) for t in s.split()), default=0.0) for s in an]

    nf = _num_features(L(a.addr_nums), L(b.addr_nums), L(a.addr_postcode), L(b.addr_postcode),
                       L(a.addr_house), L(b.addr_house))
    for j, nm in enumerate(("pc_state", "pc3_eq", "house_state", "num_jac", "num_conflict", "num_both")):
        F[nm] = nf[:, j]

    ca, cb = a.country_norm.to_numpy(), b.country_norm.to_numpy()
    F["country_state"] = np.select([(ca == "") | (cb == ""), ca == cb], [0.0, 1.0], -1.0).astype(np.float32)
    F["tgt_is_s3"] = (b.source.to_numpy() == "S3").astype(np.float32)

    # ---- context: compare against the other candidates
    ctx = pd.DataFrame({"i1": i1, "it": it, "src": b.source.to_numpy(),
                        "c_full": F.cos_full_char.to_numpy(), "c_name": F.cos_name_char.to_numpy(),
                        "c_tset": F.nm_tset.to_numpy()}, index=pairs.index)
    ctx["g1src"] = ctx.i1.astype(str) + ctx.src
    for col in ("c_full", "c_name", "c_tset"):
        F = F.join(_group_context(ctx, col, "i1", f"s1_{col}"))
        F = F.join(_group_context(ctx, col, "it", f"tg_{col}"))
    F = F.join(_group_context(ctx, "c_full", "g1src", "s1src_c_full"))
    F["s1_n_cand"] = ctx.groupby("i1").i1.transform("size").to_numpy(np.float32)
    F["tg_n_cand"] = ctx.groupby("it").it.transform("size").to_numpy(np.float32)
    return F


def stage2_features(pairs, F, p1, nt, mats, top=10):
    """Context features built from stage-1 probabilities p1 (out-of-fold on train)."""
    G = F.copy()
    G["p1"] = p1.astype(np.float32)
    ctx = pd.DataFrame({"i1": pairs.i1.to_numpy(), "it": pairs.it.to_numpy(), "p1": p1}, index=pairs.index)
    G = G.join(_group_context(ctx, "p1", "i1", "s1_p1"))
    G = G.join(_group_context(ctx, "p1", "it", "tg_p1"))
    G["s1_p1_sum"] = ctx.groupby("i1").p1.transform("sum").to_numpy(np.float32)
    G["tg_p1_sum"] = ctx.groupby("it").p1.transform("sum").to_numpy(np.float32)
    G["s1_n_p50"] = ctx.assign(h=ctx.p1 > 0.5).groupby("i1").h.transform("sum").to_numpy(np.float32)

    # support: another strong candidate of the same S1 record that looks like this one
    src = nt.source.to_numpy()
    sub = ctx.sort_values(["i1", "p1"], ascending=[True, False]).groupby("i1").head(top)
    sub = sub.reset_index().rename(columns={"index": "pid"})
    m = sub.merge(sub, on="i1", suffixes=("", "_o"))
    m = m[m.pid != m.pid_o]
    sup_same = np.zeros(len(pairs), np.float32)
    sup_cross = np.zeros(len(pairs), np.float32)
    if len(m):
        sim = rowwise_cos(mats["full_char"][1], mats["full_char"][1], m.it.to_numpy(), m.it_o.to_numpy())
        m = m.assign(sup=np.minimum(sim, m.p1_o.to_numpy()),
                     cross=src[m.it.to_numpy()] != src[m.it_o.to_numpy()])
        best = m.groupby(["pid", "cross"]).sup.max().unstack(fill_value=0.0)
        pos = pairs.index.get_indexer(best.index)
        if False in best.columns:
            sup_same[pos] = best[False].to_numpy()
        if True in best.columns:
            sup_cross[pos] = best[True].to_numpy()
    G["sup_same_src"] = sup_same
    G["sup_cross_src"] = sup_cross
    return G
