"""Glue: split -> normalised frames -> candidates -> features -> two-stage model."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import decide, features, model
from .blocking import build_views, generate_candidates
from .io_utils import read_ground_truth, read_split
from .metrics import breakdown, macro_f05
from .normalize import normalize_frame


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Split:
    """Everything derived from one split's three source files."""

    def __init__(self, split_dir, split, cfg, with_truth):
        t0 = time.time()
        raw = read_split(split_dir, split)
        self.raw = raw
        self.n1 = normalize_frame(raw[1])
        self.nt = pd.concat([normalize_frame(raw[2]), normalize_frame(raw[3])], ignore_index=True)
        log(f"{split}: S1={len(self.n1)} S2={len(raw[2])} S3={len(raw[3])} normalised ({time.time()-t0:.0f}s)")
        self.mats = build_views(self.n1, self.nt)
        self.pairs = generate_candidates(self.n1, self.nt, self.mats, cfg)
        self.pairs["s1_id"] = self.n1.entity_id.to_numpy()[self.pairs.i1]
        self.pairs["cand_id"] = self.nt.entity_id.to_numpy()[self.pairs.it]
        log(f"{split}: {len(self.pairs)} candidate pairs ({len(self.pairs)/max(1,len(self.n1)):.1f} per S1) ({time.time()-t0:.0f}s)")
        self.F = features.build_features(self.pairs, self.n1, self.nt, self.mats)
        log(f"{split}: {self.F.shape[1]} features ({time.time()-t0:.0f}s)")
        self.s1_ids = self.n1.entity_id.tolist()
        self.country_of = dict(zip(self.n1.entity_id, self.n1.country_norm))
        self.truth = None
        if with_truth:
            self.truth = read_ground_truth(Path(split_dir) / f"{split}_ground_truth.tsv")
            self.y = np.array([c in self.truth.get(s, ()) for s, c in
                               zip(self.pairs.s1_id, self.pairs.cand_id)], dtype=np.int8)

    def candidates(self):
        return self.pairs.groupby("s1_id").cand_id.apply(list).to_dict()


def eda(sp):
    t = sp.truth
    sizes = pd.Series([len(v) for v in t.values()])
    owners = pd.Series([c for v in t.values() for c in v]).value_counts()
    all_t = set(sp.nt.entity_id)
    matched = set(owners.index)
    return {
        "n_s1": len(sp.s1_ids), "n_targets": len(all_t),
        "singleton_frac": round(float((sizes == 0).mean()), 4),
        "matches_per_s1": {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
        "targets_with_multiple_owners": int((owners > 1).sum()),
        "targets_never_matched_frac": round(1 - len(matched & all_t) / max(1, len(all_t)), 4),
        "countries_s1": sp.n1.country_norm.value_counts().to_dict(),
        "countries_targets": sp.nt.country_norm.value_counts().to_dict(),
    }


class TwoStage:
    """Stage 1: LightGBM on pair features. Stage 2: LightGBM on the same
    features plus context built from stage-1 probabilities. Then isotonic
    calibration. Trained with grouped out-of-fold predictions throughout."""

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, pairs, F, y, nt, mats, rows):
        """rows: which pairs to train on (all pairs keep their context)."""
        cfg = self.cfg
        g = pairs.i1.to_numpy()[rows]
        p1, self.m1 = model.fit_oof(F.iloc[rows], y[rows], g, cfg)
        self.p1_full = np.zeros(len(pairs))
        self.p1_full[rows] = p1
        other = np.setdiff1d(np.arange(len(pairs)), rows)
        if len(other):
            self.p1_full[other] = model.predict(self.m1, F.iloc[other])
        G = features.stage2_features(pairs, F, self.p1_full, nt, mats)
        p2, self.m2 = model.fit_oof(G.iloc[rows], y[rows], g, cfg)
        # cross-fitted calibration for honest OOF probabilities, full fit for test
        oof_cal = np.zeros(len(rows))
        folds = np.unique(g) % cfg["n_folds"]
        fold_of = pd.Series(folds, index=np.unique(g)).reindex(g).to_numpy()
        for f in range(cfg["n_folds"]):
            tr, va = fold_of != f, fold_of == f
            if va.any() and tr.any():
                oof_cal[va] = model.Calibrator().fit(p2[tr], y[rows][tr])(p2[va])
        self.cal = model.Calibrator().fit(p2, y[rows])
        self.G = G
        self.p_oof = oof_cal
        return self

    def predict_rows(self, rows):
        return self.cal(model.predict(self.m2, self.G.iloc[rows]))

    def predict_new(self, pairs, F, nt, mats):
        p1 = model.predict(self.m1, F)
        G = features.stage2_features(pairs, F, p1, nt, mats)
        return self.cal(model.predict(self.m2, G))


DECISIONS = [("ef", s) for s in (0.0, 0.5, 1.0)] + [("thr", s, t) for s in (0.0, 1.0)
                                                    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]


def apply_decision(pairs_sub, p, dec, cfg):
    df = pd.DataFrame({"s1_id": pairs_sub.s1_id.to_numpy(), "cand_id": pairs_sub.cand_id.to_numpy(),
                       "i1": pairs_sub.i1.to_numpy(), "it": pairs_sub.it.to_numpy(), "p": p})
    df["p"] = decide.one_owner(df, strength=dec[1])
    if dec[0] == "ef":
        return decide.choose_sets(df, cfg)
    return decide.threshold_sets(df, dec[2])


def tune_decision(pairs_sub, p, truth, s1_ids, cfg):
    scores = {}
    for dec in DECISIONS:
        scores[dec] = macro_f05(apply_decision(pairs_sub, p, dec, cfg), truth, s1_ids)
    best = max(scores, key=scores.get)
    return best, {str(k): round(v, 4) for k, v in scores.items()}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
