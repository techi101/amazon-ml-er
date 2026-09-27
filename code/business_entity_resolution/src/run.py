"""End-to-end entry point.

  python -m src.run --root <student_resource dir> --mode cv     # validation only
  python -m src.run --root <student_resource dir> --mode full   # validation + test outputs
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from . import model
from .io_utils import write_id_lists
from .metrics import blocking_report, breakdown, macro_f05
from .pipeline import Split, TwoStage, apply_decision, eda, log, save_json, tune_decision
from .check_submission import check


def country_transfer(tr, best_dec, cfg):
    """Leave-one-country-out: train on the other countries, score this one.
    Our stand-in for France, which appears only in the test set."""
    out = {}
    s1_country = tr.n1.country_norm.to_numpy()[tr.pairs.i1.to_numpy()]
    for c, n in tr.n1.country_norm.value_counts().items():
        if n < 50 or n == len(tr.n1):
            continue
        rows_tr = np.flatnonzero(s1_country != c)
        rows_va = np.flatnonzero(s1_country == c)
        ts = TwoStage(cfg).fit(tr.pairs, tr.F, tr.y, tr.nt, tr.mats, rows_tr)
        p = ts.predict_rows(rows_va)
        ids = [s for s in tr.s1_ids if tr.country_of[s] == c]
        pred = apply_decision(tr.pairs.iloc[rows_va], p, best_dec, cfg)
        out[f"holdout={c}"] = round(macro_f05(pred, tr.truth, ids), 4)
        log(f"country transfer: train without {c!r} -> F0.5 on {c!r} = {out[f'holdout={c}']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="student_resource directory")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    ap.add_argument("--mode", choices=["cv", "full"], default="full")
    ap.add_argument("--skip-transfer", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = Path(args.root)
    P = cfg["paths"]
    rep_dir = root / P["output_dir"] / "reports"
    report = {}

    # ------------------------------------------------------------- train / CV
    tr = Split(root / P["train_dir"], "train", cfg, with_truth=True)
    report["eda"] = eda(tr)
    log(f"EDA: {report['eda']}")
    n_t = len(tr.nt)
    report["blocking_train"] = blocking_report(tr.candidates(), tr.truth, tr.s1_ids, n_t)
    log(f"blocking: {report['blocking_train']}")

    all_rows = np.arange(len(tr.pairs))
    ts = TwoStage(cfg).fit(tr.pairs, tr.F, tr.y, tr.nt, tr.mats, all_rows)
    from sklearn.metrics import roc_auc_score
    report["pair_auc_stage1"] = round(float(roc_auc_score(tr.y, ts.p1_full)), 5)
    report["pair_auc_stage2"] = round(float(roc_auc_score(tr.y, ts.p_oof)), 5)
    best_dec, dec_scores = tune_decision(tr.pairs, ts.p_oof, tr.truth, tr.s1_ids, cfg)
    report["decision_scores_oof"] = dec_scores
    report["best_decision"] = str(best_dec)
    pred = apply_decision(tr.pairs, ts.p_oof, best_dec, cfg)
    report["oof_macro_f05"] = round(macro_f05(pred, tr.truth, tr.s1_ids), 4)
    report["oof_breakdown"] = breakdown(pred, tr.truth, tr.s1_ids, tr.country_of)
    report["top_features_stage2"] = model.feature_importance(ts.m2, list(ts.G.columns))
    log(f"OOF macro F0.5 = {report['oof_macro_f05']} with decision {best_dec}; "
        f"AUC s1={report['pair_auc_stage1']} s2={report['pair_auc_stage2']}")
    log(f"breakdown: {report['oof_breakdown']}")
    if not args.skip_transfer:
        report["country_transfer"] = country_transfer(tr, best_dec, cfg)
    save_json(report, rep_dir / "validation_report.json")
    if args.mode == "cv":
        return

    # ------------------------------------------------------------------ test
    te = Split(root / P["test_dir"], "test", cfg, with_truth=False)
    p = ts.predict_new(te.pairs, te.F, te.nt, te.mats)
    matches = apply_decision(te.pairs, p, best_dec, cfg)
    out = root / P["output_dir"]
    write_id_lists(out / "candidate_pairs.tsv", te.s1_ids, te.candidates(), "candidate_entity_ids")
    write_id_lists(out / "matching_results.tsv", te.s1_ids, matches, "matched_entity_ids")
    n_m = sum(len(v) for v in matches.values())
    log(f"test: wrote {len(te.s1_ids)} rows, {n_m} matches, "
        f"{sum(1 for s in te.s1_ids if not matches.get(s))} empty rows")
    report["test_summary"] = {
        "n_s1": len(te.s1_ids), "n_pairs": len(te.pairs), "n_matches": n_m,
        "empty_frac": round(sum(1 for s in te.s1_ids if not matches.get(s)) / len(te.s1_ids), 4),
        "empty_frac_by_country": {c: round(float(np.mean([not matches.get(s) for s in te.s1_ids
                                                           if te.country_of[s] == c])), 4)
                                  for c in te.n1.country_norm.unique()},
    }
    save_json(report, rep_dir / "validation_report.json")

    problems = check(out / "matching_results.tsv", out / "candidate_pairs.tsv", root / P["test_dir"])
    log("own format check: " + ("PASS" if not problems else f"{len(problems)} problems: {problems[:5]}"))
    official = root / "utils" / "validate_submission.py"
    if official.exists():
        r = subprocess.run([sys.executable, str(official), "--matching", str(out / "matching_results.tsv"),
                            "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(root / P["test_dir"])],
                           capture_output=True, text=True)
        log("official validator:\n" + r.stdout + r.stderr)


if __name__ == "__main__":
    main()
