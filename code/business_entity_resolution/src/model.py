"""Stage 5: LightGBM pair scorer with grouped out-of-fold training + calibration."""
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold


def fit_oof(X, y, groups, cfg, seed=None):
    """Grouped K-fold (a Source-1 record never spans train and valid).

    Returns out-of-fold probabilities and the list of fold models; the fold
    models are averaged to score the test set."""
    m = cfg["model"]
    params = dict(m["lgb_params"], seed=cfg["seed"] if seed is None else seed, num_threads=0)
    oof = np.zeros(len(y), dtype=np.float64)
    models = []
    gkf = GroupKFold(n_splits=cfg["n_folds"])
    for tr, va in gkf.split(X, y, groups):
        dtr = lgb.Dataset(X.iloc[tr], y[tr])
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=m["num_boost_round"], valid_sets=[dva],
                        callbacks=[lgb.early_stopping(m["early_stopping_rounds"], verbose=False)])
        oof[va] = bst.predict(X.iloc[va], num_iteration=bst.best_iteration)
        models.append(bst)
    return oof, models


def fit_on_subset(X, y, cfg, rounds):
    params = dict(cfg["model"]["lgb_params"], seed=cfg["seed"], num_threads=0)
    return lgb.train(params, lgb.Dataset(X, y), num_boost_round=rounds)


def predict(models, X):
    return np.mean([b.predict(X, num_iteration=b.best_iteration or None) for b in models], axis=0)


class Calibrator:
    """Isotonic map from raw model score to probability, fitted on OOF scores."""

    def __init__(self):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, p, y):
        self.iso.fit(p, y)
        return self

    def __call__(self, p):
        return self.iso.predict(p)


def feature_importance(models, names, top=25):
    imp = np.mean([b.feature_importance("gain") for b in models], axis=0)
    order = np.argsort(-imp)[:top]
    return [(names[i], round(float(imp[i]) / imp.sum() * 100, 2)) for i in order]
