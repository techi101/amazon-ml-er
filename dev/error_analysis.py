"""DEV ONLY - where are the errors? Runs on the regional slice.
  ER_DATA_DIR=dev/real_subset python dev/error_analysis.py
Writes dev/reports/error_analysis.md
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).resolve().parent
src = (HERE.parent / "colab" / "er_notebook.py").read_text(encoding="utf-8")
exec(compile(src.split("# ## 8. Training split")[0], "nb", "exec"))

rep = []


def out(s=""):
    rep.append(str(s))
    print(s, flush=True)


def load(split):
    d = {s: read_tsv(f"{DATA_DIR}/{split}/{split}_source{s}.tsv") for s in (1, 2, 3)}
    return d[1], pd.concat([d[2], d[3]], ignore_index=True)


gt = read_tsv(f"{DATA_DIR}/train/train_ground_truth.tsv")
TRUE = {r.source1_entity_id: [x for x in r.matched_entity_ids.split(",") if x] for r in gt.itertuples()}
OWNER = {x: s for s, l in TRUE.items() for x in l}
s1_all, t_all = load("train")

Xs, Ms, miss_rows = [], [], []
stats = Counter()
for country in s1_all.country.unique():
    s1c = s1_all[s1_all.country == country].reset_index(drop=True)
    tc = t_all[t_all.country == country].reset_index(drop=True)
    A, B, P, idf_n, idf_a, P_raw = prepare_country(s1c, tc, country, keep_raw=True)
    s1_id, t_id = A.entity_id.to_numpy(), B.entity_id.to_numpy()
    pos_t = {x: i for i, x in enumerate(t_id)}
    pos_1 = {x: i for i, x in enumerate(s1_id)}
    raw_set = set(zip(P_raw.i1.tolist(), P_raw.it.tolist()))
    kept_set = set(zip(P.i1.tolist(), P.it.tolist()))
    for s in s1_id:
        for t in TRUE.get(s, []):
            if t not in pos_t:
                continue
            key = (pos_1[s], pos_t[t])
            where = "kept" if key in kept_set else ("pruned" if key in raw_set else "never")
            stats[f"{country}:{where}"] += 1
            if where != "kept":
                miss_rows.append((country, where, s, t, key))
    owner = np.array([OWNER.get(x, "") for x in t_id], dtype=object)
    y = (owner[P.it.to_numpy()] == s1_id[P.i1.to_numpy()]).astype(np.int8)
    Fx = features_chunked(P, A, B, idf_n, idf_a)
    Xs.append(Fx)
    Ms.append(pd.DataFrame({"s1": s1_id[P.i1.to_numpy()], "t": t_id[P.it.to_numpy()], "y": y, "country": country,
                            "qs": P.qs.to_numpy()}))
    # keep cleaned views for printing examples
    globals()[f"A_{country}"], globals()[f"B_{country}"] = A.set_index("entity_id"), B.set_index("entity_id")

raw1 = s1_all.set_index("entity_id")
rawt = t_all.set_index("entity_id")
out("# Error analysis on the regional slice\n")
out("## 1. Where each true link ends up after candidate generation")
for k, v in sorted(stats.items()):
    out(f"- {k}: {v:,}")


def show(country, s, t, extra=""):
    A, B = globals()[f"A_{country}"], globals()[f"B_{country}"]
    a, b = A.loc[s], B.loc[t]
    out(f"- {extra}\n  S1 `{raw1.loc[s].business_name}` | `{raw1.loc[s].business_address}`\n"
        f"     core2=`{a.nm_core2}` addr=`{a.ad_words}` num1=`{a.ad_num1}` st=`{a.ad_state}`\n"
        f"  T  `{rawt.loc[t].business_name}` | `{rawt.loc[t].business_address}`\n"
        f"     core2=`{b.nm_core2}` addr=`{b.ad_words}` num1=`{b.ad_num1}` st=`{b.ad_state}`")


rng = np.random.default_rng(0)
for where in ("never", "pruned"):
    rows = [r for r in miss_rows if r[1] == where]
    out(f"\n## 2{'a' if where == 'never' else 'b'}. Links {where} generated/kept ({len(rows):,}); 25 random")
    for i in rng.permutation(len(rows))[:25]:
        c, _, s, t, _ = rows[i]
        show(c, s, t)

# ---- model errors: train on half the S1 records, evaluate on the other half
X = pd.concat(Xs, ignore_index=True)
M = pd.concat(Ms, ignore_index=True)
half = pd.util.hash_array(M.s1.to_numpy().astype(object)) % 2 == 0
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=os.cpu_count(), seed=1)
mdl = lgb.train(params, lgb.Dataset(X[half], M.y[half]), num_boost_round=600)
M["p"] = mdl.predict(X)
V = M[~half].copy()
V["p_own"] = one_owner(V.t.to_numpy(), V.p.to_numpy())
V["sel"] = V.p_own >= 0.5
ids = sorted(set(s for s in TRUE if s in set(V.s1)) | set(s for s in TRUE if (pd.util.hash_array(np.array([s], dtype=object))[0] % 2 == 1)))
ids = [s for s in ids if s in raw1.index]
pred = to_sets(V.s1.to_numpy(), V.t.to_numpy(), V.sel.to_numpy())
oracle = to_sets(V.s1.to_numpy(), V.t.to_numpy(), V.y.to_numpy() == 1)
out(f"\n## 3. Scores on the held-out half ({len(ids):,} S1)")
out(f"- model (one-owner, p>=0.5): {macro_f05(pred, TRUE, ids):.4f}")
out(f"- oracle decision on our candidates (recall ceiling): {macro_f05(oracle, TRUE, ids):.4f}")
fp = V[V.sel & (V.y == 0)]
fn = V[~V.sel & (V.y == 1)]
out(f"- false matches: {len(fp):,}; missed matches among candidates: {len(fn):,}; true pairs {int(V.y.sum()):,}")
imp = pd.Series(mdl.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
out("- top features: " + ", ".join(f"{k} {v / imp.sum():.1%}" for k, v in imp.head(12).items()))
for name, df in (("False matches", fp), ("Missed matches", fn)):
    out(f"\n## 4. {name}: 25 random")
    for r in df.sample(min(25, len(df)), random_state=0).itertuples():
        true_owner = OWNER.get(r.t, "-")
        show(r.country, r.s1, r.t, f"p={r.p:.3f} p_own={r.p_own:.3f} qs={r.qs:.2f} true owner of T: {true_owner}")

(HERE / "reports" / "error_analysis.md").write_text("\n".join(rep), encoding="utf-8")
