"""DEV ONLY - train v2 on the slice, then score the FULL France test split and inspect empty rows."""
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["ER_RUN_TEST"] = "0"
HERE = Path(__file__).resolve().parent
src = (HERE.parent / "colab" / "er_v2.py").read_text(encoding="utf-8")
exec(compile(src.split("# ## 12. Format check")[0], "nb", "exec"))

FULL = r"C:\Users\Lenovo\Downloads\dataset\test"
s1 = read_tsv(f"{FULL}/test_source1.tsv")
s1 = s1[s1.country == "France"].reset_index(drop=True)
t = pd.concat([read_tsv(f"{FULL}/test_source{k}.tsv") for k in (2, 3)], ignore_index=True)
t = t[t.country == "France"].reset_index(drop=True)
A, B, P, idf_n, idf_a = prepare_country(s1, t, "France")
Xc = features_chunked(P, A, B, idf_n, idf_a)[FEATURE_COLS]
q1 = m1.predict(Xc, num_iteration=m1.best_iteration)
ii, tt = P.i1.to_numpy(), P.it.to_numpy()
cec = np.full(len(q1), -1.0, np.float32)
Gc = stage2_matrix(Xc, ii, tt, B.nm_core.to_numpy()[tt], B.ad_words.to_numpy()[tt], B.is_s3.to_numpy()[tt], q1, cec)
p = cal.predict(m2.predict(Gc[STAGE2_COLS], num_iteration=m2.best_iteration))
sel = decide(ii, tt, p, BEST_RULE)
n1 = len(A)
per = np.bincount(ii[sel], minlength=n1)
print(f"FULL France: {n1:,} S1, {per.mean():.2f} matches per S1, {(per == 0).mean():.3f} empty")
print("distribution of matches per S1:", np.bincount(per)[:10])
# owner-level view: fraction of French S2/S3 records given an owner (train: ~74%)
print(f"S2/S3 records assigned an owner: {len(np.unique(tt[sel])) / len(B):.3f}")
best = pd.DataFrame({"i1": ii, "it": tt, "p": p}).sort_values("p", ascending=False).groupby("i1").head(3)
empty = np.flatnonzero(per == 0)
rng = np.random.default_rng(0)
print("\n=== 15 empty France S1 rows with their top-3 candidates ===")
for i in rng.choice(empty, min(15, len(empty)), replace=False):
    print(f"\nS1 `{s1.business_name[i]}` | `{s1.business_address[i]}`   core2={A.nm_core2[i]!r} addr={A.ad_words[i]!r} n1={A.ad_num1[i]!r}")
    for r in best[best.i1 == i].itertuples():
        print(f"   p={r.p:.3f}  `{t.business_name[r.it]}` | `{t.business_address[r.it]}`  core2={B.nm_core2[r.it]!r} n1={B.ad_num1[r.it]!r}")
