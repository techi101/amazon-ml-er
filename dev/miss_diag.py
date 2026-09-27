"""DEV ONLY - why are full-country true links not generated? Uses the cleaned-frame cache.
  ER_CLEAN_CACHE=dev/clean_cache python dev/miss_diag.py India
"""
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
COUNTRY = sys.argv[1] if len(sys.argv) > 1 else "India"
os.environ.setdefault("ER_DATA_DIR", r"C:\Users\Lenovo\Downloads\dataset")
os.environ.setdefault("ER_OUT_DIR", str(Path(__file__).parent / "recall_out"))
src = (Path(__file__).resolve().parents[1] / "colab" / "er_v2.py").read_text(encoding="utf-8")
exec(compile(src.split("# ## 8. Stage-2 context")[0], "nb", "exec"))

gt = read_tsv(f"{DATA_DIR}/train/train_ground_truth.tsv")
TRUE = {r.source1_entity_id: [x for x in r.matched_entity_ids.split(",") if x] for r in gt.itertuples()}
OWNER = {x: s for s, l in TRUE.items() for x in l}
del gt
s1 = compact(read_tsv(f"{DATA_DIR}/train/train_source1.tsv"))
s1 = s1[s1.country == COUNTRY].reset_index(drop=True)
t = pd.concat([compact(read_tsv(f"{DATA_DIR}/train/train_source{k}.tsv")) for k in (2, 3)], ignore_index=True)
t = t[t.country == COUNTRY].reset_index(drop=True)
cache = os.environ["ER_CLEAN_CACHE"]
key = f"{cache}/{COUNTRY}_{len(s1)}_{len(t)}_{s1.entity_id.iloc[0]}_{t.entity_id.iloc[0]}"
A, B = pd.read_parquet(key + "_A.parquet"), pd.read_parquet(key + "_B.parquet")
B["is_s3"] = B.entity_id.str.startswith("S3").to_numpy(dtype=bool)
infer_states(A, B, COUNTRY)
name_df = df_counts(pd.concat([A.nm_core2, B.nm_core2])).to_dict()
addr_df = df_counts(pd.concat([A.ad_wset, B.ad_wset])).to_dict()
log("building keys")
KA, KB = build_keys(A, name_df, addr_df, COUNTRY, "s1"), build_keys(B, name_df, addr_df, COUNTRY, "t")

# sample of true links
s1_pos = {x: i for i, x in enumerate(A.entity_id.tolist())}
t_ids = B.entity_id.tolist()
links = [(s1_pos[OWNER[x]], j) for j, x in enumerate(t_ids) if x in OWNER and OWNER[x] in s1_pos]
rng = np.random.default_rng(0)
links = [links[i] for i in rng.choice(len(links), 60000, replace=False)]
L = pd.DataFrame(links, columns=["i1", "it"])
status = pd.DataFrame(index=L.index)
for kn in KEY_NAMES:
    a = pd.DataFrame({"k": np.concatenate([k for _, k in KA[kn]]), "i1": np.concatenate([r for r, _ in KA[kn]])})
    b = pd.DataFrame({"k": np.concatenate([k for _, k in KB[kn]]), "it": np.concatenate([r for r, _ in KB[kn]])})
    a, b = a[a.k != 0].drop_duplicates(), b[b.k != 0].drop_duplicates()
    ca, cb = a.k.value_counts(), b.k.value_counts()
    la = L.merge(a[a.i1.isin(L.i1)], on="i1")
    m = la.merge(b[b.it.isin(L.it)], on=["k", "it"])          # shared key for the true pair
    m["n1"], m["nt"] = m.k.map(ca).to_numpy(), m.k.map(cb).to_numpy()
    cap1, capt = KEY_CAPS[kn]
    m["ok"] = (m.n1 <= cap1) & (m.nt <= capt)
    g = m.groupby(["i1", "it"]).ok.max()
    idx = pd.MultiIndex.from_frame(L[["i1", "it"]])
    status[kn] = pd.Series(g.reindex(idx).to_numpy(), index=L.index).map({True: 2, False: 1}).fillna(0)
found = (status == 2).any(axis=1)
shared_capped = ~found & (status == 1).any(axis=1)
print(f"sample {len(L):,}: found {found.mean():.4f} | share a key but every shared key capped {shared_capped.mean():.4f} | "
      f"share no key at all {(~found & ~shared_capped).mean():.4f}")
for kn in KEY_NAMES:
    print(f"  {kn:14s} shared-but-capped on missed links: {((status[kn] == 1) & ~found).mean():.4f}")
raw1 = s1.set_index("entity_id")
rawt = t.set_index("entity_id")
for title, mask in (("CAPPED", shared_capped), ("NO SHARED KEY", ~found & ~shared_capped)):
    print(f"\n=== {title}: 12 examples ===")
    for r in L[mask].sample(min(12, int(mask.sum())), random_state=1).itertuples():
        a, b = A.iloc[r.i1], B.iloc[r.it]
        print(f"S1 `{raw1.loc[a.entity_id].business_name}` | `{raw1.loc[a.entity_id].business_address}`\n"
              f"   core2={a.nm_core2!r} alias={a.nm_alias!r} words={a.ad_words!r} n1={a.ad_num1!r} st={a.ad_state!r}\n"
              f"T  `{rawt.loc[b.entity_id].business_name}` | `{rawt.loc[b.entity_id].business_address}`\n"
              f"   core2={b.nm_core2!r} alias={b.nm_alias!r} words={b.ad_words!r} n1={b.ad_num1!r} st={b.ad_state!r}")
