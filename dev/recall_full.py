"""DEV ONLY - candidate recall on one FULL country of the training split (no model).
  python dev/recall_full.py India
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
gc.collect()
log(f"{COUNTRY}: {len(s1):,} S1, {len(t):,} S2/S3 loaded")
A, B, P, idf_n, idf_a, P_raw = prepare_country(s1, t, COUNTRY, keep_raw=True)
s1_id, t_id = A.entity_id.to_numpy(dtype=object), B.entity_id.to_numpy(dtype=object)
owner = np.array([OWNER.get(x, "") for x in t_id], dtype=object)
y_raw = owner[P_raw.it.to_numpy()] == s1_id[P_raw.i1.to_numpy()]
y = owner[P.it.to_numpy()] == s1_id[P.i1.to_numpy()]
n_true = sum(len(TRUE.get(x, [])) for x in s1_id)
log(f"true links {n_true:,} | generated {y_raw.sum():,} ({y_raw.sum() / n_true:.4f}) | kept after pruning {y.sum():,} ({y.sum() / n_true:.4f})")
km = P_raw.keymask.to_numpy()[y_raw]
for b, kn in enumerate(KEY_NAMES):
    only = (km == (1 << b)).sum()
    print(f"  {kn:14s} finds {((km >> b) & 1).sum() / n_true:.3f} of true links; only source for {only:,}")
tgt_state = np.array(B.ad_state.tolist(), dtype=object)
missed_t = np.setdiff1d(np.flatnonzero(owner != ""), P.it.to_numpy()[y])
print(f"missed targets with no state: {np.mean(tgt_state[missed_t] == ''):.3f} (all targets: {np.mean(tgt_state == ''):.3f})")
