"""DEV ONLY - regional slice of the real data (Ohio, Kerala, + Gironde in test) for fast notebook testing."""
import re
import sys
from pathlib import Path

import pandas as pd

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
PAT = re.compile(r"\b(OH|KL)\b|ohio|kerala|കേരളം|gironde|nouvelle-aquitaine", re.I)
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
for split in ("train", "test"):
    (dst / split).mkdir(parents=True, exist_ok=True)
    keep = set()
    for s in (1, 2, 3):
        d = rd(src / split / f"{split}_source{s}.tsv")
        d = d[d.business_address.str.contains(PAT)]
        d.to_csv(dst / split / f"{split}_source{s}.tsv", sep="\t", index=False)
        keep |= set(d.entity_id)
        print(split, s, len(d))
    if split == "train":
        gt = rd(src / "train/train_ground_truth.tsv")
        gt = gt[gt.source1_entity_id.isin(keep)].copy()
        gt["matched_entity_ids"] = [",".join(x for x in m.split(",") if x in keep) for m in gt.matched_entity_ids]
        gt.to_csv(dst / "train/train_ground_truth.tsv", sep="\t", index=False)
        print("gt", len(gt))
