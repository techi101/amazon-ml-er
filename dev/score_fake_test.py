"""DEV ONLY - score output/matching_results.tsv against the fake test answers, per country."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
from src.io_utils import parse_id_list, read_ground_truth, read_source  # noqa: E402
from src.metrics import breakdown, macro_f05  # noqa: E402
import pandas as pd  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "dev/fake_resource")
truth = read_ground_truth(root / "dev_test_ground_truth.tsv")
m = pd.read_csv(root / "output/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
pred = {r.source1_entity_id: parse_id_list(r.matched_entity_ids) for r in m.itertuples()}
s1 = read_source(root / "dataset/test/test_source1.tsv")
country = dict(zip(s1.entity_id, s1.country))
ids = s1.entity_id.tolist()
print("test macro F0.5:", round(macro_f05(pred, truth, ids), 4))
print(breakdown(pred, truth, ids, country))
