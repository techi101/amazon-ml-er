"""Build <team>_submission.zip in the layout the rules require.

  python tools/make_submission_zip.py --run-dir <folder with matching_results.tsv,
      candidate_pairs.tsv, validation.json> --team <team_name> [--template <official Documentation_template.md>]

Layout:
  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/src/ (pipeline), README.md, requirements.txt
  Documentation_template.md (our methodology; appended to the official template if given)
"""
import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_FILES = ["colab/er_v2.py", "colab/er_v2_jupyter.py", "colab/er_v2_kaggle.py",
             "colab/er_v2_colab.py", "colab/make_jupyter.py", "colab/make_kaggle.py",
             "colab/make_colab.py", "colab/to_ipynb.py"]

README = """# Business Entity Resolution: reproduction

All code is in `src/`. `src/er_v2.py` is the master pipeline; `src/er_v2_jupyter.py` is the
edition for a local machine (same code, local paths).

```
pip install -r requirements.txt
# dataset folder with train/ and test/ (7 .tsv files)
set ER_DATA_DIR=<path to dataset>          (PowerShell: $env:ER_DATA_DIR = "<path>")
set ER_OUT_DIR=<path to output folder>
python src/er_v2_jupyter.py
```
Writes `matching_results.tsv`, `candidate_pairs.tsv` and `validation.json` to ER_OUT_DIR, then runs
the format check (same rules as `utils/validate_submission.py`). Seed 42. Peak RAM ~12 GB with the
default ER_TRAIN_S1=120000. With an NVIDIA GPU a multilingual MiniLM cross-encoder
(Apache-2.0, 118M params, downloaded once from Hugging Face) is also trained; without a GPU it is
skipped. No external data, APIs or geocoding are used. See Documentation_template.md for the method.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--template", help="official Documentation_template.md (optional)")
    ap.add_argument("--out", help="zip path (default: Downloads/<team>_submission.zip)")
    a = ap.parse_args()
    run = Path(a.run_dir)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (run / f).exists(), f"missing {run / f}"
    doc = (ROOT / "docs/Documentation.md").read_text(encoding="utf-8")
    vj = run / "validation.json"
    if vj.exists():
        doc += ("\n\n## Appendix: validation.json of the submitted run\n\n```json\n"
                + json.dumps(json.loads(vj.read_text(encoding="utf-8")), indent=1)[:20000]
                + "\n```\n")
    if a.template:
        doc = Path(a.template).read_text(encoding="utf-8") + "\n\n---\n\n" + doc
    out = Path(a.out) if a.out else Path.home() / "Downloads" / f"{a.team}_submission.zip"
    base = "code/business_entity_resolution"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(run / "matching_results.tsv", "output/matching_results.tsv")
        z.write(run / "candidate_pairs.tsv", "output/candidate_pairs.tsv")
        for f in SRC_FILES:
            z.write(ROOT / f, f"{base}/src/{Path(f).name}")
        z.write(ROOT / "requirements.txt", f"{base}/requirements.txt")
        z.writestr(f"{base}/README.md", README)
        z.writestr("Documentation_template.md", doc)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("\n".join("  " + n for n in names))


if __name__ == "__main__":
    main()
