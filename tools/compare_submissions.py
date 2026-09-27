"""Compare two matching_results.tsv files per country (no labels needed).

  python tools/compare_submissions.py A/matching_results.tsv B/matching_results.tsv \
      --test-dir <dataset>/test
Prints matches per S1, empty-list rate, and how many links the two files share.
"""
import argparse
import csv
from collections import defaultdict


def lists(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        return {row[0]: set(x for x in (row[1] if len(row) > 1 else "").split(",") if x)
                for row in r if row}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--test-dir", required=True)
    x = ap.parse_args()
    country = {}
    with open(f"{x.test_dir}/test_source1.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            country[row[0]] = row[3] if len(row) > 3 else ""
    A, B = lists(x.a), lists(x.b)
    st = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0])
    for s, c in country.items():
        a, b = A.get(s, set()), B.get(s, set())
        for key in (c, "ALL"):
            v = st[key]
            v[0] += 1; v[1] += len(a); v[2] += len(b)
            v[3] += not a; v[4] += not b; v[5] += len(a & b); v[6] += (a == b)
    print(f"{'country':8s} {'S1':>9s} {'A/S1':>6s} {'B/S1':>6s} {'A empty':>8s} {'B empty':>8s}"
          f" {'shared/A':>9s} {'shared/B':>9s} {'same list':>9s}")
    for k in sorted(st, key=lambda k: (k == "ALL", k)):
        n, la, lb, ea, eb, sh, same = st[k]
        print(f"{k:8s} {n:9,d} {la / n:6.2f} {lb / n:6.2f} {ea / n:8.1%} {eb / n:8.1%}"
              f" {sh / max(la, 1):9.1%} {sh / max(lb, 1):9.1%} {same / n:9.1%}")


if __name__ == "__main__":
    main()
