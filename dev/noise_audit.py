"""Measure which noise operations appear in real matched pairs (train GT sample).

  python dev/noise_audit.py C:/Users/Lenovo/Downloads/dataset 20000
Writes dev/reports/noise_audit.md
"""
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
out = Path(__file__).parent / "reports" / "noise_audit.md"
out.parent.mkdir(exist_ok=True)

rd = lambda p, **kw: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, **kw)
gt = rd(root / "train/train_ground_truth.tsv").sample(N, random_state=1)
s1 = rd(root / "train/train_source1.tsv")
s1 = s1[s1.entity_id.isin(set(gt.source1_entity_id))].set_index("entity_id")
links = [(r.source1_entity_id, t) for r in gt.itertuples() for t in r.matched_entity_ids.split(",") if t]
need = {t for _, t in links}
tg = pd.concat([rd(root / f"train/train_source{s}.tsv").query("entity_id in @need") for s in (2, 3)]).set_index("entity_id")
print("pairs", len(links), "targets found", len(tg))


def script_of(s):
    names = Counter()
    for ch in s:
        if ch.isalpha() and ord(ch) > 0x24F:
            names[unicodedata.name(ch, "?").split()[0]] += 1
    return names.most_common(1)[0][0] if names else "LATIN"


def fold(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def toks(s):
    return re.findall(r"[a-z0-9]+", fold(s))


rows = []
extra_tok, drop_tok, prefix, suffix_pat, addr_extra = Counter(), Counter(), Counter(), Counter(), Counter()
scripts_name, scripts_addr = Counter(), Counter()
for sid, tid in links:
    a, b = s1.loc[sid], tg.loc[tid]
    an, bn, aa, ba = a.business_name, b.business_name, a.business_address, b.business_address
    ta, tb = toks(an), toks(bn)
    sn = script_of(bn)
    scripts_name[sn] += 1
    scripts_addr[script_of(ba)] += 1
    f = {}
    f["name_exact"] = an == bn
    f["name_exact_casefold"] = fold(an).strip() == fold(bn).strip()
    f["name_same_tokens_ordered"] = ta == tb
    f["name_same_token_set"] = set(ta) == set(tb)
    f["name_nonlatin"] = sn != "LATIN"
    f["name_domain"] = bool(re.search(r"\.(com|in|net|org|co|fr|io|biz)\b", bn.lower()))
    f["name_all_upper"] = bn.isupper()
    f["name_all_lower"] = bn.islower()
    f["name_accent_injected"] = fold(bn) != bn.lower() and fold(an) == an.lower() and sn == "LATIN"
    f["name_leet_digit"] = bool(re.search(r"[a-z][0-9][a-z]|[a-z][0-9]\b", bn.lower())) and not re.search(r"[a-z][0-9]", an.lower())
    f["name_leading_junk"] = bool(re.match(r"^\W", bn.strip()))
    f["name_brackets"] = bool(re.search(r"[\[\](){}]", bn)) and not re.search(r"[\[\](){}]", an)
    f["name_hash_number"] = bool(re.search(r"#\s*\d+", bn))
    f["name_pipe_suffix"] = "|" in bn
    f["name_dup_token"] = any(x == y for x, y in zip(tb, tb[1:]))
    f["name_double_space"] = "  " in bn
    f["name_subset"] = set(tb) < set(ta)
    f["name_superset"] = set(tb) > set(ta)
    f["name_token_overlap0"] = sn == "LATIN" and not (set(ta) & set(tb)) and not f["name_domain"]
    if sn == "LATIN" and not f["name_domain"]:
        extra_tok.update(set(tb) - set(ta))
        drop_tok.update(set(ta) - set(tb))
    m = re.match(r"^(\W+)", bn.strip())
    if m:
        prefix[m.group(1).strip()] += 1
    m = re.search(r"(\|.*|#\s*\d+|\(.*?\)|\[.*?\])\s*$", bn)
    if m:
        suffix_pat[re.sub(r"\d", "9", re.sub(r"[a-z]+", "w", m.group(1).lower()))[:20]] += 1
    xa, xb = toks(aa), toks(ba)
    f["addr_empty"] = ba.strip() == ""
    f["addr_null_literal"] = bool(re.search(r"\bnull\b", ba.lower()))
    f["addr_nonlatin"] = script_of(ba) != "LATIN"
    f["addr_exact"] = aa == ba
    f["addr_same_token_set"] = set(xa) == set(xb)
    f["addr_reordered"] = set(xa) == set(xb) and xa != xb
    na = [t for t in xa if t.isdigit()]
    nb = [t for t in xb if t.isdigit()]
    f["addr_first_num_equal"] = bool(na and nb and na[0] == nb[0])
    f["addr_first_num_lead0"] = bool(na and nb and na[0] != nb[0] and nb[0].lstrip("0") == na[0])
    f["addr_num_missing"] = bool(na) and not nb and not f["addr_empty"]
    f["addr_all_upper"] = ba.isupper()
    f["country_mismatch"] = a.country != b.country
    f["src"] = tid[:2]
    if not f["addr_empty"]:
        addr_extra.update(set(xb) - set(xa))
    rows.append(f)

df = pd.DataFrame(rows)
lines = [f"# Noise audit on {len(df):,} true (S1, target) pairs from {N:,} sampled S1 records\n",
         "## Frequency of each pattern (share of true pairs)\n", "| pattern | all | S2 | S3 |", "|---|---:|---:|---:|"]
for c in df.columns:
    if c == "src":
        continue
    lines.append(f"| {c} | {df[c].mean():.2%} | {df[df.src=='S2'][c].mean():.2%} | {df[df.src=='S3'][c].mean():.2%} |")
lines += ["\n## Scripts in target names", str(dict(scripts_name.most_common())),
          "\n## Scripts in target addresses", str(dict(scripts_addr.most_common())),
          "\n## Tokens added to names (target has, S1 lacks), top 60", str(extra_tok.most_common(60)),
          "\n## Tokens dropped from names, top 60", str(drop_tok.most_common(60)),
          "\n## Leading junk", str(prefix.most_common(20)),
          "\n## Trailing patterns (w=word, 9=digit)", str(suffix_pat.most_common(20)),
          "\n## Tokens added to addresses, top 60", str(addr_extra.most_common(60))]
ex = df.assign(sid=[l[0] for l in links], tid=[l[1] for l in links])
lines.append("\n## Examples: names with zero token overlap (not native script, not domain)")
for r in ex[ex.name_token_overlap0].head(25).itertuples():
    lines.append(f"- `{s1.loc[r.sid].business_name}` -> `{tg.loc[r.tid].business_name}` | `{s1.loc[r.sid].business_address}` -> `{tg.loc[r.tid].business_address}`")
lines.append("\n## Examples: domain names")
for r in ex[ex.name_domain].head(15).itertuples():
    lines.append(f"- `{s1.loc[r.sid].business_name}` -> `{tg.loc[r.tid].business_name}`")
lines.append("\n## Examples: native-script names")
for r in ex[ex.name_nonlatin].head(15).itertuples():
    lines.append(f"- `{s1.loc[r.sid].business_name}` -> `{tg.loc[r.tid].business_name}` | addr `{tg.loc[r.tid].business_address}`")
out.write_text("\n".join(lines), encoding="utf-8")
print("wrote", out)
