"""Our own copy of every format rule in the problem statement (the official
utils/validate_submission.py is also run when it is present)."""
from pathlib import Path


def _read(path, col):
    rows = {}
    problems = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", col]:
            problems.append(f"{Path(path).name}: bad header {header}")
        for n, line in enumerate(f, 2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 1:
                parts.append("")
            if len(parts) != 2:
                problems.append(f"{Path(path).name}:{n}: {len(parts)} columns")
                continue
            sid, ids = parts
            if sid in rows:
                problems.append(f"{Path(path).name}: duplicate row {sid}")
            lst = [x for x in ids.split(",") if x] if ids else []
            if len(lst) != len(set(lst)):
                problems.append(f"{Path(path).name}: duplicate ids in list for {sid}")
            if '"' in ids or " " in ids:
                problems.append(f"{Path(path).name}: quoting/space in list for {sid}")
            rows[sid] = lst
    return rows, problems


def _ids(path):
    with open(path, encoding="utf-8") as f:
        f.readline()
        return {line.split("\t", 1)[0].strip() for line in f if line.strip()}


def check(matching, candidate, test_dir):
    test_dir = Path(test_dir)
    s1 = _ids(test_dir / "test_source1.tsv")
    targets = _ids(test_dir / "test_source2.tsv") | _ids(test_dir / "test_source3.tsv")
    m, p1 = _read(matching, "matched_entity_ids")
    c, p2 = _read(candidate, "candidate_entity_ids")
    problems = p1 + p2
    for name, rows in (("matching", m), ("candidate", c)):
        if set(rows) != s1:
            problems.append(f"{name}: {len(s1 - set(rows))} S1 ids missing, {len(set(rows) - s1)} unknown")
        bad = [x for v in rows.values() for x in v if x not in targets]
        if bad:
            problems.append(f"{name}: {len(bad)} ids not in test S2/S3, e.g. {bad[:3]}")
    not_cand = [(s, x) for s, v in m.items() for x in v if x not in set(c.get(s, []))]
    if not_cand:
        problems.append(f"{len(not_cand)} matched ids are not in candidate list, e.g. {not_cand[:3]}")
    return problems
