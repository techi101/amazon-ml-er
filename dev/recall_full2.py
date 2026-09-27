"""DEV ONLY - like recall_full.py, but reuses the cleaned-frame cache and recomputes only the state
column (so state-matching fixes can be tested without the 17-minute cleaning)."""
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).parent
code = (HERE / "recall_full.py").read_text(encoding="utf-8").replace("sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=\"utf-8\")", "")
hook = '''
_orig_infer = infer_states


def _state_only(raw):
    out = []
    for t in fold_text(raw)[0].tolist():
        t = RE_NULL.sub(" ", RE_POBOX.sub(" ", t))
        st = ""
        for c in t.split(","):
            c = c.strip()
            if c:
                st = _state_of(c)
                if st:
                    break
        out.append(st)
    return pd.array(out, dtype="string[pyarrow]")


def infer_states(A, B, country, **kw):
    A["ad_state"] = _state_only(s1.business_address)
    B["ad_state"] = _state_only(t.business_address)
    log("  states recomputed with the current state matcher")
    _orig_infer(A, B, country, **kw)
'''
code = code.replace('log(f"{COUNTRY}: {len(s1):,} S1, {len(t):,} S2/S3 loaded")',
                    'log(f"{COUNTRY}: {len(s1):,} S1, {len(t):,} S2/S3 loaded")\n' + hook)
exec(compile(code, "recall_full2", "exec"))
