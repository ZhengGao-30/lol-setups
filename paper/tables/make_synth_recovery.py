"""Synthetic recovery of HASSE: data are drawn from the model itself (three planted patterns, 600 setups, three
snapshots, K=3, five restarts; plus a variant in which the last snapshot is missing in 40% of setups), HASSE is
refitted, and the recovered centre paths, rigidity, inertia and first-snapshot assignment are compared with the
planted values. Runs the project's own check (../code/tests/test_hasse_recovery.py), about two minutes, and caches the
parsed output so that repeated make_all.py runs do not refit."""
import ast
import contextlib
import io
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factory"))
from common import *  # noqa: E402,F401,F403

CACHE_FILE = BUILD / "synthetic_recovery.json"


def canon(path):
    """A centre path as a tuple of frozensets of blocks (block order does not matter)."""
    return tuple(frozenset(frozenset(b.strip()) for b in p.split("|")) for p in path)


def run():
    if str(PROJ / "code") not in sys.path:
        sys.path.insert(0, str(PROJ / "code"))
    from tests import test_hasse_recovery as t
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t.main()
        first = buf.getvalue()
        t.main_missing(share=0.4)
    out = buf.getvalue()
    return {"main": first, "all": out, "true": [[t.lab(t.P5[m]) for m in path] for path in t.TRUE],
            "theta": t.THETA, "iota": t.IOTA}


def parse(text):
    lls = [float(x) for x in re.findall(r"loglik (-?[\d.]+)", text)]
    ari = [float(x) for x in re.findall(r"ARI[^:]*: (-?[\d.]+)", text)]
    pats = [ast.literal_eval(l) for l in text.splitlines() if l.startswith("{'pattern'")]
    return lls, ari, pats


def build(N):
    if CACHE_FILE.exists():
        res = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    else:
        res = run()
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(res), encoding="utf-8")
    lls, ari, pats_all = parse(res["all"])
    pats = pats_all[:3]
    truth = [canon(p) for p in res["true"]]
    err_t, err_i, ok = [], [], 0
    for p in pats:
        k = [i for i, tp in enumerate(truth) if tp == canon(p["centre_path"])]
        if k:
            ok += 1
            err_t.append(abs(p["theta"] - res["theta"][k[0]]))
            err_i.append(abs(p["iota"] - res["iota"][k[0]]))
    N["synthN"] = "600"
    N["synthLLspread"] = f(max(lls) - min(lls), 1)
    N["synthPathsOK"] = f"{ok} of {len(truth)}"
    N["synthThetaErr"] = f(max(err_t), 2) if err_t else "n/a"
    N["synthIotaErr"] = f(max(err_i), 2) if err_i else "n/a"
    N["synthAri"] = f(ari[0], 3)
    N["synthAriMissing"] = f(ari[1], 3)
    return "synthetic recovery"


if __name__ == "__main__":
    run_main(build)
