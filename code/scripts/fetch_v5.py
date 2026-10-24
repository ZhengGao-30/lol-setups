"""Fetch Riot match-v5-format postgame + timeline JSON for pro games from Leaguepedia.

Leaguepedia stores them in the 'V5 data' namespace (id 10032), keyed by RiotPlatformGameId:
    V5 data:<RiotPlatformGameId>            -> postgame summary (participants, stats)
    V5 data:<RiotPlatformGameId>/Timeline   -> 60 s frames with positions + event stream
Uses the MediaWiki API (index.php?action=raw sits behind a Cloudflare challenge; do not bypass it).
Saves gzip-compressed JSON; resumable (skips files already on disk).

Usage:
    python scripts/fetch_v5.py data/manifests/v5_targets.csv [--limit N]
The targets CSV needs a column RiotPlatformGameId.
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "v5"
API = "https://lol.fandom.com/api.php"
UA = "LoLGraphTheoryResearch/0.1 (academic research; polite crawler)"
SLEEP = 2.0
BATCH = 5


def dest(gid, kind):
    return OUT / kind / f"{gid}.json.gz"


def query(titles, tries=6):
    params = {
        "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "format": "json", "formatversion": 2, "titles": "|".join(titles),
    }
    for k in range(tries):
        try:
            r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=120)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code}; retrying", file=sys.stderr)
        except (requests.RequestException, ValueError) as e:
            print(f"  {type(e).__name__}: {e}; retrying", file=sys.stderr)
        time.sleep(SLEEP * (2 ** k))
    raise RuntimeError(f"giving up on {titles[:2]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ids = (pd.read_csv(args.targets)["RiotPlatformGameId"].dropna().astype(str).str.strip())
    ids = [i for i in dict.fromkeys(ids) if i]
    if args.limit:
        ids = ids[: args.limit]
    for kind in ("postgame", "timeline"):
        (OUT / kind).mkdir(parents=True, exist_ok=True)

    # MediaWiki normalises '_' to ' ' in titles; map back to our ids
    todo = []
    for gid in ids:
        if not dest(gid, "postgame").exists():
            todo.append((f"V5 data:{gid}", gid, "postgame"))
        if not dest(gid, "timeline").exists():
            todo.append((f"V5 data:{gid}/Timeline", gid, "timeline"))
    print(f"{len(ids)} games, {len(todo)} pages to fetch")

    missing = []
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        by_title = {t.replace("_", " "): (gid, kind) for t, gid, kind in chunk}
        data = query([t for t, _, _ in chunk])
        for page in data.get("query", {}).get("pages", []):
            gid, kind = by_title.get(page["title"], (None, None))
            if gid is None:
                continue
            if page.get("missing") or "revisions" not in page:
                missing.append((gid, kind))
                continue
            content = page["revisions"][0]["slots"]["main"]["content"]
            json.loads(content)  # validate before saving
            with gzip.open(dest(gid, kind), "wt", encoding="utf-8") as f:
                f.write(content)
        done = min(b + BATCH, len(todo))
        if done % 100 < BATCH or done == len(todo):
            print(f"  {done}/{len(todo)} pages, {len(missing)} missing")
        time.sleep(SLEEP)

    if missing:
        pd.DataFrame(missing, columns=["RiotPlatformGameId", "kind"]).to_csv(
            OUT / "_missing.csv", mode="a", index=False, header=not (OUT / "_missing.csv").exists())
    print(f"done; {len(missing)} missing pages logged")


if __name__ == "__main__":
    main()
