"""Paths and loaders shared by every stage. Works locally and on a compute server ($PROJ layout)."""
import gzip
import json
import os
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
# On a compute server the inputs can live in a local temp dir (PROJ_DATA) while outputs go to
# another location (PROJ_OUT); locally both default to the repository.
DATA = Path(os.environ.get("PROJ_DATA", _REPO / "data"))
ROOT = Path(os.environ.get("PROJ_OUT", _REPO))  # outputs: ROOT / "fits", ROOT / "results"
V5 = DATA / "raw" / "v5"
MANIFESTS = DATA / "manifests"
RESULTS = ROOT / "results"

# Participant order in pro V5 JSON is role order (verified 840/840 against Leaguepedia roles);
# Riot's teamPosition field is unreliable in tournament data and is never used.
ROLES = ("Top", "Jungle", "Mid", "Bot", "Support")
ROLE_SHORT = ("T", "J", "M", "B", "S")
TEAM_IDS = {100: range(1, 6), 200: range(6, 11)}


def load_v5(gid, kind="timeline"):
    with gzip.open(V5 / kind / f"{gid}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def core_games():
    """Core video set: Worlds 2025 + MSI 2025 (one objective-rule regime, post-25.09)."""
    frames = [pd.read_csv(MANIFESTS / n) for n in ("core_worlds2025.csv", "core_msi2025.csv")]
    return pd.concat(frames, ignore_index=True)


def played_games():
    """Every played LCK + international game 2023-2026 with a Riot id (manifest only; no file check)."""
    g = pd.read_csv(MANIFESTS / "games_all.csv", dtype={"Patch": str})
    return g[g["Played"] & g["RiotPlatformGameId"].notna()]


def archive_games():
    """played_games() restricted to games whose V5 timeline is on disk (needed when reading timelines)."""
    g = played_games()
    return g[g["RiotPlatformGameId"].map(lambda i: (V5 / "timeline" / f"{i}.json.gz").exists())]


def spawns():
    return pd.read_csv(MANIFESTS / "objective_spawns.csv")


def patch_num(p):
    """Leaguepedia patch string -> sortable number. '25.1' means 25.10 and '25.2' means 25.20 (trailing zero
    lost in the source); season-start patches '25.S1.x' map to 25.00."""
    p = str(p)
    if p in ("nan", "None", ""):
        return float("nan")
    major, minor = p.split(".", 1)
    if minor.startswith("S"):
        return float(major)
    minor = minor if len(minor) == 2 else minor + "0"
    return float(f"{major}.{minor}")


def rule_regime_2509(games):
    """Games under the same objective rules as the core set: 2025, patch >= 25.09 (one grub camp of 3 at
    8:00, herald 15:00, Atakhan 20:00, baron 25:00). Verified from data: 489 games, <= 3 grub kills, first
    grub kill >= 8:04; the 265 earlier 2025 games have up to 6 grub kills from 6:04."""
    pn = games["Patch"].map(patch_num)
    return games[(games["Year"] == 2025) & (pn >= 25.09)]
