"""VOD -> game-time alignment for every cropped game .

For each $PROJ/video_crops/<name>/clock_10fps.mkv:
  fits/clock/<name>.npz      read (clock reading per 10-fps frame, s) and game_time (live game time, NaN elsewhere)
  results/clock/segments.csv one row per constant-offset segment (live or replay)
  results/clock/games.csv    one row per game with the validation diagnostics:
    frac_readable   share of frames with a confident reading
    frac_live       share of frames stamped with live game time
    misread_rate    share of confidently read frames inside live segments that disagree with the segment
    first_live_s    first live game time (the clip must start before the earliest analysis window)
    end_diff_s      last live game time minus Leaguepedia game length: > +2 s means the VOD is a different game,
                    a large negative value means the clip was cut early (the HUD is hidden for the final seconds)
Usage: python -m lolsetup.run_clock --crops $PROJ/video_crops --ffmpeg $PROJ/env/ffmpeg --workers 8
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import clock, io

MODEL = Path(__file__).with_name("clock_digits_lolesports2025.npz")


def vod_name(game_id):
    return game_id.replace("/", "_").replace(" ", "_")


def process(job):
    path, ffmpeg, out = job
    name = path.parent.name
    frames = clock.read_frames(path, ffmpeg)
    read = clock.readings(frames, clock.load_digit_model(MODEL))
    segs = clock.align(read)
    gt = clock.live_game_time(read, segs)
    np.savez_compressed(out / f"{name}.npz", read=read.astype(np.float32), game_time=gt.astype(np.float32))
    t = np.arange(len(read)) / clock.FPS
    inside = np.zeros(len(read), bool)
    for sg in segs:
        if sg["live"]:
            inside |= (t >= sg["vod_start_s"]) & (t <= sg["vod_end_s"])
    n_in = int((inside & np.isfinite(read)).sum())
    diag = {"name": name, "n_frames": len(read), "frac_readable": float(np.isfinite(read).mean()),
            "frac_live": float(np.isfinite(gt).mean()),
            "misread_rate": float(1 - np.isfinite(gt).sum() / n_in) if n_in else np.nan,
            "n_live": sum(s["live"] for s in segs), "n_replay": sum(not s["live"] for s in segs),
            "first_live_s": float(np.nanmin(gt)) if np.isfinite(gt).any() else np.nan,
            "last_live_s": float(np.nanmax(gt)) if np.isfinite(gt).any() else np.nan}
    return diag, [dict(name=name, **s) for s in segs]


def main(crops, ffmpeg, workers):
    out = io.ROOT / "fits" / "clock"
    res = io.RESULTS / "clock"
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(crops).glob("*/clock_10fps.mkv") if p.stat().st_size > 0)
    with Pool(workers) as pool:
        results = pool.map(process, [(p, ffmpeg, out) for p in paths], chunksize=1)
    games = pd.DataFrame([d for d, _ in results])
    segs = pd.DataFrame([s for _, ss in results for s in ss])
    man = pd.read_csv(io.MANIFESTS / "vod_manifest.csv")
    man["name"] = man["GameId"].map(vod_name)
    games = games.merge(man[["name", "GameId", "RiotPlatformGameId", "set", "gamelength_s"]], on="name", how="left")
    games["end_diff_s"] = games["last_live_s"] - games["gamelength_s"]
    games.to_csv(res / "games.csv", index=False)
    segs.to_csv(res / "segments.csv", index=False)
    q = games[["frac_readable", "frac_live", "misread_rate", "end_diff_s"]].describe(percentiles=[.05, .5, .95])
    print(f"games={len(games)} unmatched={games['GameId'].isna().sum()} replay segments={int(games['n_replay'].sum())}")
    print(q.round(4).to_string())
    bad = games[(games["frac_live"] < 0.5) | (games["misread_rate"] > 0.01) | (games["first_live_s"] > 120)
                | (games["end_diff_s"] > 2) | (games["end_diff_s"] < -30) | games["GameId"].isna()]
    print(f"flagged games ({len(bad)}):")
    print(bad[["name", "frac_readable", "frac_live", "misread_rate", "n_live", "n_replay", "first_live_s",
               "end_diff_s"]].to_string(index=False))
    with open(res / "flagged.json", "w") as f:
        json.dump(bad["name"].tolist(), f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main(a.crops, a.ffmpeg, a.workers)
