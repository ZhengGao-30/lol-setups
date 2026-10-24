"""Track one cropped game (one array task per game).

Inputs: video_crops/<name>/minimap_2fps.mkv, fits/clock/<name>.npz (run_clock), the V5 timeline, the navgraph.
Outputs:
  fits/tracks/<name>.parquet        production track, all V5 minutes pinned (k, game_s, pid, px, py, x, y, logit, state)
  fits/tracks/<name>_meta.json      minimap map (ax, sx, ay, sy), registration correlation, per-round detector info
  results/tracker/<name>_eval.csv   validation: templates and pins from even minutes only, error at odd minutes
  results/tracker/<name>_formation.csv  validation: formation partition from the track vs V5 at odd minutes
Usage: python -m lolsetup.run_tracker --name <name> --crops $PROJ/video_crops --ffmpeg $PROJ/env/ffmpeg
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import io, minimap, tracker
from .navgraph import NavGraph
from .run_clock import vod_name


def main(name, crops, ffmpeg):
    out, res = io.ROOT / "fits" / "tracks", io.RESULTS / "tracker"
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(io.MANIFESTS / "vod_manifest.csv")
    gid = man.loc[man["GameId"].map(vod_name) == name, "RiotPlatformGameId"].iloc[0]
    frames = minimap.read_frames(f"{crops}/{name}/minimap_2fps.mkv", ffmpeg)
    clk = np.load(io.ROOT / "fits" / "clock" / f"{name}.npz")
    fgt = tracker.frame_game_time(len(frames), clk["game_time"].astype(float))
    nav = NavGraph()
    p, corr = minimap.register(np.median(frames[::20], 0), nav.mask)
    a, deaths = tracker.v5_anchors(io.load_v5(gid))
    mins = sorted(a["minute"].unique())
    even, odd = [m for m in mins if m % 2 == 0], [m for m in mins if m % 2 == 1]
    tr_v, info_v = tracker.track(frames, fgt, p, a, deaths, nav, pin_minutes=even)
    ev = tracker.evaluate(tr_v, a, fgt, odd)
    ev.insert(0, "name", name)
    ev.to_csv(res / f"{name}_eval.csv", index=False)
    fa = tracker.formation_agreement(tr_v, a, fgt, nav, odd)
    fa.insert(0, "name", name)
    fa.to_csv(res / f"{name}_formation.csv", index=False)
    tr, info = tracker.track(frames, fgt, p, a, deaths, nav, pin_minutes=mins)
    tr.to_parquet(out / f"{name}.parquet")
    meta = {"name": name, "RiotPlatformGameId": gid, "affine": p.tolist(), "register_corr": corr,
            "production": info, "validation": info_v, "live_frames": int(np.isfinite(fgt).sum()),
            "frames": len(frames)}
    with open(out / f"{name}_meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=float)
    e = ev["err"]
    print(f"{name}: corr={corr:.3f} affine={np.round(p, 5).tolist()} held-out n={len(ev)} "
          f"median={e.median():.0f} p90={e.quantile(.9):.0f} <600={(e < 600).mean():.3f} "
          f"formation exact={fa['exact'].mean():.3f} rand={fa['rand'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    a = ap.parse_args()
    main(a.name, a.crops, a.ffmpeg)
