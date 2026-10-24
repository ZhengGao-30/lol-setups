"""Download the in-game segment of each broadcast VOD in data/manifests/vod_manifest.csv.

Video only (the analysis needs the minimap + HUD, not commentary). For per-game uploads
(<= MAX_FULL_H hours) the whole video is fetched with yt-dlp's parallel fragment downloader and
the game segment is cut locally with a lossless stream copy (fast). Multi-hour livestreams fall
back to yt-dlp section downloads (slow, ~real time). Cuts snap to the keyframe before the start,
so a file can begin a few seconds early; exact alignment is done later by clock OCR against
Riot V5 events. Footage stays local; publish derived coordinates, never video.

Requires:  pip install yt-dlp imageio-ffmpeg

Usage:
    python scripts/download_vods.py --set core_worlds2025 --limit 3     # pilot
    python scripts/download_vods.py --set core                          # Worlds 2025 + MSI 2025
    python scripts/download_vods.py --set contrast                      # off-rule contrast games
    python scripts/download_vods.py --set all --height 1080
"""
import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pandas as pd
import yt_dlp
from yt_dlp.utils import download_range_func

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "vod_manifest.csv"
OUT = ROOT / "data" / "video"
TMP = OUT / "_full"
LOG = OUT / "_download_log.csv"
PRE_ROLL_S = 5
MAX_FULL_H = 3.0


def ffmpeg_dir():
    # yt-dlp only recognises a binary literally named ffmpeg(.exe); imageio-ffmpeg ships a versioned name
    d = ROOT / "tools" / "ffmpeg"
    exe = d / "ffmpeg.exe"
    if not exe.exists():
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), exe)
    # the partial-download availability check ignores ffmpeg_location and only searches PATH
    os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    return d


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_opts(height, ffmpeg):
    return {
        "format": f"bv*[height<={height}][ext=mp4]/bv*[height<={height}]",
        "ffmpeg_location": ffmpeg,
        "merge_output_format": "mp4",
        "concurrent_fragment_downloads": 8,
        "quiet": True,
        "no_warnings": True,
        "retries": 10,
        "fragment_retries": 10,
    }


def fetch(url, name, start, end, height, ffmpeg):
    """Return (path, mode, height)."""
    with yt_dlp.YoutubeDL(base_opts(height, ffmpeg)) as ydl:
        info = ydl.extract_info(url, download=False)
    duration = info.get("duration") or 0
    target = OUT / f"{name}.mp4"
    if duration and duration <= MAX_FULL_H * 3600:
        TMP.mkdir(parents=True, exist_ok=True)
        opts = base_opts(height, ffmpeg) | {"outtmpl": str(TMP / f"{name}.%(ext)s")}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        full = next(TMP.glob(f"{name}.*"))
        cmd = [str(Path(ffmpeg) / "ffmpeg.exe"), "-v", "error", "-y", "-ss", str(start), "-to", str(end),
               "-i", str(full), "-c", "copy", "-an", str(target)]
        subprocess.run(cmd, check=True)
        full.unlink()
        return target, "full+cut", info.get("height")
    opts = base_opts(height, ffmpeg) | {
        "outtmpl": str(OUT / f"{name}.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return next(OUT.glob(f"{name}.mp4")), "sections", info.get("height")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core", choices=["core", "contrast", "core_worlds2025", "core_msi2025",
                                                       "contrast_worlds2026", "contrast_t1_2023_2024", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--count", action="store_true", help="only print how many games are still missing")
    args = ap.parse_args()

    m = pd.read_csv(MANIFEST)
    if args.set in ("core", "contrast"):
        m = m[m["set"].str.startswith(args.set)]
    elif args.set != "all":
        m = m[m["set"] == args.set]
    m = m.dropna(subset=["youtube_id", "start_s", "end_s"])
    if args.limit:
        m = m.head(args.limit)
    OUT.mkdir(parents=True, exist_ok=True)
    ffmpeg = str(ffmpeg_dir())
    todo = sum(not (OUT / f"{g.replace('/', '_').replace(' ', '_')}.mp4").exists() for g in m["GameId"])
    print(f"remaining: {todo} of {len(m)}", flush=True)
    if args.count:
        return

    for _, g in m.iterrows():
        name = g["GameId"].replace("/", "_").replace(" ", "_")
        if (OUT / f"{name}.mp4").exists():
            continue
        start = max(0, int(g["start_s"]) - PRE_ROLL_S)
        end = int(g["end_s"])
        url = f"https://www.youtube.com/watch?v={g['youtube_id']}"
        status, path, mode, height = "ok", None, None, None
        try:
            path, mode, height = fetch(url, name, start, end, args.height, ffmpeg)
        except Exception as e:  # keep going; failures are logged and can be retried
            status = f"error: {type(e).__name__}: {str(e)[:200]}"
            for p in list(TMP.glob(f"{name}.*")) + list(OUT.glob(f"{name}.*")):
                try:
                    p.unlink(missing_ok=True)
                except OSError:  # Windows: a fragment can still be held open; the next run overwrites it
                    pass
        row = {
            "GameId": g["GameId"], "set": g["set"], "youtube_id": g["youtube_id"],
            "requested_start_s": start, "requested_end_s": end, "mode": mode, "height": height,
            "file": path.name if path else None, "bytes": path.stat().st_size if path else None,
            "sha256": sha256(path) if path else None, "status": status,
        }
        print(f"{status[:60]:<60} {mode} {name} {row['bytes']}", flush=True)
        pd.DataFrame([row]).to_csv(LOG, mode="a", index=False, header=not LOG.exists())


if __name__ == "__main__":
    main()
