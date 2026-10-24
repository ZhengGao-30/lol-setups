"""Broadcast minimap <-> game coordinates, calibrated per game from the static map itself.

The minimap is an orthographic top view, so pixel = a + s * world on each axis (y flipped). The four
parameters (ax, sx, ay, sy) are found by registering the median minimap image (moving icons vanish in the
median; walls are dark, walkable ground is bright) against the navigation graph's walkable mask, which was
built from V5 positions only (navgraph.py): maximise the correlation between minimap brightness sampled at
each 100-unit cell centre and the cell's walkable indicator. No hand-picked points and no external map assets.
The fixed building positions (data/manifests/building_positions.csv, from V5 BUILDING_KILL events) are an
independent check: projected through the fitted map they must land on the tower and inhibitor icons.

Input: minimap_2fps.mkv, a 276x276 crop of the 1080p broadcast.
"""
import subprocess

import numpy as np
from scipy import ndimage, optimize

SIDE = 276
INIT = (9.0, 0.0175, 269.0, 0.0175)  # LoL Esports 2025 layout: map square ~ pixels 9..270 for world 0..15000


def read_frames(path, ffmpeg="ffmpeg", every=1):
    """RGB frames (n, 276, 276, 3); every=k keeps every k-th 2-fps frame."""
    vf = [] if every == 1 else ["-vf", f"select=not(mod(n\\,{every}))", "-vsync", "0"]
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), *vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, SIDE, SIDE, 3)


def to_pixel(x, y, p):
    ax, sx, ay, sy = p
    return ax + sx * np.asarray(x, float), ay - sy * np.asarray(y, float)


def to_world(px, py, p):
    ax, sx, ay, sy = p
    return (np.asarray(px, float) - ax) / sx, (ay - np.asarray(py, float)) / sy


def _score(p, lum, wx, wy, walk):
    px, py = to_pixel(wx, wy, p)
    v = ndimage.map_coordinates(lum, [py, px], order=1, mode="constant", cval=np.nan)
    ok = np.isfinite(v)
    if ok.sum() < 0.8 * len(v):
        return 1.0
    return -np.corrcoef(v[ok], walk[ok])[0, 1]


def register(median_rgb, walk_mask, cell=100, init=INIT):
    """Fit (ax, sx, ay, sy): coarse grid around init, then Nelder-Mead. Returns (params, correlation)."""
    lum = ndimage.gaussian_filter(median_rgb.astype(float).mean(2), 1.0)
    n = walk_mask.shape[0]
    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    wx, wy, walk = ((ix + 0.5) * cell).ravel(), ((iy + 0.5) * cell).ravel(), walk_mask.astype(float).ravel()
    args = (lum, wx, wy, walk)
    grid = [(ax, s, ay, s) for s in init[1] * np.linspace(0.95, 1.05, 11)
            for ax in init[0] + np.arange(-6, 7) for ay in init[2] + np.arange(-6, 7)]
    start = min(grid, key=lambda p: _score(p, *args))
    r = optimize.minimize(_score, start, args=args, method="Nelder-Mead",
                          options=dict(xatol=1e-5, fatol=1e-7, maxiter=4000))
    return np.asarray(r.x), float(-r.fun)
