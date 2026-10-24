"""Game-clock reading and VOD -> game-time alignment from the broadcast clock strip (no external OCR).

Input: clock_10fps.mkv, a 76x28 crop of the 'mm:ss' clock at 10 fps.
  1. binarise, find the 4 digit cells by column projection (the colon is the narrow middle run);
  2. a digit classifier learned from one development strip with a single visually verified anchor: the
     seconds-ones cells are labelled from the anchor and elapsed time (mid-second frames only) and a
     multinomial logistic regression is fitted; the broadcast font is shared, so one model serves all games;
  3. all four digits are read by the classifier -> reading in whole seconds (low-confidence frames dropped);
  4. alignment: at frames where the reading increments by exactly 1, the true game-second boundary lies
     between the two frames, so offset = reading - midpoint time; offsets are grouped into constant
     segments (a jump > JUMP_S marks a pause/replay/cut), summarised by their median, and labelled
     live or replay;
  5. live_game_time: per-frame game time, kept only where the frame's own reading agrees with its live segment.
Output: segments (vod_start_s, vod_end_s, offset_s, live): game_time = vod_time + offset on a live segment.
"""
import subprocess

import numpy as np

W, H, FPS = 76, 28, 10
THRESH = 150
JUMP_S = 0.5
REPLAY_MAX_S, REPLAY_MIN_DROP_S = 60.0, 3.0
CELL_W, CELL_H = 12, 18
DIGIT_W = 8  # typical digit width in the 76x28 strip (px)


def read_frames(path, ffmpeg="ffmpeg"):
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W)


def _split_touching(masks):
    """If one side of the colon has a single blob as wide as two digits (tight kerning, e.g. '31'),
    split it at the column with the least ink in its middle third."""
    if len(masks) != 1:
        return masks
    m = masks[0]
    cols = np.flatnonzero(m.any(0))
    x0, x1 = cols[0], cols[-1] + 1
    if x1 - x0 < 1.4 * DIGIT_W:
        return masks
    lo, hi = x0 + (x1 - x0) // 3, x1 - (x1 - x0) // 3
    cut = lo + int(np.argmin(m[:, lo:hi].sum(0)))
    a, b = m.copy(), m.copy()
    a[:, cut:] = False
    b[:, :cut] = False
    return [a, b]


def segment(img):
    """Return 4 digit bitmaps (CELL_H x CELL_W) or None if the strip does not look like 'mm:ss'.
    Uses 2-D connected components: the clock font is proportional, so adjacent digits (e.g. '31') can
    share columns and column projections would merge them. The colon is two small vertically aligned dots."""
    from scipy import ndimage
    b = img > THRESH
    lab, n = ndimage.label(b, structure=np.ones((3, 3)))
    if n == 0:
        return None
    objs = ndimage.find_objects(lab)
    comps = []
    for k, sl in enumerate(objs, 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int((lab[sl] == k).sum())
        if area >= 3:
            comps.append({"k": k, "sl": sl, "h": h, "w": w, "x0": sl[1].start, "x1": sl[1].stop, "area": area})
    if not comps:
        return None
    hmax = max(c["h"] for c in comps)
    dots = [c for c in comps if c["h"] <= hmax * 0.35]
    tall = [c for c in comps if c["h"] >= hmax * 0.6]
    if len(dots) < 2 or len(tall) < 3:  # 3 allowed: touching digits are split below
        return None
    dots.sort(key=lambda c: c["x0"])
    pair = next(((d1, d2) for d1, d2 in zip(dots, dots[1:]) if abs(d1["x0"] - d2["x0"]) <= 2), None)
    if pair is None:
        return None
    colon_x = (pair[0]["x0"] + pair[0]["x1"]) / 2
    # merge tall components that overlap strongly in x (a digit broken into pieces by the threshold)
    tall.sort(key=lambda c: c["x0"])
    groups = [[tall[0]]]
    for c in tall[1:]:
        g = groups[-1]
        gx1 = max(t["x1"] for t in g)
        if c["x0"] < gx1 - 0.5 * c["w"]:
            g.append(c)
        else:
            groups.append([c])
    left = [np.isin(lab, [t["k"] for t in g]) for g in groups if max(t["x1"] for t in g) <= colon_x]
    right = [np.isin(lab, [t["k"] for t in g]) for g in groups if min(t["x0"] for t in g) >= colon_x]
    left, right = _split_touching(left), _split_touching(right)
    if len(left) != 2 or len(right) != 2:
        return None
    cells = []
    for mask in left + right:
        rows, cols_ = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        sub = mask[rows[0]:rows[-1] + 1, cols_[0]:cols_[-1] + 1]
        cell = np.zeros((CELL_H, CELL_W))
        hh, ww = min(sub.shape[0], CELL_H), min(sub.shape[1], CELL_W)
        cell[:hh, (CELL_W - ww) // 2:(CELL_W - ww) // 2 + ww] = sub[:hh, :ww]
        cells.append(cell)
    return cells


def _cells_with_time_labels(frames, lo, hi, anchor_frame, anchor_reading, phase):
    X, y = [], []
    for k in range(lo, hi):
        if (k % FPS) not in (3, 4, 5, 6):  # mid-second frames: away from cross-fades and +-1 s jitter
            continue
        c = segment(frames[k])
        if c is None:
            continue
        X.append(c[3].ravel())
        y.append(int(np.floor(anchor_reading + phase + (k - anchor_frame) / FPS)) % 10)
    return np.array(X), np.array(y)


def anchor_phase(frames, anchor_frame, change_thresh=6, max_frames=12):
    """Fractional second elapsed at the anchor frame, from the first change of the seconds-ones digit after it.
    (Self-consistency cannot identify the phase: every phase yields internally consistent labels.)"""
    prev = segment(frames[anchor_frame])
    for k in range(anchor_frame + 1, anchor_frame + max_frames + 1):
        cur = segment(frames[k])
        if prev is not None and cur is not None and np.abs(cur[3] - prev[3]).sum() > change_thresh:
            return float(np.clip(1 - (k - anchor_frame) / FPS, 0, 0.95))
        prev = cur
    raise ValueError("no seconds-digit change found right after the anchor; pick another anchor frame")


def learn_digit_model(frames, anchor_frame, anchor_reading, span_s=600, C=0.5):
    """Self-supervised digit classifier from one visually verified anchor reading.
    Seconds-ones cells in [anchor, anchor + span] are labelled floor(anchor_reading + phase + elapsed) mod 10,
    with the phase fixed by the first digit change after the anchor (anchor_phase). A multinomial logistic
    regression on the binary cell pixels is returned as plain arrays (no pickle); held-out agreement on the
    second half of the window is reported as a diagnostic."""
    from sklearn.linear_model import LogisticRegression
    phase = anchor_phase(frames, anchor_frame)
    mid = anchor_frame + int(span_s * FPS / 2)
    hi = min(len(frames), anchor_frame + int(span_s * FPS))
    Xa, ya = _cells_with_time_labels(frames, anchor_frame, mid, anchor_frame, anchor_reading, phase)
    Xb, yb = _cells_with_time_labels(frames, mid, hi, anchor_frame, anchor_reading, phase)
    m = LogisticRegression(max_iter=3000, C=C).fit(Xa, ya)
    acc = float((m.predict(Xb) == yb).mean())
    m = LogisticRegression(max_iter=3000, C=C).fit(np.vstack([Xa, Xb]), np.r_[ya, yb])
    return {"W": m.coef_, "b": m.intercept_, "classes": m.classes_, "phase": phase, "heldout_acc": acc}


def save_digit_model(model, path):
    np.savez(path, W=model["W"], b=model["b"], classes=model["classes"],
             phase=model["phase"], heldout_acc=model["heldout_acc"])


def load_digit_model(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def classify(cell, model):
    """(digit, probability) for one cell under the logistic digit model."""
    z = model["W"] @ cell.ravel() + model["b"]
    p = np.exp(z - z.max()); p /= p.sum()
    i = int(np.argmax(p))
    return int(model["classes"][i]), float(p[i])


def readings(frames, model, min_prob=0.9):
    """Clock reading in seconds per frame (NaN where the strip is not a clean, confidently read clock)."""
    out = np.full(len(frames), np.nan)
    for i, f in enumerate(frames):
        cells = segment(f)
        if cells is None:
            continue
        digs, probs = zip(*(classify(c, model) for c in cells))
        if min(probs) < min_prob or digs[2] > 5:
            continue
        out[i] = 600 * digs[0] + 60 * digs[1] + 10 * digs[2] + digs[3]
    return out


def align(read, fps=FPS, min_n=3):
    """Constant-offset segments from second-boundary frames: game_time = vod_time + offset.
    Runs of agreeing boundary offsets are found greedily; runs with < min_n boundaries (isolated misreads)
    are dropped and neighbours whose offsets agree within JUMP_S are merged. A replay shows an earlier clock
    for a short while and is followed by live play: a segment is a replay iff it lasts <= REPLAY_MAX_S and a
    segment starting within REPLAY_MAX_S after it has an offset >= REPLAY_MIN_DROP_S higher. A pause lowers
    the offset for good and VOD frame-timing drift moves it by < 1 s; neither is a replay."""
    t = np.arange(len(read)) / fps
    idx = np.flatnonzero(np.isfinite(read[1:]) & np.isfinite(read[:-1]) & (np.diff(read) == 1)) + 1
    if len(idx) == 0:
        return []
    off = read[idx] - (t[idx] + t[idx - 1]) / 2
    runs, start = [], 0
    for k in range(1, len(off) + 1):
        if k == len(off) or abs(off[k] - np.median(off[start:k])) > JUMP_S:
            runs.append(np.arange(start, k))
            start = k
    merged = []
    for r in (r for r in runs if len(r) >= min_n):
        if merged and abs(np.median(off[r]) - np.median(off[merged[-1]])) <= JUMP_S:
            merged[-1] = np.r_[merged[-1], r]
        else:
            merged.append(r)
    segs = []
    for r in merged:
        block = off[r]
        segs.append({"vod_start_s": float(t[idx[r[0]] - 1]), "vod_end_s": float(t[idx[r[-1]]]),
                     "offset_s": float(np.median(block)), "n_boundaries": int(len(block)),
                     "mad_s": float(np.median(np.abs(block - np.median(block))))})
    for i, sg in enumerate(segs):
        later = [s2["offset_s"] for s2 in segs[i + 1:] if s2["vod_start_s"] - sg["vod_end_s"] <= REPLAY_MAX_S]
        sg["live"] = not (sg["vod_end_s"] - sg["vod_start_s"] <= REPLAY_MAX_S and later
                          and max(later) - sg["offset_s"] >= REPLAY_MIN_DROP_S)
    return segs


def live_game_time(read, segs, fps=FPS, tol=0.25):
    """Game time per clock frame, NaN unless the frame lies in a live segment AND its own reading agrees
    with that segment's offset (reading = floor(game time), up to +-tol of VOD/clock frame jitter; a misread digit
    is off by >= 1 s). Frames whose clock is hidden or replayed get NaN,
    so a replayed minimap can never be stamped with live time."""
    t = np.arange(len(read)) / fps
    gt = np.full(len(read), np.nan)
    for sg in segs:
        if not sg["live"]:
            continue
        m = (t >= sg["vod_start_s"]) & (t <= sg["vod_end_s"]) & np.isfinite(read)
        g = t[m] + sg["offset_s"]
        ok = (read[m] - g >= -1 - tol) & (read[m] - g <= tol)
        gt[np.flatnonzero(m)[ok]] = g[ok]
    return gt
