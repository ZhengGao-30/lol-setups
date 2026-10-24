"""Minimap champion tracker at 2 fps, anchored on the game's own V5 frames (no external detector weights,
no icon assets).

Per game (inputs: minimap_2fps.mkv, the clock alignment from run_clock, the V5 timeline):
  1. time base: minimap frame k shows VOD time k/2; it gets game time k/2 + offset of the nearest live clock
     frame if that frame is within LIVE_WIN_S (replays and a hidden HUD -> no game time, frame unused);
  2. calibration: minimap.register on the median frame (per game);
  3. detector: one linear filter per participant, the LDA direction w_c = S^-1 (mu_c - mu_bg) with S the
     (ridge-regularised) covariance of random minimap patches and mu_c the mean icon patch at V5 anchors
     (alive, unoccluded, re-centred within +-SHIFT px). Static map features are part of the background
     distribution and are whitened away. Self-training: after each tracking pass the class means are
     re-estimated from confident, unoccluded detections of the track (ROUNDS times; no V5 positions beyond
     the template minutes are used);
  4. observation model, fitted on the anchor frames (peaks within TRUE_PX of the V5 position vs all other top-K
     peaks): a logistic calibration gives a peak's posterior log-odds, log LR = log-odds - prior log-odds, and
     p_det = share of anchors with a true peak among the top K. A detection at peak j emits log LR(s_j) + log p_det;
     a covered or missed icon emits log(1 - p_det);
  5. tracking: per participant, Viterbi over {top-K peaks, holds, dead}. A living champion is always drawn on
     the broadcast minimap but can be covered by another icon, so 'hold' states carry the HOLD_K best previous
     positions forward; 'dead' is allowed only inside V5 death windows. Travel beyond V_MAX * dt + SLACK on the
     navigation graph costs BETA nats per 1000 units, capped at JUMP (teleport); a move into the own fountain
     area (recall, respawn) costs at most RECALL. Pins: at pin minutes the state is the V5 position.
Validation (pin and template minutes = even minutes): error of the track at held-out odd-minute V5 positions.
"""
import numpy as np
import pandas as pd
from scipy import fft, ndimage

from . import clock, io, minimap, setups

FPS = 2
P = 25  # filter side (px); icons are ~25 px across (ring included) in the 276-px minimap
SHIFT = 3  # re-centring search for anchor patches (px)
OCCLUDE_PX = 22
TRUE_PX = 4
K = 6
NMS = 9
N_BG = 20000
LAM = 0.05  # ridge, relative to the mean patch variance
MIN_LOGIT = -6.0
LIVE_WIN_S = 2.0
MATCH_S = 0.3  # an anchor is matched to a frame within this many game seconds
V_MAX, SLACK = 500.0, 150.0  # units/s, units
BETA, JUMP = 20.0, 12.0  # nats per 1000 units of excess travel; cap
RECALL, FOUNTAIN_R = 3.0, 1500.0  # nats for any move into the own fountain area (recall, respawn)
HOLD_K = 4
ABSENT_ALIVE = 25.0  # nats: 'dead' outside a death window (effectively forbidden)
DEATH_GUARD_MS = 1000
RESPAWN_LO, RESPAWN_HI = 0.8, 1.2  # certain-dead / possibly-dead fractions of the approximate death timer
MAP_LO, MAP_HI = 8, 270  # map square in the crop
ROUNDS = 2  # self-training rounds
HARVEST_LOGIT, HARVEST_CAP = 3.0, 400  # emission (log LR + log p_det) threshold


def frame_game_time(n, clock_gt):
    """Game time of each minimap frame (NaN unless a live clock frame lies within LIVE_WIN_S)."""
    t = np.arange(n) / FPS
    tc = np.arange(len(clock_gt)) / clock.FPS
    j = np.flatnonzero(np.isfinite(clock_gt))
    out = np.full(n, np.nan)
    if len(j) == 0:
        return out
    pos = np.clip(np.searchsorted(tc[j], t), 1, len(j) - 1)
    left, right = j[pos - 1], j[pos]
    near = np.where(np.abs(tc[left] - t) <= np.abs(tc[right] - t), left, right)
    ok = np.abs(tc[near] - t) <= LIVE_WIN_S
    out[ok] = t[ok] + (clock_gt[near] - tc[near])[ok]
    return out


def frame_at(fgt, g):
    """Index of the live frame nearest to game time g (or -1 if none within MATCH_S)."""
    kk = np.flatnonzero(np.isfinite(fgt))
    pos = np.clip(np.searchsorted(fgt[kk], g), 1, len(kk) - 1)
    a, b = kk[pos - 1], kk[pos]
    k = np.where(np.abs(fgt[a] - g) <= np.abs(fgt[b] - g), a, b)
    return np.where(np.abs(fgt[k] - g) <= MATCH_S, k, -1)


def v5_anchors(tl):
    """Per-minute V5 positions (minute, t_ms, pid, x, y, alive, level) and kill events per victim."""
    frames = tl["frames"] if "frames" in tl else tl["info"]["frames"]
    deaths = {}
    for fr in frames:
        for e in fr["events"]:
            if e["type"] == "CHAMPION_KILL":
                deaths.setdefault(e["victimId"], []).append((e["timestamp"], e["position"]["x"], e["position"]["y"]))
    rows = []
    for m, fr in enumerate(frames):
        pf = [fr["participantFrames"][str(i)] for i in range(1, 11)]
        alive = setups.alive_flags(pf, range(1, 11), fr["timestamp"], deaths)
        rows += [(m, fr["timestamp"], i + 1, p["position"]["x"], p["position"]["y"], bool(alive[i]), p["level"])
                 for i, p in enumerate(pf)]
    a = pd.DataFrame(rows, columns=["minute", "t_ms", "pid", "x", "y", "alive", "level"])
    return a, deaths


def death_windows(a, deaths):
    """pid -> list of (from_s, certain_end_s, possible_end_s) from V5 kills and the approximate death timer."""
    out = {}
    for pid, ds in deaths.items():
        lv = a[a["pid"] == pid].set_index("t_ms")["level"]
        for tk, _, _ in ds:
            before = lv[lv.index <= tk]
            r = setups.respawn_ms(int(before.iloc[-1]) if len(before) else 1, tk)
            out.setdefault(pid, []).append(((tk + DEATH_GUARD_MS) / 1000, (tk + RESPAWN_LO * r) / 1000,
                                            (tk + RESPAWN_HI * r) / 1000))
    return out


def _anchor_rows(a, fgt, p, minutes):
    use = a[a["minute"].isin(minutes) & a["alive"]].copy()
    use["k"] = frame_at(fgt, use["t_ms"].to_numpy() / 1000)
    use["px"], use["py"] = minimap.to_pixel(use["x"], use["y"], p)
    return use[use["k"] >= 0]


def anchor_windows(frames, fgt, a, p, minutes):
    """pid -> array of (P + 2*SHIFT)^2 RGB windows centred on clean (alive, unoccluded, inside) anchors."""
    use = _anchor_rows(a, fgt, p, minutes)
    h = P // 2 + SHIFT
    out = {}
    for pid in range(1, 11):
        wins = []
        for r in use[use["pid"] == pid].itertuples():
            others = use[(use["minute"] == r.minute) & (use["pid"] != pid)]
            if (np.hypot(others["px"] - r.px, others["py"] - r.py) < OCCLUDE_PX).any():
                continue
            cx, cy = int(round(r.px)), int(round(r.py))
            if MAP_LO + h <= cx <= MAP_HI - h and MAP_LO + h <= cy <= MAP_HI - h:
                wins.append(frames[r.k, cy - h:cy + h + 1, cx - h:cx + h + 1].astype(np.float32))
        out[pid] = np.array(wins)
    return out


def _ncc(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9))


def class_mean(wins, iters=3, keep=0.75):
    """Mean icon patch: iteratively re-centre each window within +-SHIFT on the current mean and average the
    best-matching `keep` share (drops residual occlusions, pings and the camera box)."""
    if len(wins) == 0:
        return None, 0
    crop = lambda w, dy, dx: w[SHIFT + dy:SHIFT + dy + P, SHIFT + dx:SHIFT + dx + P]
    mean = np.median(np.array([crop(w, 0, 0) for w in wins]), 0)
    shifts = [(dy, dx) for dy in range(-SHIFT, SHIFT + 1) for dx in range(-SHIFT, SHIFT + 1)]
    for _ in range(iters):
        best = [max((_ncc(crop(w, dy, dx), mean), dy, dx) for dy, dx in shifts) for w in wins]
        aligned = np.array([crop(w, dy, dx) for w, (_, dy, dx) in zip(wins, best)])
        score = np.array([b[0] for b in best])
        top = score >= np.quantile(score, 1 - keep)
        mean = aligned[top].mean(0)
    return mean, int(top.sum())


def background_stats(frames, fgt, n=N_BG, seed=0):
    """Mean and covariance of random P x P patches from live frames (no labels used)."""
    rng = np.random.default_rng(seed)
    live = np.flatnonzero(np.isfinite(fgt))
    h = P // 2
    ks = rng.choice(live, n)
    cy = rng.integers(MAP_LO + h, MAP_HI - h, n)
    cx = rng.integers(MAP_LO + h, MAP_HI - h, n)
    X = np.array([frames[k, y - h:y + h + 1, x - h:x + h + 1] for k, y, x in zip(ks, cy, cx)], np.float32)
    X = X.reshape(n, -1)
    mu = X.mean(0)
    Xc = X - mu
    return mu, (Xc.T @ Xc) / (n - 1)


def lda_filters(means, mu, cov, lam=LAM):
    """w_c = (S + lam * mean_var * I)^-1 (mu_c - mu_bg), b_c = -w_c . (mu_c + mu_bg) / 2 (missing class: never fires)."""
    d = len(mu)
    S = cov + lam * np.trace(cov) / d * np.eye(d)
    W, b = np.zeros((len(means), P, P, 3), np.float32), np.zeros(len(means))
    for i, m in enumerate(means):
        if m is None:
            b[i] = -1e3
            continue
        w = np.linalg.solve(S, m.ravel() - mu)
        W[i], b[i] = w.reshape(P, P, 3), -w @ (m.ravel() + mu) / 2
    return W, b


class LinearMatcher:
    """Scores w_c . patch(x, y) + b_c of linear filters at every pixel, via FFT ('same' alignment)."""

    def __init__(self, W, b, shape=(minimap.SIDE, minimap.SIDE)):
        self.H, self.Wd = shape
        self.fs = (self.H + P - 1, self.Wd + P - 1)
        self.wh = fft.rfft2(np.asarray(W, np.float32)[:, ::-1, ::-1], self.fs, axes=(1, 2)).astype(np.complex64)
        self.b = np.asarray(b, np.float32)

    def scores(self, imgs):
        """(B, n_filters, H, W) for a batch of frames (B, H, W, 3)."""
        ih = fft.rfft2(np.asarray(imgs, np.float32), self.fs, axes=(1, 2), workers=-1)
        s = fft.irfft2(np.einsum("tijc,bijc->btij", self.wh, ih), self.fs, axes=(2, 3), workers=-1)
        c = P // 2
        return s[:, :, c:c + self.H, c:c + self.Wd] + self.b[None, :, None, None]


def peaks(sc, k=K, floor=-np.inf):
    """Top-k local maxima (py, px), scores of one score map inside the map square (corners included: fountains
    and bases sit there; the FFT scores are zero-padded beyond the crop)."""
    lo, hi = MAP_LO, MAP_HI
    sub = sc[lo:hi, lo:hi]
    mx = ndimage.maximum_filter(sub, size=NMS)
    cand = np.flatnonzero((sub == mx) & (sub >= floor))
    top = cand[np.argsort(sub.ravel()[cand])[::-1][:k]]
    w = sub.shape[1]
    return np.stack([top // w + lo, top % w + lo]).T, sub.ravel()[top]


def build_detector(means, frames, fgt, a, p, minutes, bg, batch=32):
    """LDA filters for the given class means, then a logistic calibration of peak scores on the anchor minutes
    (peaks within TRUE_PX of the V5 position vs all other peaks)."""
    from sklearn.linear_model import LogisticRegression
    W, b = lda_filters(means, *bg)
    mt = LinearMatcher(W, b)
    use = _anchor_rows(a, fgt, p, minutes)
    X, y, n_peaks = [], [], []
    ks = use["k"].unique()
    for i in range(0, len(ks), batch):
        sc = mt.scores(frames[ks[i:i + batch]])
        for j, k in enumerate(ks[i:i + batch]):
            for r in use[use["k"] == k].itertuples():
                xy, s = peaks(sc[j, r.pid - 1])
                X += list(s)
                y += list(np.hypot(xy[:, 1] - r.px, xy[:, 0] - r.py) <= TRUE_PX)
                n_peaks.append(len(s))
    X, y = np.array(X)[:, None], np.array(y)
    lr = LogisticRegression().fit(X, y)
    prior = float(np.log(y.mean() / (1 - y.mean())))
    p_det = float(np.mean([any(hit) for hit in np.split(y, np.cumsum(n_peaks)[:-1])]))
    # observation model: detected at peak j -> log LR(s_j) + log p_det, with log LR = posterior log-odds - prior;
    # covered or missed -> log(1 - p_det)
    return mt, {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "prior": prior, "p_det": p_det}


def harvest_windows(frames, tr, min_logit=HARVEST_LOGIT, cap=HARVEST_CAP, seed=0):
    """pid -> windows around confident detections of a track (logit >= min_logit, no other tracked champion
    within OCCLUDE_PX); at most `cap` per participant, spread uniformly over the game. Uses no V5 positions."""
    rng = np.random.default_rng(seed)
    h = P // 2 + SHIFT
    pos = tr.pivot(index="k", columns="pid", values=["px", "py"])
    out = {}
    for pid in range(1, 11):
        d = tr[(tr["pid"] == pid) & (tr["state"] == "det") & (tr["logit"] >= min_logit)]
        d = d[d["px"].between(MAP_LO + h, MAP_HI - h) & d["py"].between(MAP_LO + h, MAP_HI - h)]
        if len(d) == 0:
            out[pid] = np.zeros((0, P + 2 * SHIFT, P + 2 * SHIFT, 3), np.float32)
            continue
        others = [q for q in range(1, 11) if q != pid]
        dx = pos["px"].loc[d["k"], others].to_numpy() - d["px"].to_numpy()[:, None]
        dy = pos["py"].loc[d["k"], others].to_numpy() - d["py"].to_numpy()[:, None]
        d = d[~(np.hypot(dx, dy) < OCCLUDE_PX).any(1)]
        if len(d) > cap:
            d = d.iloc[np.sort(rng.choice(len(d), cap, replace=False))]
        out[pid] = np.array([frames[r.k, int(round(r.py)) - h:int(round(r.py)) + h + 1,
                                    int(round(r.px)) - h:int(round(r.px)) + h + 1] for r in d.itertuples()], np.float32)
    return out


def candidates(frames, fgt, mt, cal, batch=32):
    """For each live frame: per participant (peaks (k, 2) as (px, py), detection emissions log LR + log p_det)."""
    out = {}
    live = np.flatnonzero(np.isfinite(fgt))
    shift = -cal["prior"] + np.log(cal["p_det"])
    for i in range(0, len(live), batch):
        ks = live[i:i + batch]
        sc = cal["a"] * mt.scores(frames[ks]) + cal["b"]
        for j, k in enumerate(ks):
            out[k] = []
            for c in range(sc.shape[1]):
                xy, s = peaks(sc[j, c], floor=MIN_LOGIT)
                out[k].append((xy[:, ::-1].astype(float), s + shift))
    return out


def viterbi(ks, g, cand, nav, p, dead_certain, dead_possible, pins, hold_em, fountain_node):
    """One participant. Returns per frame: position (px, py) (NaN when dead), emission and state label."""
    V = None
    hist = []  # per t: (positions (n, 2), nodes (n,), emissions (n + 1,), labels, backpointers)
    for t, k in enumerate(ks):
        if k in pins:
            pos, em, lab = np.array([pins[k]], float), np.array([0.0]), ["pin"]
        else:
            pos, em = cand[k]
            lab = ["det"] * len(em)
            if V is not None:  # holds: the HOLD_K best previous positions, carried forward with no evidence
                order = np.argsort(V[:-1])[::-1][:HOLD_K]
                order = order[np.isfinite(V[:-1][order])]
                pos = np.vstack([pos, hist[-1][0][order]])
                em = np.r_[em, np.full(len(order), hold_em)]
                lab = lab + ["hold"] * len(order)
        wx, wy = minimap.to_world(pos[:, 0], pos[:, 1], p)
        nodes = nav.node(wx, wy) if len(pos) else np.zeros(0, int)
        e = np.r_[em, 0.0 if dead_certain[t] else hold_em if dead_possible[t] else -ABSENT_ALIVE]
        if dead_certain[t]:
            e[:-1] = -np.inf
        if k in pins:
            e[-1] = -np.inf
        if V is None:
            V, back = e, None
        else:
            pnodes = hist[-1][1]
            full = np.zeros((len(pnodes) + 1, len(nodes) + 1))
            if len(pnodes) and len(nodes):
                d = nav.d(pnodes[:, None], nodes[None, :])
                cost = np.minimum(BETA * np.maximum(0, d - (V_MAX * (g[t] - g[t - 1]) + SLACK)) / 1000, JUMP)
                home = nav.d(nodes, fountain_node) <= FOUNTAIN_R
                cost[:, home] = np.minimum(cost[:, home], RECALL)
                full[:-1, :-1] = -cost
            tot = V[:, None] + full
            back = np.argmax(tot, 0)
            V = tot.max(0) + e
        hist.append((pos, nodes, e, lab + ["dead"], back))
    s = int(np.argmax(V))
    out_pos, out_e, out_lab = np.full((len(ks), 2), np.nan), np.full(len(ks), np.nan), [None] * len(ks)
    for t in range(len(ks) - 1, -1, -1):
        pos, _, e, lab, back = hist[t]
        out_lab[t], out_e[t] = lab[s], e[s]
        if s < len(pos):
            out_pos[t] = pos[s]
        if back is not None:
            s = int(back[s])
    return out_pos, out_e, out_lab


def _viterbi_all(frames, fgt, p, a, deaths, nav, pin_minutes, mt, cal):
    hold_em = float(np.log(1 - cal["p_det"]))
    start = a[a["minute"] == 0].set_index("pid")
    fountain = {team: int(nav.node(*start.loc[list(io.TEAM_IDS[team]), ["x", "y"]].median().to_numpy())[0])
                for team in (100, 200)}
    cand = candidates(frames, fgt, mt, cal)
    ks = np.array(sorted(cand))
    g = fgt[ks]
    dw = death_windows(a, deaths)
    pins_all = _anchor_rows(a, fgt, p, pin_minutes)
    rows = []
    for i in range(10):
        pid = i + 1
        certain, possible = np.zeros(len(ks), bool), np.zeros(len(ks), bool)
        for s0, s1, s2 in dw.get(pid, []):
            certain |= (g >= s0) & (g <= s1)
            possible |= (g > s1) & (g <= s2)
        pins = {int(r.k): (float(r.px), float(r.py)) for r in pins_all[pins_all["pid"] == pid].itertuples()}
        pos, em, lab = viterbi(ks, g, {k: cand[k][i] for k in ks}, nav, p, certain, possible, pins, hold_em,
                               fountain[100 if pid <= 5 else 200])
        wx, wy = minimap.to_world(pos[:, 0], pos[:, 1], p)
        rows.append(pd.DataFrame({"k": ks, "game_s": g, "pid": pid, "px": pos[:, 0], "py": pos[:, 1],
                                  "x": wx, "y": wy, "logit": em, "state": lab}))
    return pd.concat(rows, ignore_index=True)


def fill_holds(tr):
    """Replace 'hold' positions by linear interpolation (in time) between the bracketing detected or pinned
    positions of the same participant (state 'interp'): with no evidence while the icon is covered, the champion
    keeps moving from the last to the next confident position. Unbracketed holds stay as they are."""
    out = []
    for _, g in tr.groupby("pid", sort=False):
        g = g.sort_values("k").copy()
        t, x, y = g["game_s"].to_numpy(), g["x"].to_numpy().copy(), g["y"].to_numpy().copy()
        px, py, st = g["px"].to_numpy().copy(), g["py"].to_numpy().copy(), g["state"].to_numpy().copy()
        ia = np.flatnonzero(np.isin(st, ["det", "pin"]))
        for i in np.flatnonzero(st == "hold"):
            j = np.searchsorted(ia, i)
            if 0 < j < len(ia):
                l, r = ia[j - 1], ia[j]
                w = (t[i] - t[l]) / (t[r] - t[l])
                x[i], y[i] = x[l] + w * (x[r] - x[l]), y[l] + w * (y[r] - y[l])
                px[i], py[i] = px[l] + w * (px[r] - px[l]), py[l] + w * (py[r] - py[l])
                st[i] = "interp"
        g["x"], g["y"], g["px"], g["py"], g["state"] = x, y, px, py, st
        out.append(g)
    return pd.concat(out).sort_values(["pid", "k"]).reset_index(drop=True)


def track(frames, fgt, p, a, deaths, nav, pin_minutes, template_minutes=None, rounds=ROUNDS):
    """Track all 10 participants. Round 0 uses class means from the V5 anchors of template_minutes; each later
    round re-estimates the means from the previous track's confident detections (self-training; calibration
    stays on the anchors). Returns (long DataFrame (k, game_s, pid, px, py, x, y, logit, state), info)."""
    template_minutes = pin_minutes if template_minutes is None else template_minutes
    bg = background_stats(frames, fgt)
    wins = anchor_windows(frames, fgt, a, p, template_minutes)
    info = {"rounds": []}
    tr = None
    for rnd in range(rounds + 1):
        if tr is not None:
            more = harvest_windows(frames, tr)
            wins = {pid: np.concatenate([wins[pid], more[pid]]) if len(more[pid]) else wins[pid] for pid in wins}
        means, counts = zip(*(class_mean(wins[pid]) for pid in range(1, 11)))
        mt, cal = build_detector(means, frames, fgt, a, p, template_minutes, bg)
        tr = _viterbi_all(frames, fgt, p, a, deaths, nav, pin_minutes, mt, cal)
        info["rounds"].append({"calibration": cal, "patches": list(map(int, counts)),
                               "states": tr["state"].value_counts(normalize=True).round(4).to_dict()})
    return fill_holds(tr), info


def evaluate(tr, a, fgt, minutes):
    """Error of the track at held-out V5 positions (alive anchors in `minutes`)."""
    ev = a[a["minute"].isin(minutes) & a["alive"]].copy()
    ev["k"] = frame_at(fgt, ev["t_ms"].to_numpy() / 1000)
    ev = ev[ev["k"] >= 0].merge(tr[["k", "pid", "x", "y", "state"]].rename(columns={"x": "tx", "y": "ty"}),
                                on=["k", "pid"])
    ev["err"] = np.hypot(ev["tx"] - ev["x"], ev["ty"] - ev["y"])
    return ev


def formation_agreement(tr, a, fgt, nav, minutes):
    """Formation-level check at held-out V5 minutes: per team, the set partition of the roles alive in V5
    (geodesic proximity graph at setups.R_GROUP) from tracked vs V5 positions. Returns one row per
    (minute, team) with exact agreement and the Rand index over role pairs (frames where some V5-alive role
    is not placed by the track are skipped)."""
    ev = a[a["minute"].isin(minutes)].copy()
    ev["k"] = frame_at(fgt, ev["t_ms"].to_numpy() / 1000)
    ev = ev[ev["k"] >= 0].merge(tr[["k", "pid", "x", "y"]].rename(columns={"x": "tx", "y": "ty"}), on=["k", "pid"])
    rows = []
    for (m, team), g in ev.assign(team=np.where(ev["pid"] <= 5, 100, 200)).groupby(["minute", "team"]):
        g = g.sort_values("pid")
        alive = g["alive"].to_numpy(bool)
        if len(g) != 5 or np.isnan(g.loc[g["alive"], ["tx", "ty"]].to_numpy()).any():
            continue
        parts = []
        for cx, cy in (("x", "y"), ("tx", "ty")):
            xy = g[[cx, cy]].to_numpy(float).copy()
            xy[~alive] = 0.0
            n = nav.node(xy[:, 0], xy[:, 1])
            parts.append(setups.partition(nav.d(n[:, None], n[None, :]), alive, setups.R_GROUP))
        same = [[any(i in b and j in b for b in P_) for P_ in parts] for i in range(5) for j in range(i + 1, 5)
                if alive[i] and alive[j]]
        rows.append({"minute": m, "team": team, "n_alive": int(alive.sum()), "exact": parts[0] == parts[1],
                     "rand": float(np.mean([u == v for u, v in same])) if same else 1.0})
    return pd.DataFrame(rows)
