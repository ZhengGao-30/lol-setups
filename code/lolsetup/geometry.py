"""Map geometry: pit centres, coarse map regions, and the pit-relative battlefield quotient.

Coordinates are Riot engine units (0..~14,870). Blue base is bottom-left, red base top-right.
The river is the anti-diagonal x + y ~ RIVER; 'top side' of the map is y > x.
Region boundaries are deliberately simple bands (documented constants); they label positions for
description only and are checked visually against the occupancy mask.
"""
import numpy as np
import pandas as pd

RIVER = 14870.0
FOUNTAIN = {100: np.array([400.0, 420.0]), 200: np.array([14340.0, 14390.0])}
BASE_R = 4500.0
LANE_W = 1900.0
MID_W = 1300.0
RIVER_W = 1100.0
PIT_R = 1500.0  # primary pit radius (geodesic); sensitivity grid R in {1000, 1500, 2000}
PIT_RADII = (1000.0, 1500.0, 2000.0)

# Battlefield labels relative to a pit and a team (the 'zone quotient'); "base" = the team's OWN base.
# Dead players get the label "dead", which is outside ZONES (they are not part of any formation).
ZONES = ("pit", "own_pitside", "opp_pitside", "pit_lane", "mid", "far", "base")


def pit_centres(spawns):
    """Data-driven pit centres = median kill position per objective (Atakhan split into top/bottom)."""
    k = spawns.dropna(subset=["kill_x"]).copy()
    k["pit"] = k["objective"].map({"DRAGON": "dragon", "ELDER": "dragon", "BARON_NASHOR": "baron",
                                   "HORDE": "baron", "RIFTHERALD": "baron", "ATAKHAN": "atakhan"})
    is_top = k["kill_y"] > k["kill_x"]
    k.loc[k["pit"].eq("atakhan") & is_top, "pit"] = "atakhan_top"
    k.loc[k["pit"].eq("atakhan") & ~is_top, "pit"] = "atakhan_bot"
    return k.groupby("pit")[["kill_x", "kill_y"]].median().rename(columns={"kill_x": "x", "kill_y": "y"})


def pit_of(objective, kill_x=None, kill_y=None):
    if objective in ("DRAGON", "ELDER"):
        return "dragon"
    if objective in ("BARON_NASHOR", "HORDE", "RIFTHERALD"):
        return "baron"
    if objective == "ATAKHAN":
        if kill_x is None or pd.isna(kill_x):
            return None  # untaken Atakhan: side not observable from kills
        return "atakhan_top" if kill_y > kill_x else "atakhan_bot"
    raise ValueError(objective)


def region(x, y):
    """Coarse absolute region of map points: base_blue/base_red, top/mid/bot lane, river, jungle quadrants."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.full(x.shape, "jungle", dtype=object)
    s = x + y
    top = y > x
    out[np.abs(s - RIVER) < RIVER_W] = "river"
    out[np.abs(x - y) < MID_W] = "mid"
    top_lane = ((x < LANE_W) & (y > BASE_R)) | ((y > RIVER - LANE_W) & (x < RIVER - BASE_R))
    bot_lane = ((y < LANE_W) & (x > BASE_R)) | ((x > RIVER - LANE_W) & (y < RIVER - BASE_R))
    out[top_lane] = "top_lane"
    out[bot_lane] = "bot_lane"
    jungle = out == "jungle"
    out[jungle & top & (s < RIVER)] = "jungle_blue_top"
    out[jungle & ~top & (s < RIVER)] = "jungle_blue_bot"
    out[jungle & top & (s >= RIVER)] = "jungle_red_top"
    out[jungle & ~top & (s >= RIVER)] = "jungle_red_bot"
    out[np.hypot(x - FOUNTAIN[100][0], y - FOUNTAIN[100][1]) < BASE_R] = "base_blue"
    out[np.hypot(x - FOUNTAIN[200][0], y - FOUNTAIN[200][1]) < BASE_R] = "base_red"
    return out


def zone(x, y, team, pit_xy, d_pit, alive=True):
    """Pit-relative battlefield for a team's player(s): see ZONES."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    reg = region(x, y)
    pit_top = pit_xy[1] > pit_xy[0]
    own = "blue" if team == 100 else "red"
    opp = "red" if team == 100 else "blue"
    side_top = y > x
    out = np.full(x.shape, "far", dtype=object)
    jung_or_river = np.char.startswith(reg.astype(str), "jungle") | (reg == "river")
    same_side = side_top == pit_top
    own_half = (x + y < RIVER) if team == 100 else (x + y >= RIVER)
    out[jung_or_river & same_side & own_half] = "own_pitside"
    out[jung_or_river & same_side & ~own_half] = "opp_pitside"
    out[reg == ("top_lane" if pit_top else "bot_lane")] = "pit_lane"
    out[reg == "mid"] = "mid"
    out[reg == f"base_{opp}"] = "far"
    out[reg == f"base_{own}"] = "base"
    out[np.asarray(d_pit) <= PIT_R] = "pit"
    out[~np.broadcast_to(np.asarray(alive, bool), out.shape)] = "dead"
    return out


def block_zones(blocks, D, d_pit, zones):
    """Zone of each formation block = zone of the block's geodesic medoid (argmin of summed in-block
    distance; ties broken by smaller distance to the pit, then role order)."""
    out = []
    for B in blocks:
        B = list(B)
        cost = [(D[i, B].sum(), d_pit[i], i) for i in B]
        out.append(zones[min(cost)[2]])
    return out
