"""Shared helpers for the figure/table factory: paths, plotting style, number registry, LaTeX tables.

Every number that appears in the paper is written by a factory script into the registry (NUM) and lands in
numbers.tex as \\defV{key}{value}; the text uses \\V{key}. A missing key renders as a visible \\pending box.
Data are read from the repository's data/ folder (data/results, data/derived, ...) and, if absent there, from
paper/_cache. Inputs that are not in the public release (private model fits) are reported as missing and their
outputs are left as shipped.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
PROJ = PAPER.parent
CACHE = PAPER / "_cache"
FIG = PAPER / "figures"
TAB = PAPER / "tables"
BUILD = PAPER / "_build"
NUM_DIR = BUILD / "numbers"

# Calm single-column palette (after the "Flood and Harvest" figures): soft blue, sand/amber, brown, gray.
# The key names are kept from the old Okabe-Ito dict so that every figure script keeps working;
# blue and amber/brown stay distinguishable in greyscale and for colour-blind readers (different lightness).
C = dict(blue="#2F5D96", orange="#D39A2C", green="#6F8F6B", red="#9A5A1A", purple="#8A74A3",
         sky="#9DB6DA", yellow="#EBDDA8", black="#222222", grey="#8A8A8A", navy="#1F3F7A", sand="#E6CE9E",
         paper="#F4F4F4", ink="#333333")
STATE_C = {"trailing": C["red"], "even": C["grey"], "leading": C["blue"]}
SIDE_C = {100: C["blue"], 200: C["red"]}
COL_W, TEXT_W = 2.7, 5.5  # half and full text width of the single-column paper (in)
OBJ_LABEL = {"DRAGON": "First dragon", "DRAGON_2+": "Later dragons", "HORDE": "Grubs", "RIFTHERALD": "Herald",
             "ATAKHAN": "Atakhan", "BARON_NASHOR": "First baron", "BARON_2+": "Later barons", "ELDER": "Elder"}


class Missing(Exception):
    """A source file is not available yet; the script's outputs stay pending."""


def src(rel):
    """Resolve a result path: project mirror first, then the local cache. Raises Missing."""
    rel = rel.replace("release/build/", "")
    for base in (PROJ / "data", PROJ, CACHE):
        p = base / rel
        if p.exists():
            return p
    raise Missing(rel)


def csv(rel, **kw):
    return pd.read_csv(src(rel), **kw)


def js(rel):
    return json.loads(src(rel).read_text(encoding="utf-8"))


def _set_title(self, label, fontdict=None, loc=None, pad=None, **kw):
    """Axes.set_title replacement: 'title || subtitle' -> bold title plus italic gray subtitle."""
    t, _, sub = str(label).partition("||")
    t, sub = t.strip(), sub.strip()
    left = (loc == "left")
    y0 = 1.02
    if sub:
        self.text(0.0 if left else 0.5, y0, sub, transform=self.transAxes, ha="left" if left else "center",
                  va="bottom", fontsize=6.4, style="italic", color="#6a6a6a", clip_on=False)
        y0 = 1.02 + 0.115 * (2.0 / max(self.get_position().height * plt.gcf().get_figheight(), 0.5))
    return self.text(0.0 if left else 0.5, y0, t, transform=self.transAxes, ha="left" if left else "center",
                     va="bottom", fontsize=7.4, fontweight="bold", color=C["black"], clip_on=False)


def style():
    matplotlib.axes.Axes.set_title = _set_title
    ink = "#3a3a3a"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.55, "xtick.major.width": 0.55, "ytick.major.width": 0.55, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.edgecolor": ink, "xtick.color": ink, "ytick.color": ink,
        "axes.labelcolor": ink, "text.color": ink, "lines.linewidth": 1.2, "lines.markersize": 3.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.axisbelow": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.5,
        "axes.prop_cycle": matplotlib.cycler(color=[C[k] for k in ("blue", "orange", "green", "red", "purple", "sky", "black")]),
    })


def fill_width(fig, iters=3):
    """Stretch the subplot area so that the tight bounding box spans the full figure width (no side margins)."""
    Wf = fig.get_figwidth()
    for _ in range(iters):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        sp = fig.subplotpars
        left = sp.left - bb.x0 / Wf
        right = sp.right + (Wf - bb.x1) / Wf
        if right - left < 0.3:
            break
        fig.subplots_adjust(left=left, right=right)


def save(fig, name, fill=True):
    FIG.mkdir(exist_ok=True)
    if fill:
        try:
            fill_width(fig)
        except Exception as e:  # layouts without subplotpars (manual axes) keep their own margins
            print("fill_width skipped:", e)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    return name


class Registry:
    """Collects \\V{key} values for one script; flushed to _build/numbers/<script>.json."""

    def __init__(self, script):
        self.script, self.d = script, {}

    def __setitem__(self, k, v):
        self.d[k] = v

    def flush(self):
        NUM_DIR.mkdir(parents=True, exist_ok=True)
        (NUM_DIR / f"{self.script}.json").write_text(json.dumps(self.d, indent=1, ensure_ascii=False), encoding="utf-8")


def f(x, nd=1, pct=False, sign=False):
    """Format a number for the text (pct: proportion -> percent / percentage points)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    x = 100 * x if pct else x
    s = f"{x:+,.{nd}f}" if sign else f"{x:,.{nd}f}"
    return s.replace("-", "$-$")


def ci(lo, hi, nd=1, pct=False):
    return f"[{f(lo, nd, pct)}, {f(hi, nd, pct)}]"


def tex_escape(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


TABLE_LEADS = {'tab_data': 'The core set covers every game of two 2025 events.', 'tab_rules': 'Every objective has a fixed spawn clock, and no kill precedes its spawn.', 'tab_tracker': 'Median tracking error is small against the map width and grows with crowding.', 'tab_rlayer': 'Exact Riot frames confirm the baron contrast and the estimator dependence.', 'tab_vlayer': 'Headline contrasts concentrate after the spawn.', 'tab_selection': 'No finite $K\\leq8$ is selected on the V layer.', 'tab_patterns': 'Six patterns separate by objective class and by how they gather.', 'tab_hasse_tagnet': 'HASSE refits agree where TAGNet seeds do not.', 'tab_fights': 'The larger side loses fewer players than the square law predicts.', 'tab_qre': "Choices respond to fights, travel and player costs, and little to the objective's own value.", 'tab_shapley': 'Curvature pulls trailing teams towards contesting; team behaviour pulls them away.', 'tab_values': 'Every event is priced in win-probability points.', 'tab_vocab': 'Formation vocabularies on the R layer, by objective and time.', 'tab_qre_variants': 'All Shapley signs agree with the main fit in the six non-degenerate variants.', 'tab_team_params': 'Team parameters come with wide intervals.', 'tab_usage': 'Split-half reliability of team pattern usage is low.', 'tab_context': 'Context effects on pattern occupancy are mostly small.', 'tab_hasse_robust': '$K=6$ is best among $K=3,\\dots,6$ on the 7-snapshot grid and at both radii.', 'tab_wfit': 'The win surface fits out of fold in every game phase.', 'tab_faker': 'Faker arrives earlier but is no more central in the formation.', 'tab_t1_state': "T1's behaviour by gold state matches the rest of the league.", 'tab_ledger': 'The ledger lists every spawn of the Worlds~2025 final.'}


def write_table(name, header, rows, align=None, caption=None, label=None, star=False, size=r"\small",
                notes=None, midrules=()):
    """Write tables/<name>.tex: a complete single-column table float (booktabs, at most the text width).
    rows = list of lists of strings (already TeX). `star` is kept for old callers and ignored."""
    TAB.mkdir(exist_ok=True)
    align = align or "l" + "r" * (len(header) - 1)
    out = [f"% generated by the factory ({name}); do not edit", r"\begin{table}[t]", r"\centering", size]
    if caption:
        lead = TABLE_LEADS.get(name)
        caption = (f"\\textbf{{{lead}}} " + caption) if lead else caption
        out.append(f"\\caption{{{caption}}}")
    if label:
        out.append(f"\\label{{{label}}}")
    out += [r"\setlength{\tabcolsep}{4pt}", r"\begin{adjustbox}{max width=\linewidth}", f"\\begin{{tabular}}{{{align}}}",
            r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            out.append(r"\midrule")
        out.append(" & ".join(str(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if notes:
        out.append(f"\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\scriptsize {notes}}}")
    out.append(r"\end{table}")
    (TAB / f"{name}.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    return name


def lattice_mod():
    """Import the project's lattice library (use of ../code/lolsetup)."""
    import sys
    if str(PROJ / "code") not in sys.path:
        sys.path.insert(0, str(PROJ / "code"))
    from lolsetup import lattice
    return lattice


def run_main(build):
    """Standard entry point for make_*.py: run, flush numbers, report."""
    import sys
    name = Path(sys.argv[0]).stem
    reg = Registry(name)
    style()
    try:
        out = build(reg)
        reg.flush()
        print(f"[ok] {name}: {out}")
    except Missing as e:
        reg.flush()
        print(f"[pending] {name}: missing {e}")
