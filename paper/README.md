# Paper — Allocation Before the Clock

`main.pdf` is the paper (single-column version); `main.tex` + `sections/` + `refs.bib` are its LaTeX source.
`main_ieee.tex` + `sections_ieee/` are the IEEE two-column variant (same figures, tables and `numbers.tex`).

## Build the PDF
```
cd paper
latexmk -pdf main.tex          # or: latexmk -pdf main_ieee.tex
```
Everything the build needs is included (`numbers.tex`, `tables/*.tex`, `figures/*.pdf`). Every number in the text is `\V{key}`, defined in `numbers.tex`.

## Regenerate figures, tables and numbers (optional)
```
pip install numpy pandas matplotlib scipy pyarrow
python make_all.py             # runs figures/make_*.py and tables/make_*.py, rewrites numbers.tex and _build/pending.md
```
`factory/common.py` reads inputs from the repository's `data/` folder (`data/results`, `data/derived`, `data/navgraph`, …).
A script whose inputs are not in the public release reports `[pending]`; its figures and the shipped values in `numbers.tex` are left unchanged;
`_build/pending.md` lists what was skipped. The shipped `numbers.tex`, `figures/*.pdf` and `tables/*.tex` are the authoritative versions.

Inputs that are **not** in this repository are the private model fits (`fits/*`, e.g. per-setup parquet files, bootstrap `.npy` arrays) and a few
intermediate result folders (`results/values`, `results/r_layer`, `results/clock`, …). Figures and tables that depend on them are shipped as PDF / TeX only:
check `_build/pending.md` after a run for the exact list. The code that produces those inputs is in `../code/` (`lolsetup/`).
`tables/make_synth_recovery.py` additionally runs `../code/tests/test_hasse_recovery.py` (about two minutes).
