# ECE 285 HW1 report draft

`report.tex` is the editable source. Replace `Your Name` before submission.
It uses the unmodified official NeurIPS 2024 style, in nonanonymous preprint
mode with the first-page notice changed to identify this homework draft.
The homework did not specify a template year.

Compile from this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

The source package includes the required figures and style file, so it can also
be uploaded to Overleaf with `report.tex` as the main document. The compiled
draft is two pages, including references, at the style's normal margins and
main text size. No abstract is included to keep the report within two pages.

Evidence is from nine completed main experiments in `results_snapshot.csv`.
The debugging subset is excluded. All scores use the fixed-reference v2 protocol.
Figure sources are recorded in `figures/sources.json`. Sample grids use the
first eight saved prior draws, not manually selected examples. Rebuild assets
from the project root with `.venv\Scripts\python.exe report/prepare_figures.py`.

The draft does not claim clinical validity or that exploratory test metrics are
an untouched final benchmark. It does not report unfinished seed or width experiments.
Revise the results and figures if additional experiments are completed.

Official style archive:
https://media.neurips.cc/Conferences/NeurIPS2024/Styles.zip
