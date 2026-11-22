# Presentation-refined report

This nine-page alternative preserves the original eight-page report under `report/`
and `output/pdf/group28-dual-arm-mas.pdf`. The refined PDF is
`output/pdf/group28-dual-arm-mas-readable.pdf`; the self-contained Overleaf package
is `output/group28-readable-overleaf.zip`.

The revision displays key equations, separates learning from contact control,
adds three explanatory vector diagrams, tabulates implementation settings and
experiment coverage, and incorporates the repository URL and team-confirmed
contribution record. Existing result figures, numerical macros, comparison tables,
citations, and methodological limitations are retained.

Build from the submission root with a Python environment containing matplotlib:

```powershell
python tools/build_readable_report.py
```

The build requires `pdflatex` and `pdfinfo`. It checks for nine pages, overfull
boxes and unresolved references. To compile without regenerating the diagrams,
use `--skip-diagrams`. For Overleaf, upload the ZIP and choose `main.tex`; compile
twice. Numerical macros and original result figures are copied verbatim from
the original report and are not independent experimental results.

Template attribution is in `TEMPLATE.md`. The contribution wording was confirmed
by the user against the team's agreed record on 22 November 2026.

The title block includes all four roll numbers as recorded in the project
idea-review report. A declaration of AI use appears after the conclusion.

## LaTeX source submission

Submit a link to this entire `report-readable/` directory, not just the PDF.
`main.tex` is the entry point; compilation also needs `data.tex`,
`submission-details.tex`, `PRIMEarxiv.sty` and the `media/` directory. The Overleaf
ZIP contains all of these files. If an Overleaf project URL is specifically
required, upload the ZIP to your own Overleaf account, set `main.tex` as the main
document, and use that project's sharing controls to create the permitted link.
A GitHub source URL is not an Overleaf project URL.
