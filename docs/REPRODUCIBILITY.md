# Reproducibility and Technical Details

[Back to the project README](../README.md)

This guide covers the reference environment, saved experiment verification, fresh
evaluation and training, and the organization of model and episode records. For
a first look at the project, use the demonstration in the root README.

Run all commands below from the **repository root**, even though this guide is
stored in `docs/`. New runs write to `.work/`; published experiment records remain
separate.

## Reference environment

Use **Windows x64 and Python 3.11**. From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
& $python -m pip install -r requirements-reference.txt
```

The original runtime used Python 3.11.9 and PyTorch 2.11.0+cu128; CPU PyTorch suffices for checkpoint loading and replay. Training timings and bit-for-bit fresh simulator output are not portable guarantees. `tools/reproduce.py` explicitly imports `sfn` and `dual_arm_mas` from **this folder**, not an adjacent checkout or another editable install. PyBullet provides the Panda URDF and meshes via `pybullet_data`. The replay renderer expects Windows' Consolas font. Allow approximately 1.5 GB of free space for the bundle plus scratch verification and several minutes for full re-analysis.

## Check the saved experimental results

```powershell
& $python tools/reproduce.py hashes
& $python tools/reproduce.py verify --out check-all
```

Expected counters: **23,456** deterministic episodes, **25,280** learned alignment episodes, **1,216** corrected contact episodes in **14** groups, **12** PPO fits, and `statistics_match: true`. This derives results from raw episode records and frozen seeds, configurations, checkpoint and provenance data, then checks against the archived outputs; it does not just display the saved tables. It checks baseline coverage and paired resets, all selected actor fits, validation and logged parameter changes, and contact outcome metadata. Historical file modification times are not treated as scientific results after transport.

The checker verifies the SHA-256 manifest, restores the large deterministic JSONL and rebases original absolute keys only in a temporary copy. This preserves historical records exactly. The separate `restore` command optionally restores that file in-place for inspection; its raw SHA-256 is checked and the expanded file is ignored by Git. Do not commit the 117 MiB expanded file. The included gzip archive is sufficient. To regenerate the typeset figures/tables after verification:

```powershell
& $python tools/report_data.py --verified .work/check-all
```

This intentionally regenerates report assets. Work in a scratch copy if you also want to preserve the shipped manifest unchanged; `tools/seal.py` refreshes the manifest after deliberate report edits.

The revised report includes a vector execution-workflow diagram, matched GT/RGB PPO rows, and a stage-by-stage contact failure table. Its [failure-stage audit](../verification/contact-failure-stages.json) records the group counts, episode IDs and failure reasons for all 1,024 main open-hole trials. The generator checks each raw JSONL against the audited digest before classifying it; blocked and stress trials are not pooled with these rows. The [original editorial record](../verification/editorial-revision.json), [readability revision record](../verification/readable-editorial-revision.json), and [author/disclosure record](../verification/author-ai-declaration.json) document successive report checks. No new training was needed for these presentation changes.

## Evaluation and training commands

All names passed to `--out` must be new and relative to `.work/`. These commands do **not** append to the frozen evidence. The adapter replaces historical date gates only for new work and keeps the recorded simulation/training algorithms. A finite wall budget is checked between work units, so a final episode or update may extend slightly beyond the requested seconds. Increase the budget if coverage is incomplete.

```powershell
# Four saved-policy cases: fixed half, two learned opens, and blocked learned jam/release
& $python tools/reproduce.py demo --out replay-new --seconds 240

# Short fresh PPO and segmentation training checks, from the included data
& $python tools/reproduce.py train --regime s1 --fit 0 --steps 64 --batch-steps 64 --epochs 1 --out ppo-smoke --seconds 90
& $python tools/reproduce.py vision-train --epochs 1 --out vision-smoke --seconds 180

# Fresh evaluation of the archived model and a deterministic comparison
& $python tools/reproduce.py alignment-eval --regime s1 --fit 0 --sensing rgb --count 2 --out rgb-check --seconds 90
& $python tools/reproduce.py alignment-eval --condition drop25 --regime s2 --fit 0 --sensing rgb --count 2 --out dropout-check --seconds 90
& $python tools/reproduce.py contact-eval --group open-fit0-s1-rgb --count 2 --out contact-check --seconds 90
& $python tools/reproduce.py baseline-eval --scenario core --count 2 --out baseline-check --seconds 90
```

For a full PPO fit use `--steps 10000 --batch-steps 256 --epochs 5`, set `--fit 0`, `1` or `2` and `--regime s1`, `s2`, `s3` or `s4`, and allow an appropriate local work budget. Vision adaptation uses 16 epochs by default, reusing the bundled 192/24 train/validation images. The historical selected checkpoint is not necessarily the last checkpoint. To reproduce other impairment settings, select the named condition and allowed regime from `dual-arm-mas/configs/learned_study.json` or the corresponding baseline/contact config. Raw results, model checkpoints and telemetry from new work go under `.work/`. **Short smoke runs validate executability, not scientific performance.** Load only the package's verified checkpoints or models you trust; PyTorch training snapshots include state beyond tensor weights.

## Experiment records and checkpoints

| Path | Contents |
| --- | --- |
| `dual-arm-mas/src/`, `scripts/`, `configs/`, `tests/` | Frozen algorithms, simulator, CLIs, experiment definitions and checks. Use the `tools/` wrapper for new launches. |
| `dual-arm-mas/runs/rl-s*-40*/` | All 12 PPO fits, including training JSONL, validation rows, candidate checkpoints, selected/latest/initial weights, optimizer and RNG state. |
| `dual-arm-mas/runs/vision-adaptation/` | Training/validation RGB and masks, manifests, learning records, checkpoints and the deployed segmenter. |
| `dual-arm-mas/runs/{train-311,train-312,train-313,validation,test}/` | Seeded joint starts and public models. Contact held-out starts are under `contact-momentum-fix/`. |
| `dual-arm-mas/runs/{evaluation,learned-study,contact-study-v2}/` | Raw production experiments, configurations via linked pins, summaries, snapshots and audit records. The large deterministic log is in `evidence/` until restored. |
| `dual-arm-mas/runs/contact-study/` | **Stopped diagnostic branch:** 553 episodes predating state-preserving IK. Do not mix with the corrected contact study. |
| `peg-in-hole-sfn/sfn/`, `peg-in-hole-sfn/models/` | Only the SFN Python foundation and pretrained initialization needed here. |

Smaller pilots and diagnostics are retained as historical work, not additional test-set outcomes. Archived audits may refer to paths on the original host; use `verify` for the portable audit. The final report is the nine-page PDF linked in the root README. The earlier eight-page PDF is retained only as an archive copy.


## Final report and LaTeX source

The final submission document is the [nine-page report](../output/pdf/group28-dual-arm-mas-readable.pdf).
Its [complete LaTeX directory](../report-readable/) includes `main.tex`, numerical
macros, the style file and figures. The [Overleaf ZIP](../output/group28-readable-overleaf.zip)
contains everything needed to compile it. Upload the ZIP to Overleaf and select
`main.tex` if an Overleaf project link is required; the GitHub source directory is
also directly shareable when the course accepts a source-repository link.

Build the final report locally, from the repository root:

```powershell
& $python tools/build_readable_report.py --skip-diagrams
```

This uses the included vector diagrams and requires `pdflatex` and `pdfinfo`.
Omit `--skip-diagrams` to regenerate the three explanatory diagrams with
matplotlib. The build checks for nine pages, overfull boxes and unresolved
references, then updates the final PDF and Overleaf ZIP. Visually inspect the
result after editing. If regenerating numerical figures with `report_data.py`,
note that it targets `report/`; synchronize its numerical macros and original
result figures into `report-readable/` before rebuilding the final report.

After deliberate changes to bundled files, update and check their manifest:

```powershell
& $python tools/seal.py
& $python tools/reproduce.py hashes
```

### Earlier report version

The [original eight-page PDF](../output/pdf/group28-dual-arm-mas.pdf),
[original source](../report/) and [original Overleaf ZIP](../output/group28-overleaf.zip)
are preserved for comparison. They are not the current submission version.
The final report includes author roll numbers, the confirmed contribution record
and the declaration of AI use. If the course imposes an eight-page maximum,
resolve that requirement before submitting the nine-page final version.
