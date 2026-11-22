 
## Rebuild and verify

From the submission root, with the reference Python environment activated:

```powershell
python tools/build_readable_report.py
python tools/seal.py
python tools/reproduce.py hashes
```

The new build needs matplotlib, pdflatex and pdfinfo. Alternatively upload the
readable Overleaf ZIP, choose `main.tex`, and compile twice. If editing on Overleaf,
sync the final source back and rebuild the local package before resealing.

Inspect all nine final pages, test the repository link and grader access, and
submit the PDF and repository URL according to the portal instructions. The
report distinguishes prior thesis work, new course work, simulation results and
the original paper's hardware results. Retain those distinctions and the existing
third-party template attributions. No new experiments or training are required
for this editorial revision.

For publication, use this folder as the repository root. Exclude `.work/`, virtual
environments, caches and the expanded deterministic episode JSONL; the included
gzip archive preserves that raw evidence. Preserve `.gitattributes` so checkpoint
and evidence hashes remain stable. Existing source/model-sharing terms remain
applicable; this report revision grants no additional license to third-party work.
