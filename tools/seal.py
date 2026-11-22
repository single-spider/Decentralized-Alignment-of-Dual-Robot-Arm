"""Regenerate local file checksums after deliberate report edits."""
import os
from pathlib import Path

from reproduce import ROOT, digest, read, write


def main():
    archived = {r['path'] for r in read(ROOT / 'evidence/archives.json')}
    files = {}
    for directory, names, filenames in os.walk(ROOT):
        names[:] = sorted(set(names) - {'.git', '.venv', '.work', '__pycache__'})
        for name in sorted(filenames):
            path = Path(directory) / name
            rel = path.relative_to(ROOT).as_posix()
            if rel in archived or rel == 'manifest.sha256.json':
                continue
            if (path.parent == ROOT / 'report' and path.suffix in {'.aux', '.log', '.out', '.toc'}) or name.endswith('.synctex.gz'):
                continue
            if path.suffix in {'.pyc', '.pyo'}:
                continue
            files[rel] = digest(path)
    write(ROOT / 'manifest.sha256.json', {'algorithm': 'SHA-256', 'files': dict(sorted(files.items()))})
    print(f'Sealed {len(files)} files.')


if __name__ == '__main__':
    main()
