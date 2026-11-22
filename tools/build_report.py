"""Compile the local eight-page report and package portable Overleaf source."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from reproduce import ROOT, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip-only', action='store_true')
    args = parser.parse_args()
    source, output = ROOT / 'report', ROOT / 'output'
    (output / 'pdf').mkdir(parents=True, exist_ok=True)
    (ROOT / '.work').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='latex-', dir=ROOT / '.work') as folder:
        tmp = Path(folder)
        if not args.zip_only:
            tex = shutil.which('pdflatex')
            info = shutil.which('pdfinfo')
            require(tex and info, 'Need pdflatex and pdfinfo, or compile the ZIP on Overleaf')
            version = subprocess.check_output([tex, '--version'], text=True, errors='replace')
            flags = ['--disable-installer'] if 'MiKTeX' in version else []
            for _ in range(2):
                result = subprocess.run([tex, *flags, '-interaction=nonstopmode', '-halt-on-error',
                                         f'-output-directory={tmp}', 'main.tex'], cwd=source,
                                        text=True, errors='replace', capture_output=True, timeout=90)
                if result.returncode:
                    raise RuntimeError((result.stdout + result.stderr)[-5500:])
            pdf = tmp / 'main.pdf'
            page_info = subprocess.check_output([info, str(pdf)], text=True, errors='replace')
            require(re.search(r'^Pages:\s+8\s*$', page_info, re.M), 'The report must have exactly eight pages')
            log = (tmp / 'main.log').read_text(errors='replace')
            require(not re.search(r'Overfull|LaTeX Warning:.*(?:undefined|multiply defined)', log),
                    'Verify layout and resolve reference warnings before handing off')
            shutil.copy2(pdf, output / 'pdf/group28-dual-arm-mas.pdf')
            print('Built and checked eight pages; visually inspect changes before submission.')

        staged = tmp / 'overleaf.zip'
        with ZipFile(staged, 'w', ZIP_DEFLATED) as zip_file:
            paths = [p for p in source.iterdir() if p.suffix in ('.sty', '.tex', '.md')]
            paths += [p for p in (source / 'media').iterdir() if p.suffix in ('.pdf', '.png')]
            for path in sorted(paths):
                zip_file.write(path, path.relative_to(source).as_posix())
        with ZipFile(staged) as zip_file:
            require(zip_file.testzip() is None and 'main.tex' in zip_file.namelist(), 'Invalid Overleaf package')
        shutil.copy2(staged, output / 'group28-overleaf.zip')
        print('Packaged Overleaf source.')


if __name__ == '__main__':
    main()
