"""Build the separate, presentation-refined report without replacing the original."""
import argparse
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def diagrams():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    media = ROOT / 'report-readable/media'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': '#243845', 'pdf.fonttype': 42})

    def canvas(height):
        fig, ax = plt.subplots(figsize=(7.1, height))
        ax.set(xlim=(0, 10), ylim=(0, 3)); ax.axis('off')
        fig.subplots_adjust(left=.01, right=.99, top=.98, bottom=.02)
        return fig, ax

    def box(ax, x, y, w, h, label, warm=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle='round,pad=.04,rounding_size=.07',
                     facecolor='#f5eadc' if warm else '#e8f2f3',
                     edgecolor='#507083', linewidth=.9))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=9)

    def arrow(ax, start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                     mutation_scale=10, color='#375765', lw=1,
                     linestyle='--' if dashed else '-'))

    fig, ax = canvas(1.55)
    for i, (title, da, db) in enumerate([
        ('Peg arm only', -4, 0), ('Receiver arm only', 0, 4),
        ('Shared correction', -2, 2)]):
        x = i*3.35
        ax.text(x+1.5, 2.78, title, ha='center', weight='bold', fontsize=10)
        ax.add_patch(Rectangle((x+1.95, 1.25), .45, .65, facecolor='#117674'))
        ax.add_patch(Rectangle((x+.35, 1.1), .9, .95, facecolor='none',
                               edgecolor='#3275a1', lw=2))
        ax.text(x+2.17, .83, 'A: peg', ha='center', fontsize=8)
        ax.text(x+.8, .83, 'B: fixture', ha='center', fontsize=8)
        if da: arrow(ax, (x+2.35, 2.22), (x+1.55, 2.22))
        if db: arrow(ax, (x+.45, 2.22), (x+1.25, 2.22))
        ax.text(x+1.5, .28, f'A: {da:+d} mm    B: {db:+d} mm', ha='center', fontsize=9)
    fig.savefig(media/'allocation-intuition.pdf'); plt.close(fig)

    fig, ax = canvas(1.65)
    box(ax, .05, 1.43, 2.15, 1.02, 'Shared sensed error\nOwn joints + cost\nPermitted information')
    box(ax, 3.0, 1.43, 2.25, 1.02, 'Role-conditioned actor\n39 inputs; 64 + 64\nOne fraction per axis')
    box(ax, 6.1, 1.43, 1.8, 1.02, 'Arm A / Arm B\nLocal actions')
    box(ax, 8.65, 1.43, 1.25, 1.02, 'Shared\nenvironment')
    arrow(ax, (2.26, 1.94), (2.94, 1.94)); arrow(ax, (5.31, 1.94), (6.04, 1.94))
    arrow(ax, (7.96, 1.94), (8.59, 1.94))
    box(ax, 3.0, .05, 4.9, .85, 'Training only: critic sees true error + both joint vectors', True)
    arrow(ax, (9.25, 1.37), (7.96, .48), True)
    arrow(ax, (4.13, .96), (4.13, 1.37), True)
    ax.text(1.15, .47, 'Decentralized execution\nuses no critic', ha='center', fontsize=9)
    fig.savefig(media/'training-execution.pdf'); plt.close(fig)

    fig, ax = canvas(1.65)
    for x, label in [(0.08, 'Planar\nalignment'), (2.58, 'Measured-pose\ngate'),
                     (5.08, 'Force-regulated\ndescent'), (7.58, 'Depth + pose\n1 s hold')]:
        box(ax, x, 1.7, 2.12, .9, label)
    for x in [2.24, 4.74, 7.24]: arrow(ax, (x, 2.15), (x+.28, 2.15))
    box(ax, 4.68, .1, 2.95, .84, 'Three stalls: bounded retraction', True)
    arrow(ax, (6.14, 1.64), (6.14, 1.0))
    ax.text(1.18, .95, 'Both arms move', ha='center', fontsize=9)
    ax.text(3.65, .95, 'Reject if outside\nmeasured clearance', ha='center', fontsize=8)
    ax.text(8.77, .66, 'Contact phase:\nB holds; A inserts', ha='center', fontsize=9)
    fig.savefig(media/'insertion-stages.pdf'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-diagrams', action='store_true')
    args = parser.parse_args()
    if not args.skip_diagrams:
        diagrams()
    src = ROOT/'report-readable'
    work = ROOT/'.work/readable-build'
    work.mkdir(parents=True, exist_ok=True)
    tex = shutil.which('pdflatex')
    if not tex:
        raise RuntimeError('pdflatex is required; alternatively use the Overleaf ZIP')
    version = subprocess.check_output([tex, '--version'], text=True)
    flags = ['--disable-installer'] if 'MiKTeX' in version else []
    for _ in range(2):
        result = subprocess.run([tex, *flags, '-interaction=nonstopmode', '-halt-on-error',
                                 f'-output-directory={work}', 'main.tex'], cwd=src,
                                capture_output=True, text=True, errors='replace', timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout[-7000:])
    log = (work/'main.log').read_text(errors='replace')
    problems = re.findall(r'(?:Overfull[^\n]*|LaTeX Warning:[^\n]*(?:undefined|multiply defined)[^\n]*)', log)
    if problems:
        raise RuntimeError('\n'.join(problems))
    info = subprocess.check_output(['pdfinfo', str(work/'main.pdf')], text=True)
    if not re.search(r'^Pages:\s+9\s*$', info, re.M):
        raise RuntimeError('Expected nine pages:\n'+info)
    output = ROOT/'output'
    (output/'pdf').mkdir(parents=True, exist_ok=True)
    shutil.copy2(work/'main.pdf', output/'pdf/group28-dual-arm-mas-readable.pdf')
    with ZipFile(output/'group28-readable-overleaf.zip', 'w', ZIP_DEFLATED) as z:
        for p in sorted(src.rglob('*')):
            if p.is_file() and p.suffix in {'.tex', '.sty', '.pdf', '.png', '.md'}:
                z.write(p, p.relative_to(src).as_posix())
    print('Built nine-page alternative and Overleaf ZIP. Original report unchanged.')


if __name__ == '__main__':
    main()
