"""Create the report's figures and numerical TeX from audited statistics and demo traces."""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np

from reproduce import ROOT, RUNS, digest, read, require, write


def axes_style(ax, axis='x'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#b0bcc1')
    ax.tick_params(length=0, pad=4, labelsize=9)
    ax.grid(axis=axis, color='#dde5e9', lw=.65, zorder=0)
    ax.set_axisbelow(True)


def formats(chosen):
    return f'{chosen["mean_cost_rad2"]*1000:.3f} & {chosen["mean_steps"]:.2f} & {chosen["mean_bytes"]:.1f}'


def workflow(figures):
    fig, ax = plt.subplots(figsize=(7.1, 1.9))
    ax.set(xlim=(0, 10), ylim=(0, 3.1)); ax.axis('off')
    def box(x, y, width, height, label, color='#e8f2f3'):
        patch = FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.035,rounding_size=.07',
                               edgecolor='#507083', facecolor=color, lw=.8)
        ax.add_patch(patch); ax.text(x+width/2, y+height/2, label, ha='center', va='center', fontsize=9)
    def arrow(x, y, xx, yy, style='-|>', **kw):
        ax.add_patch(FancyArrowPatch((x, y), (xx, yy), arrowstyle=style, mutation_scale=8,
                                     lw=.9, color='#375765', **kw))
    box(.08, .95, 2.05, .94, 'RGB image\nSFN segmentation\nPlanar error estimate')
    box(3.05, 1.83, 2.28, .78, 'Arm A local policy\nOwn state + role')
    box(3.05, .2, 2.28, .78, 'Arm B local policy\nOwn state + role')
    box(3.02, 1.18, 2.34, .37, 'S2/S4 payload routing', '#f5eadc')
    box(6.14, .95, 1.55, .94, 'Prepare both\nactions, then\nexecute together')
    box(8.25, .95, 1.65, .94, 'Alignment gates\nB holds; A inserts\nForce feedback', '#f5eadc')
    arrow(2.15, 1.42, 2.65, 1.42)
    ax.plot([2.65, 2.65], [.58, 2.22], color='#375765', lw=.9)
    arrow(2.65, 2.22, 3.01, 2.22); arrow(2.65, .58, 3.01, .58)
    for y, to in [(2.22, 1.65), (.58, 1.15)]:
        arrow(5.37, y, 6.1, to)
    arrow(7.73, 1.42, 8.21, 1.42)
    arrow(4.2, 1.8, 4.2, 1.58, style='<|-|>'); arrow(4.2, 1.15, 4.2, 1.01, style='<|-|>')
    ax.text(6, 2.85, 'Planar alignment: re-observe after each synchronized command', ha='center', fontsize=9)
    ax.plot([6.92, 6.92, 1.08, 1.08], [1.93, 2.7, 2.7, 1.95], color='#375765', lw=.8, ls='--')
    arrow(1.08, 2.14, 1.08, 1.93)
    ax.text(7.85, .22, 'Contact stage: no live planar vision', fontsize=8, ha='center')
    fig.subplots_adjust(left=0, right=1, bottom=.02, top=.98)
    fig.savefig(figures/'workflow.pdf'); plt.close(fig)


def failure_stages(contact, figures):
    names = ('success', 'before_descent', 'during_descent', 'terminal_hold')
    groups, sources = {}, {}
    for summary in contact['summaries']:
        group = summary['group']
        if not group.startswith('open-'):
            continue
        path = ROOT/RUNS/'contact-study-v2'/group/'episodes.jsonl'
        require(digest(path) == contact['raw_audits'][group]['raw_sha256'], 'Contact raw hash mismatch')
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        require(len(rows) == summary['n'], 'Incorrect phase-analysis coverage')
        counts = Counter({k:0 for k in names}); reasons = Counter(); ids = {k:[] for k in names}
        for r in rows:
            ins = r['insertion']
            if r['success']:
                phase = 'success'
            elif ins.get('hold') is not None or ins['reason'].startswith('terminal_hold_'):
                phase = 'terminal_hold'
            elif ins.get('trace'):
                phase = 'during_descent'
            else:
                phase = 'before_descent'
            counts[phase] += 1; ids[phase].append(r['start_id'])
            if phase != 'success': reasons[phase+':'+r['reason']] += 1
        require(counts['success'] == summary['successes'] and sum(counts.values()) == summary['n'], 'Stage totals disagree')
        groups[group] = {'n':len(rows), 'counts':dict(counts), 'ids':ids, 'failure_reasons':dict(reasons)}
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    pools = {}
    for label, select in (
        ('learned_rgb', lambda g: g.endswith('-rgb') and '-half-' not in g),
        ('half_rgb', lambda g: g=='open-fit0-half-rgb'),
        ('s1_gt', lambda g: g=='open-fit0-s1-gt')):
        chosen = [v for k,v in groups.items() if select(k)]
        pools[label] = {'n':sum(x['n'] for x in chosen),
                        'counts':{k:sum(x['counts'][k] for x in chosen) for k in names}}
    require(sum(v['n'] for v in pools.values()) == 1024, 'Not all main open trials classified')
    result = {'scope':'All 1024 main open-hole trials; stress and blocked controls excluded from pooling.',
              'classification':'Success first; then terminal-hold failure by hold record or terminal_hold_ reason; remaining nonempty insertion traces are descent failures; all other failures occurred before descent.',
              'groups':groups,'pools':pools,'raw_sha256':sources}
    write(ROOT/'verification/contact-failure-stages.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verified', type=Path, required=True, help='Output directory from reproduce.py verify')
    args = parser.parse_args()
    evidence = args.verified.resolve()
    figures = ROOT / 'report/media'
    template_data = ROOT / 'report/data.tex'
    paths = {n: evidence / n / 'statistics.json' for n in ('deterministic', 'learned', 'contact')}
    b, l, c = [read(p) for p in paths.values()]
    require(read(evidence / 'result.json')['statistics_match'] and l['complete'] and c['complete'], 'Need complete verification first')
    require(sum(x['n'] for x in b['summaries']) == 23456 and sum(x['n'] for x in l['summaries']) == 25280 and
            c['observed_episodes'] == 1216, 'Unexpected study sizes')

    plt.rcParams.update({'axes.labelcolor': '#243845', 'xtick.color': '#243845', 'ytick.color': '#243845',
                         'text.color': '#243845', 'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'axes.labelsize': 9.5})
    workflow(figures)
    phases = failure_stages(c, figures)
    fit0 = {r['method']: r for r in b['summaries'] if r['scenario'] == 'core' and r['fit'] == 0}
    keys = ('full_info', 'local_model', 'capability', 'belief_prior', 'prediction', 'intent', 'half')
    labels = ('S0  Joint reference', 'S1  Local model', 'S2  Capability', 'S3 control  Fixed belief',
              'S3  History belief', 'S4  Explicit bid', 'Fixed half')
    colors = ['#203d51', '#117674', '#3275a1', '#80929c', '#3275a1', '#3275a1', '#80929c']
    y = np.arange(len(keys))
    mean = np.array([fit0[k]['mean_cost_rad2'] for k in keys]) * 1000
    bounds = np.array([fit0[k]['cost_bootstrap_95'] for k in keys]) * 1000
    payload = np.array([fit0[k]['mean_bytes'] for k in keys])
    fig, (ax, by) = plt.subplots(1, 2, sharey=True, figsize=(7.1, 2.75), gridspec_kw={'width_ratios': [1.25, 1]})
    ax.barh(y, mean, height=.54, color=colors, zorder=3)
    ax.errorbar(mean, y, xerr=[mean-bounds[:, 0], bounds[:, 1]-mean], fmt='none', ecolor='#243845', capsize=2, lw=.85, zorder=4)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 6.5)
    ax.set_xlabel(r'Motion cost ($10^{-3}$ rad$^2$/episode)')
    by.barh(y, payload, height=.54, color=colors, zorder=3)
    by.set_xlabel('Explicit payload (bytes/episode)'); by.set_xlim(0, 188)
    for i, (cost, amount) in enumerate(zip(mean, payload, strict=True)):
        ax.text(bounds[i, 1]+.09, i, f'{cost:.3f}', va='center', fontsize=8)
        by.text(amount+3, i, 'n/a' if i == 0 else f'{amount:.1f}' if amount else '0', va='center', fontsize=8)
    for item in (ax, by): axes_style(item)
    fig.subplots_adjust(left=.25, right=.98, top=.97, bottom=.19, wspace=.24)
    fig.savefig(figures / 'cost-payload.pdf'); plt.close(fig)

    contact = {r['group']: r for r in c['summaries']}
    open_ids = ['open-fit0-'+s+'-rgb' for s in ('s1', 's2', 's3', 's4', 'half')] + ['open-fit0-s1-gt']
    names = ['S1 / RGB', 'S2 / RGB', 'S3 / RGB', 'S4 / RGB', 'Fixed half / RGB', 'S1 / ground truth']
    rows = [contact[x] for x in open_ids]
    p = np.array([r['success_rate'] for r in rows]); ci = np.array([r['success_wilson_95'] for r in rows]); y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(7.1, 2.45))
    for i, (r, col) in enumerate(zip(rows, ['#3275a1']*4 + ['#117674', '#8f591e'], strict=True)):
        ax.errorbar(p[i], i, xerr=[[p[i]-ci[i, 0]], [ci[i, 1]-p[i]]], fmt='o', color=col, capsize=3, lw=1.4, zorder=4)
        ax.text(ci[i, 1]+.015, i, f'{r["successes"]}/{r["n"]}', ha='left', va='center', fontsize=8.5)
    ax.set_xlim(.45, 1.065); ax.set_xticks(np.linspace(.5, 1, 6)); ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel('Successful insertion (95% Wilson interval)')
    ax.set_yticks(y, names); ax.invert_yaxis(); axes_style(ax)
    fig.subplots_adjust(left=.22, right=.98, top=.96, bottom=.24)
    fig.savefig(figures / 'insertion-success.pdf'); plt.close(fig)

    traces = {k: ROOT / RUNS / 'assembly-demo-v7' / f'episode-0-{i}-learned.json' for i, k in enumerate(('Open case', 'Blocked case'))}
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.32)); ax[0].set_title('Measured insertion depth'); ax[1].set_title('Peak normal force by command')
    for (label, path), col in zip(traces.items(), ['#117674', '#a35e24'], strict=True):
        ins = read(path)['insertion']; trace = ins['trace']; x = np.arange(1, len(trace)+1)
        depth = [t['state']['depth_m']*1000 for t in trace]; forces = [t['peak_normal_force_n'] for t in trace]
        ax[0].plot(x, depth, label=label, color=col, lw=1.6); ax[1].plot(x, forces, label=label, color=col, lw=1.6)
        if ins.get('recovery', {}).get('released'):
            ax[0].plot([x[-1], x[-1]+1], [depth[-1], ins['final']['depth_m']*1000], '--o', color=col, markersize=3)
    ax[0].axhline(20, color='#61717b', ls=':', lw=1); ax[1].axhline(2, color='#61717b', ls=':', lw=1)
    ax[0].set_ylabel('Depth (mm)'); ax[1].set_ylabel('Force (N)')
    for a in ax:
        a.set_xlabel('Descent command index'); axes_style(a, 'y'); a.margins(x=.025)
        a.tick_params(labelsize=10); a.xaxis.label.set_fontsize(10); a.yaxis.label.set_fontsize(10)
    ax[0].legend(frameon=False, loc='upper left'); fig.subplots_adjust(left=.08, right=.99, top=.87, bottom=.22, wspace=.31)
    fig.savefig(figures / 'contact-traces.pdf'); plt.close(fig)

    learned = {(r['fit'], r['regime'], r['sensing']): r for r in l['summaries'] if r['condition'] == 'core'}
    gt_delta = next(x for x in c['paired_comparisons'] if (x['first'], x['second']) == ('open-fit0-s1-rgb', 'open-fit0-s1-gt'))
    blocked = [r for k, r in contact.items() if k.startswith('blocked-')]
    open_groups = [r for k, r in contact.items() if k.startswith('open-')]
    learned_open = [r for r in open_groups if '-half-' not in r['group'] and r['group'].endswith('-rgb')]
    pct = lambda f: f'{100*f:.1f}'
    fit_gap = 100*(fit0['local_model']['mean_cost_rad2']/fit0['full_info']['mean_cost_rad2']-1)
    macros = {
      'LocalGap': f'{fit_gap:.1f}', 'LocalCost': f'{fit0["local_model"]["mean_cost_rad2"]*1000:.3f}',
      'OracleCost': f'{fit0["full_info"]["mean_cost_rad2"]*1000:.3f}',
      'CapPayload': f'{fit0["capability"]["mean_bytes"]:.1f}', 'BidPayload': f'{fit0["intent"]["mean_bytes"]:.1f}',
      'RgbAlignRange': f'{pct(min(r["success_rate"] for r in learned.values() if r["sensing"]=="rgb"))}--{pct(max(r["success_rate"] for r in learned.values() if r["sensing"]=="rgb"))}',
      'LearnedContactRange': f'{pct(min(r["success_rate"] for r in learned_open))}--{pct(max(r["success_rate"] for r in learned_open))}',
      'HalfInserted': str(contact['open-fit0-half-rgb']['successes']), 'TruthInserted': str(contact['open-fit0-s1-gt']['successes']),
      'ContactDiff': pct(gt_delta['success_difference']),
      'ContactDiffCI': ', '.join(pct(v) for v in gt_delta['success_paired_bootstrap_95']),
      'BlockedInserted': str(sum(r['successes'] for r in blocked)), 'BlockedN': str(sum(r['n'] for r in blocked)),
      'BlockedReleased': str(sum(r['released_recoveries'] for r in blocked)), 'BlockedTried': str(sum(r['recovery_attempts'] for r in blocked)),
      'OpenReleased': str(sum(r['released_recoveries'] for r in open_groups)), 'OpenTried': str(sum(r['recovery_attempts'] for r in open_groups)),
    }
    pool = phases['pools']['learned_rgb']['counts']
    for name, key in [('LearnedPreFail','before_descent'), ('LearnedDescentFail','during_descent'), ('LearnedHoldFail','terminal_hold')]:
        macros[name] = str(pool[key])
    for name, reason in [('LearnedPoseReject','alignment_gate'), ('LearnedVisionReject','estimated_alignment_gate')]:
        macros[name] = str(sum(v['failure_reasons'].get('before_descent:'+reason, 0)
                               for k,v in phases['groups'].items() if k.endswith('-rgb') and '-half-' not in k))
    slash = '\\'; end = ' '+slash*2
    lines = ['% Generated from the verified episode analyses. Do not hand-edit numerical cells.']
    lines.extend(r'\newcommand{'+slash+k+'}{'+v+'}' for k,v in macros.items())
    table = []
    table.append(r'\multicolumn{5}{@{}l}{\textit{Deterministic allocation references (GT only)}}'+end)
    for name, key in [('S0 joint reference','full_info'), ('S1 local model','local_model'), ('Fixed half','half')]:
        r = fit0[key]
        table.append(f'{name} & 100.0 & {r["mean_cost_rad2"]*1000:.3f} & -- & --'+end)
    table.append(r'\midrule')
    table.append(r'\multicolumn{5}{@{}l}{\textit{Learned policies: same frozen actor under GT and RGB}}'+end)
    for g in ('s1','s2','s3','s4'):
        r = learned[0,g,'rgb']; gt = learned[0,g,'gt']
        table.append(f'PPO {g.upper()} & {pct(gt["success_rate"])} & {gt["mean_cost_rad2"]*1000:.3f} & {pct(r["success_rate"])} & {r["mean_cost_rad2"]*1000:.3f}'+end)
    lines.extend([r'\newcommand{\AlignmentRows}{%', *table, '}'])
    rows = []
    for name, key in [('Learned RGB (six conditions)','learned_rgb'),('Fixed-half RGB','half_rgb'),('S1 / ground truth','s1_gt')]:
        v = phases['pools'][key]; count = v['counts']
        rows.append(name+f' & {v["n"]} & {count["before_descent"]} & {count["during_descent"]} & {count["terminal_hold"]} & {count["success"]}'+end)
    lines.extend([r'\newcommand{\FailureRows}{%', *rows, '}'])
    template_data.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    record = {'statistic_sha256': {k:digest(v) for k,v in paths.items()},
              'published_statistic_sha256': {k:digest(ROOT/RUNS/f/'analysis/statistics.json') for k,f in (
                  ('deterministic','evaluation'),('learned','learned-study'),('contact','contact-study-v2'))},
              'trace_sha256': {p.name:digest(p) for p in traces.values()}, 'verification_sha256':digest(evidence/'result.json'),
              'outputs_sha256': {p.relative_to(ROOT).as_posix():digest(p) for p in (
                  template_data, ROOT/'verification/contact-failure-stages.json',
                  *(figures/f for f in ('workflow.pdf','cost-payload.pdf','insertion-success.pdf','contact-traces.pdf')))},
              'figure_source_sha256': digest(Path(__file__))}
    write(ROOT/'verification/report-data-provenance.json', record)


if __name__ == '__main__':
    main()
