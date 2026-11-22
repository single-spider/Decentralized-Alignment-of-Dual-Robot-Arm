"""Standalone grader entry point. Study code and historical records are immutable inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PureWindowsPath
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MAS = Path('dual-arm-mas')
RUNS = MAS / 'runs'
CONFIG = MAS / 'configs'
FOUNDATION = Path('peg-in-hole-sfn')


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def confined(root, rel):
    candidate = (root / rel).resolve()
    require(candidate.is_relative_to(root.resolve()) and candidate != root.resolve(), f'Unsafe path: {rel}')
    return candidate


def restore(root):
    for row in read(root / 'evidence/archives.json'):
        target = confined(root, row['path'])
        archive = confined(root, row['archive'])
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                tmp = Path(stream.name)
                try:
                    with gzip.open(archive, 'rb') as src:
                        shutil.copyfileobj(src, stream, 2**20)
                    stream.close()
                    require(tmp.stat().st_size == row['size'] and digest(tmp) == row['sha256'], 'Bad archive payload')
                    require(not target.exists(), 'Concurrent restore: refusing to overwrite')
                    tmp.rename(target)
                finally:
                    tmp.unlink(missing_ok=True)
        require(target.stat().st_size == row['size'] and digest(target) == row['sha256'], f'Restore mismatch: {target}')
        print(f'Restored or verified {row["path"]}: {row["sha256"]}', flush=True)


def check_manifest():
    manifest = read(ROOT / 'manifest.sha256.json')
    for rel, expected in manifest['files'].items():
        target = confined(ROOT, rel)
        require(target.is_file() and digest(target) == expected, f'Bundle hash mismatch: {rel}')
    print(f'Verified {len(manifest["files"])} bundled hashes.', flush=True)


def boot(root):
    os.chdir(root)
    sys.path[:0] = [str(root / p) for p in (FOUNDATION, MAS / 'src', MAS / 'scripts')]
    for name in ('sfn', 'dual_arm_mas'):
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve().is_relative_to(root), f'{name} imported from outside bundle')
        print(f'{name}: {Path(module.__file__).relative_to(root).as_posix()}', flush=True)


def fresh_out(rel):
    out = confined(ROOT / '.work', rel)
    require(not out.exists(), f'Output already exists: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def rebase_legacy_key(raw, scratch):
    win = PureWindowsPath(raw)
    if not win.is_absolute():
        relative = Path(*win.parts)
        return str(confined(scratch, relative))
    parts = win.parts
    roots = [i for i, part in enumerate(parts) if part in (MAS.name, FOUNDATION.name)]
    require(len(roots) == 1, f'Unrecognized historical pin: {raw}')
    return str(confined(scratch, Path(*parts[roots[0]:])))


def relocate_keys(path, field, scratch, *, baseline=False):
    data = read(path)
    relocated = {}
    for p, sha in data[field].items():
        key = rebase_legacy_key(p, scratch)
        require(key not in relocated or relocated[key] == sha, f'Conflicting source pins: {key}')
        current = Path(key)
        if not current.is_file() or digest(current) != sha:
            # Baseline provenance predates some analysis-only edits. Only its
            # explicit historical snapshots can satisfy those original pins.
            archived = path.parent / 'source_snapshot' / current.relative_to(scratch)
            require(baseline and archived.is_file() and digest(archived) == sha,
                    f'Relocated source does not match historical SHA-256: {p}')
        relocated[key] = sha
    data[field] = relocated
    write(path, data)


def close_tree(a, b, loc='result'):
    if isinstance(a, dict) and isinstance(b, dict):
        require(a.keys() == b.keys(), f'Mismatched fields at {loc}')
        for k in a:
            close_tree(a[k], b[k], f'{loc}.{k}')
    elif isinstance(a, list) and isinstance(b, list):
        require(len(a) == len(b), f'Mismatched length at {loc}')
        for i, (x, y) in enumerate(zip(a, b)):
            close_tree(x, y, f'{loc}[{i}]')
    elif isinstance(a, float) and isinstance(b, (float, int)):
        require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), f'Numeric mismatch at {loc}: {a} != {b}')
    else:
        require(a == b, f'Mismatch at {loc}: {a} != {b}')


def verify(out):
    check_manifest()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='sfn-g28-') as folder:
        scratch = Path(folder).resolve()
        for p in (MAS / 'src', MAS / 'scripts', CONFIG, FOUNDATION, Path('evidence')):
            shutil.copytree(ROOT / p, scratch / p, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ['evaluation', 'learned-study', 'contact-study-v2', 'contact-momentum-fix', 'test', 'validation',
                     'train-311', 'train-312', 'train-313', 'vision-adaptation']:
            shutil.copytree(ROOT / RUNS / name, scratch / RUNS / name,
                            ignore=shutil.ignore_patterns('episodes.jsonl') if name == 'evaluation' else None)
        for regime in ('s1', 's2', 's3', 's4'):
            for seed in (401, 402, 403):
                p = RUNS / f'rl-{regime}-{seed}'
                shutil.copytree(ROOT / p, scratch / p)
        restore(scratch)
        base = RUNS / 'evaluation'
        learned = RUNS / 'learned-study'
        contact = RUNS / 'contact-study-v2'
        relocate_keys(scratch / base / 'provenance.json', 'sha256', scratch, baseline=True)
        frozen_paths = [scratch / learned / 'core', *(scratch / learned / name for name in (
            'drop25', 'delay1', 'unequal_limits', 'rgb_occlusion', 'history0', 'history1'))]
        for path in frozen_paths:
            for meta in path.rglob('provenance.json'):
                relocate_keys(meta, 'pins', scratch)
        try:
            boot(scratch)
            import analyze
            import analyze_learned
            import analyze_assembly

            analyze.analyze(base / 'episodes.jsonl', out / 'deterministic')
            analyze_learned.run(argparse.Namespace(config=CONFIG / 'learned_study.json', out=out / 'learned',
                                                   allow_partial=False, skip_baseline=False))
            analyze_assembly.run(argparse.Namespace(config=CONFIG / 'contact_study_v2.json', allow_partial=False))
            (out / 'contact').mkdir()
            for name in ('statistics.json', 'summary.md'):
                shutil.copy2(contact / 'analysis' / name, out / 'contact' / name)
            b, l, c = [read(out / sub / 'statistics.json') for sub in ('deterministic', 'learned', 'contact')]
            expected = {key: read(ROOT / p / 'analysis/statistics.json') for key, p in (
                ('deterministic', base), ('learned', learned), ('contact', contact))}
            close_tree(b, expected['deterministic'], 'deterministic')
            for key in ('summaries', 'comparisons', 'deterministic_baselines', 'selection_audit', 'continuation_audit'):
                close_tree(l[key], expected['learned'][key], f'learned.{key}')
            for key in ('summaries', 'paired_comparisons', 'raw_audits'):
                close_tree(c[key], expected['contact'][key], f'contact.{key}')
            for curr, old in zip(l['training'], expected['learned']['training'], strict=True):
                # Copy/checkout mtime is not evidence of historical wall span.
                for key in ('fit', 'regime', 'seed', 'steps', 'updates', 'log_sha256', 'checkpoint_parameter_evidence',
                            'recorded_update_wall_seconds', 'training_provenance_sha256'):
                    close_tree(curr[key], old[key], f'training.{key}')
            result = {'deterministic': sum(x['n'] for x in b['summaries']),
                      'learned_alignment': sum(x['n'] for x in l['summaries']),
                      'corrected_contact': c['observed_episodes'], 'contact_groups': len(c['summaries']),
                      'ppo_fits': len(l['training']), 'statistics_match': True,
                      'scope': 'Recomputed from saved episode records; not a rerun of all simulations.'}
            require(list(result.values())[:5] == [23456, 25280, 1216, 14, 12] and l['complete'] and c['complete'],
                    'Study incomplete or unexpected episode counts')
            write(out / 'result.json', result)
            print(json.dumps(result), flush=True)
        finally:
            os.chdir(ROOT)


def bundle_provenance():
    paths = list((ROOT / MAS / 'src/dual_arm_mas').glob('*.py')) + list((ROOT / MAS / 'scripts').glob('*.py'))
    paths += list((ROOT / CONFIG).glob('*.json')) + list((ROOT / FOUNDATION / 'sfn').rglob('*.py'))
    paths += [Path(__file__).resolve()]
    return {'time_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
            'revision': 'standalone bundle; exact source hashes below', 'execution_scope': 'new reproduction run',
            'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'pybullet', 'Pillow', 'torch')},
            'sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}}


def execute(args, out):
    boot(ROOT)
    require(0 < args.seconds <= 86400, 'Use a finite positive work budget (at most 24 hours per invocation)')
    end = datetime.now(timezone.utc) + timedelta(seconds=args.seconds)
    deadline = time.monotonic() + args.seconds
    gate = lambda: time.monotonic() < deadline
    cfg = read(CONFIG / 'learned_study.json')
    if args.action == 'demo':
        import demo_assembly as mod
        mod.run(argparse.Namespace(out=out, seconds=args.seconds, window_end=end.isoformat()))
        result = read(out / 'result.json')
        require(result['completed_episodes'] == result['expected_episodes'] == 4, 'Replay did not finish all four cases')
        return
    if args.action == 'train':
        import train_rl as mod
        mod.allowed_time, mod.provenance = gate, bundle_provenance
        starts = RUNS / f'train-{311 + args.fit}'
        mod.run(argparse.Namespace(out=out, starts=starts / 'starts.json', public=starts / 'public.json',
                                   regime=args.regime, seed=401 + args.fit, steps=args.steps, seconds=args.seconds,
                                   batch_steps=args.batch_steps, epochs=args.epochs, lr=.0003, resume=None, eval_only=False))
        require((out / 'latest.pt').is_file(), 'Budget ended without a completed PPO update')
        return
    if args.action == 'vision-train':
        import adapt_vision as mod
        mod.allowed_time = gate
        out.mkdir()
        for p in ('train.npz', 'train.json', 'validation.npz', 'validation.json'):
            shutil.copy2(RUNS / 'vision-adaptation' / p, out / p)
        mod.run(argparse.Namespace(out=out, seed=702, count=192, epochs=args.epochs, seconds=args.seconds, resume=None))
        require((out / 'latest.pt').is_file(), 'Budget ended without a completed vision epoch')
        return
    if args.action == 'alignment-eval':
        import evaluate_learned as mod
        mod.allowed_time = gate
        fit = cfg['fits'][args.fit]
        condition = {}
        if args.condition != 'core':
            condition = next((c for c in cfg['robustness'] if c['name'] == args.condition), None)
            require(condition is not None and args.fit == 0 and args.regime in condition['regimes'] and
                    args.sensing == condition['sensing'], 'Use the condition’s declared fit, regime and sensing')
        checkpoint = Path(fit['policies'][args.regime]['checkpoint'])
        segment = Path(cfg['perception']) if args.sensing == 'rgb' else None
        require(digest(checkpoint) == fit['policies'][args.regime]['sha256'] and
                (segment is None or digest(segment) == cfg['perception_sha256']), 'Frozen weights changed')
        mod.run(argparse.Namespace(out=out, seconds=args.seconds, checkpoint=checkpoint, public=Path(fit['public']),
                                   starts=Path(cfg['test_starts']), sensing=args.sensing,
                                   count=min(args.count, condition.get('count', cfg['core_count'])),
                                   seed=cfg['evaluation_seed'], decoder=cfg['decoder'], drop=condition.get('drop', 0.),
                                   delay=condition.get('delay', 0), b_limit_factor=condition.get('b_limit_factor', 1.),
                                   occlusion=condition.get('occlusion', 0.), history_length=condition.get('history_length', 4),
                                   segmentation=segment,
                                   crop=segment is not None, images=False))
        require((out / 'episodes.jsonl').is_file(), 'No alignment episode completed')
        return
    if args.action == 'contact-eval':
        import evaluate_assembly_v2 as mod
        study = read(CONFIG / 'contact_study_v2.json')
        group = next((g for g in study['groups'] if g['id'] == args.group), None)
        require(group is not None, f'Unknown contact group: {args.group}')
        out.mkdir()
        group['count'] = min(group['count'], args.count)
        study['output'] = out.relative_to(ROOT).as_posix()
        config = out / 'fresh-config.json'
        write(config, study)
        mod.run(argparse.Namespace(config=config, group=args.group, seconds=args.seconds, window_end=end.isoformat()))
        require((out / args.group / 'episodes.jsonl').is_file(), 'No contact episode completed')
        return
    if args.action == 'baseline-eval':
        import evaluate as mod
        mod.provenance = bundle_provenance
        study = read(CONFIG / 'evaluation.json')
        selected = [g for g in study['scenarios'] if g['name'] == args.scenario]
        require(selected, f'Unknown baseline scenario: {args.scenario}')
        for g in selected:
            g['count'] = min(args.count, g['count'])
        study['scenarios'], study['wrapup_at'] = selected, end.isoformat()
        out.mkdir()
        config = out / 'fresh-config.json'
        write(config, study)
        mod.run(argparse.Namespace(config=config, out=out, scenario=args.scenario, seconds=args.seconds))
        require((out / 'episodes.jsonl').stat().st_size > 0, 'No baseline episode completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('hashes', help='Check the immutable shipped file hashes')
    sub.add_parser('restore', help='Restore compressed raw evidence into the bundle and verify bytes')
    sub.add_parser('verify', help='Regenerate full saved-study statistics without resimulating').add_argument('--out', required=True)
    for name in ('demo', 'train', 'vision-train', 'alignment-eval', 'contact-eval', 'baseline-eval'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--out', required=True, help='New name or relative path within .work/')
        cmd.add_argument('--seconds', type=float, default=240)
        if name in ('train', 'alignment-eval'):
            cmd.add_argument('--fit', type=int, choices=(0, 1, 2), default=0)
            cmd.add_argument('--regime', choices=('s1', 's2', 's3', 's4'), default='s1')
        if name == 'train':
            cmd.add_argument('--steps', type=int, default=10000)
            cmd.add_argument('--batch-steps', type=int, default=256)
        if name in ('train', 'vision-train'):
            cmd.add_argument('--epochs', type=int, default=5 if name == 'train' else 16)
        if name.endswith('-eval'):
            cmd.add_argument('--count', type=int, default=1)
        if name == 'alignment-eval':
            cmd.add_argument('--sensing', choices=('rgb', 'gt'), default='rgb')
            cmd.add_argument('--condition', default='core', help='Core or a named robustness row in learned_study.json')
        if name == 'contact-eval':
            cmd.add_argument('--group', default='open-fit0-s1-rgb')
        if name == 'baseline-eval':
            cmd.add_argument('--scenario', default='core')
    args = parser.parse_args()
    if args.action == 'hashes':
        check_manifest()
    elif args.action == 'restore':
        restore(ROOT)
    else:
        require(not args.out or not Path(args.out).is_absolute(), '--out must be relative to .work/')
        for key in ('count', 'steps', 'batch_steps', 'epochs'):
            require(getattr(args, key, 1) > 0, f'--{key.replace("_", "-")} must be positive')
        out = fresh_out(args.out)
        if args.action == 'verify':
            verify(out)
        else:
            execute(args, out)


if __name__ == '__main__':
    main()
