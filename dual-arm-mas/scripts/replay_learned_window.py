"""Fresh-process frozen demo with an explicitly supplied current calendar gate."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
import demo_learned


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    end=datetime.fromisoformat(args.window_end)
    if end.tzinfo is None or datetime.now(timezone.utc)>=end:raise ValueError('Supply a future timezone-aware window end')
    cfg_path=Path('dual-arm-mas/configs/learned_study.json');cfg=json.loads(cfg_path.read_text())
    for path,sha in cfg['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('Frozen scientific source changed: '+path)
    old=json.loads(Path('dual-arm-mas/runs/integrated-demo-neural-crop-v5/replay.json').read_text())
    for path,sha in old['pins'].items():
        if digest(path)!=sha:raise ValueError('Accepted demo dependency changed: '+path)
    if args.out.exists():raise ValueError('Use a new replay output directory')
    policy=cfg['fits'][0]['policies']['s1']
    if digest(policy['checkpoint'])!=policy['sha256'] or digest(cfg['perception'])!=cfg['perception_sha256']:
        raise ValueError('Frozen weights changed')
    started=datetime.now(timezone.utc).isoformat()
    demo_learned.allowed_time=lambda:datetime.now(timezone.utc)<end
    config=argparse.Namespace(checkpoint=Path(policy['checkpoint']),out=args.out,
        starts=Path('dual-arm-mas/runs/validation/starts.json'),public=Path(cfg['fits'][0]['public']),
        segmentation=Path(cfg['perception']),crop=True,ids='0,4',include_mode_failure=True)
    demo_learned.replay(config)
    shutil.copy2(__file__,args.out/'source_snapshot'/Path(__file__).name)
    (args.out/'renewed_replay_provenance.json').write_text(json.dumps({
        'started_utc':started,'ended_utc':datetime.now(timezone.utc).isoformat(),'window_end':args.window_end,
        'wrapper_sha256':digest(__file__),'original_entrypoint_sha256':digest(demo_learned.__file__),
        'study_config_sha256':digest(cfg_path),'replay_json_sha256':digest(args.out/'replay.json'),
        'scope':'Only the between-episode calendar gate is replaced; original accepted-demo dependencies, actors and perception verified unchanged.'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--window-end',required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
