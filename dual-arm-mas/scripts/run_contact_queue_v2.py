"""Scheduling-only queue for disjoint contact groups within ONE bounded slice."""
import argparse,time,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from evaluate_assembly_v2 import run

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('dual-arm-mas/configs/contact_study_v2.json'))
 p.add_argument('--groups',required=True);p.add_argument('--seconds',type=float,default=240);p.add_argument('--window-end',required=True)
 args=p.parse_args();began=time.monotonic();end=datetime.fromisoformat(args.window_end)
 if end.tzinfo is None:raise ValueError('Timezone-aware end required')
 print(json.dumps({'scheduler_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'groups':args.groups,'seconds':args.seconds,'window_end':args.window_end}),flush=True)
 for group in args.groups.split(','):
  remaining=args.seconds-(time.monotonic()-began)
  if remaining<5 or datetime.now(timezone.utc)>=end:break
  run(argparse.Namespace(config=args.config,group=group,seconds=remaining,window_end=args.window_end))
