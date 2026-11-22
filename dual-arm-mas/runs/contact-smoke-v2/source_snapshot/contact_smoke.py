"""First dynamic-contact acceptance: centered passage vs physical blocked hole."""
import argparse,json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dual_arm_mas.contact_world import ContactWorld,insert


def run(args):
    args.out.mkdir(parents=True,exist_ok=False);began=time.monotonic();rows=[]
    for blocked in [False,True]:
        with ContactWorld() as world:
            world.attach_parts(blocked=blocked)
            settle=world.advance(world.motor_targets,240)
            # Guard against reintroducing kinematic attachment/robot stepping.
            def forbidden(*a,**kw):raise AssertionError('Reset/sync in physics execution')
            original_sync=world.sync_parts;original_set=world.set_joints
            original_advance=world.advance
            def guarded(targets,steps=None):
                world.sync_parts=forbidden;world.set_joints=forbidden
                try:return original_advance(targets,steps)
                finally:world.sync_parts=original_sync;world.set_joints=original_set
            world.advance=guarded
            result=insert(world)
            rows.append({'blocked':blocked,'settle':settle,'result':result,'seconds':time.monotonic()-began})
            (args.out/f'case-{int(blocked)}.json').write_text(json.dumps(rows[-1],indent=2))
            print(json.dumps({'blocked':blocked,'success':result['success'],'reason':result['reason'],
                'depth_mm':result['final']['depth_m']*1000,'steps':len(result['trace']),
                'peak_force_n':max([r['peak_normal_force_n'] for r in result['trace']],default=0),
                'grasp_drift':result['final']['grasp_drift']}),flush=True)
    passed=rows[0]['result']['success'] and not rows[1]['result']['success'] and any(x['peak_normal_force_n']>0 for x in rows[1]['result']['trace'])
    result={'passed':passed,'scope':'Single nominal centered and blocked-hole controls only, not RL/vision integration or insertion study.',
        'execution_mode':'dynamic joint motors, gravity -9.81, massive collision parts, fixed constraints; no per-tick resets/sync',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).resolve().parents[1]/'src/dual_arm_mas/contact_world.py']},'cases':rows}
    (args.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'passed':bool(passed),'seconds':time.monotonic()-began}))
    if not passed:raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args())
