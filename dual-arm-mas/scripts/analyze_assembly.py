"""Audit and paired summaries for the separately frozen contact-transfer study."""
from pathlib import Path
import argparse,json,hashlib,math
from collections import Counter
import numpy as np

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def wilson(k,n):
 if not n:return [0.,1.]
 z=1.95996398454;p=k/n;d=1+z*z/n;center=(p+z*z/(2*n))/d;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0.,center-radius),min(1.,center+radius)]
def run(args):
 cfg=json.loads(args.config.read_text());root=Path(cfg['output']);out=root/'analysis';groups={};summaries=[];audits={};missing=[]
 if list(root.rglob('writer.lock')):raise ValueError('Wait for bounded writers before analyzing')
 for path,h in cfg['runtime_source_sha256'].items():
  if digest(path)!=h:raise ValueError('Current frozen contact source mismatch: '+path)
 starts=json.loads(Path(cfg['starts']).read_text())['starts'];assert digest(cfg['starts'])==cfg['starts_sha256']
 known={x['id']:x for x in starts}
 for group in cfg['groups']:
  folder=root/group['id'];raw=folder/'episodes.jsonl';expected={x['id'] for x in starts[:group['count']]}
  if not raw.exists():missing.append({'group':group['id'],'remaining':group['count']});continue
  provenance=json.loads((folder/'provenance.json').read_text());assert provenance['group']==group
  for path,h in provenance['pins'].items():
   assert digest(path)==h,path
   if Path(path).suffix=='.py':assert digest(folder/'source_snapshot'/Path(path).name)==h,path
  blob=raw.read_bytes();assert blob.endswith(b'\n');rows=[json.loads(line) for line in blob.splitlines()]
  ids={r['start_id'] for r in rows};assert len(ids)==len(rows) and ids<=expected
  if ids!=expected:missing.append({'group':group['id'],'remaining':len(expected-ids)})
  for r in rows:
   assert r['group']==group['id'] and r['method']==group['method'] and r['regime']==group['regime'] and r['fit']==group['fit'] and r['sensing']==group['sensing']
   assert r['initial_joints']==known[r['start_id']]['q']
   assert r['success']==r['insertion']['success']
   for t in r['alignment']['trace']:
    if t['local_features'] is not None:
     for i,n in enumerate(['A','B']):np.testing.assert_allclose(t['local_features'][i][:3],np.clip(np.array(t['actor_error'][n])/[.002,.002,.04],-10,10),atol=1e-6)
   if group['sensing']=='rgb':
    valid=[d['error'] for d in r['alignment']['perceptions'] if d['valid']]
    for t,e in zip(r['alignment']['trace'],valid):np.testing.assert_allclose(t['actor_error']['A'],e,atol=1e-12)
   if r['success']:
    ins=r['insertion'];assert ins['reason']=='inserted' and ins['hold']['stop_reason'] is None
    assert ins['final']['depth_m']>=.0195 and max(abs(x) for x in ins['final']['tip_in_fixture_m'][:2])<=.001 and abs(ins['final']['error_m_m_rad'][2])<=np.deg2rad(1)
   for key in ['alignment_cost_rad2','insertion_endpoint_cost_rad2','alignment_joint_travel_rad','insertion_endpoint_joint_travel_rad']:
    assert math.isfinite(r[key]) and r[key]>=0
  groups[group['id']]=rows;audits[group['id']]={'raw_sha256':digest(raw),'provenance_sha256':digest(folder/'provenance.json'),'n':len(rows),'missing_ids':sorted(expected-ids)}
  n=len(rows);k=sum(r['success'] for r in rows);costs=[r['alignment_cost_rad2']+r['insertion_endpoint_cost_rad2'] for r in rows]
  peaks=[];penetrations=[]
  for r in rows:
   ins=r['insertion'];phases=[r['settle']]+[t['dynamic_execution_contact'] for t in r['alignment']['trace'] if t.get('dynamic_execution_contact')]+ins['trace']
   if ins.get('hold'):phases.append(ins['hold'])
   if ins.get('recovery',{}).get('execution'):phases.append(ins['recovery']['execution'])
   states=[ins['final']]+([ins['stopped']] if ins.get('stopped') else [])
   force=max([x['peak_normal_force_n'] for x in phases]+[x['normal_force_n'] for x in states]+[0.])
   penetration=max([x['peak_penetration_m'] for x in phases]+[x['penetration_m'] for x in states]+[0.])
   assert math.isfinite(force) and math.isfinite(penetration)
   peaks.append(force);penetrations.append(penetration)
  recoveries=[r['insertion']['recovery'] for r in rows if r['insertion'].get('recovery',{}).get('attempted')]
  sensor_errors=[]
  for r in rows:
   for d in r['alignment']['perceptions']:
    if d['valid']:
     error=np.array(d['error'])-np.array(d['true_error_diagnostic_only']);error[2]=(error[2]+np.pi)%(2*np.pi)-np.pi;sensor_errors.append(abs(error))
  summaries.append({'sensing_observations':len(sensor_errors),'sensing_axis_mae_m_m_rad':np.mean(sensor_errors,axis=0).tolist() if sensor_errors else None,
   'sensing_axis_p95_m_m_rad':np.quantile(sensor_errors,.95,axis=0).tolist() if sensor_errors else None,'group':group['id'],'n':n,'successes':k,'success_rate':k/n,'success_wilson_95':wilson(k,n),
   'mean_endpoint_command_cost_rad2':float(np.mean(costs)),'success_conditioned_cost_rad2':float(np.mean([c for c,r in zip(costs,rows) if r['success']])) if k else None,
   'mean_endpoint_joint_travel_rad':float(np.mean([r['alignment_joint_travel_rad']+r['insertion_endpoint_joint_travel_rad'] for r in rows])),
   'mean_alignment_commands':float(np.mean([r['alignment_commands'] for r in rows])),'mean_insertion_commands':float(np.mean([r['insertion_commands'] for r in rows])),
   'mean_simulation_seconds':float(np.mean([r['simulation_seconds'] for r in rows])),'mean_payload_bytes':float(np.mean([r['payload_bytes'] for r in rows])),
   'mean_peak_force_n':float(np.mean(peaks)),'max_peak_force_n':max(peaks),'max_penetration_m':max(penetrations),'peak_force_scope':'Peg-fixture aggregate across logged settling, alignment, descent, recovery and hold phases; not robot/environment contact force.','failure_categories':dict(Counter(r['reason'] for r in rows if not r['success'])),
   'recovery_attempts':len(recoveries),'released_recoveries':sum(x['released'] for x in recoveries),
   'mean_final_depth_mm':float(np.mean([r['insertion']['final']['depth_m']*1000 for r in rows]))})
 rng=np.random.default_rng(cfg['bootstrap_seed']);contrasts=[]
 pairs=[(f'open-fit0-{reg}-rgb','open-fit0-half-rgb') for reg in ['s1','s2','s3','s4']]
 pairs += [('open-fit0-s1-rgb','open-fit0-s1-gt')]+[(f'open-fit0-{reg}-rgb','open-fit0-s1-rgb') for reg in ['s2','s3','s4']]
 for g in cfg['groups']:
  if not g['id'].startswith('open-'):pairs.append((g['id'],f"open-fit{g['fit']}-{g['regime']}-{g['sensing']}"))
 for first,second in pairs:
  if first not in groups or second not in groups:continue
  a={r['start_id']:r for r in groups[first]};b={r['start_id']:r for r in groups[second]};ids=sorted(a.keys()&b.keys())
  if not ids:continue
  d=np.array([a[i]['alignment_cost_rad2']+a[i]['insertion_endpoint_cost_rad2']-b[i]['alignment_cost_rad2']-b[i]['insertion_endpoint_cost_rad2'] for i in ids])
  sd=np.array([int(a[i]['success'])-int(b[i]['success']) for i in ids]);ix=rng.integers(0,len(ids),(cfg['bootstrap_resamples'],len(ids)))
  contrasts.append({'first':first,'second':second,'paired_n':len(ids),'mean_cost_difference_rad2':float(d.mean()),'cost_paired_bootstrap_95':np.quantile(d[ix].mean(1),[.025,.975]).tolist(),
   'success_difference':float(sd.mean()),'success_paired_bootstrap_95':np.quantile(sd[ix].mean(1),[.025,.975]).tolist()})
 result={'complete':not missing,'config_sha256':digest(args.config),'analysis_source_sha256':digest(__file__),
  'observed_episodes':sum(len(x) for x in groups.values()),'expected_episodes':cfg['expected_episodes'],'missing':missing,
  'summaries':summaries,'paired_comparisons':contrasts,'raw_audits':audits,'limits':cfg['limits']}
 if missing and not args.allow_partial:raise ValueError('Incomplete contact study: '+str(missing))
 out.mkdir(exist_ok=True);(out/'statistics.json').write_text(json.dumps(result,indent=2))
 lines=['# '+('COMPLETE' if result['complete'] else 'INCOMPLETE')+' contact-transfer extension','',cfg['scope'],'',
  '|Group|N|Inserted|Mean cost rad2|Peak force N|Released/attempts|','|---|---:|---:|---:|---:|---:|']
 for x in summaries:lines.append(f"|{x['group']}|{x['n']}|{x['successes']}|{x['mean_endpoint_command_cost_rad2']:.6f}|{x['max_peak_force_n']:.3f}|{x['released_recoveries']}/{x['recovery_attempts']}|")
 (out/'summary.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'complete':result['complete'],'episodes':result['observed_episodes'],'remaining':sum(x['remaining'] for x in missing)}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('dual-arm-mas/configs/contact_study_v2.json'));p.add_argument('--allow-partial',action='store_true');run(p.parse_args())
