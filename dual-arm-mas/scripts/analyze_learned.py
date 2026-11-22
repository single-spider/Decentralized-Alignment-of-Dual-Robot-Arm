"""Audited learned-study statistics; incomplete groups cannot certify delivery."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,collections,hashlib,json,sys
import numpy as np
from analyze import bootstrap

TOL=np.array([.001,.001,np.deg2rad(1)])


def wilson(successes,n):
    if not n:return None
    z=1.959963984540054;p=successes/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [0. if successes==0 else float(max(0,center-radius)),
            1. if successes==n else float(min(1,center+radius))]


def summarize(rows,rng):
    values=list(rows.values());n=len(values);successes=[r for r in values if r['success']]
    cost=np.array([r['motion_cost_rad2'] for r in values]);error=np.array([r['final_error'] for r in values])
    sensing=[];latencies=[];residuals=[];valid_sensing=[];prediction=[];overshoots=[];oscillations=[];same_direction=[];alignment_shares={'A_cheaper_X':[],'B_cheaper_X':[]}
    for row in values:
        count=0;same=0;axis_flips=np.zeros(3,dtype=int)
        for t in row['trace']:
            before=np.array(t['before']['error']);after=np.array(t['after']['error'])
            flipped=(before*after<0)&(abs(before)>TOL)&(abs(after)>TOL)
            count+=int(np.any(flipped));axis_flips+=flipped
            a,b=np.array(t['executed']['A']),np.array(t['executed']['B'])
            same+=int(np.any(a*b>1e-10))
            if 'own_H_diagnostic_only' in t and abs(a[0])+abs(b[0])>1e-7:
                h=t['own_H_diagnostic_only'];key='A_cheaper_X' if h['A'][0][0]<h['B'][0][0] else 'B_cheaper_X'
                alignment_shares[key].append(abs(a[0])/(abs(a[0])+abs(b[0])))
            for pred in t.get('prediction_diagnostic_only',{}).values():
                for estimate,label in zip(pred['predicted_fraction'],pred['executed_fraction_label']):
                    if label is not None:prediction.append(abs(estimate-label))
        overshoots.append(count);same_direction.append(same)
        oscillations.append(bool(np.any(axis_flips>=2)))
        sensing += [d['sensing_error'] for d in row.get('sensing_diagnostics',[]) if 'sensing_error' in d]
        latencies += [d['latency_seconds'] for d in row.get('sensing_diagnostics',[]) if 'latency_seconds' in d]
        residuals += [d['template_residual_px2'] for d in row.get('sensing_diagnostics',[]) if 'template_residual_px2' in d]
        valid_sensing += [d['valid'] for d in row.get('sensing_diagnostics',[])]
    out={'n':n,'successes':len(successes),'success_rate':len(successes)/n,'success_wilson_95':wilson(len(successes),n),
        'mean_cost_rad2':float(cost.mean()),'cost_bootstrap_95':bootstrap(cost,rng),
        'success_conditioned_cost_rad2':float(np.mean([r['motion_cost_rad2'] for r in successes])) if successes else None,
        'mean_joint_travel_rad':float(np.mean([r['joint_travel_rad'] for r in values])),
        'mean_steps':float(np.mean([r['steps'] for r in values])),
        'mean_simulated_seconds':float(np.mean([r['steps']*.1 for r in values])),
        'mean_bytes':float(np.mean([r['bytes'] for r in values])),
        'mean_messages':float(np.mean([r['messages'] for r in values])),
        'mean_received_messages_reconstructed':float(np.mean([r['received_messages_reconstructed'] for r in values])),
        'receipt_accounting':'Derived from frozen per-episode routing RNG and delay timing; transmitted counts/payloads are directly logged. Undelivered includes reset/episode-end in-flight messages, not only dropout.',
        'mean_rounds':float(np.mean([r['rounds'] for r in values])),
        'invalid_actions':sum(r['invalid_actions'] for r in values),
        'failure_categories':dict(collections.Counter(r['failure'] for r in values if not r['success'])),
        'terminal_per_axis_mae_m_m_rad':abs(error).mean(axis=0).tolist(),
        'mean_terminal_xy_mm':float(np.linalg.norm(error[:,:2],axis=1).mean()*1000),
        'mean_terminal_yaw_deg':float(np.rad2deg(abs(error[:,2])).mean()),
        'mean_overshoot_steps':float(np.mean(overshoots)),'mean_same_world_direction_steps':float(np.mean(same_direction)),
        'oscillation_episode_fraction':float(np.mean(oscillations)),
        'oscillation_definition':'At least two outside-tolerance true-error sign reversals in the SAME axis within an episode.',
        'prediction_fraction_mae':float(np.mean(prediction)) if prediction else None,
        'allocation_A_X_share_by_cheaper_arm':{k:{'commands':len(v),'mean_A_share':float(np.mean(v)) if v else None} for k,v in alignment_shares.items()}}
    if sensing:
        e=abs(np.array(sensing));out['sensing_observations']=len(e)
        out['sensing_per_axis_mae_m_m_rad']=e.mean(axis=0).tolist();out['sensing_per_axis_p95_m_m_rad']=np.quantile(e,.95,axis=0).tolist()
    if valid_sensing:out['valid_sensing_fraction']=float(np.mean(valid_sensing))
    if latencies:
        out['pose_inference_latency_seconds']={'mean':float(np.mean(latencies)),'p95':float(np.quantile(latencies,.95)),
            'scope':'Segmentation plus template estimation only; excludes renderer and policy. Wall-clock, not simulated time.'}
    if residuals:out['mean_template_residual_px2']=float(np.mean(residuals))
    return out


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_continuation():
    path=Path('dual-arm-mas/configs/learned_continuation_20261120.json')
    if not path.exists():
        if any(Path('dual-arm-mas/runs/learned-study').rglob('continuation_invocations.jsonl')):
            raise ValueError('Continuation records exist but scheduling manifest is missing')
        return None
    manifest=json.loads(path.read_text());sha=digest(path)
    if digest(manifest['adapter'])!=manifest['adapter_sha256'] or digest(manifest['study_config'])!=manifest['study_config_sha256']:
        raise ValueError('Continuation scheduling branch changed')
    original_total=0;new_total=0;groups=[]
    for name,prefix in manifest['original_prefixes'].items():
        directory=Path(name);raw_path=directory/'episodes.jsonl'
        if (directory/'writer.lock').exists():raise ValueError('Active continuation writer: '+name)
        raw=raw_path.read_bytes() if raw_path.exists() else b''
        original=raw[:prefix['bytes']]
        if len(original)!=prefix['bytes'] or hashlib.sha256(original).hexdigest()!=prefix['sha256']:
            raise ValueError('Original November7 evidence prefix changed: '+name)
        if len(original.splitlines())!=prefix['episodes']:raise ValueError('Prefix episode count mismatch')
        new_ids=[json.loads(line)['start_id'] for line in raw[prefix['bytes']:].splitlines()]
        covered=[];n=prefix['episodes'];invocation_path=directory/'continuation_invocations.jsonl'
        recovery_path=directory/'continuation_recoveries.jsonl'
        recoveries={r['failed_started_utc']:r for r in (json.loads(line) for line in recovery_path.read_text().splitlines())} if recovery_path.exists() else {}
        if invocation_path.exists():
            for line in invocation_path.read_text().splitlines():
                record=json.loads(line)
                if record['manifest_sha256']!=sha or record['adapter_sha256']!=manifest['adapter_sha256']:
                    raise ValueError('Unpinned continuation invocation')
                if record['status']!='returned':
                    recovery=recoveries.get(record['started_utc'])
                    if recovery is None:raise ValueError('Continuation error requires explicit recovery audit: '+name)
                    if hashlib.sha256(raw[:recovery['raw_bytes_at_recovery']]).hexdigest()!=recovery['raw_sha256_at_recovery']:
                        raise ValueError('Recovered raw prefix changed')
                    if recovery['before_n']!=record['before_n']:raise ValueError('Recovery does not match failed invocation')
                    record={**record,'after_n':recovery['after_n'],'new_start_ids':recovery['new_start_ids']}
                if record['before_n']!=n or record['after_n']-n!=len(record['new_start_ids']):
                    raise ValueError('Broken continuation append chain')
                if datetime.fromisoformat(record['started_utc'])>=datetime.fromisoformat(manifest['window_end_ist']):
                    raise ValueError('Invocation began outside renewed window')
                covered += record['new_start_ids'];n=record['after_n']
        if len(covered)!=len(set(covered)) or set(covered)!=set(new_ids) or len(covered)!=len(new_ids):
            raise ValueError('New episodes lack exact scheduling-adapter provenance: '+name)
        original_total+=prefix['episodes'];new_total+=len(new_ids)
        groups.append({'group':name,'original_n':prefix['episodes'],'new_n':len(new_ids),'recovered_terminal_errors':len(recoveries)})
    return {'manifest_sha256':sha,'adapter_sha256':manifest['adapter_sha256'],
        'original_byte_identical_episodes':original_total,'continuation_episodes':new_total,
        'window_end_ist':manifest['window_end_ist'],'groups':groups,
        'scope':'Explicit between-episode calendar-gate replacement only; frozen mathematics/checkpoint/asset/runtime guards remain active.'}


def audit_initial_state(row,starts):
    if row['start_id'] not in starts:raise ValueError('Unknown paired starting state')
    if row['trace']:
        actual=row['trace'][0]['before']['q'];expected=starts[row['start_id']]['q']
        if any(not np.allclose(actual[n],expected[n],atol=1e-10,rtol=0) for n in ['A','B']):
            raise ValueError('Episode does not use the claimed paired actual joint state')
    elif row.get('failure')!='perception_invalid':
        raise ValueError('Unexplained empty episode trace')


def audit_contract(contract,directory,cfg,fit,regime,sensing,condition,count):
    """Reject mislabeled or stale evidence, even if episode counts look complete."""
    expected={'regime':regime,'sensing':sensing,'decoder':cfg['decoder'],
              'seed':cfg['evaluation_seed'],'count':count,
              'checkpoint_sha256':fit['policies'][regime]['sha256'],
              'inspection_crop':bool(cfg['inspection_crop'] and sensing=='rgb')}
    for key,default in [('drop',0.),('delay',0),('b_limit_factor',1.),('occlusion',0.),('history_length',4)]:
        expected[key]=condition.get(key,default)
    for key,value in expected.items():
        if contract.get(key)!=value:raise ValueError(f'Provenance {key} mismatch: {directory}')
    pins={str(Path(k).resolve()):v for k,v in contract['pins'].items()}
    required=dict(cfg['source_sha256'])
    required[cfg['test_starts']]=cfg['test_starts_sha256']
    required[fit['public']]=digest(fit['public'])
    if sensing=='rgb':required[cfg['perception']]=cfg['perception_sha256']
    for path,sha in required.items():
        if pins.get(str(Path(path).resolve()))!=sha or digest(path)!=sha:
            raise ValueError('Frozen source/asset mismatch: '+path)
    # Also audit imported SFN dependencies not enumerated in the study manifest.
    for path,sha in pins.items():
        if digest(path)!=sha:raise ValueError('Runtime dependency changed: '+path)
        if Path(path).suffix=='.py':
            snapshot=directory/'source_snapshot'/Path(path).name
            if not snapshot.exists() or digest(snapshot)!=sha:raise ValueError('Source snapshot mismatch: '+str(snapshot))
    if digest(fit['policies'][regime]['checkpoint'])!=expected['checkpoint_sha256']:
        raise ValueError('Selected actor checkpoint changed')


def load_group(directory,expected_ids,regime,sensing,cfg,fit,condition,count):
    path=directory/'episodes.jsonl';meta=directory/'provenance.json'
    if not path.exists():return {},{'missing_directory':str(directory)}
    if (directory/'writer.lock').exists():raise ValueError('Active/unaudited writer: '+str(directory))
    contract=json.loads(meta.read_text());rows={}
    audit_contract(contract,directory,cfg,fit,regime,sensing,condition,count)
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):raise ValueError('Partial JSONL record: '+str(path))
            row=json.loads(line);i=row['start_id']
            if i in rows or i not in expected_ids:raise ValueError('Duplicate/unexpected start key: '+str(path))
            if row['regime']!=regime or row['sensing']!=sensing:raise ValueError('Condition label mismatch')
            if row['steps']!=len(row['trace']):raise ValueError('Trace/step count mismatch')
            for field in ['motion_cost_rad2','joint_travel_rad']:
                if not np.isfinite(row[field]) or row[field]<0:raise ValueError('Invalid outcome scalar')
                if not np.isclose(row[field],sum(t[field] for t in row['trace']),rtol=1e-9,atol=1e-12):
                    raise ValueError('Outcome/trace aggregate mismatch')
            if not np.isfinite(row['final_error']).all():raise ValueError('Nonfinite terminal error')
            if row['success'] and (np.any(abs(np.array(row['final_error']))>TOL) or row['failure'] is not None):
                raise ValueError('Success contradicts terminal evidence')
            for step,record in enumerate(row['trace']):
                expected_bytes={'s1':0,'s2':48,'s3':0,'s4':24}[regime]
                communication=record['communication']
                if communication['bytes']!=expected_bytes or communication['messages']!=(2 if expected_bytes else 0):
                    raise ValueError('Learned payload accounting contradicts regime')
                wire=communication['payload_hex']
                if set(wire)!=({'A','B'} if expected_bytes else set()) or sum(len(bytes.fromhex(v)) for v in wire.values())!=expected_bytes:
                    raise ValueError('Serialized wire payload contradicts accounting')
                if sensing=='rgb':
                    diagnostic=row['sensing_diagnostics'][step]
                    if not diagnostic['valid'] or not all(np.array_equal(record['actor_error'][n],diagnostic['error']) for n in ['A','B']):
                        raise ValueError('Vision actor did not use the logged pre-action image estimate')
            rng=np.random.default_rng(contract['seed']+i+1);received=0
            for step in range(len(row['trace'])):
                for _ in ['A','B']:
                    dropped=rng.random()<contract['drop']
                    received+=int(regime in ['s2','s4'] and not dropped and not (contract['delay'] and step==0))
            row['received_messages_reconstructed']=received
            rows[i]=row
    if contract['regime']!=regime or contract['sensing']!=sensing:raise ValueError('Provenance label mismatch')
    return rows,{'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'provenance_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),
                 'missing_start_ids':sorted(expected_ids-set(rows)),'contract':contract}


def baseline_comparisons(groups,cfg,rng):
    """Reuse the completed deterministic suite, never simulate it again."""
    directory=Path('dual-arm-mas/runs/evaluation');raw=directory/'episodes.jsonl'
    audit=json.loads((directory/'completion_audit.json').read_text())
    if not audit['passed'] or digest(raw)!=audit['episodes_sha256']:
        raise ValueError('Saved deterministic evidence does not match its completed audit')
    provenance=json.loads((directory/'provenance.json').read_text())
    # Same measured physical environment and objective, not merely similar labels.
    for filename in ['world.py','episode.py']:
        source=Path('dual-arm-mas/src/dual_arm_mas')/filename
        if provenance['sha256'].get(str(source.resolve()))!=digest(source):
            raise ValueError('Baseline physics/cost no longer matched: '+filename)
    baselines={};keys=set();starts={s['id']:s for s in json.loads(Path(cfg['test_starts']).read_text())['starts']}
    with raw.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):raise ValueError('Partial baseline record')
            row=json.loads(line);key=(row['scenario'],row['fit'],row['method'],row['start_id'])
            if key in keys:raise ValueError('Duplicate baseline key')
            keys.add(key)
            if row['scenario']!='core':continue
            audit_initial_state(row,starts)
            # Keep only outcomes: baseline trajectories can be large.
            row.pop('trace',None)
            baselines.setdefault((row['fit'],row['method']),{})[row['start_id']]=row
    if len(keys)!=audit['expected_keys']:raise ValueError('Baseline count mismatch')
    comparisons=[]
    for (condition,fit,regime,sensing),rows in groups.items():
        if condition!='core':continue
        for method in ['a_only','b_only','half','full_info','local_convention','local_model','capability','prediction','intent']:
            baseline_fit=fit if (fit,method) in baselines else 0
            other=baselines.get((baseline_fit,method),{});ids=sorted(set(rows)&set(other))
            if not ids:continue
            d=[rows[i]['motion_cost_rad2']-other[i]['motion_cost_rad2'] for i in ids]
            comparisons.append({'fit':fit,'regime':regime,'sensing':sensing,'baseline':method,'baseline_fit':baseline_fit,
                'paired_n':len(ids),'mean_paired_cost_difference_rad2':float(np.mean(d)),
                'paired_cost_bootstrap_95':bootstrap(d,rng),
                'paired_success_difference':float(np.mean([int(rows[i]['success'])-int(other[i]['success']) for i in ids])),
                'interpretation':'Different controller family; RGB also changes sensing. Not an information-only causal effect.'})
    summaries=[]
    for (fit,method),rows in sorted(baselines.items()):
        values=list(rows.values());n=len(values);successes=sum(r['success'] for r in values)
        summaries.append({'fit':fit,'method':method,'n':n,'success_rate':successes/n,'success_wilson_95':wilson(successes,n),
            'mean_cost_rad2':float(np.mean([r['motion_cost_rad2'] for r in values])),
            'mean_steps':float(np.mean([r['steps'] for r in values])),
            'mean_joint_travel_rad':float(np.mean([r['joint_travel_rad'] for r in values])),
            'mean_bytes':float(np.mean([r['bytes'] for r in values]))})
    return {'raw_sha256':audit['episodes_sha256'],'audited_episode_count':len(keys),'summaries':summaries,'paired_comparisons':comparisons}


def training_diagnostics(cfg):
    import torch
    results=[]
    for fit in cfg['fits']:
        for regime in ['s1','s2','s3','s4']:
            directory=Path(f"dual-arm-mas/runs/rl-{regime}-{fit['seed']}")
            path=directory/'training.jsonl'
            if not path.exists():continue
            records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not records:continue
            previous_update=0;previous_steps=0;previous_wall=0.;recorded_seconds=0.;restart_boundaries=0
            for row in records:
                if row['update']<=previous_update or row['steps']<=previous_steps:raise ValueError('Training log is not strictly advancing')
                if row['wall_seconds']<previous_wall:
                    previous_wall=0.;restart_boundaries+=1
                recorded_seconds+=row['wall_seconds']-previous_wall
                previous_wall=row['wall_seconds'];previous_steps=row['steps'];previous_update=row['update']
                for key in ['loss','max_preclip_gradient_norm','last_approx_kl','actor_parameter_change_l2_this_invocation']:
                    if not np.isfinite(row[key]):raise ValueError('Nonfinite training diagnostic')
            parameter_evidence={}
            if not (directory/'writer.lock').exists():
                initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
                final=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
                if final['steps']!=records[-1]['steps'] or final['update']!=records[-1]['update'] or final['seed']!=fit['seed']:
                    raise ValueError('Training checkpoint/log/seed mismatch')
                for name,candidate_path in [('latest',directory/'latest.pt')]+(
                    [('selected',Path(fit['policies'][regime]['checkpoint']))] if regime in fit['policies'] else []):
                    state=final if name=='latest' else torch.load(candidate_path,map_location='cpu',weights_only=False)
                    delta=sum(float(((v-initial['model'][k])**2).sum()) for k,v in state['model'].items() if k.startswith('actor.'))**.5
                    if delta<=0:raise ValueError('Trained actor unchanged from initialization')
                    parameter_evidence[name]={'steps':state['steps'],'sha256':digest(candidate_path),'actor_change_l2_from_initial':delta}
            provenance_path=directory/'provenance.json';provenance=json.loads(provenance_path.read_text())
            start_timestamp=datetime.fromisoformat(provenance['runtime']['time_utc']).timestamp()
            end_timestamp=(directory/'latest.pt').stat().st_mtime
            results.append({'fit':fit['fit'],'regime':regime,'seed':fit['seed'],'steps':records[-1]['steps'],
                'updates':records[-1]['update'],'episodes':records[-1]['episodes'],'log_sha256':digest(path),
                'checkpoint_parameter_evidence':parameter_evidence,'active_training_writer':(directory/'writer.lock').exists(),
                'recorded_update_wall_seconds':recorded_seconds,'recorded_steps_per_second':records[-1]['steps']/recorded_seconds,
                'initial_provenance_to_latest_checkpoint_wall_seconds':end_timestamp-start_timestamp,
                'checkpoint_wall_end_utc':datetime.fromtimestamp(end_timestamp,timezone.utc).isoformat(),
                'wall_span_scope':'Initial provenance timestamp through latest checkpoint filesystem modification time, including idle gaps/concurrent work; not dedicated compute time.',
                'training_provenance_sha256':digest(provenance_path),
                'detected_resume_boundaries':restart_boundaries,
                'timing_caveat':'Reconstructed from invocation-local cumulative timestamps, detecting decreases; excludes startup/checkpoint tail and idle gaps. Not exact end-to-end elapsed wall time.',
                'max_preclip_gradient_norm':max(r['max_preclip_gradient_norm'] for r in records),
                'maximum_recorded_actor_change_l2':max(r['actor_parameter_change_l2_this_invocation'] for r in records),
                'training_invalid_episodes':sum(r['invalid_episodes'] for r in records),
                'learning_curve':[{k:r[k] for k in ['steps','success_rate','mean_reward','mean_cost_rad2','mean_steps','loss','last_approx_kl']} for r in records]})
    return results


def selection_audit(cfg):
    """Check actual validation records, not just a 'selected' status string."""
    import torch
    ids={s['id'] for s in json.loads(Path('dual-arm-mas/runs/validation/starts.json').read_text())['starts'][:24]}
    results=[]
    for fit in cfg['fits']:
        if fit['status']!='selected_validation_only':continue
        for regime,policy in fit['policies'].items():
            checkpoint=Path(policy['checkpoint']);summary_path=Path(policy['validation_summary'])
            if digest(checkpoint)!=policy['sha256'] or digest(summary_path)!=policy['validation_summary_sha256']:
                raise ValueError('Selected policy/validation digest mismatch')
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state['steps']!=policy['selected_training_steps'] or state['regime']!=regime or state['seed']!=fit['seed']:
                raise ValueError('Selected actor metadata mismatch')
            meta=json.loads((summary_path.parent/'provenance.json').read_text())
            rows=[json.loads(line) for line in (summary_path.parent/'episodes.jsonl').read_text().splitlines()]
            if len(rows)!=len(ids) or {r['start_id'] for r in rows}!=ids or meta['checkpoint_sha256']!=policy['sha256']:
                raise ValueError('Selected validation coverage/checkpoint mismatch')
            if meta['sensing']!='rgb' or meta['decoder']!=cfg['decoder']:
                raise ValueError('Selected validation was not the declared RGB/mean condition')
            candidates=fit.get('candidate_validation',{}).get(regime)
            if candidates:
                if len(candidates)!=2:raise ValueError('Candidate count changed')
                for candidate in candidates:
                    path=Path(candidate['validation_summary']);actual=json.loads(path.read_text())
                    if digest(path)!=candidate['validation_summary_sha256'] or digest(candidate['checkpoint'])!=candidate['sha256']:
                        raise ValueError('Candidate validation changed')
                    if actual['episodes']!=24 or any(actual[k]!=candidate[k] for k in ['success_rate','mean_cost_rad2','mean_steps']):
                        raise ValueError('Candidate selection metrics do not match evidence')
                chosen=min(candidates,key=lambda c:(-c['success_rate'],c['mean_cost_rad2'],c['mean_steps']))
                if chosen['sha256']!=policy['sha256']:raise ValueError('Selection contradicts the predeclared rule')
            results.append({'fit':fit['fit'],'regime':regime,'selected_steps':state['steps'],'validation_n':len(rows),
                'selected_rgb_includes_adapted_neural_weights':any(sha==cfg['perception_sha256'] for sha in meta['pins'].values()),
                'two_candidate_rule_recomputed':bool(candidates),
                'protocol_note':('Fit0 S1 late candidate was originally validated with color-only RGB, not neural RGB. Its later same-neural validation is diagnostic only; the pre-test selected actor remains fixed.' if fit['fit']==0 and regime=='s1' else None)})
    return results


def run(args):
    continuation=audit_continuation()
    cfg=json.loads(args.config.read_text(encoding='utf-8'));manifest=json.loads(Path(cfg['test_starts']).read_text())
    groups={};audits={};missing=[];rng=np.random.default_rng(cfg['bootstrap_seed'])
    starts_by_id={s['id']:s for s in manifest['starts']}
    for fit in cfg['fits']:
        for regime in ['s1','s2','s3','s4']:
            for sensing in cfg['core_sensing']:
                key=('core',fit['fit'],regime,sensing);directory=Path('dual-arm-mas/runs/learned-study/core')/f"fit-{fit['fit']}"/regime/sensing
                ids={s['id'] for s in manifest['starts'][:cfg['core_count']]}
                rows,audit=load_group(directory,ids,regime,sensing,cfg,fit,{},cfg['core_count']);audits['/'.join(map(str,key))]=audit
                if set(rows)!=ids:missing.append({'group':key,'expected':len(ids),'actual':len(rows)})
                if rows:
                    for row in rows.values():audit_initial_state(row,starts_by_id)
                    groups[key]=rows
    for condition in cfg['robustness']:
        for regime in condition['regimes']:
            key=(condition['name'],0,regime,condition['sensing'])
            directory=Path('dual-arm-mas/runs/learned-study')/condition['name']/'fit-0'/regime/condition['sensing']
            ids={s['id'] for s in manifest['starts'][:condition['count']]}
            rows,audit=load_group(directory,ids,regime,condition['sensing'],cfg,cfg['fits'][0],condition,condition['count']);audits['/'.join(map(str,key))]=audit
            if set(rows)!=ids:missing.append({'group':key,'expected':len(ids),'actual':len(rows)})
            if rows:
                for row in rows.values():audit_initial_state(row,starts_by_id)
                groups[key]=rows
    pending=[f['fit'] for f in cfg['fits'] if f['status']!='selected_validation_only']
    training=training_diagnostics(cfg);selection=selection_audit(cfg)
    unfinished_training=[{'fit':f['fit'],'regime':r} for f in cfg['fits'] for r in ['s1','s2','s3','s4']
        if not any(t['fit']==f['fit'] and t['regime']==r and t['steps']>=cfg['training_target_steps_per_regime_per_seed'] for t in training)]
    if (missing or pending or unfinished_training) and not args.allow_partial:raise ValueError(f'Study incomplete: {len(missing)} groups; pending policy fits {pending}; unfinished training {unfinished_training}')
    summaries=[]
    for key,rows in groups.items():summaries.append({'condition':key[0],'fit':key[1],'regime':key[2],'sensing':key[3],**summarize(rows,rng)})
    comparisons=[]
    def compare(first,second,label):
        if first not in groups or second not in groups:return
        a,b=groups[first],groups[second];ids=sorted(set(a)&set(b));d=[a[i]['motion_cost_rad2']-b[i]['motion_cost_rad2'] for i in ids]
        comparisons.append({'first':first,'second':second,'comparison_type':label,'paired_n':len(ids),
            'mean_paired_cost_difference_rad2':float(np.mean(d)),'paired_cost_bootstrap_95':bootstrap(d,rng),
            'paired_success_difference':float(np.mean([int(a[i]['success'])-int(b[i]['success']) for i in ids]))})
    for fit in cfg['fits']:
        for regime in ['s1','s2','s3','s4']:compare(('core',fit['fit'],regime,'rgb'),('core',fit['fit'],regime,'gt'),'same learned checkpoint, actual sensing change')
        for sensing in ['gt','rgb']:
            for regime in ['s2','s3','s4']:compare(('core',fit['fit'],regime,sensing),('core',fit['fit'],'s1',sensing),'learned regime comparison, separate fitted policies/validation selection')
    for condition in cfg['robustness']:
        for regime in condition['regimes']:compare((condition['name'],0,regime,condition['sensing']),('core',0,regime,condition['sensing']),'same starts/checkpoint, impairment or history ablation')
    variability=[]
    for regime in ['s1','s2','s3','s4']:
        for sensing in ['gt','rgb']:
            selected=[s for s in summaries if s['condition']=='core' and s['regime']==regime and s['sensing']==sensing]
            if selected:variability.append({'regime':regime,'sensing':sensing,'fits':len(selected),
                'mean_costs_rad2':[s['mean_cost_rad2'] for s in selected],'success_rates':[s['success_rate'] for s in selected],
                'between_fit_cost_sample_sd':float(np.std([s['mean_cost_rad2'] for s in selected],ddof=1)) if len(selected)>1 else None})
    args.out.mkdir(parents=True,exist_ok=True)
    baseline=baseline_comparisons(groups,cfg,rng) if not args.skip_baseline else None
    result={'complete':not missing and not pending and not unfinished_training and baseline is not None,
        'missing_groups':missing,'pending_fits':pending,'unfinished_training':unfinished_training,'training':training,'selection_audit':selection,'deterministic_baselines':baseline,
        'continuation_audit':continuation,
        'bootstrap_seed':cfg['bootstrap_seed'],'bootstrap_resamples':2000,'summaries':summaries,'comparisons':comparisons,'fitting_variability':variability,
        'audits':audits,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),'analysis_source_sha256':digest(__file__),
        'bootstrap_helper_sha256':digest(Path(__file__).with_name('analyze.py')),
        'limitations':'Wilson marginal success intervals and episode bootstrap condition on the filtered distribution/model. Three actor fits share starts and one perception fit. Squared command cost is subdivision dependent; failed episodes retained.'}
    (args.out/'statistics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# '+('COMPLETE learned evaluation' if result['complete'] else 'INCOMPLETE learned evaluation — not final evidence'),'','|Condition|Fit|Regime|Sensing|N|Success|Cost rad²|Steps|Bytes|','|---|---:|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"|{s['condition']}|{s['fit']}|{s['regime']}|{s['sensing']}|{s['n']}|{s['success_rate']:.4f}|{s['mean_cost_rad2']:.7f}|{s['mean_steps']:.3f}|{s['mean_bytes']:.2f}|")
    (args.out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'complete':result['complete'],'groups':len(groups),'missing_groups':len(missing),'episodes':sum(len(r) for r in groups.values())}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('dual-arm-mas/configs/learned_study.json'))
    p.add_argument('--out',type=Path,default=Path('dual-arm-mas/runs/learned-study/analysis'));p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--skip-baseline',action='store_true',help='Exploratory partial output only; prevents completion certification');run(p.parse_args())
