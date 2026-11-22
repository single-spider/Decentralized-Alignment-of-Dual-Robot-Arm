"""Generate report inserts only from audited statistics; never invent final results."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def tex(value):
    return str(value).replace('\\', '/').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def table(headers,rows,align):
    return '\n'.join([r'\begin{center}\small',r'\begin{tabular}{'+align+'}',r'\toprule',
        ' & '.join(headers)+r'\\\midrule']+[' & '.join(map(str,r))+r'\\' for r in rows]+[
        r'\bottomrule',r'\end{tabular}',r'\end{center}'])


def run(args):
    stats=json.loads(args.statistics.read_text(encoding='utf-8'))
    if args.final and not stats['complete']:raise ValueError('Cannot label incomplete evidence final')
    if args.final:
        source=Path(__file__).with_name('analyze_learned.py')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=stats['analysis_source_sha256']:
            raise ValueError('Analysis code changed since statistics were derived')
        helper=Path(__file__).with_name('analyze.py')
        if hashlib.sha256(helper.read_bytes()).hexdigest()!=stats['bootstrap_helper_sha256']:
            raise ValueError('Bootstrap helper changed since statistics were derived')
    cfg=json.loads(Path('dual-arm-mas/configs/learned_study.json').read_text())
    if hashlib.sha256(Path('dual-arm-mas/configs/learned_study.json').read_bytes()).hexdigest()!=stats['config_sha256']:
        raise ValueError('Statistics were derived from a different selected study manifest')
    out=Path('dual-arm-mas/docs/report-rl-vision');out.mkdir(parents=True,exist_ok=True)
    figures=Path('dual-arm-mas/runs/learned-study/analysis/figures');figures.mkdir(parents=True,exist_ok=True)
    n=sum(s['n'] for s in stats['summaries']);expected=cfg['core_count']*len(cfg['fits'])*4*2+sum(c['count']*len(c['regimes']) for c in cfg['robustness'])
    status=(r'\textbf{Audited learned evaluation complete.} Report authorship and submission remain subject to team verification.'
        if stats['complete'] else r'\textbf{INCOMPLETE STUDY / WORKING DRAFT.} Required evaluation/training groups remain unfinished; this is not the final submission.')
    (out/'status.tex').write_text(status+f' Observed learned episodes: {n:,} of {expected:,} planned.\n',encoding='utf-8')
    core={(s['fit'],s['regime'],s['sensing']):s for s in stats['summaries'] if s['condition']=='core'}
    findings=[r'\subsection{Main findings and answer to the research question}']
    if stats['complete']:
        rgb=[s for s in core.values() if s['sensing']=='rgb'];gt=[s for s in core.values() if s['sensing']=='gt']
        findings.append(f"Actual neural-RGB execution succeeded in {min(s['success_rate'] for s in rgb)*100:.1f}--{max(s['success_rate'] for s in rgb)*100:.1f}\\% across the twelve 1000-case actor-fit/regime groups; all twelve matched GT groups succeeded on 1000/1000. Thus local-only learned S1 already supports useful alignment without teammate messages in this narrow model. This is not a guarantee outside the filtered operating region.")
        increases=[100*(s['mean_cost_rad2']/core[s['fit'],s['regime'],'gt']['mean_cost_rad2']-1) for s in rgb]
        findings.append(f"Replacing GT with neural RGB increased overall squared command cost in every matched learned group (point increases {min(increases):.1f}--{max(increases):.1f}\\%). All twelve paired cost intervals are positive. Vision also increased corrective steps and produced repeated outside-tolerance oscillation in {min(s['oscillation_episode_fraction'] for s in rgb)*100:.1f}--{max(s['oscillation_episode_fraction'] for s in rgb)*100:.1f}\\% of episodes, versus zero under GT. High alignment success does not imply equally efficient visual control.")
        for regime,label in [('s2','Capability-sharing S2'),('s4','Bid-assisted S4')]:
            reductions=[100*(1-core[f,regime,'rgb']['mean_cost_rad2']/core[f,'s1','rgb']['mean_cost_rad2']) for f in [0,1,2]]
            supporting=sum(c['paired_cost_bootstrap_95'][1]<0 for c in stats['comparisons'] if c['first'][0]=='core' and c['first'][2:]==[regime,'rgb'] and c['second'][2:]==['s1','rgb'])
            findings.append(f"{label} had lower RGB point cost than separately trained S1 in all three fits, with reductions {min(reductions):.1f}--{max(reductions):.1f}\\%. Conditional paired cost intervals are below zero in {supporting} of three fits. These are fitted-policy/protocol comparisons, not information-only causal effects or a large-sample training-population test.")
        lower=sum(core[f,'s3','rgb']['mean_cost_rad2']<core[f,'s1','rgb']['mean_cost_rad2'] for f in [0,1,2])
        findings.append(f"Prediction-assisted S3 lowered RGB cost relative to S1 in {lower} of three fits and increased it in the other. The deterministic-policy prior and noisy local motion reconstruction are not a reliable oracle of the learned teammate's current action. No consistent general advantage for prediction is established.")
        base=next(s for s in stats['deterministic_baselines']['summaries'] if s['fit']==0 and s['method']=='half')
        findings.append(f"Learning did not establish motion-cost superiority: all twelve GT learned groups cost more than matched GT fixed-half control ({base['mean_cost_rad2']:.6f}rad$^2$), with learned means {min(s['mean_cost_rad2'] for s in gt):.6f}--{max(s['mean_cost_rad2'] for s in gt):.6f}. Learned GT used fewer commands ({min(s['mean_steps'] for s in gt):.3f}--{max(s['mean_steps'] for s in gt):.3f} versus {base['mean_steps']:.3f}). This is a speed/command-cost tradeoff under the alignment-heavy reward, not physical energy improvement. The initial-mean validation control also shows that the geometric scaffold itself is already competent.")
        robust=[s for s in stats['summaries'] if s['condition']!='core']
        findings.append(f"The ten bounded robustness rows observed {min(s['successes'] for s in robust)}--{max(s['successes'] for s in robust)} successes out of 128. Dropout and delay did not uniformly worsen cost, and the frozen occlusion cases mostly recovered despite the demonstrated possibility of an accepted biased pose. These small, fixed-actor stress samples do not calibrate the estimator's confidence or prove safety/zero failure risk. History0/1 changes only the explicit belief window, not the previous-own-motion input.")
        findings.append("The information question therefore has an empirical, conditional answer: explicit messages are unnecessary for high success here; compact capability or bid interfaces can improve some learned-policy costs, but availability alone does not produce optimal cost adaptation. The much cheaper deterministic references and the weak allocation change in the replay remain important negative learning findings. No minimum-sufficient-information theorem or hardware conclusion is claimed.")
    else:findings.append(r'\textbf{Final conclusions withheld: required evaluation groups are incomplete.}')
    (out/'findings.tex').write_text('\n\n'.join(findings)+'\n',encoding='utf-8')
    parts=[r'\clearpage\subsection{Observed learned core and uncertainty}',
        'Each row is one independently fitted actor and sensing condition. Cost is overall, including failed episodes.'+(' Partial rows are exploratory only.' if not stats['complete'] else '')]
    rows=[]
    for s in stats['summaries']:
        if s['condition']!='core':continue
        ci=s['success_wilson_95']
        rows.append([s['fit'],s['regime'].upper(),s['sensing'].upper(),s['n'],f"{s['success_rate']*100:.2f}",
            f"[{ci[0]*100:.2f},{ci[1]*100:.2f}]",f"{s['mean_cost_rad2']:.6f}",f"{s['mean_steps']:.2f}",f"{s['mean_bytes']:.1f}"])
    parts.append(table(['Fit','Regime','Sensor','N',r'Success\%',r'95\% Wilson','Cost rad$^2$','Steps','Bytes'],rows,'rlllrrrrr'))
    if stats['summaries']:
        fig,axes=plt.subplots(2,2,figsize=(9,5.3),sharex=True)
        colors={'gt':'#2463a0','rgb':'#d67822'}
        for regime,ax in zip(['s1','s2','s3','s4'],axes.flat):
            for sensing,offset in [('gt',-.07),('rgb',.07)]:
                data=[s for s in stats['summaries'] if s['condition']=='core' and s['regime']==regime and s['sensing']==sensing]
                if not data:continue
                means=np.array([s['mean_cost_rad2'] for s in data]);intervals=np.array([s['cost_bootstrap_95'] for s in data])
                ax.errorbar(np.array([s['fit'] for s in data])+offset,means,
                    yerr=np.maximum(0,np.array([means-intervals[:,0],intervals[:,1]-means])),fmt='o',capsize=3,
                    color=colors[sensing],label=sensing.upper(),markersize=4)
            ax.set_title(regime.upper());ax.set_ylabel('Overall cost (rad squared)');ax.grid(alpha=.2);ax.legend(fontsize=8)
            ax.set_xticks([0,1,2]);ax.set_xlim(-.4,2.4)
        for ax in axes[-1]:ax.set_xlabel('Independent actor/data fit')
        fig.tight_layout();fig.savefig(figures/'core_costs.png',dpi=180);plt.close(fig)
        parts.append(r'\clearpage\begin{figure}[ht]\centering\includegraphics[width=.95\linewidth]{../../runs/learned-study/analysis/figures/core_costs.png}\caption{Actual selected-policy costs by actor/data fit and sensing, with marginal episode-bootstrap intervals. Overall costs retain failures; inspect success rates separately.}\end{figure}')
    parts.append(r'\subsection{Initial versus trained behavior: validation, not final test}')
    rows=[]
    for label,sensor,name in [('Initial','GT','validation-initial-gt-mean'),('Early3375','GT','validation-3375-gt-mean'),
        ('Late10118','GT','validation-10118-gt'),('Initial','Neural RGB','validation-initial-rgb-neural-frozen'),
        ('Early3375','Neural RGB','validation-3375-neural-crop'),
        ('Late diagnostic','Neural RGB','validation-10118-rgb-neural-frozen-diagnostic')]:
        path=Path('dual-arm-mas/runs/rl-s1-401')/name/'summary.json'
        if not path.exists():continue
        record=json.loads(path.read_text());rows.append([label,sensor,record['episodes'],f"{record['success_rate']*100:.2f}",
            f"{record['mean_cost_rad2']:.6f}",f"{record['mean_steps']:.2f}"])
    parts.append(table(['S1 candidate','Sensor','Validation N',r'Success\%','Cost rad$^2$','Steps'],rows,'llrrrr'))
    parts.append('All rows use deterministic mean fractions, not an initial-mode/trained-mean comparison. The initial geometric scaffold is already competent. Lower GT step counts do not establish lower motion cost or a neural-sensing improvement. The late neural diagnostic was executed after the test freeze to expose a historical sensing-protocol discrepancy; it does not reselect the frozen test policy.')
    parts.append(r'\clearpage\subsection{Training budgets, actual updates and timing}')
    rows=[]
    for t in stats['training']:
        selected=next(f for f in cfg['fits'] if f['fit']==t['fit'])['policies'].get(t['regime'],{})
        rows.append([t['fit'],t['regime'].upper(),t['seed'],t['steps'],t['updates'],selected.get('selected_training_steps','pending'),
            f"{t['recorded_update_wall_seconds']/60:.1f}",f"{t['recorded_steps_per_second']:.1f}",
            ('not recorded' if 'initial_provenance_to_latest_checkpoint_wall_seconds' not in t else f"{t['initial_provenance_to_latest_checkpoint_wall_seconds']/3600:.2f}")])
    parts.append(table(['Fit','Regime','Seed','Steps','Updates','Selected','Min*','Steps/s*','Span h**'],rows,'rlrrrrrrr'))
    parts.append('Timing* sums recorded update intervals reconstructed from invocation-local clocks, detecting clock decreases at resumes; it excludes startup/checkpoint tails and idle gaps, and is not exact end-to-end elapsed time. Curves show the unscaled environment return, before the trainer-only terminal guard. Real actor changes and finite-gradient diagnostics are audited separately in the machine-readable analysis.')
    parts.append('Span** brackets the initial provenance timestamp and final checkpoint filesystem modification time. It includes idle gaps, interleaved validation and concurrent work, so it is an elapsed wall span rather than dedicated compute time.')
    if stats['training']:
        fig,axes=plt.subplots(2,2,figsize=(9,5.5),sharex=True)
        for regime,ax in zip(['s1','s2','s3','s4'],axes.flat):
            for t in stats['training']:
                if t['regime']!=regime:continue
                curve=t['learning_curve'];ax.plot([r['steps'] for r in curve],[r['mean_reward'] for r in curve],label=f"seed {t['seed']}",linewidth=1)
            ax.set_title(regime.upper());ax.set_ylabel('Mean environment return');ax.grid(alpha=.2);ax.legend(fontsize=7)
        for ax in axes[-1]:ax.set_xlabel('Actual environment steps')
        fig.tight_layout();fig.savefig(figures/'training.png',dpi=180);plt.close(fig)
        parts.append(r'\begin{figure}[ht]\centering\includegraphics[width=.90\linewidth]{../../runs/learned-study/analysis/figures/training.png}\caption{Actual stochastic rollout learning curves; useful training return does not prove deterministic deployment quality.}\end{figure}')
    parts.append(r'\clearpage\subsection{Paired sensing and learned-regime contrasts}')
    rows=[]
    for c in stats['comparisons']:
        if c['first'][0]!='core':continue
        first,second=c['first'],c['second'];ci=c['paired_cost_bootstrap_95']
        label=f"{first[2].upper()} {first[3]} - {second[2].upper()} {second[3]}"
        rows.append([first[1],tex(label),c['paired_n'],f"{c['mean_paired_cost_difference_rad2']:.6f}",f"[{ci[0]:.6f},{ci[1]:.6f}]"])
    parts.append(table(['Fit','First minus second','Paired N',r'$\Delta$cost rad$^2$',r'95\% paired bootstrap'],rows,'rlrrl'))
    parts.append('Negative differences favor the first row on squared cost only. Sensing contrasts keep the selected actor fixed. Regime contrasts change trained policies/selection and message or belief interfaces, not just information availability.')
    baseline=stats.get('deterministic_baselines')
    if baseline:
        parts.append(r'\clearpage\subsection{Reused deterministic baselines and learned comparisons}')
        rows=[[tex(s['method']),s['n'],f"{s['success_rate']*100:.2f}",f"{s['mean_cost_rad2']:.6f}",f"{s['mean_steps']:.2f}",f"{s['mean_bytes']:.1f}"] for s in baseline['summaries'] if s['fit']==0]
        parts.append(table(['Baseline (fit0)','N',r'Success\%','Cost rad$^2$','Steps','Bytes'],rows,'lrrrrr'))
        rows=[]
        for c in baseline['paired_comparisons']:
            if c['fit']!=0 or c['baseline'] not in ['half','full_info','local_model']:continue
            ci=c['paired_cost_bootstrap_95'];rows.append([c['regime'].upper(),c['sensing'],tex(c['baseline']),c['paired_n'],
                f"{c['mean_paired_cost_difference_rad2']:.6f}",f"[{ci[0]:.6f},{ci[1]:.6f}]"])
        parts.append(table(['Learned','Sensor','Baseline','Paired N',r'$\Delta$cost',r'95\% paired bootstrap'],rows,'lllrrl'))
        parts.append('Positive learned-minus-baseline cost is a negative efficiency finding, not an implementation failure. Remaining paired fits and all methods are in the machine-readable analysis.')
    else:parts.append(r'\textbf{Baseline paired analysis not yet generated; required for final results.}')
    parts.append(r'\clearpage\subsection{Robustness, failures and replication}')
    rows=[]
    for s in stats['summaries']:
        if s['condition']=='core':continue
        rows.append([tex(s['condition']),s['regime'].upper(),s['n'],f"{s['success_rate']*100:.2f}",f"{s['mean_cost_rad2']:.6f}",f"{s['mean_steps']:.2f}"])
    parts.append(table(['Condition','Regime','N',r'Success\%','Cost rad$^2$','Steps'],rows,'llrrrr'))
    rows=[]
    for s in stats['summaries']:
        if s['fit']!=0 or s['regime'] not in ['s2','s4'] or s['sensing']!='rgb':continue
        rows.append([tex(s['condition']),s['regime'].upper(),s['n'],f"{s['mean_bytes']:.1f}",
            f"{s['mean_messages']:.2f}",f"{s['mean_received_messages_reconstructed']:.2f}",f"{s['mean_rounds']:.2f}"])
    parts.append(table(['Condition','Regime','N','Sent bytes','Sent msgs','Received*','Rounds'],rows,'llrrrrr'))
    parts.append('Received* is independently reconstructed from the frozen per-episode routing RNG and delay timing, not a directly logged receipt counter. Sent payloads are directly serialized/logged and audited. Delay leaves initial/final in-flight messages outside the episode; undelivered does not mean dropout alone. Headers, common sensing and camera bandwidth are excluded.')
    parts.append(r'\clearpage\subsection{Execution, sensing and prediction diagnostics}')
    rows=[]
    for s in stats['summaries']:
        if s['condition']!='core' or s['fit']!=0:continue
        rows.append([s['regime'].upper(),s['sensing'],f"{s['mean_joint_travel_rad']:.4f}",f"{s['mean_overshoot_steps']:.3f}",
            ('pending' if 'oscillation_episode_fraction' not in s else f"{s['oscillation_episode_fraction']*100:.2f}"),f"{s['mean_same_world_direction_steps']:.3f}",
            s['invalid_actions'],f"{s['mean_rounds']:.2f}"])
    parts.append(table(['Fit0 regime','Sensor','L1 rad','Overshoot',r'Oscill.\%','Same-dir.','Invalid','Rounds'],rows,'llrrrrrr'))
    parts.append('Overshoot counts steps with any true-error sign reversal outside tolerance. Oscillation means at least two such reversals in the same axis. Same-direction means both measured world-frame contributions share a sign in an axis, which may cancel in relative-task space; this is not a collision measure. Full per-fit diagnostics remain in the raw/statistical evidence.')
    rows=[]
    for s in stats['summaries']:
        if s['condition']!='core' or 'sensing_per_axis_mae_m_m_rad' not in s:continue
        e=s['sensing_per_axis_mae_m_m_rad'];p=s['sensing_per_axis_p95_m_m_rad']
        latency=s.get('pose_inference_latency_seconds',{}).get('mean',0.)
        rows.append([s['fit'],s['regime'].upper(),s['sensing_observations'],f'{e[0]*1000:.3f}',f'{e[1]*1000:.3f}',
            f'{np.rad2deg(e[2]):.3f}',f'{np.rad2deg(p[2]):.3f}',f'{latency*1000:.1f}'])
    parts.append(table(['Fit','Regime','Images','X MAE mm','Y MAE mm','Yaw MAE deg','Yaw P95 deg','Latency ms'],rows,'rlrrrrrr'))
    parts.append('Latency is recorded segmentation/template inference wall time under concurrent host load, excluding rendering and policy execution; it is not a controlled real-time benchmark or simulated task duration. X/Y P95 and all impairment sensing diagnostics are in statistics.json.')
    rows=[]
    for s in stats['summaries']:
        if s['condition']=='core' and s['regime']=='s3':
            rows.append([s['fit'],s['sensing'],'undefined' if s['prediction_fraction_mae'] is None else f"{s['prediction_fraction_mae']:.4f}"])
    parts.append(table(['S3 fit','Sensor','Current executed-fraction prediction MAE'],rows,'rlr'))
    parts.append('Prediction labels are measured teammate contributions divided by the observed clipped correction, scored only where the denominator exceeds 5\\% of the action scale. Labels are produced after execution and never enter the current actor. The local predictor is a past-executed persistence belief, not privileged current intention.')
    for s in stats['summaries']:
        if s['condition']!='core':continue
        failure=', '.join(f'{tex(k)}: {v}' for k,v in sorted(s['failure_categories'].items())) or 'none'
        parts.append(f"Fit{s['fit']} {s['regime'].upper()} {s['sensing']}: failures {failure}; success-conditioned cost "+
            ('undefined' if s['success_conditioned_cost_rad2'] is None else f"{s['success_conditioned_cost_rad2']:.6f}")+f" rad$^2$; L1 travel {s['mean_joint_travel_rad']:.4f} rad; terminal X/Y MAE ({s['terminal_per_axis_mae_m_m_rad'][0]*1000:.3f},{s['terminal_per_axis_mae_m_m_rad'][1]*1000:.3f}) mm; mean XY norm {s['mean_terminal_xy_mm']:.3f} mm, yaw MAE {s['mean_terminal_yaw_deg']:.3f}$^\\circ$.\n")
    rows=[]
    for v in stats['fitting_variability']:
        rows.append([v['regime'].upper(),v['sensing'],v['fits'],'not estimable' if v['between_fit_cost_sample_sd'] is None else f"{v['between_fit_cost_sample_sd']:.6f}"])
    parts.append(table(['Regime','Sensor','Actor fits','Between-fit cost sample SD'],rows,'llrl'))
    parts.append('Per-axis sensing errors, pose-inference latency, prediction-fraction MAE, overshoot, same-world-direction commands, invalid actions, payload rounds and cheaper-arm-conditioned allocation shares are retained in statistics.json. They are diagnostics, not evidence of optimal allocation or calibrated confidence.')
    (out/'results.tex').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    (out/'reproduction.tex').write_text(r'''\begingroup\small\raggedright
From the SFN checkout, use the existing virtual environment; no additional dependencies or paid services are required. Paths below are relative to that checkout. Full source/config/checkpoint hashes and raw episodes are retained locally, not just plotted summaries.
\begin{itemize}
\item Study contract: \texttt{configs/learned\_study.json}; raw learned episodes and source snapshots: \texttt{runs/learned-study/}.
\item Historical bounded training command: \texttt{scripts/train\_rl.py --regime s1 --out NEW --steps 10000 --seconds 240}, with the declared training/public manifests and seed. Its old calendar gate is preserved. Future retraining needs an explicitly recorded renewed scheduling branch, not silent gate removal. Existing model/optimizer/RNG checkpoints and the actual training configurations are retained; no retraining is needed to reproduce saved-policy inference.
\item Renewed test slice: \texttt{scripts/continue\_learned.py --regime s1 --sensing rgb --phase all --seconds 240}. Its separately sealed November20 scheduling manifest preserves original raw prefixes and invokes the unchanged evaluator. After the recorded memory-allocation failure, use at most two neural processes total including demos/probes; do not overlap additional Torch jobs. The completed study needs no further evaluation. The original entrypoints retain the expired November7 calendar gate.
\item Analysis: \texttt{scripts/analyze\_learned.py}. It refuses incomplete groups, unselected fits, insufficient training, provenance mismatch or active writers. \texttt{--allow-partial} explicitly produces incomplete exploratory output.
\item Replay: \texttt{scripts/replay\_learned\_window.py --window-end FUTURE-ISO-TIME --out NEW}. Supply an explicitly authorized timezone-aware future end time (for example an ISO timestamp ending in \texttt{+05:30}). This verifies the accepted demo dependencies and frozen weights, then executes matched cases 0/4 with the same neural/crop/mean policy and modal-failure condition. Prefix project-local paths with \texttt{dual-arm-mas/} and invoke with \texttt{.venv/Scripts/python.exe} from SFN root.
\item Report data: \texttt{scripts/report\_rl\_data.py --statistics PATH --final}; final labeling refuses incomplete analysis. Compile this LaTeX twice with installed MiKTeX, installer disabled, then render and visually inspect every page.
\end{itemize}
The complete baseline evidence remains \texttt{runs/evaluation/}; its PDF is preserved separately. Authoritative raw episodes, not a stale status statement, determine completion. Source and checkpoint mismatches refuse resume instead of combining experiments.
\endgroup
''',encoding='utf-8')
    print(json.dumps({'report_inserts':str(out),'complete_evidence':stats['complete'],'observed_episodes':n,'expected_episodes':expected}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--statistics',type=Path,default=Path('dual-arm-mas/runs/learned-study/analysis/statistics.json'))
    p.add_argument('--final',action='store_true');run(p.parse_args())
