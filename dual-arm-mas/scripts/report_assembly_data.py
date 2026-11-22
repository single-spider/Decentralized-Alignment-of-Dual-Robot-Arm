"""Contact report inserts from an audited analysis snapshot; final refuses partial."""
import argparse,json,hashlib,math
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%')
def label(s):return tex(s.replace('open-fit0-','').replace('-fit0-','-'))
def run(args):
 cfg=json.loads(args.config.read_text());path=Path(cfg['output'])/'analysis/statistics.json';s=json.loads(path.read_text())
 if s['config_sha256']!=sha(args.config) or s['analysis_source_sha256']!=sha('dual-arm-mas/scripts/analyze_assembly.py'):raise ValueError('Stale analysis source/config')
 if not s['complete'] and not args.draft:raise ValueError('Contact study incomplete; final report refused')
 out=Path('dual-arm-mas/docs/report-assembly');out.mkdir(exist_ok=True)
 status='Audited contact-transfer evaluation complete.' if s['complete'] else 'INCOMPLETE contact-transfer evaluation; not a final assembly report.'
 (out/'contact_status.tex').write_text(r'\noindent\textbf{'+status+'} Observed '+str(s['observed_episodes'])+' of '+str(s['expected_episodes'])+r' planned contact episodes. Original alignment evidence is separately identified below.\par'+'\n')
 summary={x['group']:x for x in s['summaries']}
 open_rows=[x for x in s['summaries'] if x['group'].startswith('open-') and '-half-' not in x['group'] and x['group'].endswith('-rgb')]
 lines=[]
 if s['complete']:
  local=summary['open-fit0-s1-rgb'];oracle=summary['open-fit0-s1-gt'];half=summary['open-fit0-half-rgb']
  lines.append(r'\paragraph{Observed answer for actual contact assembly.} '+f"Across six learned-RGB open conditions, insertion success ranged {100*min(x['success_rate'] for x in open_rows):.1f}--{100*max(x['success_rate'] for x in open_rows):.1f}\\%. Fit0 S1/S2/S3/S4 inserted "+
   '/'.join(str(summary[f'open-fit0-{reg}-rgb']['successes']) for reg in ['s1','s2','s3','s4'])+f" of 128 each. The matched fixed-half RGB baseline inserted {half['successes']}/128, and the identical S1 actor with GT inserted {oracle['successes']}/128. These are insertion rates, not the much higher historical kinematic-alignment rates.")
  contrast=next(x for x in s['paired_comparisons'] if x['first']=='open-fit0-s1-rgb' and x['second']=='open-fit0-s1-gt');ci=contrast['success_paired_bootstrap_95']
  lines.append(f"Replacing GT with neural RGB reduced S1 insertion by {-100*contrast['success_difference']:.1f} percentage points (paired difference 95\\% interval [{100*ci[0]:.1f},{100*ci[1]:.1f}] points for RGB minus GT). True-yaw safety-gate rejections despite estimated alignment are a major observed transfer limitation; no true-error fallback corrects the actor.")
  lines.append('All four first-fit learned RGB groups cost more than fixed-half in the overall endpoint-squared-displacement surrogate; their conditional paired cost intervals are above zero. Thus actual contact execution does not establish an RL motion-cost advantage. Capability, prediction and bid-assisted fitted policies have lower point costs than S1 here, but protocol/policy changes and failed episodes prevent an information-only or energy interpretation. Success-conditioned costs and success uncertainty remain essential.')
  blocked=[summary[g] for g in ['blocked-fit0-s1-rgb','blocked-fit0-s4-rgb']];njam=sum(x['recovery_attempts'] for x in blocked);released=sum(x['released_recoveries'] for x in blocked)
  opens=[x for x in s['summaries'] if x['group'].startswith('open-')]
  lines.append(f"The two 32-case blocked conditions produced no false insertion successes. Of {njam} actual jam/recovery attempts, {released} released; the remaining blocked starts were rejected before descent. This does not imply universal recovery: across open groups only {sum(x['released_recoveries'] for x in opens)}/{sum(x['recovery_attempts'] for x in opens)} bounded recovery attempts released. Force/penetration stops, timeouts and terminal-hold failures are retained. The small actuator/message stress tests do not establish a monotonic degradation law or hardware safety.")
 lines += [r'\begin{center}\small',r'\begin{tabular}{lrrrrr}\toprule',r'Contact condition & N & Inserted\% & Cost rad$^2$ & Peak N & Released/attempted\\\midrule']
 for row in s['summaries']:
  lines.append(tex(row['group'])+f" & {row['n']} & {row['success_rate']*100:.1f} & {row['mean_endpoint_command_cost_rad2']:.5f} & {row['max_peak_force_n']:.2f} & {row['released_recoveries']}/{row['recovery_attempts']}"+r'\\')
 lines.extend([r'\bottomrule\end{tabular}\end{center}',r'Peak peg--fixture normal force includes logged settling, alignment, descent, recovery and hold, not merely the descent trace; it is not a certified bound. Failed episodes remain in overall cost. Blocked conditions are intended negative controls; insertion failure is not automatically a successful safe rejection or release.'])
 for row in s['summaries']:
  ci=row['success_wilson_95'];fail=', '.join(f'{tex(k)}={v}' for k,v in row['failure_categories'].items()) or 'none'
  conditioned='undefined' if row['success_conditioned_cost_rad2'] is None else f"{row['success_conditioned_cost_rad2']:.6f}"
  lines.append(tex(row['group'])+f": Wilson95\\% [{100*ci[0]:.1f},{100*ci[1]:.1f}], success-conditioned cost {conditioned}, mean alignment/insertion commands {row['mean_alignment_commands']:.2f}/{row['mean_insertion_commands']:.2f}, endpoint L1 travel {row['mean_endpoint_joint_travel_rad']:.4f} rad, simulated time {row['mean_simulation_seconds']:.1f} s, mean payload {row['mean_payload_bytes']:.1f} bytes. Failures: "+fail+r'\par')
 lines.extend([r'\begin{center}\small',r'\begin{tabular}{llrrl}\toprule',r'First & Second & Paired N & $\Delta$cost & 95\% paired bootstrap\\\midrule'])
 for row in s['paired_comparisons']:
  ci=row['cost_paired_bootstrap_95'];lines.append(label(row['first'])+' & '+label(row['second'])+f" & {row['paired_n']} & {row['mean_cost_difference_rad2']:.6f} & [{ci[0]:.6f},{ci[1]:.6f}]"+r'\\')
 lines.extend([r'\bottomrule\end{tabular}\end{center}',r'Regime contrasts compare distinct fitted policies and protocols, not information-only causal effects. RGB-versus-GT holds the S1 actor fixed. Episode bootstrap intervals do not substitute for independent training-run replication. Per-condition command counts, L1 travel, success intervals, force/release/failure data and raw hashes remain in the accompanying statistics.'])
 lines += [r'\begin{center}\small',r'\begin{tabular}{llrrl}\toprule',r'First & Second & Paired N & $\Delta$success pp & 95\% paired bootstrap\\\midrule']
 for row in s['paired_comparisons']:
  ci=row['success_paired_bootstrap_95'];lines.append(label(row['first'])+' & '+label(row['second'])+f" & {row['paired_n']} & {100*row['success_difference']:.2f} & [{100*ci[0]:.2f},{100*ci[1]:.2f}]"+r'\\')
 lines += [r'\bottomrule\end{tabular}\end{center}',r'Stress contrasts use the same first 32 starts from the corresponding open group, not an unpaired comparison to its entire 128-case sample. Blocked contrasts describe an intentional obstruction intervention, not an expected insertion-success improvement.',r'\begin{center}\small',r'\begin{tabular}{lrrrrr}\toprule',r'Condition & Images & X MAE mm & Y MAE mm & Yaw MAE deg & Yaw P95 deg\\\midrule']
 for row in s['summaries']:
  if not row['sensing_observations']:continue
  e=row['sensing_axis_mae_m_m_rad'];p95=row['sensing_axis_p95_m_m_rad']
  lines.append(label(row['group'])+f" & {row['sensing_observations']} & {1000*e[0]:.3f} & {1000*e[1]:.3f} & {math.degrees(e[2]):.3f} & {math.degrees(p95[2]):.3f}"+r'\\')
 lines += [r'\bottomrule\end{tabular}\end{center}',r'Sensing diagnostics use logged valid pre-action planar images; invalid images and phase-gate failures remain explicit episode outcomes. These estimates are not used during changing-height insertion. No calibrated-confidence or real-camera accuracy claim follows.']
 (out/'contact_results.tex').write_text('\n\n'.join(lines)+'\n')
 print(json.dumps({'contact_complete':s['complete'],'observed':s['observed_episodes'],'expected':s['expected_episodes']}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('dual-arm-mas/configs/contact_study_v2.json'));p.add_argument('--draft',action='store_true');run(p.parse_args())
