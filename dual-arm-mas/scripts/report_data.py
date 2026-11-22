"""Generate report tables from audited outcomes; never runs/tunes controllers."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone


LABELS = {"a_only": "A-only", "b_only": "B-only", "half": "50/50",
          "full_info": "Full-info reference", "local_convention": "Fixed local convention",
          "local_model": "Adaptive local", "capability": "Capability (S2)",
          "belief_prior": "Fixed belief", "prediction": "Prediction (S3)", "intent": "Intent (S4)"}
SCENARIOS = {"noise": "Sensed noise", "drop25": "25\\% drops", "delay1": "One-step delay",
             "unequal_limits": "B half limits", "tight_tolerance": "Tight tolerance",
             "loose_tolerance": "Loose tolerance", "history1": "History 1",
             "history8": "History 8", "float16": "Float16"}


def table(headers, rows, caption, columns=None):
    return "\n".join([r"\begin{center}\begin{minipage}{\linewidth}\centering\small",
        r"\refstepcounter{table}\textbf{Table \thetable:} "+caption+r"\par\smallskip",
        r"\begin{tabular}{"+(columns or "l"+"r"*(len(headers)-1))+"}", r"\toprule",
        " & ".join(headers)+r" \\", r"\midrule",
        *[" & ".join(row)+r" \\" for row in rows], r"\bottomrule", r"\end{tabular}",
        r"\end{minipage}\end{center}", ""])


def generate(evaluation, report):
    analysis = evaluation / "analysis"
    stats_path = analysis / "statistics.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    audit = json.loads((evaluation / "completion_audit.json").read_text())
    assert audit["passed"] and audit["episodes"] == 23456
    assert hashlib.sha256((evaluation / "episodes.jsonl").read_bytes()).hexdigest() == audit["episodes_sha256"]
    summaries = stats["summaries"]
    assert sum(s["n"] for s in summaries) == audit["episodes"]
    assert all(s["success_rate"] == 1 and s["invalid_actions"] == 0 for s in summaries)
    core = {s["method"]: s for s in summaries if s["scenario"] == "core" and s["fit"] == 0}
    report.mkdir(parents=True, exist_ok=True)
    def write(name, content):
        (report / name).write_text(content, encoding="utf-8")
    write("results.tex", "\\newcommand{\\CoreN}{1,000}\n\\newcommand{\\TotalN}{23,456}\n")
    rows = [[LABELS[m], f"{s['mean_cost_rad2']*1000:.4f}",
             f"[{s['cost_bootstrap_95'][0]*1000:.4f}, {s['cost_bootstrap_95'][1]*1000:.4f}]",
             f"{s['mean_joint_travel_rad']:.3f}", f"{s['mean_steps']:.3f}", f"{s['mean_bytes']:.2f}"] for m,s in core.items()]
    write("core_table.tex", "All core methods succeeded on all 1,000 paired starts, with no invalid actions. "
          "Overall and success-conditioned motion costs therefore coincide.\n" + table(
        ["Method", "$10^3 C$", "95\\% cost CI", "$L_1$ (rad)", "Steps", "Bytes"], rows,
        "Core fit 0. Cost is in $10^{-3}$ rad$^2$; intervals are episode-bootstrap means. "
        "Full-info's zero payload is an oracle accounting convention, not free distributed access."))
    rows = [[LABELS[m], f"{s['mean_terminal_xy_mm']:.3f}", f"{s['mean_terminal_yaw_deg']:.3f}",
             f"{s['mean_overshoot_steps']:.3f}", f"{s['mean_same_world_direction_steps']:.3f}"] for m,s in core.items()]
    write("error_table.tex", table(["Method", "XY (mm)", "Yaw (deg)", "Overshoots", "Same-dir."], rows,
        "Fit-0 mean terminal XY norm, absolute yaw, and per-episode event counts. Same-direction "
        "counts a step with any same-sign world action component; it is not a cooperation failure."))
    pairs = [p for p in stats["paired_comparisons"] if p["scenario"] == "core"]
    rows = [[str(p["fit"]), LABELS[p["method"]]+" minus "+LABELS[p["control"]],
             f"{p['mean_paired_cost_difference_rad2']*1e6:.2f}",
             f"[{p['paired_bootstrap_95'][0]*1e6:.2f}, {p['paired_bootstrap_95'][1]*1e6:.2f}]"] for p in pairs]
    write("comparison_table.tex", table(["Fit", "Paired comparison", "$10^6\\Delta C$", "95\\% paired CI"], rows,
        "Paired differences in $10^{-6}$ rad$^2$, 1,000 matched starts each. Negative favours the "
        "first method. Intent changes reconciliation, so its contrast is not information alone.", "rlrr"))
    rows = [[LABELS[f["method"]], *[f"{v*1e3:.5f}" for v in f["cost_means_rad2"]],
             f"{f['between_fit_sample_sd_rad2']*1e3:.5f}"] for f in stats["fitting_variability"]]
    write("fit_table.tex", table(["Method", "Fit 0", "Fit 1", "Fit 2", "Sample SD"], rows,
        "Cost means and between-fit sample SD in $10^{-3}$ rad$^2$. All fits share test starts; "
        "these are three public-model fits, not independent RL training experiments."))
    conditions = {(p["scenario"],p["method"]): p for p in stats["condition_comparisons"]}
    stress = [s for s in summaries if s["scenario"] != "core"]
    rows = []
    for s in stress:
        p = conditions[s["scenario"], s["method"]]
        rows.append([SCENARIOS[s["scenario"]], LABELS[s["method"]], f"{s['mean_cost_rad2']*1e3:.4f}",
                     f"{p['mean_paired_cost_difference_rad2']*1e6:+.2f}",
                     f"{s['mean_steps']:.3f}", f"{s['mean_bytes']:.2f}"])
    write("stress_table.tex", table(["Condition", "Method", "$10^3 C$", "$10^6\\Delta C$", "Steps", "Bytes"], rows,
        "All bounded conditions: 128 paired starts per row, 100\\% observed success, no invalid actions. "
        "$\\Delta C$ is against the SAME 128 core starts/method, not the full-core mean. "
        "Condition paired intervals are retained in statistics.json.", "llrrrr"))
    cap = core["capability"]; local = core["local_model"]; pred = core["prediction"]; belief = core["belief_prior"]
    cap_gain = 100*(local["mean_cost_rad2"]-cap["mean_cost_rad2"])/local["mean_cost_rad2"]
    pred_gain = 100*(belief["mean_cost_rad2"]-pred["mean_cost_rad2"])/belief["mean_cost_rad2"]
    write("findings.tex", r"\subsection{Interpretation of the core comparison}"+"\n" +
        f"Capability sharing reduced mean cost by {cap_gain:.2f}\\% versus the strong adaptive local control "
        f"in fit 0, at {cap['mean_bytes']:.2f} payload bytes per episode. Its aggregate cost and steps "
        "match the full-information reference to displayed precision, as the shared deterministic rule predicts. "
        "This is a small conditional gain, not a global-optimality result. Intent used half the per-step "
        "payload and approached the reference, but its reconciliation rule is an algorithmic change.\n\n" +
        f"History prediction reduced cost by {pred_gain:.2f}\\% against its matched fixed-belief solver "
        f"(fit 0), with the same sign in fits 1 and 2. It took {pred['mean_steps']:.3f} versus "
        f"{belief['mean_steps']:.3f} steps and left greater mean terminal error, despite every episode succeeding. "
        "It costs substantially more than the allocation family while finishing sooner. Squared command-level "
        "joint cost is subdivision-dependent: splitting a displacement into more commands can lower its sum "
        "of squares. The fixed 24 interpolation ticks per command do not remove that dependence. "
        "Neither this cross-family ranking nor the modest history gain proves superior intent inference. "
        "A-only also has lower mean $L_1$ travel than the reference despite greater squared cost: "
        "which motion metric is valued matters.\n")
    write("stress_findings.tex", r"\paragraph{Stress interpretation.} "+
        "No bounded impairment caused an observed failure. Noise-cost intervals against matched core starts "
        "include zero for all tested methods. One-step delay increased capability cost, but slightly reduced "
        "intent cost and steps on this subset; stale messages are not universally beneficial or harmful. "
        "B's half limit increased steps but lowered prediction's squared-motion cost, illustrating the "
        "subdivision tradeoff. Tighter/looser tolerances change the task and stopping criterion, not just "
        "controller quality. History 8 produced identical outcomes to history 4 on this subset; episodes "
        "are short, so this does not establish general memory sufficiency. Float16 halved payload with "
        "identical displayed capability cost and a negligible intent-cost change.\n\n" +
        "Zero observed failures is not zero risk. Starts are feasibility-filtered, stress samples contain "
        "only 128 configurations, repeated fits share starts, and this study did not exercise contact, "
        "dynamics, unknown policies, independent sensing errors or many training replications.\n")
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "statistics_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "episodes_sha256": audit["episodes_sha256"],
        "derived_only": True,
        "tools_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name("analyze.py")]}}
    (analysis / "report_provenance.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Generated audited report tables in {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, default=Path("dual-arm-mas/runs/evaluation"))
    parser.add_argument("--report", type=Path, default=Path("dual-arm-mas/docs/report"))
    args = parser.parse_args()
    generate(args.evaluation, args.report)
