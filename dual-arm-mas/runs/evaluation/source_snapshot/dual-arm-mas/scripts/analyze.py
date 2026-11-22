"""Paired episode uncertainty and independent public-model fitting variability."""
from pathlib import Path
import argparse
import json
import numpy as np


def bootstrap(values, rng, samples=2000):
    values = np.asarray(values)
    # Bounded batches avoid a bootstrap-by-all-episodes memory spike.
    means = []
    for _ in range(0, samples, 100):
        indices = rng.integers(len(values), size=(min(100, samples-len(means)), len(values)))
        means.extend(values[indices].mean(axis=1).tolist())
    return np.quantile(means, [.025, .975]).tolist()


def analyze(path, out):
    groups = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            groups.setdefault((row["scenario"], row["fit"], row["method"]), {})[row["start_id"]] = row
    rng = np.random.default_rng(704)
    summaries, comparisons = [], []
    for (scenario, fit, method), byid in groups.items():
        rows = list(byid.values())
        cost = [e["motion_cost_rad2"] for e in rows]
        successes = [e for e in rows if e["success"]]
        error = np.array([e["final_error"] for e in rows])
        summaries.append({"scenario": scenario, "fit": fit, "method": method, "n": len(rows),
            "success_rate": float(np.mean([e["success"] for e in rows])),
            "success_bootstrap_95": bootstrap([e["success"] for e in rows], rng),
            "mean_cost_rad2": float(np.mean(cost)), "cost_bootstrap_95": bootstrap(cost, rng),
            "success_conditioned_cost_rad2": float(np.mean([e["motion_cost_rad2"] for e in successes])) if successes else None,
            "mean_joint_travel_rad": float(np.mean([e["joint_travel_rad"] for e in rows])),
            "mean_steps": float(np.mean([e["steps"] for e in rows])),
            "mean_bytes": float(np.mean([e["bytes"] for e in rows])),
            "mean_messages": float(np.mean([e["messages"] for e in rows])),
            "invalid_actions": sum(e["invalid_actions"] for e in rows),
            "mean_overshoot_steps": float(np.mean([e["overshoot_steps"] for e in rows])),
            "mean_same_world_direction_steps": float(np.mean([e["same_world_direction_steps"] for e in rows])),
            "mean_terminal_xy_mm": float(np.mean(np.linalg.norm(error[:, :2], axis=1)) * 1000),
            "mean_terminal_yaw_deg": float(np.mean(abs(error[:, 2])) * 180 / np.pi)})
        control = {"capability": "local_model", "prediction": "belief_prior", "intent": "local_model"}.get(method)
        if control and (scenario, fit, control) in groups:
            other = groups[scenario, fit, control]
            matched = sorted(set(byid) & set(other))
            differences = [byid[i]["motion_cost_rad2"] - other[i]["motion_cost_rad2"] for i in matched]
            comparisons.append({"scenario": scenario, "fit": fit, "method": method, "control": control, "n": len(matched),
                "mean_paired_cost_difference_rad2": float(np.mean(differences)),
                "paired_bootstrap_95": bootstrap(differences, rng),
                "interpretation": "Same-family information comparison" if method != "intent" else "Protocol/reconciliation comparison, not information alone"})
    variability = []
    for method in sorted({k[2] for k in groups}):
        fits = [s for s in summaries if s["scenario"] == "core" and s["method"] == method]
        if len(fits) > 1:
            means = [s["mean_cost_rad2"] for s in fits]
            variability.append({"method": method, "fitting_runs": len(fits), "cost_means_rad2": means,
                "between_fit_sample_sd_rad2": float(np.std(means, ddof=1)),
                "warning": "Only three independently fitted public priors/predictors, no RL runs; paired episodes shared across fits"})
    out.mkdir(parents=True, exist_ok=True)
    (out / "statistics.json").write_text(json.dumps({"bootstrap_seed": 704, "bootstrap_samples": 2000,
        "summaries": summaries, "paired_comparisons": comparisons, "fitting_variability": variability,
        "limitations": "Episode bootstrap is conditional on simulator/distribution/fitted model; all-success bootstrap degenerates and is not proof of zero failure risk"}, indent=2))
    lines = ["# Measured evaluation summary", "", "|Scenario|Fit|Method|N|Success|Cost rad²|Steps|Bytes|Invalid|", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"|{s['scenario']}|{s['fit']}|{s['method']}|{s['n']}|{s['success_rate']:.3f}|{s['mean_cost_rad2']:.6f}|{s['mean_steps']:.2f}|{s['mean_bytes']:.1f}|{s['invalid_actions']}|")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        core = [s for s in summaries if s["scenario"] == "core" and s["fit"] == 0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].barh([s["method"] for s in core], [s["mean_cost_rad2"] for s in core], color="#37677f")
        axes[0].set_xlabel("Measured squared joint motion (rad²)")
        axes[1].scatter([s["mean_bytes"] for s in core], [s["mean_cost_rad2"] for s in core])
        for s in core:
            axes[1].annotate(s["method"], (s["mean_bytes"], s["mean_cost_rad2"]), fontsize=7)
        axes[1].set_xlabel("Payload bytes / episode (shared sensing excluded)")
        axes[1].set_ylabel("Squared joint motion (rad²)")
        fig.tight_layout()
        fig.savefig(out / "cost_communication.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    print(f"Analyzed {sum(len(v) for v in groups.values())} episodes, {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.episodes, args.out)
