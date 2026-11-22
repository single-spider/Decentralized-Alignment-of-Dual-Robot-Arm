# Decentralized Alignment of Dual Robot Arms

**Group 28 · Foundations of AI: Multiagent Systems · IIT Gandhinagar**

How should two robot arms share a small alignment correction when each has only
limited information about the other? This project studies that question using
two Franka Panda arms in PyBullet: one holds a peg, the other a receiving fixture.
We compare decentralized coordination rules and learned policies, then test
whether RGB-based alignment carries over to force-controlled insertion.

**[Final report](output/pdf/group28-dual-arm-mas-readable.pdf)** ·
**[Complete LaTeX source](report-readable/)** ·
**[Overleaf ZIP](output/group28-readable-overleaf.zip)** ·
**[Recorded demonstration](dual-arm-mas/runs/assembly-demo-v7/contact-assembly.gif)**

## Project Overview

Both arms can correct the same relative position and orientation error. However,
their configurations can make one arm's motion more expensive than the other's.
Acting without coordination can also cause over-correction or unnecessary
movement. Our objective is to understand how much teammate information is needed
to coordinate effectively, and how sensing affects the resulting assembly task.

The course project implements the shared two-arm environment, information-sharing
protocols, cost-aware allocation baselines, cooperative policy training, adapted
RGB perception for a moving receiver, and paired alignment and contact studies.

## Team Members

| Member | Roll number |
| --- | --- |
| Gaurav Srivastava | 23110113 |
| Aayush Bundel | 23110005 |
| Laxmidhar Swine | 23110326 |
| Roshin Gelasious | 23110280 |

## Methodology

1. **Estimate relative error.** Use ground-truth pose for controlled allocation
   comparisons, or neural RGB segmentation and geometric estimation for visual
   execution.
2. **Allocate planar corrections.** Each arm selects its action from the same
   pre-step observation, subject to its information regime. Both actions are
   prepared before execution in the shared simulation.
3. **Learn cooperative policies.** Train role-conditioned PPO actors with a
   centralized critic; execution uses local observations and permitted messages.
4. **Evaluate contact insertion separately.** After alignment and a measured-pose
   check, the receiver arm holds while a deterministic force-feedback controller
   regulates the peg arm's descent and bounded retraction.

| Regime | Information used for coordination |
| --- | --- |
| S0: joint reference | Both arms' capability, used by a centralized allocator |
| S1: local model | Own cost and a shared prior; no teammate messages |
| S2: capability exchange | The teammate's local cost matrix |
| S3: history-based belief | Past shared error changes and own executed motion; no messages |
| S4: explicit bids | Pre-action bids interpreted through a shared allocation convention |

Decentralized execution still relies on common sensing and agreed coordination
conventions. The study distinguishes kinematic alignment from dynamic insertion;
successful alignment alone is not counted as successful assembly.

## Results and Demonstration

| Study | Evaluation coverage | Key finding |
| --- | --- | --- |
| Deterministic alignment | 23,456 episodes | On 1,000 matched starts for fit 0, the strongest no-message rule and joint reference both achieved 100% alignment; the local rule cost only 1.1% more in squared joint motion. |
| Learned alignment | 25,280 episodes; 12 actor/regime fits | Neural RGB alignment success was 99.2–99.9%. Under ground-truth sensing, learned policies used more squared motion than fixed half. |
| Dynamic contact | 1,216 trials | Learned RGB insertion success ranged from 68.0–79.7% across six open-hole conditions. Fixed-half RGB inserted on 104/128 starts; S1 with ground truth on 118/128. |

Capability exchange closed the small deterministic cost gap to displayed
precision, but richer information did not consistently improve performance.
In the contact study, the four learned RGB policies for fit 0 also used more
motion than fixed-half RGB. High visual alignment success therefore did not
translate into equally high insertion success.

| Successful insertion | Blocked opening: jam and release |
| --- | --- |
| ![Successful simulated insertion with depth and camera views](assets/screenshots/inserted.png) | ![Simulated blocked-hole run after bounded retraction](assets/screenshots/jam-release.png) |

Watch the [RGB-to-insertion replay](dual-arm-mas/runs/assembly-demo-v7/contact-assembly.gif).
PPO chooses the planar alignment corrections; force feedback controls axial
insertion. The blocked controls produced no false insertions in 64 trials, and
all 53 attempted blocked-hole retractions released. Recovery was less reliable
in open-hole trials, so this is a controlled jam response rather than a general
recovery guarantee.

All results are **simulation-only**, with fixed grasp constraints and a limited
geometry family. Squared joint motion is a movement metric, not energy; the
force checks do not constitute hardware safety certification.

Detailed results: [deterministic allocation](dual-arm-mas/runs/evaluation/analysis/summary.md) ·
[learned alignment](dual-arm-mas/runs/learned-study/analysis/summary.md) ·
[contact insertion](dual-arm-mas/runs/contact-study-v2/analysis/summary.md).

## Installation

The supplied setup targets **Windows x64 and Python 3.11**. From the repository
root in PowerShell:

```powershell
py -3.11 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
& $python -m pip install -r requirements-reference.txt
```

CPU PyTorch is sufficient to load the included models and run the demonstration.
Allow approximately 1.5 GB for the repository and working outputs. See the
[environment notes](docs/REPRODUCIBILITY.md#reference-environment) for runtime
versions and rendering requirements.

## How to Run

### Demonstration

After installation, run four saved-policy cases: a fixed-half baseline, two
learned open-hole cases, and a learned blocked-hole case with jam/retraction.

```powershell
& $python tools/reproduce.py demo --out replay-new --seconds 240
```

Outputs are written to `.work/replay-new/`. Use a new `--out` name each time.
If the time limit is insufficient on your machine, use a new name and increase
`--seconds`.

### Evaluation

These short examples evaluate saved policies on two starts each; they are quick
execution checks, not reproductions of the full reported success rates.

```powershell
& $python tools/reproduce.py alignment-eval --regime s1 --fit 0 --sensing rgb --count 2 --out rgb-check --seconds 90
& $python tools/reproduce.py contact-eval --group open-fit0-s1-rgb --count 2 --out contact-check --seconds 90
& $python tools/reproduce.py baseline-eval --scenario core --count 2 --out baseline-check --seconds 90
```

### Training

To train a PPO fit with the study's 10,000-step minimum:

```powershell
& $python tools/reproduce.py train --regime s1 --fit 0 --steps 10000 --batch-steps 256 --epochs 5 --out ppo-s1-new --seconds 3600
```

The time allowance may need adjustment for your machine. The
[training guide](docs/REPRODUCIBILITY.md#evaluation-and-training-commands)
also covers other regimes and fits, vision adaptation, and robustness conditions.

## Repository Structure

| Directory | Contents |
| --- | --- |
| `dual-arm-mas/src/` | Two-arm environment, coordination methods, learned policies and contact control |
| `dual-arm-mas/configs/`, `scripts/`, `tests/` | Experiment settings, entry points and implementation checks |
| `dual-arm-mas/runs/` | Trained models, recorded trials, summaries and demonstrations |
| `peg-in-hole-sfn/` | Reused SFN perception and Panda simulation components |
| `report-readable/` | Final report's complete LaTeX source and figures; compile `main.tex` |
| `output/` | Final PDF and portable Overleaf ZIP |
| `tools/` | Demonstration, training, evaluation, analysis and report-build utilities |
| `docs/` | Detailed reproducibility and environment instructions |
| `evidence/`, `verification/` | Compressed episode data and verification records |

## Reproducing Results

The saved episode records allow the reported statistics to be recomputed without
retraining or rerunning every simulation:

```powershell
& $python tools/reproduce.py verify --out check-all
```

This checks the three studies, including all 12 PPO fits, against the recorded
results. For hash checks, checkpoints, full command options, report rebuilding
and earlier report versions, see [Reproducibility and Technical Details](docs/REPRODUCIBILITY.md).

## References and Acknowledgments

The perception pipeline builds on **Xie et al., “Learning to Fill the Seam by
Vision: Sub-millimeter Peg-in-hole on Unseen Shapes in Real World,” ICRA 2022**
([paper](https://arxiv.org/abs/2204.07776),
[original implementation](https://github.com/xieliang555/SFN)).
The reused [single-arm implementation](https://single-spider.github.io/SFN/)
was developed for Gaurav Srivastava's MTech thesis under Prof. Harish P M.
The decentralized two-arm coordination and evaluations described here are the
course-project extension.

The final report cites PPO and cooperative multi-agent PPO and includes the
team's contribution record. AI tools assisted project development and report
preparation; the authors retain responsibility for the work and final content.
LaTeX template credits are provided in the [template attribution](report-readable/TEMPLATE.md).
