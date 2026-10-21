# Complete workflow study

This study runs the public Python and native Rust optimizers through complete
small experiments, measuring both elapsed time and optimization quality. It
extends the [fixed-input component comparison](../README.md).

Each environment is a separate experiment. The local Windows machine, hosted
Linux x86-64, hosted Windows x86-64 and hosted macOS Apple Silicon reports retain
their own hardware, Python/dependency versions, compiler, affinity and source
hashes. Compare backends within one report. Timings from different runners are
not pooled into an overall speedup.

## Workloads and controls

| Workflow | Target budget | Initialization / prior information | Quality measure |
| --- | ---: | --- | --- |
| Scalar BO, normalized Branin | 16 | Four shared initial observations, 128-candidate grid search | Simple regret, lower is better |
| Scalar BO, Hartmann-6 | 16 | Four shared initial observations, 128-candidate grid search | Simple regret, lower is better |
| Independent IEIPV MOBO, OKA2 | 16 | Four shared initial observations, 128 candidates and bounded refinement | Hypervolume / analytic optimum, higher is better |
| TAF-R, normalized Branin | 16 | Source-only start with three shared 12-observation source tasks | Simple regret, lower is better |
| Small NAF PPO training + held-out BO | 16 | Two PPO iterations, two 16-step episodes per iteration, three shared training tasks | Held-out simple regret, lower is better |

Every workload uses seeds `0, 1, 2, 3, 4`. Scalar and MOBO initial observations
match exactly across the two implementations; subsequent candidate pools and
random streams differ. Transfer fixtures contain the same data and saved GP
parameters, and each implementation must confirm that all three sources loaded
without warnings. Offline preparation of the fixed fixtures is excluded; loading
and fitting sources during optimizer construction is timed.

The neural run includes **64 training objective evaluations plus 16 held-out
target evaluations**. The small network has two hidden layers of width eight.
It uses the final policy after two PPO iterations, with actual training history
and evaluation counts checked. This exercises training and inference; it is not
evidence of a converged neural policy. Python networks use float32 PyTorch and
native Rust uses float64.

Scalar/transfer GP fitting uses matching iteration and restart caps (50 and two).
Python warms ordinary scalar GP refits from the preceding fitted parameters;
native Rust starts from its configured parameters. MOBO uses independent GPs and
MAP fitting with the same caps. Python refines acquisition candidates with one
L-BFGS-B start limited to eight iterations; Rust uses its native coordinate
refinement. Matching budgets and caps does not make these different numerical
and search routines identical. Complete configurations are retained in every run.

## Timing and checks

Both implementations measure elapsed wall time internally, starting before
optimizer construction. Timing includes model/source loading and fitting,
suggestions, objective calls, observations, result histories and neural training.
Imports, process startup, compilation, shared fixture preparation, JSON output
and independent verification are excluded. PyTorch's one-time lazy optimizer
imports are materialized before measurement. Backend order alternates by
workflow and seed. Numerical thread pools are limited to one; Windows and Linux
processes share one permitted logical CPU. macOS does not provide supported hard
process affinity, and reports that scheduler placement is uncontrolled.

Before timing, fixed objective evaluations must agree across implementations.
After each run, every observation is independently reevaluated in Python and
checked for finite values, correct shape and bounds. The runner checks target
budget, shared initialization, monotonic times, source loading and actual neural
training counts. Scalar quality is known optimum minus best observed value;
OKA2 quality is exact dominated hypervolume divided by its analytic optimum.
Metrics are recomputed after timing using the same reference evaluator.

Tables and figures use medians and interquartile ranges across the five seeds.
A time ratio is the Python median divided by the Rust median. Read achieved
quality alongside that ratio: faster completion can accompany different or worse
solutions. Small budgets, five seeds and shared hosted runners support a bounded
implementation comparison, rather than paper-scale or language-only conclusions.

## Reproduce

Use a reference Python environment and the exact upstream source revision:

```sh
git clone https://github.com/yichiliao/openbo.git ../python-openbo
git -C ../python-openbo checkout 3ceda2d34225d2f1601fc57815cd821fbc78f37d
python -m pip install -e ../python-openbo threadpoolctl matplotlib
python tools/benchmark_workflows.py --python-checkout ../python-openbo --reference-revision 3ceda2d34225d2f1601fc57815cd821fbc78f37d --build --seeds 0 1 2 3 4 --output results/benchmarks/workflows-local.json
python tools/plot_workflows.py --reports results/benchmarks/workflows-local.json --output-dir results/benchmarks/figures
```

The manual **Workflow performance and quality study** GitHub Actions workflow
runs the same study on Linux x86-64, Windows x86-64 and macOS ARM64, using the
official Python baseline's locked numerical dependency versions. Download its
per-host artifacts for full reports and `pip freeze` provenance. Local versions
can differ; each report records what was actually used. Each run has a bounded
180-second timeout, and validated partial results are retained if a later run
fails. Partial or failed runs are never substituted for a complete report.
