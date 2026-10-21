# Python and Rust speed comparison

[The measured report](windows-x64.json) retains every timing sample, repetition
count, quartile, numerical check, input fingerprint, source hash and environment
detail. [The input specification](windows-x64-inputs.json) and the two benchmark
implementations are sufficient to regenerate the deterministic data. The README
table and [chart](comparison.svg) use the same report, without an aggregate score.

## What was compared

The Python reference is the maintainer's OpenBO revision
[`3ceda2d34225d2f1601fc57815cd821fbc78f37d`](https://github.com/yichiliao/openbo/tree/3ceda2d34225d2f1601fc57815cd821fbc78f37d),
checked against the upstream `main` branch on 2026-10-21. It is a separate,
unmodified source snapshot, rather than the Python fork with additional fixes.
The Rust implementation is OpenBO Rust 0.1.0, built in release mode. The report
records hashes of the actual reference modules, Rust numerical source, benchmark
drivers and executable; its Git status also records the benchmark additions that
were awaiting commit when measured.

| Workload | Work per timed call |
| --- | --- |
| Branin objective batch | Evaluate 65,536 normalized 2D inputs, without noise |
| Small GP | Construct, fit 64 observations, predict mean and latent variance at 256 queries, 3D |
| Larger GP | Construct, fit 256 observations, predict at 1,024 queries, 3D |
| Expected improvement | Score 65,536 supplied means/variances for maximization |
| Exact 2D hypervolume | Compute volume from 256 points, including Pareto filtering; 128 are nondominated |
| BO candidate scoring | Fit the larger GP, predict at the same 1,024 candidates, compute EI and select its maximum |

Both GP implementations use an RBF kernel, ARD lengthscales `[0.2, 0.35, 0.5]`,
signal variance `1.3`, noise variance `0.0001`, prior mean `0.2`, no target
standardization and no hyperparameter optimization. Hypervolume maximizes both
objectives relative to `[0, 0]`. Both backends generate identical float64 inputs
using the same SplitMix64 specification and seed `20261021`.

Before any timing, the runner checks matching input-bit fingerprints, finite
outputs and numerical agreement at `rtol=1e-7`, `atol=1e-9`. It also verifies the
hypervolume front size and identical selected candidate for BO scoring.

## How timing works

Each backend runs computations in-process. Compilation, imports, process startup,
input generation, JSON transport and correctness comparisons are excluded.
Public API validation, allocation and output deallocation are included. Python's
cyclic garbage collector is disabled during timing; normal reference counting
and allocation remain enabled.

The recorded run has five blocks per workload, alternating which backend runs
first. Each block calibrates its repetition count to at least 100 ms, warms up
with three calls, then records seven samples. Thus each reported median and
interquartile range uses 35 per-call measurements per backend. Raw elapsed nanoseconds and
repetition counts are retained. The ratio is Python median divided by Rust
median; values below one mean Rust took longer.

The machine was an Intel Core Ultra 7 155H running Windows 11 Pro x64, on AC power
with the Balanced power plan. Both processes were pinned to the same logical
CPU, and NumPy/BLAS thread pools were limited to one thread. Versions and the
effective affinity are recorded in the JSON. Compilation and other validation
jobs were stopped before measuring. Operating-system activity and thermal state
can still affect results; the quartiles describe observed variation on this
machine.

These measurements compare the current implementations and numerical routines.
Python uses NumPy/SciPy, including general dense GP solves, `np.vectorize(math.erf)`
for EI and a cell-grid hypervolume method. Rust uses nalgebra triangular solves,
statrs normal probabilities and a hypervolume sweep. Differences therefore
include algorithms, allocation and library backends as well as language overhead.
These component timings do not measure complete optimization runs or solution
quality. Hyperparameter fitting, random streams and acquisition searches differ
between the packages; see [PORTING.md](../../PORTING.md).

## Reproduce

Use a Python environment with the Python reference project's dependencies,
`threadpoolctl` and Matplotlib. The Rust library itself does not require Python.
For the recorded versions, see the report's `metadata` object. From the Rust
repository root:

```sh
git clone https://github.com/yichiliao/openbo.git ../python-openbo
git -C ../python-openbo checkout 3ceda2d34225d2f1601fc57815cd821fbc78f37d
python -m pip install -e ../python-openbo threadpoolctl matplotlib
python tools/benchmark_python_rust.py --python-checkout ../python-openbo --reference-revision 3ceda2d34225d2f1601fc57815cd821fbc78f37d --build --blocks 5 --samples-per-block 7 --target-ms 100 --output results/benchmarks/local.json
python tools/plot_benchmark.py --report results/benchmarks/local.json
```

The second tool writes SVG and PNG charts and prints the Markdown comparison
table. The benchmark currently requires Windows or Linux CPU-affinity support.
Use `--cpu N` to select an allowed logical CPU, or omit it to choose the first
allowed CPU. Increasing `--blocks`, `--samples-per-block` or `--target-ms` permits
longer measurements. Both tools provide `--help` without importing optional
Python dependencies. The Rust timing helper rejects debug builds.
