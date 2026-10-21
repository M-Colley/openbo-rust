"""Compare matched CPU numerical workloads with a separately installed Python OpenBO.

Run from this standalone Rust repository in the reference Python environment::

    python tools/benchmark_python_rust.py --python-checkout /path/to/official-openbo --reference-revision SHA --build

The script checks identical generated inputs and numerical outputs BEFORE timing.
Each backend is timed in-process after calibration and warmup. Process startup,
imports, RNG/input preparation, validation comparisons, and JSON I/O are excluded.
Results compare the existing implementations and their numerical backends, not
language overhead alone; no end-to-end optimizer equivalence is claimed.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASK = (1 << 64) - 1
WORKLOADS = (
    ("branin_batch", "Branin objective (65,536 points, 2D)"),
    ("gp_small", "GP fit + predict (64 train, 256 query, 3D)"),
    ("gp_medium", "GP fit + predict (256 train, 1,024 query, 3D)"),
    ("ei_batch", "Expected improvement (65,536 predictions)"),
    ("hypervolume_2d", "2D hypervolume (128 front + 128 dominated)"),
    ("bo_candidate_scoring", "BO candidate scoring (256 train, 1,024 candidates, 3D)"),
)
SPECIFICATION = {
    "seed": 20261021,
    "objective_points": 65536,
    "ei_points": 65536,
    "gp_small_train": 64,
    "gp_small_query": 256,
    "gp_medium_train": 256,
    "gp_medium_query": 1024,
    "hv_front": 128,
    "hv_dominated": 128,
}
GP_PARAMETERS = {
    "lengthscale": [0.2, 0.35, 0.5], "variance": 1.3, "noise": 1e-4,
    "kernel_type": "rbf", "standardize_targets": False,
    "optimize_hyperparameters": False, "mean_constant": 0.2,
}


class Generator:
    """SplitMix64 uniforms with exactly representable top-53-bit conversion."""

    def __init__(self, seed: int):
        self.state = seed

    def uniform(self) -> float:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
        value ^= value >> 31
        return (value >> 11) / 9007199254740992.0

    def matrix(self, rows: int, columns: int):
        return [[self.uniform() for _ in range(columns)] for _ in range(rows)]


def command_text(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_arrays(arrays) -> str:
    value = 0xCBF29CE484222325
    for array in arrays:
        for byte in array.astype("<f8", copy=False).tobytes(order="C"):
            value = ((value ^ byte) * 0x100000001B3) & MASK
    return f"fnv1a64-f64le:{value:016x}"


def git_metadata(directory: Path) -> dict:
    top = command_text(["git", "-C", str(directory), "rev-parse", "--show-toplevel"])
    if not top or Path(top).resolve() != directory.resolve():
        return {"commit": None, "status": None}
    commit = command_text(["git", "-C", str(directory), "rev-parse", "HEAD"])
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        return {"commit": None, "status": None}
    return {"commit": commit, "status": command_text(["git", "-C", str(directory), "status", "--short"])}


def hardware() -> dict:
    result = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "logical_cpus": os.cpu_count()}
    if sys.platform == "win32":
        result["power_plan"] = command_text(["powercfg", "/getactivescheme"])
        raw = command_text(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"])
        try:
            result["cpu"] = json.loads(raw)
        except json.JSONDecodeError:
            result["cpu_query"] = raw
    return result


@contextlib.contextmanager
def affinity(cpu: int | None):
    """Pin this process; benchmark child processes inherit the same CPU mask."""
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        process = kernel.GetCurrentProcess()
        old, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(process, ctypes.byref(old), ctypes.byref(system)):
            raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
        selected = (old.value & -old.value).bit_length() - 1 if cpu is None else cpu
        if selected < 0 or not old.value & (1 << selected):
            raise ValueError(f"logical CPU {selected} is outside allowed mask {old.value:#x}")
        if not kernel.SetProcessAffinityMask(process, 1 << selected):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
        effective = ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(process, ctypes.byref(effective), ctypes.byref(system)) or effective.value != 1 << selected:
            raise RuntimeError("failed to verify effective benchmark affinity")
        try:
            yield {"logical_cpu": selected, "effective_mask": hex(effective.value), "original_mask": hex(old.value), "child_affinity": "inherits pinned parent mask"}
        finally:
            kernel.SetProcessAffinityMask(process, old.value)
    elif hasattr(os, "sched_setaffinity"):
        original = os.sched_getaffinity(0)
        selected = min(original) if cpu is None else cpu
        if selected not in original:
            raise ValueError(f"logical CPU {selected} is outside allowed CPUs {sorted(original)}")
        os.sched_setaffinity(0, {selected})
        try:
            yield {"logical_cpu": selected, "effective_cpus": sorted(os.sched_getaffinity(0)), "original_cpus": sorted(original), "child_affinity": "inherits pinned parent CPUs"}
        finally:
            os.sched_setaffinity(0, original)
    else:
        raise RuntimeError("this matched benchmark requires affinity support (Windows or sched_setaffinity)")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(blocks: list[dict]) -> dict:
    raw, repetitions, normalized = [], [], []
    for block in blocks:
        repeats = block["repetitions"]
        for value in block["raw_elapsed_ns"]:
            if not math.isfinite(value) or value <= 0 or repeats < 1:
                raise ValueError("invalid or zero benchmark duration")
            raw.append(value)
            repetitions.append(repeats)
            normalized.append(value / repeats / 1e6)
    return {"median_ms": percentile(normalized, 0.5), "p25_ms": percentile(normalized, 0.25), "p75_ms": percentile(normalized, 0.75), "sample_count": len(raw), "samples_ms": normalized, "raw_elapsed_ns": raw, "repetitions": repetitions, "blocks": blocks}


def time_python(operation, samples: int, warmups: int, target_ms: float) -> dict:
    def elapsed(repetitions: int) -> int:
        start = time.perf_counter_ns()
        for _ in range(repetitions):
            result = operation()
            del result
        return time.perf_counter_ns() - start

    repetitions = 1
    # Calibrate with headroom so warmed/faster samples still exceed the minimum.
    target = target_ms * 1e6 * 2.0
    while True:
        duration = elapsed(repetitions)
        if duration >= target:
            break
        repetitions *= min(16, max(2, math.ceil(target / max(duration, 1))))
        if repetitions > 10_000_000:
            raise RuntimeError("workload too short for reliable calibration")
    for _ in range(warmups):
        result = operation()
        del result
    return {"backend": "python", "repetitions": repetitions, "raw_elapsed_ns": [elapsed(repetitions) for _ in range(samples)], "warmup_calls": warmups, "timer": "time.perf_counter_ns", "calibration_target_ms": target_ms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python-checkout", type=Path, help="official Python OpenBO checkout/archive containing src/openbo; omitted uses installed package")
    parser.add_argument("--reference-revision", help="declared Python source SHA, especially for an archive without .git")
    parser.add_argument("--binary", type=Path, help="release examples/benchmark_compare executable")
    parser.add_argument("--build", action="store_true", help="compile the release Rust example before benchmarking")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/local.json")
    parser.add_argument("--cpu", type=int, help="logical CPU to pin both backends to; default first allowed CPU")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--samples-per-block", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--target-ms", type=float, default=30.0)
    args = parser.parse_args()
    if args.blocks < 1 or args.samples_per_block < 1 or args.blocks * args.samples_per_block < 11 or args.warmups < 1:
        parser.error("need at least 11 samples per backend and positive block/sample/warmup counts")
    if not math.isfinite(args.target_ms) or args.target_ms < 1.0:
        parser.error("--target-ms must be finite and at least 1 ms")
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"
    source_root = None
    if args.python_checkout is not None:
        source_root = args.python_checkout.expanduser().resolve() / "src"
        if not (source_root / "openbo/__init__.py").is_file():
            parser.error("--python-checkout must contain src/openbo/__init__.py")
        sys.path.insert(0, str(source_root))
    try:
        import numpy as np
        import openbo
        import scipy
        import threadpoolctl
        from openbo.acquisition.ei import expected_improvement_maximization
        from openbo.benchmarks.hypervolume import hypervolume, pareto_mask
        from openbo.models.gp_scratch import GPScratch
        from openbo.test_functions.synthetic import branin
    except ImportError as error:
        parser.error(f"optional comparison needs the Python OpenBO reference, NumPy, SciPy and threadpoolctl ({error})")
    reference_file = Path(openbo.__file__).resolve()
    if source_root is not None and not reference_file.is_relative_to(source_root):
        parser.error("a different Python OpenBO was imported; use a fresh Python process")
    reference_package = reference_file.parent
    reference_files = {name: sha256_file(reference_package / name) for name in ["__init__.py", "models/gp_scratch.py", "models/kernels.py", "acquisition/ei.py", "benchmarks/hypervolume.py", "test_functions/synthetic.py"]}
    reference_git = git_metadata(source_root.parent if source_root is not None else reference_package)
    executable = "benchmark_compare.exe" if sys.platform == "win32" else "benchmark_compare"
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")).expanduser().resolve()
    if args.build:
        subprocess.run(["cargo", "build", "--locked", "--release", "--example", "benchmark_compare"], cwd=ROOT, check=True)
    binary = args.binary.expanduser().resolve() if args.binary is not None else target / "release/examples" / executable
    if not binary.is_file():
        parser.error(f"release benchmark helper not found: {binary}; use --build or --binary")
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec_file = args.output.with_name(args.output.stem + "-inputs.json")
    spec_file.write_text(json.dumps(SPECIFICATION, indent=2) + "\n", encoding="utf-8")

    data = {}
    data["branin_batch"] = (np.asarray(Generator(SPECIFICATION["seed"] ^ 1).matrix(SPECIFICATION["objective_points"], 2), dtype=np.float64),)
    for workload, small in [("gp_small", True), ("gp_medium", False)]:
        generator = Generator(SPECIFICATION["seed"] ^ (2 if small else 3))
        prefix = "gp_small" if small else "gp_medium"
        x = np.asarray(generator.matrix(SPECIFICATION[prefix + "_train"], 3), dtype=np.float64)
        y = (x[:, 0] * 2.0 + x[:, 1] * 0.5) - x[:, 2] * 0.25
        query = np.asarray(generator.matrix(SPECIFICATION[prefix + "_query"], 3), dtype=np.float64)
        data[workload] = (x, y, query)
    data["bo_candidate_scoring"] = data["gp_medium"]
    generator = Generator(SPECIFICATION["seed"] ^ 4)
    pairs = np.asarray([[2.0 * generator.uniform() - 1.0, 0.25 + generator.uniform()] for _ in range(SPECIFICATION["ei_points"])], dtype=np.float64)
    data["ei_batch"] = (np.ascontiguousarray(pairs[:, 0]), np.ascontiguousarray(pairs[:, 1]))
    generator = Generator(SPECIFICATION["seed"] ^ 5)
    n = SPECIFICATION["hv_front"]
    front = [[0.5 + (i + 1) / (n + 1), 1.5 - (i + 1) / (n + 1)] for i in range(n)]
    dominated = [[0.05 + 0.35 * generator.uniform(), 0.05 + 0.35 * generator.uniform()] for _ in range(SPECIFICATION["hv_dominated"])]
    data["hypervolume_2d"] = (np.asarray(front + dominated, dtype=np.float64),)
    digests = {key: digest_arrays(arrays) for key, arrays in data.items()}
    best = float(np.max(data["gp_medium"][1]))

    def gp_operation(workload: str):
        x, y, query = data[workload]
        model = GPScratch(**GP_PARAMETERS)
        model.fit(x, y)
        return model.posterior(query)

    def pipeline():
        mean, variance = gp_operation("gp_medium")
        # Incumbent calculation is part of both pipelines, outside the GP helper.
        incumbent = float(np.max(data["gp_medium"][1]))
        values = expected_improvement_maximization(mean, variance, incumbent)
        index = int(np.argmax(values))
        return index, float(values[index])

    operations = {
        "branin_batch": lambda: branin(data["branin_batch"][0]),
        "gp_small": lambda: gp_operation("gp_small"),
        "gp_medium": lambda: gp_operation("gp_medium"),
        "ei_batch": lambda: expected_improvement_maximization(*data["ei_batch"], 0.2),
        "hypervolume_2d": lambda: hypervolume(data["hypervolume_2d"][0], [0.0, 0.0]),
        "bo_candidate_scoring": pipeline,
    }
    parameters = {
        "branin_batch": {"points": SPECIFICATION["objective_points"], "input_dim": 2, "noise_std": 0.0},
        "gp_small": {"training_points": SPECIFICATION["gp_small_train"], "query_points": SPECIFICATION["gp_small_query"], "input_dim": 3, **GP_PARAMETERS},
        "gp_medium": {"training_points": SPECIFICATION["gp_medium_train"], "query_points": SPECIFICATION["gp_medium_query"], "input_dim": 3, **GP_PARAMETERS},
        "ei_batch": {"points": SPECIFICATION["ei_points"], "best_y": 0.2, "mean_range": [-1.0, 1.0], "variance_range": [0.25, 1.25]},
        "hypervolume_2d": {"points": n + SPECIFICATION["hv_dominated"], "nondominated_points": n, "objectives": 2, "reference": [0.0, 0.0]},
        "bo_candidate_scoring": {"training_points": SPECIFICATION["gp_medium_train"], "candidate_points": SPECIFICATION["gp_medium_query"], "input_dim": 3, "best_y": best, "acquisition": "EI", "selection": "argmax, first tie", **GP_PARAMETERS},
    }
    metadata = {
        "hardware": hardware(), "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "threadpoolctl": threadpoolctl.__version__,
        "rustc": command_text(["rustc", "--version", "--verbose"]), "cargo": command_text(["cargo", "--version"]),
        "rust_binary": str(binary), "rust_binary_sha256": sha256_file(binary), "rust_git": git_metadata(ROOT),
        "python_openbo": str(reference_file), "python_git": reference_git, "python_reference_files_sha256": reference_files,
        "specification_sha256": sha256_file(spec_file),
        "rust_files_sha256": {name: sha256_file(ROOT / name) for name in ["Cargo.toml", "Cargo.lock", "src/core.rs", "src/acquisition.rs", "examples/benchmark_compare.rs", "tools/benchmark_python_rust.py"]},
    }
    rows = []
    with affinity(args.cpu) as effective_affinity, threadpoolctl.threadpool_limits(limits=1):
        metadata["affinity"] = effective_affinity
        metadata["threadpools"] = threadpoolctl.threadpool_info()
        if any(pool.get("num_threads", 1) != 1 for pool in metadata["threadpools"]):
            raise RuntimeError("reference BLAS/threadpool is not limited to one thread")
        verified = json.loads(subprocess.run([str(binary), "--input", str(spec_file), "--mode", "verify"], capture_output=True, text=True, check=True).stdout)
        if verified["profile"] != "release":
            raise ValueError("Rust benchmark must be a release build")
        metadata["rust_upstream_revision"] = verified["upstream_revision"]
        validations = {}
        for workload, _ in WORKLOADS:
            native = verified["outputs"][workload]
            if native["input_digest"] != digests[workload]:
                raise AssertionError(f"{workload}: exact input-bit fingerprint differs")
            actual, expected = [], []
            reference = operations[workload]()
            if workload in ("branin_batch", "ei_batch"):
                actual, expected = [native["output"]["values"]], [reference]
            elif workload.startswith("gp_"):
                actual = [native["output"]["mean"], native["output"]["variance"]]
                expected = list(reference)
            elif workload == "hypervolume_2d":
                if native["output"]["front_size"] != n or int(pareto_mask(data[workload][0]).sum()) != n:
                    raise AssertionError("hypervolume front size differs from prescribed workload")
                actual, expected = [[native["output"]["value"]]], [[reference]]
            else:
                if native["output"]["index"] != reference[0]:
                    raise AssertionError("candidate-scoring argmax differs")
                actual, expected = [[native["output"]["score"]], native["output"]["point"]], [[reference[1]], data["gp_medium"][2][reference[0]]]
            max_absolute = max_relative = 0.0
            values = 0
            for native_array, reference_array in zip(actual, expected):
                native_array = np.asarray(native_array, dtype=np.float64)
                reference_array = np.asarray(reference_array, dtype=np.float64)
                if native_array.shape != reference_array.shape or not np.isfinite(native_array).all() or not np.isfinite(reference_array).all():
                    raise AssertionError(f"{workload}: invalid output shape/value")
                np.testing.assert_allclose(native_array, reference_array, rtol=1e-7, atol=1e-9, err_msg=workload)
                error = np.abs(native_array - reference_array)
                max_absolute = max(max_absolute, float(error.max(initial=0.0)))
                max_relative = max(max_relative, float((error / np.maximum(np.abs(reference_array), 1e-12)).max(initial=0.0)))
                values += native_array.size
            validations[workload] = {"passed": True, "input_digest": digests[workload], "values_compared": values, "max_absolute_error": max_absolute, "max_relative_error": max_relative, "rtol": 1e-7, "atol": 1e-9}
        print("All exact-input and numerical-output comparisons passed before timing.", flush=True)
        # Keep ordinary Python allocation/reference counting; disable only cyclic
        # GC so unpredictable collections do not affect one backend's samples.
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for number, (workload, label) in enumerate(WORKLOADS):
                recorded = {"python": [], "rust": []}
                orders = []
                for block in range(args.blocks):
                    order = ["python", "rust"] if (number + block) % 2 == 0 else ["rust", "python"]
                    orders.append(order)
                    for backend in order:
                        if backend == "python":
                            sample = time_python(operations[workload], args.samples_per_block, args.warmups, args.target_ms)
                        else:
                            response = subprocess.run([str(binary), "--input", str(spec_file), "--mode", "time", "--workload", workload, "--samples", str(args.samples_per_block), "--warmups", str(args.warmups), "--min-sample-ms", str(args.target_ms)], capture_output=True, text=True, check=True)
                            sample = json.loads(response.stdout)
                            if sample["input_digest"] != digests[workload]:
                                raise AssertionError("timed Rust inputs changed after verification")
                        sample["block"] = block
                        recorded[backend].append(sample)
                python_summary, rust_summary = summarize(recorded["python"]), summarize(recorded["rust"])
                row = {"id": workload, "label": label, "parameters": parameters[workload], "validation": validations[workload], "python": python_summary, "rust": rust_summary, "backend_order": orders, "speedup_python_over_rust": python_summary["median_ms"] / rust_summary["median_ms"]}
                rows.append(row)
                print(f"{label}: Python {python_summary['median_ms']:.6f} ms, Rust {rust_summary['median_ms']:.6f} ms, ratio {row['speedup_python_over_rust']:.3f}x", flush=True)
        finally:
            if was_enabled:
                gc.enable()
    report = {
        "schema_version": 1, "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "reference_revision": args.reference_revision or reference_git["commit"], "metadata": metadata,
        "methodology": {"scope": "matched numerical workloads of current implementations, not language-only speed or full optimization runs", "dtype": "float64", "cpu_threads": 1, "blocks": args.blocks, "samples_per_block": args.samples_per_block, "warmup_calls_per_block": args.warmups, "minimum_calibration_sample_ms": args.target_ms, "timed_work": "public API evaluation, validation, allocation, output consumption/deallocation", "excluded": ["process startup", "imports", "input generation", "JSON I/O", "correctness comparisons", "compilation"], "statistics": "per-call warmed median and linear-interpolated 25th/75th percentiles; raw block timings and repetitions retained", "backend_differences": ["Python uses NumPy/SciPy BLAS; Rust uses nalgebra scalar/native kernels", "Python GP uses generic dense solves; Rust exploits triangular solves", "Python EI uses np.vectorize(math.erf); Rust evaluates statrs erfc", "Python hypervolume uses aligned cells after Pareto sorting; Rust uses pairwise Pareto filtering and recursive sweep"]},
        "specification": SPECIFICATION, "workloads": rows,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
