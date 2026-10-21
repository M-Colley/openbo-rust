"""Compare complete CPU BO workflows, reporting time AND optimization quality.

Requires a separate official Python OpenBO checkout/install. Imports, process
startup, compilation and shared fixture preparation are excluded. Optimizer
construction, source loading/fitting, ask/objective/tell and (for NAF) tiny PPO
training are included. This is a small implementation study, not a language-only
speed test or evidence that distinct search/learning algorithms are equivalent.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Resolve the stdlib-only sibling explicitly, also under Python -I -S.
_helper_spec = importlib.util.spec_from_file_location("_benchmark_cpu_helpers", ROOT / "tools/benchmark_python_rust.py")
assert _helper_spec is not None and _helper_spec.loader is not None
_helpers = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helpers)
Generator, affinity = _helpers.Generator, _helpers.affinity
command_text, git_metadata = _helpers.command_text, _helpers.git_metadata
hardware, sha256_file = _helpers.hardware, _helpers.sha256_file
WORKLOADS = {
    "bo_branin": ("BO / normalized Branin", "normalized_branin", 2),
    "bo_hartmann6": ("BO / Hartmann-6", "hartmann6", 6),
    "mobo_oka2": ("MOBO / independent IEIPV / OKA2", "oka2", 3),
    "taf_branin": ("TAF-R / normalized Branin", "normalized_branin", 2),
    "naf_train_branin": ("Tiny NAF PPO training + BO / normalized Branin", "normalized_branin", 2),
}
SOURCE_TASKS = (
    {"translation": [-0.04, 0.03], "scaling": 0.95},
    {"translation": [-0.02, -0.04], "scaling": 1.0},
    {"translation": [0.04, -0.03], "scaling": 1.05},
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-checkout", type=Path, help="official source checkout/archive containing src/openbo; default installed package")
    parser.add_argument("--reference-revision", help="declared exact upstream revision, required for an archive without Git")
    parser.add_argument("--build", action="store_true", help="build the release Rust example before measurement")
    parser.add_argument("--binary", type=Path, help="release benchmark_workflows executable")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/workflows-local.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--workloads", choices=list(WORKLOADS), nargs="+", default=list(WORKLOADS))
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--optimizer-maxiter", type=int, default=50)
    parser.add_argument("--optimizer-restarts", type=int, default=2)
    parser.add_argument("--cpu", type=int, help="pin both backends to this logical CPU; default first allowed where supported")
    parser.add_argument("--timeout", type=float, default=180, help="maximum seconds per backend/seed/workflow")
    # Worker mode is an implementation detail. It excludes startup/imports using
    # an internal clock rather than subtracting an estimated process overhead.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=list(WORKLOADS), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 2 <= args.budget <= 100 or not 0 < args.n_init < args.budget:
        parser.error("require 2 <= budget <= 100 and 0 < n-init < budget")
    if min(args.candidates, args.optimizer_maxiter, args.optimizer_restarts) < 1 or args.timeout <= 0:
        parser.error("counts and timeout must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("seeds must be distinct unsigned 32-bit integers")
    if len(set(args.workloads)) != len(args.workloads):
        parser.error("workloads must be distinct")
    if args.worker and (args.input is None or args.workload is None or args.seed is None):
        parser.error("worker requires input, workload and seed")
    return args


def dependencies(checkout: Path | None):
    # These limits must precede NumPy/Torch imports in BOTH parent and workers.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if checkout is not None:
        source = checkout.resolve() / "src"
        if not (source / "openbo/__init__.py").is_file():
            raise ValueError(f"Python checkout lacks src/openbo: {checkout}")
        sys.path.insert(0, str(source))
    import numpy as np
    import openbo
    import scipy
    import threadpoolctl
    import torch
    from openbo.benchmarks.hypervolume import hypervolume
    from openbo.optimizers.bo_scratch import ScratchConfig, ScratchSequentialOptimizer
    from openbo.optimizers.bo_taf import TAFConfig, TAFSequentialOptimizer
    from openbo.optimizers.mobo import MOBOConfig, MOBOSequentialOptimizer
    from openbo.optimizers.naf import (
        BRANIN_GP_HYPERPARAMETERS,
        NAFConfig,
        NAFSequentialOptimizer,
        train_naf,
    )
    from openbo.test_functions.metabo_families import BraninTask, normalized_branin
    from openbo.test_functions.multi_objective import get_mo_function_spec, oka2
    from openbo.test_functions.synthetic import HARTMANN6_MAXIMUM, hartmann6
    reference = Path(openbo.__file__).resolve()
    if checkout is not None and not reference.is_relative_to(checkout.resolve() / "src"):
        raise RuntimeError(f"wrong Python reference imported: {reference}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Materialize Torch's lazy optimizer imports before starting the run clock.
    parameter = torch.nn.Parameter(torch.zeros(1))
    warm_optimizer = torch.optim.Adam([parameter])
    parameter.sum().backward()
    warm_optimizer.step()
    return {"np": np, "openbo": openbo, "scipy": scipy, "threadpoolctl": threadpoolctl, "torch": torch,
        "hypervolume": hypervolume, "ScratchConfig": ScratchConfig, "ScratchSequentialOptimizer": ScratchSequentialOptimizer,
        "TAFConfig": TAFConfig, "TAFSequentialOptimizer": TAFSequentialOptimizer,
        "MOBOConfig": MOBOConfig, "MOBOSequentialOptimizer": MOBOSequentialOptimizer,
        "BRANIN_GP_HYPERPARAMETERS": BRANIN_GP_HYPERPARAMETERS, "NAFConfig": NAFConfig,
        "NAFSequentialOptimizer": NAFSequentialOptimizer, "train_naf": train_naf,
        "BraninTask": BraninTask, "normalized_branin": normalized_branin,
        "get_mo_function_spec": get_mo_function_spec, "oka2": oka2,
        "HARTMANN6_MAXIMUM": HARTMANN6_MAXIMUM, "hartmann6": hartmann6}


@contextlib.contextmanager
def study_affinity(cpu):
    if sys.platform == "darwin":
        if cpu is not None:
            raise ValueError("macOS does not expose hard process CPU affinity; omit --cpu")
        yield {"enforced": False, "reason": "macOS has no supported hard process affinity API; scheduler placement uncontrolled"}
    else:
        with affinity(cpu) as actual:
            yield {"enforced": True, **actual}


def objective_map(dep):
    return {"normalized_branin": dep["normalized_branin"], "hartmann6": dep["hartmann6"], "oka2": dep["oka2"]}


def worker(args, dep):
    np = dep["np"]
    spec = json.loads(args.input.read_text(encoding="utf-8"))
    selected = next(run for run in spec["runs"] if run["seed"] == args.seed)
    _, function, dim = WORKLOADS[args.workload]
    objective = objective_map(dep)[function]
    initial = np.asarray(selected[f"initial_{dim}d"], dtype=np.float64)
    bounds = [(0.0, 1.0)] * dim
    common = {"bounds": bounds, "n_init": spec["n_init"], "n_candidates": spec["n_candidates"], "n_starts": 1, "seed": args.seed}
    details = {"shared_initial_design": args.workload not in ("taf_branin", "naf_train_branin"),
               "optimize_hyperparameters": True, "optimizer_maxiter": spec["optimizer_maxiter"],
               "optimizer_restarts": spec["optimizer_restarts"], "bounds": bounds}
    observations = []
    training_seconds, training = 0.0, None
    source_names = []
    with dep["threadpoolctl"].threadpool_limits(limits=1):
        threadpools = dep["threadpoolctl"].threadpool_info()
        if any(pool["num_threads"] != 1 for pool in threadpools):
            raise RuntimeError("reference worker did not enforce one thread")
        started = time.perf_counter_ns()
        if args.workload == "naf_train_branin":
            config = dep["NAFConfig"](dim=2, budget=spec["budget"], hidden=(8, 8), points_per_dim=9,
                n_local_maxima=2, n_local=32, n_iterations=2, steps_per_iteration=2 * spec["budget"],
                epochs=2, minibatch_size=8, seed=args.seed)
            tasks = [dep["BraninTask"](tuple(task["translation"]), task["scaling"]) for task in spec["training_tasks"]]
            result = dep["train_naf"](config, dep["BRANIN_GP_HYPERPARAMETERS"], task_pool=tasks)
            training_seconds = (time.perf_counter_ns() - started) / 1e9
            training = {"iterations": 2, "episodes_per_iteration": 2, "episode_budget": spec["budget"],
                "objective_evaluations": 4 * spec["budget"], "mean_return": result.mean_return,
                "mean_final_log_regret": result.mean_final_log_regret, "policy": "final_af"}
            result.af.eval()
            optimizer = dep["NAFSequentialOptimizer"](result.af, dep["BRANIN_GP_HYPERPARAMETERS"], config.budget, config.adaptive_set())
            details.update({"optimize_hyperparameters": False, "naf_config": config.__dict__, "training_tasks": spec["training_tasks"], "gp": dep["BRANIN_GP_HYPERPARAMETERS"].to_dict()})
            n_init = 0
        elif args.workload == "mobo_oka2":
            problem = dep["get_mo_function_spec"]("oka2")
            config = dep["MOBOConfig"](**common, v_ref=problem.v_ref, w_ref=problem.w_ref,
                model="independent", acquisition="ieipv", optimizer_maxiter=8)
            optimizer = dep["MOBOSequentialOptimizer"](config)
            optimizer.gp.optimizer_maxiter = spec["optimizer_maxiter"]
            optimizer.gp.optimizer_restarts = spec["optimizer_restarts"]
            details.update({"model": "independent", "acquisition": "ieipv", "acquisition_optimizer": "SciPy L-BFGS-B, one start, maxiter=8",
                "v_ref": list(problem.v_ref), "w_ref": list(problem.w_ref), "optimize_noise": True,
                "hyperfit_log_lengthscale_bounds": [math.log(1e-2), math.log(10)], "hyperfit_log_noise_bounds": [math.log(1e-8), 0.0]})
            n_init = spec["n_init"]
        else:
            if args.workload == "taf_branin":
                config = dep["TAFConfig"](**{**common, "n_init": 0}, n_iter=spec["budget"],
                    taf_run_dir=spec["source_directory"], taf_weight_mode="taf_r", search_strategy="grid", track_acquisition=False)
                with warnings.catch_warnings(record=True) as source_warnings:
                    warnings.simplefilter("always")
                    optimizer = dep["TAFSequentialOptimizer"](config)
                source_names = [source.name for source in optimizer.source_surrogates]
                if len(source_names) != 3 or source_warnings:
                    raise AssertionError(f"expected exactly 3 sources without loading warnings, got {source_names}, {source_warnings}")
                gp = optimizer.target_gp
                n_init = 0
            else:
                config = dep["ScratchConfig"](**common, search_strategy="grid")
                optimizer = dep["ScratchSequentialOptimizer"](config)
                gp = optimizer.gp
                n_init = spec["n_init"]
            gp.optimizer_maxiter = spec["optimizer_maxiter"]
            gp.optimizer_restarts = spec["optimizer_restarts"]
            details.update({"acquisition_optimizer": "SciPy Owen-scrambled Sobol pool, argmax",
                "hyperfit_log_lengthscale_bounds": [-6.0, 4.0], "hyperfit_log_variance_bounds": [-8.0, 6.0], "optimize_noise": False})
        for index in range(spec["budget"]):
            step = time.perf_counter_ns()
            x = initial[index:index + 1] if index < n_init else optimizer.suggest()
            y = np.asarray(objective(x), dtype=np.float64)
            optimizer.observe(x, y)
            elapsed = time.perf_counter_ns()
            observations.append({"evaluation": index + 1, "x": x.reshape(-1).tolist(), "y": y.reshape(-1).tolist(),
                "cumulative_seconds": (elapsed - started) / 1e9, "evaluation_seconds": (elapsed - step) / 1e9})
        total_seconds = (time.perf_counter_ns() - started) / 1e9
    print(json.dumps({"backend": "python", "workload": args.workload, "seed": args.seed, "function": function,
        "total_seconds": total_seconds, "training_seconds": training_seconds, "training": training,
        "target_evaluations": len(observations), "observations": observations, "configuration": details,
        "numeric_backend": "NumPy/SciPy float64; Torch PPO/network float32", "source_names": source_names, "threadpools": threadpools}, allow_nan=False))


def build_specification(args, dep, directory):
    np = dep["np"]
    source_directory = directory / "sources"
    for subdirectory in ("gp_states", "trajectories"):
        (source_directory / subdirectory).mkdir(parents=True, exist_ok=True)
    sources = []
    for index, task in enumerate(SOURCE_TASKS):
        x = np.asarray(Generator(20261021 + index).matrix(12, 2))
        y = task["scaling"] * dep["normalized_branin"](x - task["translation"])
        state = {"kernel_type": "rbf", "lengthscale": [0.235, 0.578], "variance": 2.0,
            "noise": 1e-10, "standardize_targets": False, "optimize_hyperparameters": False,
            "mean_constant": 0.0, "y_mean": 0.0, "y_std": 1.0}
        name = f"shared_source_{index}"
        (source_directory / "gp_states" / f"{name}.json").write_text(json.dumps({"gp_state": state}), encoding="utf-8")
        (source_directory / "trajectories" / f"{name}.json").write_text(json.dumps({"x_values": x.tolist(), "y_values": y.tolist()}), encoding="utf-8")
        sources.append({"name": name, "task": task, "x": x.tolist(), "y": y.tolist(), "gp_state": state})
    return {"budget": args.budget, "n_init": args.n_init, "n_candidates": args.candidates,
        "optimizer_maxiter": args.optimizer_maxiter, "optimizer_restarts": args.optimizer_restarts,
        "source_directory": str(source_directory.resolve()), "source_fixture": sources, "training_tasks": SOURCE_TASKS,
        "runs": [{"seed": seed, **{f"initial_{dim}d": Generator(123456 + seed * 1009 + dim).matrix(args.n_init, dim) for dim in (2, 3, 6)}} for seed in args.seeds],
        "objective_checks": [{"function": name, "x": Generator(987654 + dim).matrix(12, dim)} for name, dim in [("normalized_branin", 2), ("hartmann6", 6), ("oka2", 3)]]}


def validate_run(run, spec, dep):
    np = dep["np"]
    rows = run["observations"]
    budget = spec["budget"]
    if len(rows) != budget or run["target_evaluations"] != budget or not math.isfinite(run["total_seconds"]) or run["total_seconds"] <= 0:
        raise AssertionError("invalid run length/time")
    _, function, dim = WORKLOADS[run["workload"]]
    x, y = np.asarray([r["x"] for r in rows]), np.asarray([r["y"] for r in rows])
    expected_objectives = 2 if function == "oka2" else 1
    if x.shape != (budget, dim) or y.shape != (budget, expected_objectives) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x < 0) or np.any(x > 1):
        raise AssertionError("invalid observation shape, bounds or finiteness")
    expected = np.asarray(objective_map(dep)[function](x)).reshape(y.shape)
    np.testing.assert_allclose(y, expected, rtol=1e-10, atol=1e-10)
    cumulative = [r["cumulative_seconds"] for r in rows]
    if any(not math.isfinite(t) or t <= 0 for t in cumulative) or any(b < a for a, b in itertools.pairwise(cumulative)) or cumulative[-1] > run["total_seconds"]:
        raise AssertionError("invalid cumulative history")
    if any(r["evaluation"] != i + 1 or not math.isfinite(r["evaluation_seconds"]) or r["evaluation_seconds"] < 0 for i, r in enumerate(rows)):
        raise AssertionError("invalid evaluation history")
    if run["workload"] not in ("taf_branin", "naf_train_branin"):
        seed_spec = next(item for item in spec["runs"] if item["seed"] == run["seed"])
        np.testing.assert_array_equal(x[:spec["n_init"]], seed_spec[f"initial_{dim}d"])
    if function == "oka2":
        problem = dep["get_mo_function_spec"]("oka2")
        quality = [dep["hypervolume"](y[:i], problem.v_ref) / problem.true_hypervolume for i in range(1, budget + 1)]
        metric = "hypervolume_fraction_of_analytic_optimum"
        if any(v < -1e-10 or v > 1 + 1e-10 for v in quality):
            raise AssertionError("OKA2 hypervolume outside analytic bounds")
    else:
        optimum = (-0.39788735772973816 + 54.44) / 51.44 if function == "normalized_branin" else dep["HARTMANN6_MAXIMUM"]
        quality = (optimum - np.maximum.accumulate(y[:, 0])).tolist()
        metric = "simple_regret"
        if min(quality) < -1e-9:
            raise AssertionError("observations exceed known optimum")
    run["quality"] = {"metric": metric, "direction": "maximize" if function == "oka2" else "minimize", "history": quality, "final": quality[-1]}
    if run["workload"] == "taf_branin" and sorted(run["source_names"]) != [f"shared_source_{i}" for i in range(3)]:
        raise AssertionError("transfer source names/count did not match shared fixture")
    if run["workload"] == "naf_train_branin":
        training = run["training"]
        if not 0 < run["training_seconds"] < run["total_seconds"] or training["iterations"] != 2 or training["episodes_per_iteration"] != 2 or training["episode_budget"] != budget or training["objective_evaluations"] != 4 * budget or training["policy"] != "final_af":
            raise AssertionError("invalid/missing actual neural training")
        for name in ("mean_return", "mean_final_log_regret"):
            if len(training[name]) != 2 or not all(math.isfinite(v) for v in training[name]):
                raise AssertionError("invalid neural training history")
    run["validation"] = {"passed": True, "maximum_objective_error": float(np.max(np.abs(y - expected))), "shared_initial_design": run["workload"] not in ("taf_branin", "naf_train_branin")}


def statistics(values, np):
    return {"median": float(np.median(values)), "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)), "values": values}


def main():
    args = parse_args()
    dep = dependencies(args.python_checkout)
    if args.worker:
        worker(args, dep)
        return
    binary = args.binary or Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "release/examples" / ("benchmark_workflows.exe" if os.name == "nt" else "benchmark_workflows")
    binary = binary.resolve()
    if args.build:
        subprocess.run(["cargo", "build", "--locked", "--release", "--example", "benchmark_workflows", "--manifest-path", str(ROOT / "Cargo.toml")], check=True)
    if not binary.is_file():
        raise ValueError(f"release benchmark binary missing: {binary}; use --build or --binary")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture_directory = ROOT / "results/benchmarks/workflow-fixtures" / args.output.stem
    spec = build_specification(args, dep, fixture_directory)
    specification_path = args.output.with_name(args.output.stem + "-inputs.json")
    specification_path.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    reference_file = Path(dep["openbo"].__file__).resolve()
    reference_root = args.python_checkout.resolve() if args.python_checkout else reference_file.parent
    reference_git = git_metadata(reference_root)
    if not args.reference_revision and not reference_git["commit"]:
        raise ValueError("--reference-revision required when reference has no own Git repository")
    if args.reference_revision and reference_git["commit"] and args.reference_revision != reference_git["commit"]:
        raise ValueError("declared reference revision differs from reference Git HEAD")
    module_hashes = {str(Path(module.__file__).resolve().relative_to(reference_file.parent)): sha256_file(Path(module.__file__)) for name, module in list(sys.modules.items()) if name.startswith("openbo") and getattr(module, "__file__", None) and Path(module.__file__).is_file()}
    rust_files = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", Path(__file__), ROOT / "tools/benchmark_python_rust.py", ROOT / "examples/benchmark_workflows.rs", *sorted((ROOT / "src").rglob("*.rs"))]
    metadata = {"hardware": hardware(), "python": sys.version, "numpy": dep["np"].__version__, "scipy": dep["scipy"].__version__,
        "torch": dep["torch"].__version__, "threadpoolctl": dep["threadpoolctl"].__version__, "rustc": command_text(["rustc", "--version", "--verbose"]),
        "python_openbo": str(reference_file), "python_git": reference_git, "python_files_sha256": module_hashes,
        "rust_git": git_metadata(ROOT), "rust_binary": str(binary), "rust_binary_sha256": sha256_file(binary),
        "rust_files_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in rust_files},
        "specification_sha256": sha256_file(specification_path)}
    metadata["python_packages"] = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "botorch", "gpytorch", "threadpoolctl")}
    metadata["command"] = [sys.executable, *sys.argv]
    runs, order = [], []
    with study_affinity(args.cpu) as actual_affinity:
        metadata["affinity"] = actual_affinity
        # Check shared objective definitions before starting any workflow timers.
        native = json.loads(subprocess.run([str(binary), "--input", str(specification_path), "--verify"], capture_output=True, text=True, check=True, timeout=args.timeout).stdout)
        for check, actual in zip(spec["objective_checks"], native["objective_checks"], strict=True):
            expected = dep["np"].asarray(objective_map(dep)[check["function"]](dep["np"].asarray(check["x"]))).reshape(len(check["x"]), -1)
            dep["np"].testing.assert_allclose(actual["y"], expected, rtol=1e-10, atol=1e-10)
        training_x = dep["np"].asarray(next(check for check in spec["objective_checks"] if check["function"] == "normalized_branin")["x"])
        for task, actual in zip(spec["training_tasks"], native["training_task_checks"], strict=True):
            expected = dep["BraninTask"](tuple(task["translation"]), task["scaling"])(training_x)
            dep["np"].testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        print("Shared objective checks passed before workflow timing.", flush=True)
        for number, workload in enumerate(args.workloads):
            for index, seed in enumerate(args.seeds):
                backends = ["python", "rust"] if (number + index) % 2 == 0 else ["rust", "python"]
                order.append({"workload": workload, "seed": seed, "backends": backends})
                for backend in backends:
                    if backend == "rust":
                        command = [str(binary), "--input", str(specification_path), "--workload", workload, "--seed", str(seed)]
                    else:
                        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--input", str(specification_path), "--workload", workload, "--seed", str(seed)]
                        if args.python_checkout:
                            command += ["--python-checkout", str(args.python_checkout.resolve())]
                    response = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
                    if response.returncode:
                        raise RuntimeError(f"{backend}/{workload}/seed{seed} failed: {response.stderr}\n{response.stdout}")
                    run = json.loads(response.stdout)
                    validate_run(run, spec, dep)
                    runs.append(run)
                    print(f"{workload}, seed={seed}, {backend}: {run['total_seconds']:.4f}s; {run['quality']['metric']}={run['quality']['final']:.6g}", flush=True)
                    # Preserve partial validated results if a later workflow fails.
                    partial = args.output.with_name(args.output.stem + "-partial.json")
                    partial.write_text(json.dumps({"runs": runs, "metadata": metadata}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summaries = []
    for workload in args.workloads:
        row = {"id": workload, "label": WORKLOADS[workload][0], "target_evaluations": args.budget, "seeds": args.seeds}
        for backend in ("python", "rust"):
            selected = [run for run in runs if run["workload"] == workload and run["backend"] == backend]
            row[backend] = {"total_seconds": statistics([run["total_seconds"] for run in selected], dep["np"]),
                "training_seconds": statistics([run["training_seconds"] for run in selected], dep["np"]),
                "final_quality": statistics([run["quality"]["final"] for run in selected], dep["np"]),
                "quality_metric": selected[0]["quality"]["metric"], "quality_direction": selected[0]["quality"]["direction"],
                "quality_curve": [statistics([run["quality"]["history"][i] for run in selected], dep["np"]) for i in range(args.budget)],
                "cumulative_time_curve_seconds": [statistics([run["observations"][i]["cumulative_seconds"] for run in selected], dep["np"]) for i in range(args.budget)]}
        row["ratio_of_median_total_time_python_over_rust"] = row["python"]["total_seconds"]["median"] / row["rust"]["total_seconds"]["median"]
        summaries.append(row)
    report = {"schema_version": 1, "study": f"complete CPU ask/tell workflows, small {len(args.seeds)}-seed implementation study", "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "reference_revision": args.reference_revision or reference_git["commit"], "metadata": metadata, "specification": spec,
        "methodology": {"seeds": args.seeds, "runs_per_backend_per_workflow": len(args.seeds), "timeout_per_run_seconds": args.timeout, "cpu_threads": 1,
            "timing": "internal wall clock; optimizer construction + source loading/fits + ask/objective/tell; NAF additionally includes tiny PPO training",
            "excluded": ["process startup", "imports", "Torch lazy optimizer warmup", "shared fixture generation/writing", "build", "JSON serialization", "independent objective/quality verification"],
            "quality": "Every returned point is reevaluated independently outside timing; scalar simple regret, OKA2 hypervolume / analytic optimum; medians and IQRs across seeds",
            "same_inputs": f"identical target objectives/bounds/budget; identical serialized first {args.n_init} points in scalar BO and MOBO; TAF-R starts source-only using identical 3 x 12 fixtures; NAF uses identical 3 training tasks and settings",
            "differences": ["Numerical implementations and hyperparameter optimization algorithms differ; time ratios are not pure Rust-versus-Python language effects",
                "Python acquisition pools use Owen-scrambled Sobol; native pools use shifted Sobol in dimensions up to 4 and shifted Halton for Hartmann-6; NAF local clouds also differ in Sobol scrambling; suggestions and quality are expected to differ",
                "Scalar/TAF GP hyperparameters use bounded ML; MO independent GP uses MAP with matching iteration/restart caps but backend-specific solver/gradients",
                "Scalar ML fit: Python SciPy L-BFGS-B uses finite-difference gradients and accepts successful fits; native projected BFGS uses analytic kernel gradients and retains the best finite parameters; equal iteration caps do not imply equal convergence or computation",
                "Python scalar/TAF optimizers keep the mutated GP and warm-start future fits; Rust scalar/TAF fits fresh GPs from config each suggestion",
                "MO acquisition refinement: Python SciPy L-BFGS-B one start/maxiter8; Rust native coordinate refinement one start; trajectories are not algorithmically identical",
                "Python MO observe also maintains its public hypervolume history inside timing; Rust public observe defers result summaries",
                "NAF uses independent native initialization/action RNG/Adam implementations, Python Torch float32 networks versus native Rust float64; report final policy after only 2 PPO iterations, not a converged research policy",
                "Shared source trajectories/fixed hyperparameter artifacts are generated before timing; source loading and first posterior fitting during construction are included; offline source hyperparameter training is not measured",
                f"Only {len(args.seeds)} seeds and small {args.budget}-evaluation budgets; quality/time results are descriptive for these workloads, not a paper-scale conclusion",
                "macOS hard CPU affinity unavailable and explicitly recorded; all workers remain sequential with numerical thread limits 1"]},
        "backend_order": order, "runs": runs, "workloads": summaries}
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    args.output.with_name(args.output.stem + "-partial.json").unlink(missing_ok=True)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
