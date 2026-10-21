"""Plot complete-workflow study reports without pooling distinct machines.

    python tools/plot_workflows.py --reports docs/benchmarks/workflows/*.json

Matplotlib and NumPy are optional development dependencies, not Rust requirements.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import textwrap
from pathlib import Path

LABELS = {
    "bo_branin": "BO / Branin",
    "bo_hartmann6": "BO / Hartmann-6",
    "mobo_oka2": "MOBO / OKA2",
    "taf_branin": "TAF-R / Branin",
    "naf_train_branin": "NAF training + BO",
}
COLORS = {"python": "#2864A2", "rust": "#C5532C"}


def host_label(report: dict, path: Path) -> str:
    hardware = report["metadata"]["hardware"]
    os_name = hardware["platform"].split("-")[0]
    cpu = hardware.get("cpu", {})
    processor = cpu.get("Name", "") if isinstance(cpu, dict) else ""
    processor = processor or hardware.get("processor") or hardware["machine"]
    version = report["metadata"]["python"].split()[0]
    return f"{path.stem} · {os_name} {hardware['machine']}\n{processor} · Python {version}"


def save(figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = stem.with_suffix(".svg")
    figure.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    figure.savefig(stem.with_suffix(".png"), dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks/workflows"))
    args = parser.parse_args()
    reports = []
    paths = [Path(match) for pattern in args.reports for match in (sorted(glob.glob(str(pattern))) or [str(pattern)])]
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if "runs" not in report or "workloads" not in report:
            continue
        if not report["runs"] or not all(run["validation"]["passed"] for run in report["runs"]):
            parser.error(f"Every run must pass validation: {path}")
        reports.append((path, report))
    if not reports:
        parser.error("No complete, validated workflow reports supplied.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "svg.hashsalt": "openbo-workflow-study",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    columns = min(2, len(reports))
    rows = math.ceil(len(reports) / columns)
    figure_width = 11 if columns == 1 else 7.3 * columns
    figure, axes = plt.subplots(rows, columns, figsize=(figure_width, 4.2 * rows + 1), squeeze=False)
    tables = ["# Complete workflow results", "", "Each environment is measured independently. Ratios compare backends within that environment; they do not pool platforms.", ""]
    for index, (path, report) in enumerate(reports):
        axis = axes.flat[index]
        workloads = report["workloads"]
        for i, workload in enumerate(workloads):
            for backend, offset in [("python", -0.13), ("rust", 0.13)]:
                result = workload[backend]["total_seconds"]
                median, low, high = result["median"], result["p25"], result["p75"]
                if not 0 < low <= median <= high:
                    parser.error(f"Invalid timing interval: {path}/{workload['id']}/{backend}")
                axis.plot([low, high], [i + offset] * 2, color=COLORS[backend], linewidth=3)
                axis.scatter([median], [i + offset], color=COLORS[backend], s=45,
                             label=backend.title() if i == 0 else None, zorder=4)
            if i % 2 == 0:
                axis.axhspan(i - 0.45, i + 0.45, color="#F2F5F8", zorder=0)
        axis.set_xscale("log")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_xlabel("Seconds per complete run · log scale · lower is faster")
        axis.set_yticks(range(len(workloads)), [LABELS.get(w["id"], w["label"]) for w in workloads])
        axis.set_ylim(len(workloads) - 0.5, -0.5)
        axis.grid(axis="x", color="#D8DFE6", zorder=0)
        axis.set_axisbelow(True)
        axis.set_title(host_label(report, path), fontsize=10, loc="left", pad=15)
        axis.legend(frameon=False, ncol=2, fontsize=9)
        tables += [f"## {path.stem}", "", host_label(report, path).replace("\n", "; "), "",
                   "| Workflow | Python seconds | Rust seconds | Python/Rust time | Final quality Python / Rust |",
                   "| --- | ---: | ---: | ---: | ---: |"]
        for workload in workloads:
            python, rust = workload["python"], workload["rust"]
            metric = "HV fraction ↑" if python["quality_direction"] == "maximize" else "regret ↓"
            ratio = python["total_seconds"]["median"] / rust["total_seconds"]["median"]
            tables.append(f"| {LABELS.get(workload['id'], workload['label'])} | {python['total_seconds']['median']:.4g} | "
                          f"{rust['total_seconds']['median']:.4g} | {ratio:.2f}× | "
                          f"{python['final_quality']['median']:.4g} / {rust['final_quality']['median']:.4g} ({metric}) |")
        tables += ["", f"[Raw report]({path.name}) · [Quality trajectories]({path.stem}-quality.svg)", ""]
        quality, quality_axes = plt.subplots(2, 3, figsize=(13.5, 8))
        for position, workload in enumerate(workloads):
            plot = quality_axes.flat[position]
            for backend in ("python", "rust"):
                curve = workload[backend]["quality_curve"]
                evaluations = list(range(1, len(curve) + 1))
                plot.plot(evaluations, [point["median"] for point in curve], color=COLORS[backend], label=backend.title(), linewidth=2)
                plot.fill_between(evaluations, [point["p25"] for point in curve], [point["p75"] for point in curve],
                                  color=COLORS[backend], alpha=0.14)
            maximize = workload["python"]["quality_direction"] == "maximize"
            plot.set_ylabel("Hypervolume / analytic optimum ↑" if maximize else "Simple regret ↓")
            plot.set_title(LABELS.get(workload["id"], workload["label"]), fontsize=11)
            plot.set_xlabel("Target objective evaluations")
            plot.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            plot.grid(alpha=0.2)
            plot.legend(frameon=False, fontsize=9)
        for unused in list(quality_axes.flat)[len(workloads):]:
            unused.axis("off")
            unused.text(0, 0.8, "Lines: medians across seeds\nBands: interquartile ranges\n\nLower regret is better.\nHigher hypervolume fraction is better.\n\nNAF includes tiny PPO training;\nquality curves show its held-out run.",
                        transform=unused.transAxes, va="top", fontsize=10, linespacing=1.7, color="#465B6B")
        quality.suptitle(f"Optimization quality · {host_label(report, path)}", fontsize=13, x=0.05, ha="left", y=0.98)
        quality.text(0.05, 0.02, "Different search, numerical and training backends produce different trajectories. Small budgets and five seeds are descriptive, not paper-scale evidence.",
                     fontsize=9, color="#465B6B")
        quality.tight_layout(rect=(0.02, 0.06, 1, 0.88))
        save(quality, args.output_dir / f"{path.stem}-quality")
        plt.close(quality)
    for unused in list(axes.flat)[len(reports):]:
        unused.axis("off")
    figure.suptitle("OpenBO workflows: time to finish the evaluation budget", fontsize=17, x=0.04, ha="left", y=0.98)
    footer = "Dots: median run times. Lines: interquartile ranges across seeds. Each machine is a separate experiment; see quality curves and raw data alongside these timings."
    figure.text(0.04, 0.025, textwrap.fill(footer, width=130 if columns == 1 else 200),
                fontsize=9, color="#465B6B")
    figure.tight_layout(rect=(0.01, 0.07, 1, 0.93), h_pad=3, w_pad=3)
    save(figure, args.output_dir / "workflow-timing")
    plt.close(figure)
    (args.output_dir / "RESULTS.md").write_text("\n".join(tables), encoding="utf-8")
    print(f"Rendered {len(reports)} independent environments into {args.output_dir}")


if __name__ == "__main__":
    main()
