"""Render the measured Python/Rust benchmark report with optional Matplotlib.

Example::

    python tools/plot_benchmark.py --report docs/benchmarks/windows-x64.json

Plotting is optional and is not needed to build or run OpenBO Rust.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def milliseconds(value: float) -> str:
    return f"{value:.3f}" if value < 1 else f"{value:.2f}"


def workload_label(row: dict) -> str:
    parameters = row["parameters"]
    if row["id"] == "branin_batch":
        return f"Branin objective batch\n{parameters['points']:,} points · {parameters['input_dim']}D"
    if row["id"] in ("gp_small", "gp_medium"):
        return (f"GP fit + posterior\n{parameters['training_points']:,} train / "
                f"{parameters['query_points']:,} query · {parameters['input_dim']}D")
    if row["id"] == "ei_batch":
        return f"Expected improvement\n{parameters['points']:,} candidates"
    if row["id"] == "hypervolume_2d":
        front = parameters["nondominated_points"]
        return (f"Exact {parameters['objectives']}D hypervolume\n{front:,} front + "
                f"{parameters['points'] - front:,} dominated")
    if row["id"] == "bo_candidate_scoring":
        return (f"BO candidate scoring\nGP {parameters['training_points']:,} / "
                f"{parameters['candidate_points']:,} · {parameters['input_dim']}D + EI")
    return row["label"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="output stem, without a suffix")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report["workloads"]
    if not rows or not all(row["validation"]["passed"] for row in rows):
        parser.error("Every benchmark workload must pass numerical validation.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "svg.hashsalt": "openbo-python-rust-benchmark",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    colors = {"python": "#2864A2", "rust": "#C5532C"}
    figure = plt.figure(figsize=(14, 7.5), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=[3.4, 2], wspace=0.09)
    axis = figure.add_subplot(grid[0])
    table = figure.add_subplot(grid[1], sharey=axis)
    all_values = []
    for i, row in enumerate(rows):
        for backend, offset in [("python", -0.13), ("rust", 0.13)]:
            result = row[backend]
            median = result["median_ms"]
            low, high = result["p25_ms"], result["p75_ms"]
            if not all(math.isfinite(value) for value in [low, median, high]) or not 0 < low <= median <= high:
                parser.error(f"Invalid positive timing interval for {row['id']}/{backend}.")
            all_values.extend([low, high])
            y = i + offset
            axis.plot([low, high], [y, y], color=colors[backend], linewidth=3)
            axis.scatter([median], [y], color=colors[backend], s=52, zorder=4,
                         label=backend.title() if i == 0 else None)
        ratio = row["python"]["median_ms"] / row["rust"]["median_ms"]
        result_text = f"{ratio:.2f}× faster" if ratio >= 1 else f"{1 / ratio:.2f}× slower"
        table.text(0.03, i, milliseconds(row["python"]["median_ms"]),
                   color=colors["python"], va="center", fontsize=11)
        table.text(0.95, i, milliseconds(row["rust"]["median_ms"]),
                   color=colors["rust"], va="center", fontsize=11)
        table.text(1.85, i, result_text, va="center", fontsize=11)
        if i % 2 == 0:
            axis.axhspan(i - 0.46, i + 0.46, color="#F2F5F8", zorder=0)
            table.axhspan(i - 0.46, i + 0.46, color="#F2F5F8", zorder=0)
    axis.set_xscale("log")
    axis.set_xlim(min(all_values) / 2.2, max(all_values) * 2.2)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_xlabel("Milliseconds per call · logarithmic scale · lower is faster", labelpad=12)
    axis.set_yticks(range(len(rows)), [workload_label(row) for row in rows])
    axis.tick_params(axis="y", length=0, pad=15)
    axis.grid(axis="x", color="#D8DFE6", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["left"].set_visible(False)
    axis.set_ylim(len(rows) - 0.5, -0.65)
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False, ncol=2)
    table.set_xlim(0, 3.4)
    table.axis("off")
    for x, heading in [(0.03, "Python ms"), (0.95, "Rust ms"), (1.85, "Rust result")]:
        table.text(x, -0.63, heading, fontsize=10, color="#516173", weight="bold", va="bottom")
    figure.text(0.04, 0.96, "OpenBO: Python vs Rust", fontsize=23, weight="bold", color="#183047")
    threads = report["methodology"]["cpu_threads"]
    figure.text(0.04, 0.915,
                f"Matched numerical workloads · warmed, in-process calls · float64 · {threads} CPU thread",
                fontsize=12, color="#516173")
    hardware = report["metadata"]["hardware"]
    cpu = hardware.get("cpu", {})
    processor = cpu.get("Name", hardware.get("processor", hardware["machine"])) if isinstance(cpu, dict) else hardware["machine"]
    python_version = report["metadata"]["python"].split()[0]
    rust_version = report["metadata"]["rustc"].splitlines()[0]
    figure.text(0.04, 0.88,
                f"{processor} · {hardware['platform']} · Python {python_version} · {rust_version} · {report['created_utc'][:10]}",
                fontsize=9, color="#516173")
    figure.text(0.04, 0.065,
                "Dots: medians. Lines: interquartile ranges. Raw samples, workload parameters, host details and source hashes accompany the chart.",
                fontsize=9.5, color="#516173")
    figure.text(0.04, 0.038,
                "Current implementations use different numerical routines. These are component timings, not full optimizer or solution-quality comparisons.",
                fontsize=9.5, color="#516173")
    figure.subplots_adjust(left=0.265, right=0.99, top=0.81, bottom=0.17)
    output = args.output or args.report.with_name("comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output.with_suffix(".svg")
    figure.savefig(svg_path, metadata={"Date": None})
    # Matplotlib adds trailing spaces to path data; keep generated Git diffs clean.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    figure.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(figure)
    print(f"Saved {output.with_suffix('.svg')} and {output.with_suffix('.png')}")
    print("\n| Workload | Python (ms) | Rust (ms) | Rust result |")
    print("| --- | ---: | ---: | ---: |")
    for row in rows:
        ratio = row["python"]["median_ms"] / row["rust"]["median_ms"]
        result_text = f"{ratio:.2f}× faster" if ratio >= 1 else f"{1 / ratio:.2f}× slower"
        print(f"| {row['label']} | {milliseconds(row['python']['median_ms'])} | "
              f"{milliseconds(row['rust']['median_ms'])} | {result_text} |")


if __name__ == "__main__":
    main()
