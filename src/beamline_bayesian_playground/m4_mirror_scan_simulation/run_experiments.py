"""Run the agent repeatedly for several iteration counts and plot RMSE vs the real scan."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from ax.api.client import Client

from beamline_bayesian_playground.m4_mirror_scan_simulation.compare_ax_scan import (
    DEFAULT_SCAN,
    load_scan,
    predict_scan,
)

ITERATION_COUNTS = (10, 20, 30, 60)
REPEATS = 10
SIGNAL = "S1Yfw"
METRIC = "beam_focus_evaluator"
OUTPUT_DIR = Path("rmse_mapping_runs")
AGENT = Path(__file__).with_name("agent.py")


def run_agents(output_dir: Path) -> None:
    for iterations in ITERATION_COUNTS:
        directory = output_dir / str(iterations)
        existing = len(list(directory.glob("*.json"))) if directory.exists() else 0
        for repeat in range(existing, REPEATS):
            print(f"Running {iterations} iterations, repeat {repeat + 1}/{REPEATS}", flush=True)
            subprocess.run(
                [sys.executable, str(AGENT), "--iterations", str(iterations),
                 "--output-dir", str(directory)],
                check=True,
            )


def rmse(snapshot: Path, scan) -> float:
    client = Client.load_from_json_file(str(snapshot))
    data = predict_scan(client, scan, SIGNAL, METRIC, "m4_ty", "m4_rx")
    return float(np.sqrt(np.mean(np.square(data["prediction"] - data["real"]))))


def plot(output_dir: Path) -> None:
    scan = load_scan(DEFAULT_SCAN, SIGNAL)
    figure = go.Figure()
    for iterations in ITERATION_COUNTS:
        values = [rmse(p, scan) for p in sorted((output_dir / str(iterations)).glob("*.json"))]
        print(iterations, np.round(values, 4).tolist())
        figure.add_trace(
            go.Scatter(x=[iterations] * len(values), y=values, mode="markers", name=str(iterations))
        )
    figure.update_layout(
        title="RMSE of Ax prediction vs real scan",
        xaxis={"title": "Iterations", "tickvals": list(ITERATION_COUNTS)},
        yaxis={"title": "RMSE"},
        showlegend=False,
    )
    figure.write_html(str(output_dir / "rmse_scatter.html"), include_plotlyjs=True)
    print(f"Wrote {output_dir / 'rmse_scatter.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not args.plot_only:
        run_agents(args.output_dir)
    plot(args.output_dir)
