"""Compare saved M4 agent runs, grouped by iteration count."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ax.api.client import Client
from ax.core.observation import ObservationFeatures
from plotly.subplots import make_subplots

from beamline_bayesian_playground.m4_mirror_scan_simulation.compare_ax_scan import (
    DEFAULT_SCAN,
    fit_model,
    load_scan,
    predict_scan,
)
from beamline_bayesian_playground.m4_mirror_scan_simulation.run_experiments import (
    METRIC,
    SIGNAL,
)
from beamline_bayesian_playground.m4_mirror_scan_simulation.simulation_devices import (
    GridInterpolator,
)

TY = "m4_ty"
RX = "m4_rx"


@dataclass(frozen=True)
class RunMetrics:
    snapshot: Path
    rmse: float
    predicted_best_error: float
    actual_best_error: float
    predicted_best_ty: float
    predicted_best_rx: float
    predicted_best_value: float
    predicted_best_actual: float
    actual_best_ty: float
    actual_best_rx: float
    actual_best_value: float
    actual_best_prediction: float


def calculate_metrics(
    snapshot: Path, scan: pd.DataFrame, interpolator: GridInterpolator
) -> RunMetrics:
    client = Client.load_from_json_file(str(snapshot))
    experiment = client._experiment  # noqa: SLF001
    adapter = fit_model(client)
    comparison = predict_scan(client, scan, SIGNAL, METRIC, TY, RX, adapter=adapter)
    rmse = float(np.sqrt(np.mean(np.square(comparison["prediction"] - comparison["real"]))))

    sampled: list[tuple[float, float]] = []
    for trial in experiment.trials.values():
        if not trial.status.is_completed:
            continue
        for arm in trial.arms:
            ty, rx = arm.parameters[TY], arm.parameters[RX]
            if not isinstance(ty, (int, float)) or not isinstance(rx, (int, float)):
                raise ValueError(f"{snapshot}: non-numeric sampled coordinates")
            sampled.append((float(ty), float(rx)))
    if not sampled:
        raise ValueError(f"{snapshot}: no completed trial locations")

    # Bilinear interpolation and boundary clamping are convex combinations of scan values.
    minima = scan.loc[scan[SIGNAL] == scan[SIGNAL].min()]
    true_optima = [
        {TY: float(ty), RX: float(rx)}
        for ty, rx in minima[["M4S:Ty", "M4S:Rx"]].itertuples(index=False, name=None)
        if experiment.search_space.check_membership(
            {TY: float(ty), RX: float(rx)}, raise_error=False
        )
    ]
    if not true_optima:
        raise ValueError(
            f"{snapshot}: no global scan minimum is inside the search bounds; "
            "cannot determine the bounded interpolated optimum"
        )
    true_best = true_optima[0]
    means, _ = adapter.predict(
        [
            ObservationFeatures(parameters={TY: ty, RX: rx})
            for ty, rx in [*sampled, (true_best[TY], true_best[RX])]
        ]
    )
    predictions = np.asarray(means[METRIC], dtype=float)
    if predictions.shape != (len(sampled) + 1,) or not np.isfinite(predictions).all():
        raise ValueError(f"{snapshot}: invalid predictions at optimum locations")
    best_index = int(np.argmin(predictions[:-1]))
    best_ty, best_rx = sampled[best_index]
    best_actual = interpolator.interpolate(best_rx, best_ty)
    true_ty, true_rx = float(true_best[TY]), float(true_best[RX])
    true_value = interpolator.interpolate(true_rx, true_ty)
    best_prediction = float(predictions[best_index])
    true_prediction = float(predictions[-1])
    return RunMetrics(
        snapshot=snapshot,
        rmse=rmse,
        predicted_best_error=abs(best_prediction - best_actual),
        actual_best_error=abs(true_prediction - true_value),
        predicted_best_ty=best_ty,
        predicted_best_rx=best_rx,
        predicted_best_value=best_prediction,
        predicted_best_actual=best_actual,
        actual_best_ty=true_ty,
        actual_best_rx=true_rx,
        actual_best_value=true_value,
        actual_best_prediction=true_prediction,
    )


def comparison_figure(iterations: int, results: dict[str, list[RunMetrics]]) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Scan RMSE",
            "Error at model's best sampled location",
            "Error at actual interpolated optimum",
        ),
        vertical_spacing=0.13,
    )
    colors = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    )
    for group_index, (group, runs) in enumerate(results.items()):
        color = colors[group_index % len(colors)]
        for row, attribute in enumerate(
            ("rmse", "predicted_best_error", "actual_best_error"), start=1
        ):
            custom = [
                [
                    run.snapshot.name, run.predicted_best_ty, run.predicted_best_rx,
                    run.predicted_best_value, run.predicted_best_actual, run.actual_best_ty,
                    run.actual_best_rx, run.actual_best_value, run.actual_best_prediction,
                ]
                for run in runs
            ]
            figure.add_trace(
                go.Scatter(
                    x=[group] * len(runs),
                    y=[getattr(run, attribute) for run in runs],
                    customdata=custom,
                    mode="markers",
                    name=group,
                    legendgroup=group,
                    showlegend=row == 1,
                    marker={"color": color, "size": 9, "opacity": 0.8},
                    hovertemplate=(
                        "%{x}<br>Snapshot: %{customdata[0]}<br>Error: %{y:.4f}"
                        "<br>Predicted-best Ty/Rx: %{customdata[1]:.3f}, %{customdata[2]:.3f}"
                        "<br>Predicted/actual FWHM there: %{customdata[3]:.3f} / "
                        "%{customdata[4]:.3f}"
                        "<br>Actual-best Ty/Rx: %{customdata[5]:.3f}, %{customdata[6]:.3f}"
                        "<br>Actual/predicted FWHM there: %{customdata[7]:.3f} / "
                        "%{customdata[8]:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
    for row in range(1, 4):
        figure.update_xaxes(title_text="Run directory", row=row, col=1)
        figure.update_yaxes(title_text="FWHM error", rangemode="tozero", row=row, col=1)
    figure.update_layout(
        title=f"{iterations} iterations: model prediction errors",
        height=1400,
        legend_title_text="Run directory",
    )
    return figure


def plot_directories(directories: list[Path], output_dir: Path) -> list[Path]:
    if not directories:
        raise ValueError("No run directories found")
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"Run directory does not exist: {directory}")
    labels = [str(directory) for directory in directories]
    if len(set(labels)) != len(labels):
        raise ValueError("Run directories must be distinct")
    snapshots = {
        str(directory): {
            int(child.name): sorted(child.glob("*.json"))
            for child in directory.iterdir()
            if child.is_dir() and child.name.isdecimal() and list(child.glob("*.json"))
        }
        for directory in directories
    }
    counts = sorted({count for groups in snapshots.values() for count in groups})
    if not counts:
        raise ValueError("No snapshots found in numeric iteration directories")

    scan = load_scan(DEFAULT_SCAN, SIGNAL)
    interpolator = GridInterpolator(scan.set_index(["M4S:Ty", "M4S:Rx"])[[SIGNAL]])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for count in counts:
        results: dict[str, list[RunMetrics]] = {}
        for label, groups in snapshots.items():
            if count not in groups:
                print(f"No snapshots for {count} iterations in {label}")
                continue
            results[label] = []
            for snapshot in groups[count]:
                print(f"Analyzing {snapshot}", flush=True)
                results[label].append(calculate_metrics(snapshot, scan, interpolator))
        output = output_dir / f"comparison_{count}.html"
        comparison_figure(count, results).write_html(str(output), include_plotlyjs=True)
        print(f"Wrote {output}", flush=True)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        type=Path,
        nargs="*",
        help="Run directories (default: rmse* directories in the current directory)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("rmse_comparison_plots"))
    args = parser.parse_args()
    directories = args.directories or sorted(
        path for path in Path(".").glob("rmse*") if path.is_dir() and path != args.output_dir
    )
    plot_directories(directories, args.output_dir)


if __name__ == "__main__":
    main()
