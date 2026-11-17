"""Compare an Ax client snapshot with the original measured scan points."""
# Shamlessly vibe coded.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ax.adapter.base import Adapter
from ax.api.client import Client
from ax.core.observation import ObservationFeatures
from ax.core.types import TParameterization
from plotly.subplots import make_subplots

from beamline_bayesian_playground.m4_mirror_scan_simulation.scan_grid import scan_grid_indices

SCAN_DIRECTORY = Path(__file__).parent / "real_data" / "spec_scans"
DEFAULT_SCAN = SCAN_DIRECTORY / "m4s_mirror_ue48_fwhm_12_11_26.dat"


def load_scan(path: Path, signal: str) -> pd.DataFrame:
    """Load measured positions and one signal, without binning or interpolation."""
    data = pd.read_csv(path, sep=r"\s+", engine="python")
    columns = ["M4S:Ty", "M4S:Rx", signal]
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    scan = data[columns].copy()
    if scan.empty or not np.isfinite(scan.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} must contain finite positions and {signal} values")
    return scan


def fit_model(client: Client) -> Adapter:
    """Fit the model-based node, including while the strategy is sampling Sobol points."""
    experiment = client._experiment  # noqa: SLF001
    strategy = client._generation_strategy  # noqa: SLF001
    adapter: Adapter | None
    model_node = strategy.nodes_by_name.get("MBM")
    if model_node is not None and len(model_node.generator_specs) == 1:
        spec = model_node.generator_specs[0]
        spec.fit(experiment=experiment)
        adapter = spec.fitted_adapter
    else:
        adapter = strategy.fit(experiment=experiment)
    if adapter is None:
        raise ValueError("The restored Ax generation strategy does not provide a fitted model")
    return adapter


def predict_scan(
    client: Client,
    scan: pd.DataFrame,
    signal: str,
    metric: str,
    ty_name: str,
    rx_name: str,
    *,
    inside_search_space: bool = False,
    adapter: Adapter | None = None,
) -> pd.DataFrame:
    """Predict at measured coordinates, optionally restricting to the Ax search space."""
    experiment = client._experiment  # noqa: SLF001
    expected = {ty_name, rx_name}
    if len(expected) != 2 or set(experiment.parameters) != expected:
        raise ValueError(
            f"Ax experiment parameters are {sorted(experiment.parameters)}; "
            f"expected exactly {sorted(expected)}. Set --ty-parameter and --rx-parameter."
        )
    if metric not in experiment.metrics:
        raise ValueError(
            f"Ax metric {metric!r} not found. Available: {', '.join(experiment.metrics)}"
        )
    ty = scan["M4S:Ty"].to_numpy(dtype=float)
    rx = scan["M4S:Rx"].to_numpy(dtype=float)
    rows, columns, ty_groups, rx_groups = scan_grid_indices(ty, rx)
    ty_axis = np.bincount(ty_groups, weights=ty, minlength=rows) / np.bincount(
        ty_groups, minlength=rows
    )
    rx_axis = np.bincount(rx_groups, weights=rx, minlength=columns) / np.bincount(
        rx_groups, minlength=columns
    )
    parameters: list[TParameterization] = [
        {ty_name: float(ty), rx_name: float(rx)}
        for ty, rx in scan[["M4S:Ty", "M4S:Rx"]].itertuples(index=False, name=None)
    ]
    in_design = np.array(
        [experiment.search_space.check_membership(p, raise_error=False) for p in parameters]
    )
    selected = in_design if inside_search_space else np.ones(len(scan), dtype=bool)
    if not selected.any():
        raise ValueError("No measured scan positions are inside the Ax search space")
    selected_parameters = [p for p, keep in zip(parameters, selected, strict=True) if keep]

    if adapter is None:
        adapter = fit_model(client)
    # Client.predict rejects out-of-design readbacks; the fitted adapter permits extrapolation.
    means, covariance = adapter.predict(
        observation_features=[ObservationFeatures(parameters=p) for p in selected_parameters]
    )
    if metric not in means or metric not in covariance or metric not in covariance[metric]:
        raise ValueError(f"The fitted Ax model cannot predict metric {metric!r}")
    predicted = np.asarray(means[metric], dtype=float)
    variance = np.asarray(covariance[metric][metric], dtype=float)
    if predicted.shape != (len(selected_parameters),) or variance.shape != predicted.shape:
        raise ValueError("Ax returned a prediction count that does not match the selected scan")
    if not np.isfinite(predicted).all() or not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError("Ax returned non-finite predictions or invalid variance")
    result = scan.loc[selected, ["M4S:Ty", "M4S:Rx"]].copy()
    result["real"] = scan.loc[selected, signal].to_numpy(dtype=float)
    result["prediction"] = predicted
    result["absolute_error"] = np.abs(predicted - result["real"].to_numpy())
    result["sem"] = np.sqrt(variance)
    result["in_design"] = in_design[selected]
    result["grid_row"] = ty_groups[selected]
    result["grid_column"] = rx_groups[selected]
    result.attrs["ty_axis"] = ty_axis
    result.attrs["rx_axis"] = rx_axis
    result.attrs["completed_trials"] = sum(
        t.status.is_completed for t in experiment.trials.values()
    )
    return result


def comparison_figure(data: pd.DataFrame, signal: str, metric: str) -> go.Figure:
    """Plot one unsmoothed heatmap cell per measured position."""
    if data.empty:
        raise ValueError("No measured scan positions to plot")
    ty_groups = data["grid_row"].to_numpy(dtype=int)
    rx_groups = data["grid_column"].to_numpy(dtype=int)
    ty_min, ty_max = int(ty_groups.min()), int(ty_groups.max())
    rx_min, rx_max = int(rx_groups.min()), int(rx_groups.max())
    ty_groups, rx_groups = ty_groups - ty_min, rx_groups - rx_min
    rows, columns = ty_max - ty_min + 1, rx_max - rx_min + 1

    def as_grid(values: np.ndarray) -> np.ndarray:
        grid = np.full((rows, columns), np.nan)
        grid[ty_groups, rx_groups] = values
        return grid

    ty_grid = as_grid(data["M4S:Ty"].to_numpy(dtype=float))
    rx_grid = as_grid(data["M4S:Rx"].to_numpy(dtype=float))
    ty_axis = data.attrs["ty_axis"][ty_min : ty_max + 1]
    rx_axis = data.attrs["rx_axis"][rx_min : rx_max + 1]
    in_design_grid = as_grid(data["in_design"].to_numpy(dtype=float))
    customdata = np.stack((ty_grid, rx_grid, in_design_grid), axis=-1)
    rmse = float(np.sqrt(np.mean(np.square(data["prediction"] - data["real"]))))
    limits = (
        float(min(data["real"].min(), data["prediction"].min())),
        float(max(data["real"].max(), data["prediction"].max())),
    )
    panels = [
        ("real", f"Measured {signal}", "Viridis", limits),
        ("prediction", f"Ax prediction: {metric}", "Viridis", limits),
        ("absolute_error", f"Absolute error (RMSE = {rmse:.4g})", "Reds", None),
        ("sem", "Ax predictive SEM", "Blues", None),
    ]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[panel[1] for panel in panels],
        horizontal_spacing=0.18,
        vertical_spacing=0.17,
    )
    for index, (column, title, colorscale, color_limits) in enumerate(panels):
        row, col = divmod(index, 2)
        figure.add_trace(
            go.Heatmap(
                x=rx_axis.tolist(),
                y=ty_axis.tolist(),
                z=as_grid(data[column].to_numpy(dtype=float)).tolist(),
                zsmooth=False,
                hoverongaps=False,
                name=title,
                colorscale=colorscale,
                zmin=color_limits[0] if color_limits is not None else None,
                zmax=color_limits[1] if color_limits is not None else None,
                colorbar={
                    "title": column,
                    "x": 0.44 if col == 0 else 1.0,
                    "y": 0.79 if row == 0 else 0.21,
                    "len": 0.38,
                },
                customdata=customdata.tolist(),
                hovertemplate=(
                    "Measured Rx: %{customdata[1]:.6f}<br>Measured Ty: %{customdata[0]:.6f}"
                    "<br>Value: %{z:.6g}"
                    "<br>In Ax search space (1=yes, 0=no): %{customdata[2]:.0f}<extra></extra>"
                ),
            ),
            row=row + 1,
            col=col + 1,
        )
        figure.update_xaxes(title_text="M4S:Rx", row=row + 1, col=col + 1)
        figure.update_yaxes(title_text="M4S:Ty", row=row + 1, col=col + 1)
    figure.update_layout(
        title=(
            f"Measured scan vs Ax model ({len(data)} points; "
            f"{data.attrs['completed_trials']} completed Ax trials)"
        ),
        height=900,
        width=1400,
        showlegend=False,
        plot_bgcolor="white",
    )
    return figure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Ax Client JSON snapshot")
    parser.add_argument("--scan", type=Path, default=DEFAULT_SCAN, help="SPEC .dat scan file")
    parser.add_argument("--signal", default="S1Yfw", help="Measured scan signal (default: S1Yfw)")
    parser.add_argument(
        "--metric", default="beam_focus_evaluator", help="Ax metric (default: beam_focuser)"
    )
    parser.add_argument("--ty-parameter", default="m4_ty", help="Ax Ty parameter name")
    parser.add_argument("--rx-parameter", default="m4_rx", help="Ax Rx parameter name")
    parser.add_argument(
        "--inside-search-space",
        action="store_true",
        help="Exclude measured points outside the Ax search space",
    )
    parser.add_argument("--output", type=Path, default=Path("ax_scan_comparison.html"))
    args = parser.parse_args(argv)

    scan = load_scan(args.scan, args.signal)
    client = Client.load_from_json_file(str(args.snapshot))
    comparison = predict_scan(
        client,
        scan,
        args.signal,
        args.metric,
        args.ty_parameter,
        args.rx_parameter,
        inside_search_space=args.inside_search_space,
    )
    if args.inside_search_space:
        print(
            f"Excluded {len(scan) - len(comparison)}/{len(scan)} points outside the search space."
        )
    else:
        outside = int((~comparison["in_design"]).sum())
        if outside:
            print(
                f"Warning: {outside}/{len(scan)} points are outside the Ax search space; "
                "their model predictions are extrapolations.",
                file=sys.stderr,
            )
    figure = comparison_figure(comparison, args.signal, args.metric)
    figure.write_html(str(args.output), include_plotlyjs=True, full_html=True)
    print(f"Wrote {args.output} ({len(comparison)} measured points)")


if __name__ == "__main__":
    main()
