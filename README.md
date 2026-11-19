[![CI](https://github.com/evvaaaa/beamline-bayesian-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/evvaaaa/beamline-bayesian-playground/actions/workflows/ci.yml)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# beamline_bayesian_playground

Simulation devices (interpolating real beamline data) and a queueserver which allow for tweaking of bluesky-adaptive bayesian optimization agents.

This module is basically my own playground for running simulation bayesian agents. Each of the submodules for running an agent client
will become exceptionally messy.

What            | Where
:---:           | :---:
Source          | <https://github.com/evvaaaa/beamline-bayesian-playground>
Docker          | `docker run ghcr.io/evvaaaa/beamline-bayesian-playground:latest`
Releases        | <https://github.com/evvaaaa/beamline-bayesian-playground/releases>

# Usage Guide

### Build and start the queueserver container

Build the container `blop-testing` if I haven't got around to making CI container build work.

```
docker build -t blop-testing .
```

... then start the container with:

```
docker compose up
```

### M4 mirror scan exploration

The M4 agent starts at the center and uses 16 space-filling initialization
trials (including the center). It then alternates one model-based minimization
trial with two Sobol space-filling trials. The model-based phase uses
`qUpperConfidenceBound` with `beta=50.0`, favoring both low predicted FWHM and
uncertain points. This prioritizes learning the whole scan over finding the
minimum quickly; it does not guarantee lower or monotonic RMSE. Change
`EXPLORATION_BETA` or `INITIALIZATION_TRIALS` in `agent.py` to adjust the
tradeoff.

Run one scan with:

```
python -m beamline_bayesian_playground.m4_mirror_scan_simulation.agent --iterations 30
```

Use `--acquisition default` for Ax's previous acquisition strategy. To run
ten scans each at 10, 20, 30 and 60 iterations and plot their RMSE against
the measured scan:

```
python -m beamline_bayesian_playground.m4_mirror_scan_simulation.run_experiments
```

The new results go to `rmse_mapping_runs/`, separate from the previous
`rmse_exploration_runs/` and `rmse_runs/` baselines. The script resumes based
on existing JSON snapshots in that directory, so use a separate `--output-dir`
when comparing different settings. Use `--plot-only --output-dir rmse_exploration_runs`
to replot the previous exploration results. The RMSE includes measured points
outside the search space, where predictions are extrapolations.

To compare saved `rmse*` run directories, create one interactive HTML page per
iteration count:

```
python -m beamline_bayesian_playground.m4_mirror_scan_simulation.plot_run_comparison
```

The pages in `rmse_comparison_plots/` compare scan RMSE, absolute prediction
error at the model's best completed sampled point, and absolute prediction
error at the true interpolated minimum. RMSE uses all measured scan readbacks,
including those outside the search space. The best sampled point is chosen by
the fitted model's lowest predicted FWHM; both best-point errors compare model
predictions to the simulator's interpolated FWHM. The actual minimum is
restricted to each experiment's bounds. The script requires a global measured
minimum to lie inside those bounds, which establishes the interpolated minimum
because interpolation and boundary clamping are convex combinations of scan
values. Hover over points for the snapshot, coordinates and FWHM values.

Pass specific run directories as positional arguments and use `--output-dir`
to select a different output location.
