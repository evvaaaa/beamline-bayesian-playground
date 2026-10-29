[![CI](https://github.com/evvaaaa/beamline-bayesian-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/evvaaaa/beamline-bayesian-playground/actions/workflows/ci.yml)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# beamline_bayesian_playground

Simulation devices (with interpolated real beamline data) and a queueserver which allow for tweaking of bluesky-adaptive bayesian optimization agents.

What            | Where
:---:           | :---:
Source          | <https://github.com/evvaaaa/beamline-bayesian-playground>
Docker          | `docker run ghcr.io/evvaaaa/beamline-bayesian-playground:latest`
Releases        | <https://github.com/evvaaaa/beamline-bayesian-playground/releases>

# TODO: A slightly nicer interface for setting up play around scripts

# Usage Guide

## 1)

Install the package using uv:

```
uv sync --group client
```

## 2)

Build the container `blop-testing` if I haven't got around to making CI container build work.

```
docker build -t blop-testing .
```

... then start the container with:

```
docker compose up
```

## 3)

Run a simulation:

```
python -m beamline_bayesian_playground.m4_mirror_scan_simulation
```

You can play around with each of these submodule's `__main__.py` to tweak agent parameters.
