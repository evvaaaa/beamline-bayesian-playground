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

You can play around with each of these submodule's `__main__.py` to tweak agent parameters, for example

### Trying to maximize the fwhm of y in `m4_mirror_scan_simulation`:

<img width="1913" height="2100" alt="image" src="https://github.com/user-attachments/assets/29dee5e1-e51b-40da-abbe-7e09a4ffd066" />


### 10 iterations (moves of the motors):

<img width="1698" height="1270" alt="fwhm-10-iterations" src="https://github.com/user-attachments/assets/54415ef6-ffad-4721-aeb0-dd90c3611d39" />

### 20 iterations (moves of the motors):

<img width="1698" height="1270" alt="fwhm-20-iterations" src="https://github.com/user-attachments/assets/8b880b3d-cc55-407a-ae73-08ed1a42ced9" />

### 40 iterations (moves of the motors):

<img width="1698" height="1270" alt="fwhm-40-iterations" src="https://github.com/user-attachments/assets/2b2923fe-272b-4f48-863a-046cc6984ea4" />
