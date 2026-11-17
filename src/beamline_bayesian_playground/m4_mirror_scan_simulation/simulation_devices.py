"""Beam focus devices using full width half maximum data from a real fluorescence camera.

There are simulation motors for virtual axes ty and rx on the mirror 4 hexapod. The fake
beam position monitor gives the full width half maximum of the waveform summing over x and y
on the grayscaled image, captured at a given mirror (ty, rx) position.

These full width half maximum values depend on the m4 mirror position, and give back data based on
the position of the fake mirror motors, interpolated from the measured scan positions.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from ophyd_async.core import (
    StandardReadable,
    soft_signal_r_and_setter,
)
from ophyd_async.core import (
    StandardReadableFormat as Fmt,
)
from ophyd_async.sim import SimMotor

from beamline_bayesian_playground.m4_mirror_scan_simulation.scan_grid import scan_grid_indices

# ===== Data loading

DATA_DIRECTORY = Path(__file__).parent / "real_data" / "spec_scans"


def read_spec_data(path: Path, x_signal: str, y_signal: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read FWHM values indexed by their unrounded measured mirror positions."""
    data = pd.read_csv(path, sep=r"\s+", engine="python")
    columns = ["M4S:Ty", "M4S:Rx", x_signal, y_signal]
    if data.empty or not np.isfinite(data[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Scan data in {path} must contain finite positions and signals")
    if data.duplicated(["M4S:Ty", "M4S:Rx"]).any():
        raise ValueError(f"Scan data in {path} contains duplicate measured positions")
    data = data.set_index(["M4S:Ty", "M4S:Rx"])
    return data[[x_signal]], data[[y_signal]]  # type: ignore


class GridInterpolator:
    """Interpolate locally on a grid of unrounded measured positions."""

    def __init__(self, grid: pd.DataFrame) -> None:
        ty = grid.index.get_level_values("M4S:Ty").to_numpy(dtype=float)
        rx = grid.index.get_level_values("M4S:Rx").to_numpy(dtype=float)
        values = grid.iloc[:, 0].to_numpy(dtype=float)
        if not np.isfinite(np.column_stack((ty, rx, values))).all():
            raise ValueError("Grid positions and values must be finite")
        rows, columns, ty_groups, rx_groups = scan_grid_indices(ty, rx)
        points = np.empty((rows, columns, 2))
        self._values = np.empty((rows, columns))
        points[ty_groups, rx_groups] = np.column_stack((ty, rx))
        self._values[ty_groups, rx_groups] = values

        p00 = points[:-1, :-1].reshape(-1, 2)
        p10 = points[1:, :-1].reshape(-1, 2)
        p01 = points[:-1, 1:].reshape(-1, 2)
        p11 = points[1:, 1:].reshape(-1, 2)
        self._p00 = p00
        self._a = p10 - p00
        self._b = p01 - p00
        self._c = p11 - p10 - p01 + p00
        corners = np.stack((p00, p10, p01, p11), axis=1)
        self._minimum = corners.min(axis=1)
        self._maximum = corners.max(axis=1)
        self._corners = np.stack(
            (
                self._values[:-1, :-1].ravel(),
                self._values[1:, :-1].ravel(),
                self._values[:-1, 1:].ravel(),
                self._values[1:, 1:].ravel(),
            ),
            axis=1,
        )
        for u, v in ((0, 0), (0, 1), (1, 0), (1, 1)):
            if np.any(self._cross(self._a + v * self._c, self._b + u * self._c) <= 0):
                raise ValueError("Scan grid contains overlapping or degenerate cells")

        boundary_points = np.concatenate(
            (points[0], points[1:, -1], points[-1, -2::-1], points[-2:0:-1, 0])
        )
        boundary_values = np.concatenate(
            (
                self._values[0],
                self._values[1:, -1],
                self._values[-1, -2::-1],
                self._values[-2:0:-1, 0],
            )
        )
        self._boundary_points = boundary_points
        self._boundary_values = boundary_values

    @staticmethod
    def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def interpolate(self, rx_position: float, ty_position: float) -> float:
        if not np.isfinite([rx_position, ty_position]).all():
            raise ValueError("Motor positions must be finite")
        point = np.array([ty_position, rx_position])
        candidates = np.flatnonzero(
            np.all(point >= self._minimum - 1e-9, axis=1)
            & np.all(point <= self._maximum + 1e-9, axis=1)
        )
        for i in candidates:
            a, b, c = self._a[i], self._b[i], self._c[i]
            delta = point - self._p00[i]
            determinant = self._cross(a, b)
            u, v = self._cross(delta, b) / determinant, self._cross(a, delta) / determinant
            for _ in range(12):
                residual = self._p00[i] + u * a + v * b + u * v * c - point
                du, dv = a + v * c, b + u * c
                determinant = self._cross(du, dv)
                if abs(determinant) < 1e-15:
                    break
                u -= self._cross(residual, dv) / determinant
                v -= self._cross(du, residual) / determinant
                if np.max(np.abs(residual)) < 1e-9:
                    break
            if -1e-9 <= u <= 1 + 1e-9 and -1e-9 <= v <= 1 + 1e-9:
                residual = self._p00[i] + u * a + v * b + u * v * c - point
                if np.max(np.abs(residual)) < 1e-7:
                    z00, z10, z01, z11 = self._corners[i]
                    return float(
                        z00 * (1 - u) * (1 - v)
                        + z10 * u * (1 - v)
                        + z01 * (1 - u) * v
                        + z11 * u * v
                    )

        starts = self._boundary_points
        ends = np.roll(starts, -1, axis=0)
        crossing = (starts[:, 0] > point[0]) != (ends[:, 0] > point[0])
        intersections = starts[crossing, 1] + (
            (point[0] - starts[crossing, 0])
            * (ends[crossing, 1] - starts[crossing, 1])
            / (ends[crossing, 0] - starts[crossing, 0])
        )
        if np.count_nonzero(intersections > point[1]) % 2:
            raise ValueError("Could not locate a scan grid cell for the requested position")
        directions = ends - starts
        fractions = np.clip(
            np.sum((point - starts) * directions, axis=1) / np.sum(directions * directions, axis=1),
            0,
            1,
        )
        projections = starts + fractions[:, np.newaxis] * directions
        nearest = np.argmin(np.sum((projections - point) ** 2, axis=1))
        fraction = fractions[nearest]
        values = self._boundary_values
        return float(
            values[nearest] * (1 - fraction) + values[(nearest + 1) % len(values)] * fraction
        )


def interpolate_grid(grid: pd.DataFrame, rx_position: float, ty_position: float) -> float:
    """Bilinearly interpolate measured positions; clamp outside the scan boundary."""
    return GridInterpolator(grid).interpolate(rx_position, ty_position)


# ===== Devices


class Mirror(StandardReadable):
    def __init__(self, name: str = "") -> None:
        super().__init__(name=name)
        with self.add_children_as_readables():
            self.ty = SimMotor(units="dial")
            self.rx = SimMotor(units="dial")


m4_ty = SimMotor(units="dial", name="m4_ty")
m4_rx = SimMotor(units="dial", name="m4_rx")


class BeamPositionMonitor(StandardReadable):
    get_simulation_values: Callable[[], Awaitable[tuple[float, float]]]

    def __init__(self, scan_data: Path, x_signal: str, y_signal: str, name: str = "") -> None:
        super().__init__(name=name)
        self._x_grid, self._y_grid = read_spec_data(scan_data, x_signal, y_signal)
        self._x_interpolator = GridInterpolator(self._x_grid)
        self._y_interpolator = GridInterpolator(self._y_grid)
        with self.add_children_as_readables(Fmt.HINTED_SIGNAL):
            self.fwhm_x, self.fwhm_x_test_setter = soft_signal_r_and_setter(float)
            self.fwhm_y, self.fwhm_y_test_setter = soft_signal_r_and_setter(float)

    async def read(self):
        rx, ty = await asyncio.gather(
            m4_rx.user_readback.get_value(),
            m4_ty.user_readback.get_value(),
        )
        self.fwhm_x_test_setter(self._x_interpolator.interpolate(rx, ty))
        self.fwhm_y_test_setter(self._y_interpolator.interpolate(rx, ty))
        return await super().read()


mocked_bpm_ue48_12_11_26 = BeamPositionMonitor(
    DATA_DIRECTORY / "m4s_mirror_ue48_fwhm_12_11_26.dat", "S1Xfw", "S1Yfw", name="bpm"
)
mocked_bpm_u17_09_26 = BeamPositionMonitor(
    DATA_DIRECTORY / "m4s_mirror_u17_fwhm_09_26.dat", "S2Xfw", "S2Yfw", name="bpm"
)
mocked_bpm_ue48_09_26 = BeamPositionMonitor(
    DATA_DIRECTORY / "m4s_mirror_ue48_fwhm_09_26.dat", "S2Xfw", "S2Yfw", name="bpm"
)
