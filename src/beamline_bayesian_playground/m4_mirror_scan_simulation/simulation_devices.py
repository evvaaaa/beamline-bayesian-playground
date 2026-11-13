"""Beam focus devices using full width half maximum data from a real fluorescence camera.

There are simulation motors for virtual axes ty and rx on the mirror 4 hexapod. The fake
beam position monitor gives the full width half maximum of the waveform summing over x and y
on the grayscaled image, captured at a given mirror (ty, rx) position.

These full width half maximum values depend on the m4 mirror position, and give back data based on
the position of the fake mirror motors, interpolated from the real scan data over those mirrors.
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

# ===== Data loading

DATA_DIRECTORY = Path(__file__).parent.parent / "real_data"


def read_spec_data(path: Path, x_signal: str, y_signal: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the spec ascii scan data and gives dataframes of fwhm of each of x and y."""

    data = pd.read_csv(path, sep=r"\s+", engine="python")
    data["Ty_bin"], data["Rx_bin"] = data["M4S:Ty"].round(0), data["M4S:Rx"].round(0)
    return data.pivot(index="Ty_bin", columns="Rx_bin", values=x_signal), data.pivot(
        index="Ty_bin", columns="Rx_bin", values=y_signal
    )


def interpolate_grid(grid: pd.DataFrame, rx_position: float, ty_position: float) -> float:
    """Shamelessly gpt promted bilinear interpolation."""
    ty_axis = grid.index.to_numpy(dtype=float)
    rx_axis = grid.columns.to_numpy(dtype=float)
    z = grid.to_numpy(dtype=float)

    ty = np.clip(ty_position, ty_axis[0], ty_axis[-1])
    rx = np.clip(rx_position, rx_axis[0], rx_axis[-1])

    i1 = np.searchsorted(ty_axis, ty)
    j1 = np.searchsorted(rx_axis, rx)
    i0 = max(i1 - 1, 0)
    j0 = max(j1 - 1, 0)
    i1 = min(i1, len(ty_axis) - 1)
    j1 = min(j1, len(rx_axis) - 1)

    y0, y1 = ty_axis[i0], ty_axis[i1]
    x0, x1 = rx_axis[j0], rx_axis[j1]

    q00 = z[i0, j0]
    q01 = z[i0, j1]
    q10 = z[i1, j0]
    q11 = z[i1, j1]

    if x1 == x0 and y1 == y0:
        return float(q00)
    if x1 == x0:
        ty_t = 0.0 if y1 == y0 else (ty - y0) / (y1 - y0)
        return float(q00 * (1 - ty_t) + q10 * ty_t)
    if y1 == y0:
        rx_t = (rx - x0) / (x1 - x0)
        return float(q00 * (1 - rx_t) + q01 * rx_t)

    rx_t = (rx - x0) / (x1 - x0)
    ty_t = (ty - y0) / (y1 - y0)

    return float(
        q00 * (1 - rx_t) * (1 - ty_t)
        + q01 * rx_t * (1 - ty_t)
        + q10 * (1 - rx_t) * ty_t
        + q11 * rx_t * ty_t
    )


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
        with self.add_children_as_readables(Fmt.HINTED_SIGNAL):
            self.fwhm_x, self.fwhm_x_test_setter = soft_signal_r_and_setter(float)
            self.fwhm_y, self.fwhm_y_test_setter = soft_signal_r_and_setter(float)

    async def read(self):
        rx, ty = await asyncio.gather(
            m4_rx.user_readback.get_value(),
            m4_ty.user_readback.get_value(),
        )
        self.fwhm_x_test_setter(interpolate_grid(self._x_grid, rx, ty))
        self.fwhm_y_test_setter(interpolate_grid(self._y_grid, rx, ty))
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
