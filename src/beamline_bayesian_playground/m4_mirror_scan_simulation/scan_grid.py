"""Reconstruct grid membership from unrounded scan motor readbacks."""

import numpy as np


def _groups(values: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    ordered = np.sort(values)
    gaps = np.diff(ordered)
    splits = np.sort(np.argsort(gaps)[-(count - 1) :])
    boundaries = (ordered[splits] + ordered[splits + 1]) / 2
    inside = np.delete(gaps, splits)
    separation = float(np.min(gaps[splits]) / max(float(np.max(inside)), np.finfo(float).eps))
    return np.searchsorted(boundaries, values), separation


def scan_grid_indices(ty: np.ndarray, rx: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return grid dimensions and the Ty/Rx cell indices for each measured point."""
    if ty.ndim != 1 or rx.ndim != 1 or len(ty) != len(rx):
        raise ValueError("Scan positions must be one-dimensional arrays of equal length")
    if not np.isfinite(ty).all() or not np.isfinite(rx).all():
        raise ValueError("Scan positions must be finite")
    n = len(ty)
    candidates = []
    for rows in range(2, n // 2 + 1):
        if n % rows:
            continue
        columns = n // rows
        ty_groups, ty_separation = _groups(ty, rows)
        rx_groups, rx_separation = _groups(rx, columns)
        if len(np.unique(ty_groups * columns + rx_groups)) == n:
            candidates.append((ty_separation * rx_separation, rows, columns, ty_groups, rx_groups))
    if not candidates:
        raise ValueError("Scan positions must form a complete two-dimensional grid")
    _, rows, columns, ty_groups, rx_groups = max(candidates, key=lambda item: item[0])
    return rows, columns, ty_groups, rx_groups
