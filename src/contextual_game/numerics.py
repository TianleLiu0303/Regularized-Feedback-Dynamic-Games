"""Small dependency-free numerical differentiation helpers."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def gradient(function: Callable[[Array], float], point: Array, step: float) -> Array:
    x = np.asarray(point, dtype=float)
    result = np.empty_like(x)
    for i in range(x.size):
        delta = np.zeros_like(x)
        delta[i] = step
        result[i] = (function(x + delta) - function(x - delta)) / (2.0 * step)
    return result


def hessian(function: Callable[[Array], float], point: Array, step: float) -> Array:
    x = np.asarray(point, dtype=float)
    size = x.size
    result = np.empty((size, size), dtype=float)
    center = function(x)
    for i in range(size):
        ei = np.zeros_like(x)
        ei[i] = step
        result[i, i] = (function(x + ei) - 2.0 * center + function(x - ei)) / (step * step)
        for j in range(i + 1, size):
            ej = np.zeros_like(x)
            ej[j] = step
            value = (
                function(x + ei + ej)
                - function(x + ei - ej)
                - function(x - ei + ej)
                + function(x - ei - ej)
            ) / (4.0 * step * step)
            result[i, j] = value
            result[j, i] = value
    return 0.5 * (result + result.T)


def project_psd(matrix: Array, floor: float) -> Array:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T
