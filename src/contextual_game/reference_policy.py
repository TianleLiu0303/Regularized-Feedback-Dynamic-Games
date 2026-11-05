"""Gaussian action references in local control-deviation coordinates."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GaussianReference:
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
