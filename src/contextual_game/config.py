"""Typed configuration objects and JSON loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PlannerConfig:
    """Numerical settings used by the nonlinear forward-game solver."""

    scenario: str = "interhub"
    dt: float = 0.1
    horizon: int = 50
    outer_iterations: int = 30
    step_size: float = 0.5
    tolerance: float = 1.0e-3
    temperature: float = 0.99
    reference_source: str = "demonstration"
    coefficient_mode: str = "adaptive"
    fixed_lambda: float = 2.0
    lambda_min: float = 0.5
    lambda_max: float = 5.0
    lambda_sigma: float = 1.2
    finite_difference_step: float = 1.0e-3  # legacy config field; analytic primitives do not use it
    hessian_floor: float = 1.0e-3
    seed: int = 42

    def __post_init__(self) -> None:
        if self.reference_source not in {"cost", "demonstration", "none"}:
            raise ValueError("unknown reference_source")
        if self.coefficient_mode not in {"adaptive", "fixed"} or self.fixed_lambda <= 0:
            raise ValueError("invalid coefficient configuration")
        if self.dt <= 0 or self.tolerance <= 0 or self.hessian_floor <= 0:
            raise ValueError("dt, tolerance and curvature floor must be positive")
        if self.horizon < 1 or self.outer_iterations < 1:
            raise ValueError("horizon and outer_iterations must be positive")
        if not 0.0 < self.step_size <= 1.0:
            raise ValueError("step_size must be in (0, 1]")
        if not 0.0 < self.lambda_min < self.lambda_max:
            raise ValueError("require 0 < lambda_min < lambda_max")
        if self.temperature <= 0.0 or self.lambda_sigma <= 0.0:
            raise ValueError("temperature and lambda_sigma must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> "PlannerConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls(**json.load(stream))
