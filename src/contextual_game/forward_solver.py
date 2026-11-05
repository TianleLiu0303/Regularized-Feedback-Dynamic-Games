"""Outer-loop nonlinear forward-game solver (paper Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .config import PlannerConfig
from .costs import GameCost
from .dynamics import UnicycleDynamics
from .geometry import adaptive_kl_coefficient
from .local_model import build_local_model
from .rakl_solver import LocalPolicy, solve_local_game, cost_induced_references
from .reference_policy import GaussianReference

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PlannerResult:
    states: Array
    controls: Array
    feedback: Array
    feedforward: Array
    covariance: Array
    regularization_history: Array
    residual_history: Array
    converged: bool


class ForwardGameSolver:
    """Iteratively solve frozen local Gaussian feedback Nash games."""

    def __init__(
        self,
        dynamics: UnicycleDynamics,
        cost: GameCost,
        config: PlannerConfig,
        lower_control: Array,
        upper_control: Array,
        reference_provider=None,
        obstacle_distance=None,
        envelope_radii: Array | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.cost = cost
        self.config = config
        if not np.isclose(config.dt,dynamics.dt):
            raise ValueError("planner dt differs from dynamics dt")
        self.reference_provider = reference_provider
        self.obstacle_distance = obstacle_distance
        self.envelope_radii = None if envelope_radii is None else np.asarray(envelope_radii, dtype=float).copy()
        if self.envelope_radii is not None:
            if (self.envelope_radii.shape != (dynamics.num_agents,)
                    or not np.isfinite(self.envelope_radii).all() or np.any(self.envelope_radii <= 0)):
                raise ValueError("one finite positive envelope radius per agent is required")
        elif config.reference_source != "none" and config.coefficient_mode == "adaptive":
            raise ValueError("adaptive planning requires explicit circular-envelope radii")
        if config.reference_source == "demonstration" and reference_provider is None:
            raise ValueError("demonstration planning requires a fitted training library/provider")
        self.lower_control = np.broadcast_to(lower_control, (dynamics.num_agents, 2)).reshape(-1)
        self.upper_control = np.broadcast_to(upper_control, (dynamics.num_agents, 2)).reshape(-1)
        if np.any(self.lower_control >= self.upper_control):
            raise ValueError("each lower control bound must be below its upper bound")

    def _regularization(self, states: Array) -> Array:
        values = np.empty((self.config.horizon, self.dynamics.num_agents), dtype=float)
        if self.config.reference_source == "none":
            return np.zeros_like(values)
        if self.config.coefficient_mode == "fixed":
            return np.full_like(values, self.config.fixed_lambda)
        for t in range(self.config.horizon):
            for i in range(self.dynamics.num_agents):
                values[t, i] = adaptive_kl_coefficient(
                    states[t],
                    i,
                    self.config.lambda_min,
                    self.config.lambda_max,
                    self.config.lambda_sigma,
                    envelope_radii=self.envelope_radii,
                    obstacle_distance=(float("inf") if self.obstacle_distance is None
                                       else self.obstacle_distance(states[t], i)),
                )
        return values

    def solve(self, initial_state: Array, initial_controls: Array | None = None) -> PlannerResult:
        cfg = self.config
        agents = self.dynamics.num_agents
        if initial_controls is None:
            controls = np.zeros((cfg.horizon, self.dynamics.control_dim), dtype=float)
        else:
            controls = np.asarray(initial_controls, dtype=float).copy()
        if controls.shape != (cfg.horizon, self.dynamics.control_dim):
            raise ValueError("initial_controls has the wrong shape")
        controls = np.clip(controls, self.lower_control, self.upper_control)
        states = self.dynamics.rollout(np.asarray(initial_state, dtype=float), controls)
        residuals: list[float] = []
        lambda_history: list[Array] = []
        local_policy: LocalPolicy | None = None

        for _ in range(cfg.outer_iterations):
            model = build_local_model(
                self.dynamics,
                self.cost,
                states,
                controls,
                cfg.finite_difference_step,
                cfg.hessian_floor,
            )
            if cfg.reference_source == "demonstration":
                references = self.reference_provider(states, controls)
            elif cfg.reference_source == "cost":
                references = cost_induced_references(model, cfg.temperature)
            else:
                references = [[GaussianReference(np.zeros(2), np.eye(2))
                               for _ in range(agents)] for _ in range(cfg.horizon)]
            regularization = self._regularization(states)
            lambda_history.append(regularization.copy())
            local_policy = solve_local_game(model, regularization, references)
            new_states = np.empty_like(states)
            new_controls = np.empty_like(controls)
            new_states[0] = initial_state
            for t in range(cfg.horizon):
                state_deviation = new_states[t] - states[t]
                state_deviation[2::4] = (state_deviation[2::4] + np.pi) % (2 * np.pi) - np.pi
                action = (controls[t] - cfg.step_size*local_policy.feedforward[t].ravel()
                          - (local_policy.feedback[t] @ state_deviation).ravel())
                new_controls[t] = np.clip(action,self.lower_control,self.upper_control)
                new_states[t + 1] = self.dynamics.step(new_states[t], new_controls[t])
            state_delta = new_states - states
            state_delta[:, 2::4] = (state_delta[:, 2::4] + np.pi) % (2 * np.pi) - np.pi
            residual = float(np.linalg.norm(np.concatenate((state_delta.ravel(), (new_controls-controls).ravel()))))
            residuals.append(residual)
            states, controls = new_states, new_controls
            if residual < cfg.tolerance:
                break

        if local_policy is None:
            raise RuntimeError("solver performed no iterations")
        return PlannerResult(
            states=states,
            controls=controls,
            feedback=local_policy.feedback,
            feedforward=local_policy.feedforward,
            covariance=local_policy.covariance,
            regularization_history=np.stack(lambda_history),
            residual_history=np.asarray(residuals),
            converged=bool(residuals[-1] < cfg.tolerance),
        )
