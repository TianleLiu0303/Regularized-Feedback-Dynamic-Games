"""Cost primitives shared by inverse inference and forward planning."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .geometry import positions

Array = NDArray[np.float64]


@dataclass(frozen=True)
class GameCost:
    """Weighted primitives from the paper's CAMP/RM experiments.

    CAMP weights correspond to ``[goal, separation, effort]``. When
    ``lane_half_width`` is set, weights correspond to
    ``[goal, separation, lane, effort]``.
    """

    goals: Array
    weights: Array
    control_metric: Array
    lane_half_width: float | None = None
    separation_epsilon: float = 1.0e-12
    routes: tuple | None = None

    def __post_init__(self) -> None:
        goals = np.asarray(self.goals, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        metric = np.asarray(self.control_metric, dtype=float)
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "control_metric", metric)
        num_agents = len(goals)
        feature_dim = 4 if self.lane_half_width is not None else 3
        if goals.shape != (num_agents, 2):
            raise ValueError("goals must have shape [N, 2]")
        if weights.shape != (num_agents, feature_dim):
            raise ValueError(f"weights must have shape [N, {feature_dim}]")
        if metric.shape != (num_agents, num_agents, 2, 2):
            raise ValueError("control_metric must have shape [N, N, 2, 2]")
        if self.routes is not None and len(self.routes) != num_agents:
            raise ValueError("one assigned route per agent is required")
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1):
            raise ValueError("weights must be nonnegative, finite and sum to one")

    @property
    def num_agents(self) -> int:
        return len(self.goals)

    @property
    def feature_dim(self) -> int:
        return self.weights.shape[1]

    def features(self, state: Array, control: Array, agent: int, terminal: bool = False) -> Array:
        pos = positions(state)
        action = np.asarray(control, dtype=float).reshape(self.num_agents, 2)
        goal = float(np.sum((pos[agent] - self.goals[agent]) ** 2))
        squared_distance = np.sum((pos - pos[agent]) ** 2, axis=1)
        mask = np.arange(self.num_agents) != agent
        separation = float(-np.log(np.maximum(squared_distance[mask], self.separation_epsilon)).sum())
        effort = 0.0 if terminal else float(
            sum(action[j] @ self.control_metric[agent, j] @ action[j] for j in range(self.num_agents))
        )
        if self.lane_half_width is None:
            return np.array([goal, separation, effort], dtype=float)
        if self.routes is None:
            excess = max(abs(float(pos[agent, 1])) - self.lane_half_width, 0.0)
        else:
            route = self.routes[agent]
            excess = max(abs(route.project(pos[agent])[1]) - route.half_width, 0.0)
        return np.array([goal, separation, excess * excess, effort], dtype=float)

    def stage(self, state: Array, control: Array, agent: int) -> float:
        return float(self.weights[agent] @ self.features(state, control, agent))

    def position_derivatives(self, state: Array, agent: int) -> tuple[Array, Array]:
        """Exact gradient/Hessian in joint position coordinates, before clipping.

        Stage and terminal state derivatives coincide: effort has no state
        dependence. At polyline switches we use the currently selected segment;
        at the lane hinge and separation floor we use the inactive branch.
        """
        pos = positions(state)
        n = self.num_agents
        grad = np.zeros(2*n)
        curvature = np.zeros((2*n,2*n))
        own = slice(2*agent,2*agent+2)
        weight = self.weights[agent]
        identity = np.eye(2)
        grad[own] = 2*weight[0]*(pos[agent]-self.goals[agent])
        curvature[own,own] = 2*weight[0]*identity
        delta = pos[agent]-pos
        squared = np.einsum('ni,ni->n',delta,delta)
        others = np.flatnonzero((np.arange(n) != agent) & (squared > self.separation_epsilon))
        delta,squared = delta[others],squared[others]
        pair_gradient = -2*weight[1]*delta/squared[:,None]
        pair_hessian = weight[1]*(4*delta[:,:,None]*delta[:,None,:]/squared[:,None,None]**2
                                 - 2*identity/squared[:,None,None])
        indices = 2*others[:,None]+np.arange(2)
        grad[own] += pair_gradient.sum(axis=0)
        grad[indices.ravel()] -= pair_gradient.ravel()
        curvature[own,own] += pair_hessian.sum(axis=0)
        curvature[indices[:,:,None],indices[:,None,:]] += pair_hessian
        curvature[own,indices.ravel()] -= pair_hessian.transpose(1,0,2).reshape(2,-1)
        curvature[indices.ravel(),own] -= pair_hessian.reshape(-1,2)
        if self.lane_half_width is not None:
            if self.routes is None:
                lateral, half_width, normal = pos[agent,1], self.lane_half_width, np.array([0.,1.])
            else:
                route = self.routes[agent]
                _,lateral,tangent = route.project(pos[agent])
                half_width,normal = route.half_width, np.array([-tangent[1],tangent[0]])
            excess = abs(lateral)-half_width
            if excess > 0:
                grad[own] += 2*weight[2]*excess*np.sign(lateral)*normal
                curvature[own,own] += 2*weight[2]*np.outer(normal,normal)
        return grad,curvature

    def terminal(self, state: Array, agent: int) -> float:
        zero_control = np.zeros(2 * self.num_agents, dtype=float)
        return float(self.weights[agent] @ self.features(state, zero_control, agent, terminal=True))

    def trajectory_features(self, states: Array, controls: Array, agent: int) -> Array:
        total = np.zeros(self.feature_dim, dtype=float)
        for state, control in zip(states[:-1], controls, strict=True):
            total += self.features(state, control, agent)
        total += self.features(states[-1], np.zeros(2 * self.num_agents), agent, terminal=True)
        return total

    def trajectory_cost(self, states: Array, controls: Array, agent: int) -> float:
        return float(self.weights[agent] @ self.trajectory_features(states, controls, agent))
