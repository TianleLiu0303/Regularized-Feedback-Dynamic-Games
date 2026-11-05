"""Distances and state-dependent KL regularization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def positions(joint_state: Array) -> Array:
    """Extract an ``[N, 2]`` position matrix from a joint state."""
    return np.asarray(joint_state, dtype=float).reshape(-1, 4)[:, :2]


def nearest_agent_distance(joint_state: Array, agent: int) -> float:
    pos = positions(joint_state)
    if len(pos) == 1:
        return float("inf")
    delta = pos - pos[agent]
    distance = np.linalg.norm(delta, axis=1)
    distance[agent] = np.inf
    return float(np.min(distance))


def envelope_clearance(
    joint_state: Array,
    agent: int,
    envelope_radii: Array,
    obstacle_distance: float = float("inf"),
) -> float:
    """Nonnegative minimum envelope clearance; obstacle distance is centre-to-set.

    ``obstacle_distance=inf`` represents the empty static-obstacle set.
    The envelope is used for scheduling, not for footprint collision scoring.
    """
    pos = positions(joint_state)
    radii = np.asarray(envelope_radii, dtype=float)
    if radii.shape != (len(pos),) or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("one finite positive envelope radius per agent is required")
    if np.isnan(obstacle_distance) or obstacle_distance < 0:
        raise ValueError("obstacle_distance must be a nonnegative centre-to-set distance")
    distances = np.linalg.norm(pos - pos[agent], axis=1) - radii[agent] - radii
    distances[agent] = np.inf
    return float(max(0.0, min(np.min(distances), obstacle_distance - radii[agent])))


def adaptive_kl_coefficient(
    joint_state: Array,
    agent: int,
    lambda_min: float,
    lambda_max: float,
    sigma: float,
    *,
    envelope_radii: Array,
    obstacle_distance: float = float("inf"),
) -> float:
    """Increase reference reliance as circular-envelope clearance decreases."""
    distance = envelope_clearance(joint_state, agent, envelope_radii, obstacle_distance)
    risk = np.exp(-(distance * distance) / (2.0 * sigma * sigma))
    return float(lambda_min + (lambda_max - lambda_min) * risk)
