"""Fixed dynamically feasible proposal generation for Algorithm 2."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from ..costs import GameCost
from ..dynamics import UnicycleDynamics

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CandidateSet:
    """The expert is always candidate zero, followed by M proposals."""

    states: Array
    controls: Array


def generate_candidate_set(
    dynamics: UnicycleDynamics,
    initial_state: Array,
    expert_controls: Array,
    proposal_count: int,
    noise_std: Array,
    lower_control: Array,
    upper_control: Array,
    rng: np.random.Generator,
    expert_states: Array | None = None,
) -> CandidateSet:
    expert_controls = np.asarray(expert_controls, dtype=float)
    candidates = [expert_controls]
    lower = np.broadcast_to(lower_control, (dynamics.num_agents, 2)).reshape(-1)
    upper = np.broadcast_to(upper_control, (dynamics.num_agents, 2)).reshape(-1)
    scale = np.broadcast_to(noise_std, (dynamics.num_agents, 2)).reshape(-1)
    for _ in range(proposal_count):
        noisy = expert_controls + rng.normal(0.0, scale, size=expert_controls.shape)
        candidates.append(np.clip(noisy, lower, upper))
    controls = np.stack(candidates)
    states = np.stack([dynamics.rollout(initial_state, item) for item in controls])
    if expert_states is not None:
        if np.shape(expert_states) != states[0].shape or not np.allclose(expert_states[0], initial_state):
            raise ValueError("expert states must match the observed window and initial state")
        states[0] = expert_states
    return CandidateSet(states=states, controls=controls)


def precompute_features(
    candidates: CandidateSet,
    cost: GameCost,
) -> Array:
    """Return stage-plus-terminal task features [N,C,F] for every candidate."""
    count = len(candidates.states)
    agents = cost.num_agents
    features = np.empty((agents, count, cost.feature_dim), dtype=float)
    for c in range(count):
        for i in range(agents):
            features[i, c] = cost.trajectory_features(
                candidates.states[c], candidates.controls[c], i
            )
    return features
