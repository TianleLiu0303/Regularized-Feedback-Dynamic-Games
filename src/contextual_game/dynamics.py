"""Joint 4D unicycle dynamics used in the paper's experiments."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class UnicycleDynamics:
    """Independent unicycles coupled through game costs.

    Each agent has state ``[px, py, heading, speed]`` and control
    ``[yaw_rate, acceleration]``. The joint vectors concatenate agents.
    """

    num_agents: int
    dt: float = 0.1

    @property
    def state_dim(self) -> int:
        return 4 * self.num_agents

    @property
    def control_dim(self) -> int:
        return 2 * self.num_agents

    def step(self, state: Array, control: Array) -> Array:
        x = np.asarray(state, dtype=float).reshape(self.num_agents, 4)
        u = np.asarray(control, dtype=float).reshape(self.num_agents, 2)
        nxt = x.copy()
        nxt[:, 0] += self.dt * x[:, 3] * np.cos(x[:, 2])
        nxt[:, 1] += self.dt * x[:, 3] * np.sin(x[:, 2])
        nxt[:, 2] += self.dt * u[:, 0]
        nxt[:, 3] += self.dt * u[:, 1]
        nxt[:, 2] = (nxt[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
        return nxt.reshape(-1)

    def rollout(self, initial_state: Array, controls: Array) -> Array:
        controls = np.asarray(controls, dtype=float)
        states = np.empty((len(controls) + 1, self.state_dim), dtype=float)
        states[0] = initial_state
        for t, action in enumerate(controls):
            states[t + 1] = self.step(states[t], action)
        return states

    def linearize(self, state: Array) -> tuple[Array, list[Array]]:
        """Return exact Jacobians A and B_i for the Euler update."""
        x = np.asarray(state, dtype=float).reshape(self.num_agents, 4)
        a = np.eye(self.state_dim, dtype=float)
        b = [np.zeros((self.state_dim, 2), dtype=float) for _ in range(self.num_agents)]
        for i, (_, _, heading, speed) in enumerate(x):
            base = 4 * i
            a[base, base + 2] = -self.dt * speed * np.sin(heading)
            a[base, base + 3] = self.dt * np.cos(heading)
            a[base + 1, base + 2] = self.dt * speed * np.cos(heading)
            a[base + 1, base + 3] = self.dt * np.sin(heading)
            b[i][base + 2, 0] = self.dt
            b[i][base + 3, 1] = self.dt
        return a, b
