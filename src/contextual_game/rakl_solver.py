"""Finite-horizon proximity-adaptive KL local feedback-game recursion."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .local_model import LocalGameModel
from .reference_policy import GaussianReference

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LocalPolicy:
    feedback: Array
    feedforward: Array
    covariance: Array
    value_hessians: Array
    value_gradients: Array


def solve_local_game(
    model: LocalGameModel,
    regularization: Array,
    references: list[list[GaussianReference]],
) -> LocalPolicy:
    """Solve the frozen local game's coupled policy/value backward recursions.

    ``feedback[t, i]`` stores P_t^i and the policy mean is
    ``-P_t^i delta_x - alpha_t^i``.
    """
    horizon, agents = regularization.shape
    if not np.isfinite(regularization).all() or np.any(regularization < 0):
        raise ValueError("regularization must be finite and nonnegative")
    state_dim = model.a[0].shape[0]
    action_dim = model.b[0][0].shape[1]
    joint_action_dim = agents * action_dim
    feedback = np.zeros((horizon, agents, action_dim, state_dim), dtype=float)
    feedforward = np.zeros((horizon, agents, action_dim), dtype=float)
    covariance = np.zeros((horizon, agents, action_dim, action_dim), dtype=float)
    value_q = np.array(model.terminal_q)
    value_linear = np.array(model.terminal_linear)
    value_hessians = np.empty((horizon + 1, agents, state_dim, state_dim))
    value_gradients = np.empty((horizon + 1, agents, state_dim))
    value_hessians[-1], value_gradients[-1] = value_q, value_linear
    players = np.arange(agents)
    diagonal_rows = (players[:,None]*action_dim + np.arange(action_dim)).ravel()
    diagonal_cols = players.repeat(action_dim)

    for t in range(horizon - 1, -1, -1):
        b = np.asarray(model.b[t])
        joint_b = b.transpose(1,0,2).reshape(state_dim,joint_action_dim)
        bt = b.swapaxes(-1,-2)
        bz = bt @ value_q
        r = np.asarray(model.r[t])
        linear_u = np.asarray(model.linear_control[t])
        means = np.array([ref.mean for ref in references[t]])
        inverse_reference = np.linalg.inv(np.array([ref.covariance for ref in references[t]]))
        weighted_precision = regularization[t,:,None,None]*inverse_reference
        own_r = r[players,players]
        # All response blocks share B_i^T Z_i. Solve both right-hand sides
        # together, using one factorization of the same coupled Nash system.
        coupled = bz.reshape(joint_action_dim,state_dim) @ joint_b
        blocks = coupled.reshape(joint_action_dim,agents,action_dim)
        blocks[diagonal_rows,diagonal_cols] += (own_r+weighted_precision).reshape(-1,action_dim)
        rhs_p = (bz @ model.a[t]).reshape(joint_action_dim,state_dim)
        rhs_alpha = (np.einsum('ima,ia->im',bt,value_linear) + linear_u[players,players]
                     - np.einsum('imn,in->im',weighted_precision,means)).ravel()

        try:
            solution = np.linalg.solve(coupled, np.column_stack([rhs_p,rhs_alpha]))
        except np.linalg.LinAlgError as error:
            raise RuntimeError(f"coupled Nash system is singular at stage {t}") from error

        stacked_p,stacked_alpha = solution[:,:state_dim],solution[:,state_dim]
        feedback[t] = stacked_p.reshape(agents,action_dim,state_dim)
        feedforward[t] = stacked_alpha.reshape(agents,action_dim)
        positive = regularization[t] > 0
        precision = ((own_r+bz@b)[positive]/regularization[t,positive,None,None]
                     + inverse_reference[positive])
        covariance[t,positive] = np.linalg.inv(precision)

        closed_loop = model.a[t] - joint_b @ stacked_p
        drift = -joint_b @ stacked_alpha
        next_q = np.asarray(model.q[t]) + closed_loop.T @ value_q @ closed_loop
        next_linear = np.asarray(model.linear_state[t]) + np.einsum(
            'ab,ib->ia',closed_loop.T,value_linear+value_q@drift)
        # Omit only exactly-zero terms. Cross-player effort terms remain
        # supported; the paper's diagonal effort metric avoids N^2 products.
        for i,j in np.argwhere(np.any(r != 0,axis=(-2,-1)) | np.any(linear_u != 0,axis=-1)):
            p_j = feedback[t,j]
            pr = p_j.T @ r[i,j]
            next_q[i] += pr @ p_j
            next_linear[i] += pr @ feedforward[t,j] - p_j.T @ linear_u[i,j]
        weighted_p = feedback[t].swapaxes(-1,-2) @ weighted_precision
        next_q += weighted_p @ feedback[t]
        next_linear += np.einsum('iam,im->ia',weighted_p,feedforward[t]+means)
        value_q = .5*(next_q+next_q.swapaxes(-1,-2))
        value_linear = next_linear
        value_hessians[t], value_gradients[t] = value_q, value_linear

    return LocalPolicy(feedback, feedforward, covariance, value_hessians, value_gradients)


def cost_induced_references(model: LocalGameModel, temperature: float):
    """Boltzmann Q reference at delta_x=0 after an auxiliary unregularized solve.

    Opponents use the auxiliary feedback policies; these references are frozen
    for the subsequent KLGame baseline solve.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    horizon, agents = len(model.a), len(model.b[0])
    d = model.b[0][0].shape[1]
    dummy = [[GaussianReference(np.zeros(d), np.eye(d)) for _ in range(agents)] for _ in range(horizon)]
    auxiliary = solve_local_game(model, np.zeros((horizon, agents)), dummy)
    result = []
    for t in range(horizon):
        row = []
        own_drifts = np.array([model.b[t][j] @ auxiliary.feedforward[t,j] for j in range(agents)])
        joint_drift = -own_drifts.sum(axis=0)
        for i in range(agents):
            b = model.b[t][i]
            z = auxiliary.value_hessians[t + 1, i]
            drift = joint_drift + own_drifts[i]
            h = model.r[t][i][i] + b.T @ z @ b
            g = model.linear_control[t][i][i] + b.T @ (auxiliary.value_gradients[t + 1, i] + z @ drift)
            np.linalg.cholesky(h)
            row.append(GaussianReference(-np.linalg.solve(h, g), temperature * np.linalg.inv(h)))
        result.append(row)
    return result
