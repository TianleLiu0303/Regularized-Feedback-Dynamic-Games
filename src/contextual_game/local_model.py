"""Linearize dynamics and quadratize each player's cost."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .costs import GameCost
from .dynamics import UnicycleDynamics
from .numerics import project_psd

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LocalGameModel:
    a: list[Array]
    b: list[list[Array]]
    q: list[list[Array]]
    linear_state: list[list[Array]]
    r: list[list[list[Array]]]
    linear_control: list[list[list[Array]]]
    terminal_q: list[Array]
    terminal_linear: list[Array]


def build_local_model(
    dynamics: UnicycleDynamics,
    cost: GameCost,
    nominal_states: Array,
    nominal_controls: Array,
    difference_step: float,
    hessian_floor: float,
) -> LocalGameModel:
    """Use analytic primitive derivatives and the paper's eigenvalue floors.

    ``difference_step`` is retained for existing callers/configuration files;
    the built-in primitives no longer require finite differences.
    """
    horizon = len(nominal_controls)
    agents = dynamics.num_agents
    position_indices = (4*np.arange(agents)[:,None]+np.arange(2)).ravel()

    def state_expansion(state, player):
        gradient, curvature = cost.position_derivatives(state, player)
        linear = np.zeros(dynamics.state_dim)
        quadratic = np.zeros((dynamics.state_dim,dynamics.state_dim))
        linear[position_indices] = gradient
        # All other state rows/columns are exactly zero. Clipping this position
        # block is equivalent to clipping the full joint-state Hessian.
        quadratic[np.ix_(position_indices,position_indices)] = project_psd(curvature,0.)
        return linear,quadratic

    # Costs/weights are frozen throughout the cycle, and these exact control
    # Hessians do not depend on the nominal state or control.
    control_curvature = cost.weights[:,-1,None,None,None]*(cost.control_metric+cost.control_metric.swapaxes(-1,-2))
    control_blocks = [[project_psd(control_curvature[i,j],hessian_floor if i == j else 0.)
                       for j in range(agents)] for i in range(agents)]
    a_list: list[Array] = []
    b_list: list[list[Array]] = []
    q_list: list[list[Array]] = []
    lx_list: list[list[Array]] = []
    r_list: list[list[list[Array]]] = []
    lu_list: list[list[list[Array]]] = []

    for t in range(horizon):
        state = nominal_states[t]
        control = nominal_controls[t]
        a_t, b_t = dynamics.linearize(state)
        a_list.append(a_t)
        b_list.append(b_t)
        q_row: list[Array] = []
        lx_row: list[Array] = []
        r_players: list[list[Array]] = []
        lu_players = np.einsum('ijab,jb->ija',control_curvature,control.reshape(agents,2))
        for i in range(agents):
            linear,quadratic = state_expansion(state,i)
            lx_row.append(linear)
            q_row.append(quadratic)
            r_players.append(control_blocks[i])
        q_list.append(q_row)
        lx_list.append(lx_row)
        r_list.append(r_players)
        lu_list.append(lu_players)

    terminal_q: list[Array] = []
    terminal_linear: list[Array] = []
    final_state = nominal_states[-1]
    for i in range(agents):
        linear,quadratic = state_expansion(final_state,i)
        terminal_linear.append(linear)
        terminal_q.append(quadratic)
    return LocalGameModel(
        a=a_list,
        b=b_list,
        q=q_list,
        linear_state=lx_list,
        r=r_list,
        linear_control=lu_list,
        terminal_q=terminal_q,
        terminal_linear=terminal_linear,
    )
