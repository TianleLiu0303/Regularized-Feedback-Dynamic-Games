"""Generate synthetic experts from supplied initial states and goals."""
from dataclasses import replace
import numpy as np
from .costs import GameCost
from .dynamics import UnicycleDynamics
from .forward_solver import ForwardGameSolver


def generate_expert(initial_state,goals,config,seed,domain="camp",routes=None):
    """Dirichlet weights held fixed; unregularized iterative LQ feedback game.

    The caller supplies the nominal layout to reproduce its actual scenarios.
    The convergence flag must be inspected before treating a rollout as expert.
    """
    if domain not in {"camp","rm"}:
        raise ValueError("synthetic domain must be camp or rm")
    if domain == "rm" and routes is None:
        raise ValueError("RM requires assigned routes")
    state=np.asarray(initial_state,float).reshape(-1)
    n=len(state)//4
    rng=np.random.default_rng(seed)
    omega=rng.dirichlet(np.ones(3 if domain == "camp" else 4),size=n)
    metric=np.zeros((n,n,2,2))
    for i in range(n):
        metric[i,i]=np.diag([1.,.25])
    cost=GameCost(np.asarray(goals),omega,metric,lane_half_width=None if routes is None else routes[0].half_width,
                  routes=routes)
    bounds=([-1.2,-2.],[1.2,2.]) if domain == "camp" else ([-.3,-3.],[.3,2.])
    solver=ForwardGameSolver(UnicycleDynamics(n,config.dt),cost,replace(config,reference_source="none"),*bounds)
    result=solver.solve(state)
    return result,omega


def perturb_positions(initial_state,rng,std=.2):
    state=np.asarray(initial_state,float).reshape(-1,4).copy()
    state[:,:2]+=rng.normal(0,std,state[:,:2].shape)
    return state.ravel()
