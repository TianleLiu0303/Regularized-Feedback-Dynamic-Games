"""Receding-horizon control with one objective update per measured cycle."""
from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np
from .demonstration import DemonstrationReference
from .forward_solver import ForwardGameSolver


@dataclass
class ClosedLoopResult:
    states: np.ndarray
    controls: np.ndarray
    weights: np.ndarray
    cycle_ms: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    residual_history: np.ndarray

    def solver_diagnostics(self):
        """Numeric traces; NaN pads unused iterations after tolerance stopping."""
        final = self.residual_history[np.arange(len(self.iterations)),self.iterations-1]
        return dict(converged=self.converged,iterations=self.iterations,
                    final_residual=final,residual_history=self.residual_history,
                    stop_reason=np.where(self.converged,'tolerance','iteration_limit'))


def run_closed_loop(dynamics,cost,config,history,infer_weights,cycles,lower,upper,
                    library=None,observe=None,obstacle_distance=None,envelope_radii=None):
    """observe(x,u) is the environment/hardware transition, default joint dynamics.

    Zero feasible controls initialize cycle zero. Later cycles shift controls
    one step and repeat the last control, then re-roll from the measured state.
    No recorded future states enter this function.
    """
    history = np.asarray(history,float).copy()
    if history.ndim != 3 or history.shape[1:] != (dynamics.num_agents,4):
        raise ValueError("history shape must be [H,N,4]")
    if cycles < 1:
        raise ValueError("cycles must be positive")
    observed_length = len(history)
    states,controls,weights,times,converged = [history[-1].ravel()],[],[],[],[]
    iterations = np.empty(cycles,dtype=int)
    residual_history = np.full((cycles,config.outer_iterations),np.nan)
    initial_controls = None
    transition = dynamics.step if observe is None else observe
    for cycle in range(cycles):
        start = perf_counter()
        omega = np.asarray(infer_weights(history.copy()),float)
        frozen_cost = replace(cost,weights=omega)
        provider = None
        if config.reference_source == "demonstration":
            if library is None:
                raise ValueError("demonstration controller requires a library")
            provider = DemonstrationReference(library,history,cost.goals,cost.routes,config.dt,observed_length)
        solver = ForwardGameSolver(dynamics,frozen_cost,config,lower,upper,
                                   reference_provider=provider,obstacle_distance=obstacle_distance,
                                   envelope_radii=envelope_radii)
        plan = solver.solve(states[-1],initial_controls)
        # Execute the final damped/projected nominal first action, not raw alpha.
        command = plan.controls[0].copy()
        times.append((perf_counter()-start)*1000)
        measured = np.asarray(transition(states[-1].copy(),command.copy()),float).reshape(-1)
        if measured.shape != states[-1].shape or not np.isfinite(measured).all():
            raise ValueError("environment returned an invalid measured state")
        states.append(measured)
        controls.append(command)
        weights.append(omega)
        converged.append(plan.converged)
        iterations[cycle] = len(plan.residual_history)
        residual_history[cycle,:iterations[cycle]] = plan.residual_history
        history = np.concatenate([history,measured.reshape(1,dynamics.num_agents,4)])[-observed_length:]
        initial_controls = np.vstack([plan.controls[1:],plan.controls[-1:]])
    return ClosedLoopResult(np.stack(states),np.stack(controls),np.stack(weights),np.array(times),
                            np.array(converged),iterations,residual_history)
