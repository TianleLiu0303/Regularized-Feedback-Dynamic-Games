"""Paper metrics: event-average joint errors, footprint events and route progress."""
import numpy as np
from .demonstration import reconstruct_controls
from .pipeline import event_cost, event_rules


def expert_history_weight_trace(event, infer_weights):
    """Score-only inference on common recorded rolling histories, including the last.

    Index zero uses the initial history; the last index uses the final recorded
    observation. These weights do not enter the executed closed-loop planner.
    """
    h = event.history_length
    return np.stack([infer_weights(event.states[end-h:end].copy())
                     for end in range(h, len(event.states)+1)])


def evaluation_controls(event, states):
    """Use the same causal reconstruction for executed and recorded motion."""
    trajectory = np.asarray(states, dtype=float).reshape(-1, event.agents, 4)
    if trajectory.shape != event.recorded.shape or not np.allclose(trajectory[0], event.history[-1]):
        raise ValueError("control scoring requires the full window from the common observed initial state")
    observed_and_executed = np.concatenate([event.history, trajectory[1:]])
    return reconstruct_controls(observed_and_executed, event.dt)[event.history_length-1:]


def event_metrics(event, states, controls, cycle_ms=None, *, estimated_weights=None,
                  control_diagonal=(1., .25), effort_diagonal=(30., 1.)):
    predicted = np.asarray(states).reshape(-1,event.agents,4)
    recorded = event.recorded
    controls = np.asarray(controls).reshape(-1,event.agents,2)
    if predicted.shape != recorded.shape or len(controls) != len(recorded)-1:
        raise ValueError("metrics require the full common evaluation window, including unsuccessful encounters")
    if not np.isfinite(predicted).all() or not np.isfinite(controls).all():
        raise ValueError("cannot score nonfinite trajectories or controls")
    error = np.linalg.norm(predicted[1:,:,:2]-recorded[1:,:,:2],axis=-1)
    rules = event_rules(event)
    flags = [rules.flags(x) for x in predicted[1:]]
    result = dict(event_id=event.event_id,group=event.group,scene=event.domain,
                  collision=bool(any(c.any() for c,o in flags)))
    if event.domain == "interhub":
        progress = lambda trajectory:sum(r.project(trajectory[-1,i,:2])[0]-r.project(trajectory[0,i,:2])[0]
                                        for i,r in enumerate(event.routes))
        denominator = progress(recorded)
        if denominator <= 0:
            raise ValueError("relative progress requires positive recorded joint route progress")
        scored_controls = evaluation_controls(event, predicted)
        result.update(jADE=float(error.mean()),jFDE=float(error[-1].mean()),
                      offroad=bool(any(o.any() for c,o in flags)),
                      E_u=float(np.mean(np.sum(scored_controls**2*np.asarray(effort_diagonal),axis=-1))),
                      P_rel=float(progress(predicted)/denominator))
    else:
        if event.true_weights is None or event.expert_controls is None or estimated_weights is None:
            raise ValueError("synthetic metrics require true_weights, exact expert_controls, "
                             "and weights inferred on the final recorded expert history")
        estimate = np.asarray(estimated_weights, float)
        truth = event.true_weights
        if estimate.shape != truth.shape or not np.isfinite(estimate).all() or np.any(np.linalg.norm(estimate,axis=1) == 0):
            raise ValueError("estimated weights do not match the synthetic ground truth")
        cosine = np.sum(truth*estimate,axis=1)/(np.linalg.norm(truth,axis=1)*np.linalg.norm(estimate,axis=1))
        cost = event_cost(event, truth, control_diagonal)
        true_cost = np.array([cost.trajectory_cost(recorded.reshape(len(recorded),-1),
                                                  event.expert_controls.reshape(len(controls),-1),i)
                              for i in range(event.agents)])
        executed_cost = np.array([cost.trajectory_cost(predicted.reshape(len(predicted),-1),
                                                      controls.reshape(len(controls),-1),i)
                                  for i in range(event.agents)])
        result.update(D_tra=float(error.mean()),D_par=float(1-np.clip(cosine,-1,1).mean()),
                      D_par_protocol="final_recorded_expert_history",
                      D_cost=float(np.abs(true_cost-executed_cost).mean()))
    if cycle_ms is not None:
        result["cycle_ms"] = np.asarray(cycle_ms).tolist()
    return result


def aggregate(rows):
    if not rows:
        raise ValueError("no evaluated events")
    scenes = {r.get("scene", "interhub") for r in rows}
    if len(scenes) != 1:
        raise ValueError("aggregate one scene family at a time")
    keys = ("jADE","jFDE","E_u","P_rel") if scenes == {"interhub"} else ("D_cost","D_tra","D_par")
    result = {k:float(np.mean([r[k] for r in rows])) for k in keys}
    if scenes != {"interhub"}:
        result.update({k+"_std":float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else None for k in keys})
    result.update(events=len(rows),collision_count=sum(r["collision"] for r in rows),
                  CR=100*float(np.mean([r["collision"] for r in rows])))
    if scenes == {"interhub"}:
        result["OR"] = 100*float(np.mean([r["offroad"] for r in rows]))
    timing = [t for r in rows for t in r.get("cycle_ms",[])]
    if timing:
        result.update(mean_cycle_ms=float(np.mean(timing)),p95_cycle_ms=float(np.percentile(timing,95)))
    return result
