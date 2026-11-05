"""Build fixed inverse candidates and training-only retrieval libraries."""
import numpy as np
from .costs import GameCost
from .dynamics import UnicycleDynamics
from .demonstration import descriptors,reconstruct_controls,DemonstrationLibrary
from .inverse.candidates import generate_candidate_set,precompute_features
from .road import DiskRules, RoadRules


def control_bounds(domain):
    if domain == "camp":
        return np.array([-1.2, -2.]), np.array([1.2, 2.])
    if domain in {"rm", "interhub"}:
        return np.array([-.3, -3.]), np.array([.3, 2.])
    raise ValueError("unsupported scene domain")


def event_rules(event):
    return DiskRules(event.sizes) if event.domain == "camp" else RoadRules(event.sizes, event.polygons)


def event_envelope_radii(event):
    """Fixed disk radii for CAMP and circumscribed-rectangle radii for road scenes."""
    if event.domain == "camp":
        return DiskRules(event.sizes).radii
    return np.linalg.norm(event.sizes, axis=1) / 2


def event_cost(event, weights=None, control_diagonal=(1., .25)):
    n = event.agents
    metric = np.zeros((n,n,2,2))
    for i in range(n):
        metric[i,i] = np.diag(control_diagonal)
    dimension = 3 if event.domain == "camp" else 4
    if weights is None:
        weights = np.full((n,dimension),1/dimension)
    return GameCost(event.goals,weights,metric,
                    lane_half_width=None if event.domain == "camp" else event.routes[0].half_width,
                    routes=event.routes or None)


def training_examples(events, manifest, proposals=30, seed=42, *, noise_std=(.2, 1.),
                      lower_control=None, upper_control=None, control_diagonal=(1., .25)):
    """One fixed joint candidate set per scene/event, reused throughout training."""
    rng = np.random.default_rng(seed)
    allowed = set(manifest["groups"]["train"])
    examples = []
    for event in events:
        if event.group not in allowed:
            raise ValueError("training_examples received a non-training event")
        controls = (event.expert_controls if event.expert_controls is not None else
                    reconstruct_controls(event.states,event.dt)[event.history_length-1:])
        controls = controls.reshape(-1,2*event.agents)
        expert = event.recorded.reshape(-1,4*event.agents)
        lower, upper = control_bounds(event.domain)
        lower = lower if lower_control is None else np.asarray(lower_control, float)
        upper = upper if upper_control is None else np.asarray(upper_control, float)
        candidates = generate_candidate_set(UnicycleDynamics(event.agents,event.dt),expert[0],controls,
                                            proposals,np.asarray(noise_std),lower,upper,rng,
                                            expert_states=expert)
        features = precompute_features(candidates,event_cost(event, control_diagonal=control_diagonal))
        examples.append(dict(histories=event.history.transpose(1,0,2),map_tokens=event.map_tokens(),
                             features=features,event_id=event.event_id,group=event.group))
    if not examples:
        raise ValueError("no training events")
    return examples


def build_libraries(events, manifest, neighbors=20, *, bandwidth=None, covariance_floor=(.05**2, .10**2)):
    allowed = set(manifest["groups"]["train"])
    partitions = {}
    for event in events:
        if event.group not in allowed:
            raise ValueError("library build received a non-training event")
        key = (event.domain,event.agents)
        contexts,actions,groups = partitions.setdefault(key,([],[],[]))
        controls = reconstruct_controls(event.states,event.dt)
        if event.expert_controls is not None:
            controls[event.history_length-1:] = event.expert_controls
        for t in range(event.history_length-1,len(event.states)-1):
            h = event.states[t-event.history_length+1:t+1]
            for i,context in enumerate(descriptors(h,event.goals,event.routes or None,event.dt)):
                contexts.append(context)
                actions.append(controls[t,i])
                groups.append(event.group)
    return {key:DemonstrationLibrary.fit(z,u,g,domain=key[0],agents=key[1],allowed_groups=allowed,
                                        neighbors=neighbors,bandwidth=bandwidth,covariance_floor=covariance_floor)
            for key,(z,u,g) in partitions.items()}
