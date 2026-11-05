"""Causal descriptors and direct Gaussian action references (no cost weights)."""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from .reference_policy import GaussianReference

DESCRIPTOR_VERSION = 2  # Eq. (7): ego frame at each history sample.


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


@lru_cache(maxsize=5)
def _smoothing_coefficients(samples):
    design = np.vander(np.arange(1-samples,1,dtype=float),
                       min(2,samples-1)+1, increasing=True)
    coefficients = np.linalg.pinv(design)[0]
    coefficients.setflags(write=False)
    return coefficients


def smooth_causal(values):
    """End-point Savitzky--Golay, degree 2, five samples; no future access.

    A startup prefix uses all available samples and degree min(2,n-1).
    Consumers requiring exactly five past samples must exclude that prefix.
    """
    values = np.asarray(values, float)
    result = np.empty_like(values)
    for t in range(len(values)):
        window = values[max(0,t-4):t+1]
        result[t] = _smoothing_coefficients(len(window)) @ window
    return result


def reconstruct_controls(states, dt=0.1):
    """Output [time-1,N,2]; label at t describes completed transition t->t+1."""
    states = np.asarray(states, float)
    if states.ndim != 3 or states.shape[-1] != 4 or dt <= 0:
        raise ValueError("states must be [time,N,4] with positive dt")
    heading = smooth_causal(np.unwrap(states[:,:,2], axis=0))
    speed = smooth_causal(states[:,:,3])
    return np.stack([wrap(np.diff(heading, axis=0))/dt, np.diff(speed, axis=0)/dt], axis=-1)


def descriptor(history, target, goals, routes=None, dt=0.1):
    """H states through query time; neighbors sorted once by current distance/bearing.

    Historical relative motion uses the ego frame at each history sample.
    Goals and road geometry use the current ego frame.
    Relative velocities use speed/heading; headings enter as sine/cosine.
    """
    h = np.asarray(history,float)
    if h.ndim != 3 or h.shape[2] != 4 or len(h) < 2:
        raise ValueError("history must have shape [H>=2,N,4]")
    recent = reconstruct_controls(h,dt)[-1]
    velocity = h[:,:,3,None] * np.stack([np.cos(h[:,:,2]),np.sin(h[:,:,2])],axis=-1)
    return _descriptor(h,target,goals,routes,recent,velocity)


def descriptors(history, goals, routes=None, dt=0.1):
    """All target descriptors, sharing only target-independent history work."""
    h = np.asarray(history,float)
    if h.ndim != 3 or h.shape[2] != 4 or len(h) < 2:
        raise ValueError("history must have shape [H>=2,N,4]")
    recent = reconstruct_controls(h,dt)[-1]
    velocity = h[:,:,3,None] * np.stack([np.cos(h[:,:,2]),np.sin(h[:,:,2])],axis=-1)
    return [_descriptor(h,i,goals,routes,recent,velocity) for i in range(h.shape[1])]


def _descriptor(h,target,goals,routes,recent,velocity):
    ego = h[-1,target]
    c,s = np.cos(ego[2]), np.sin(ego[2])
    rotation = np.array([[c,-s],[s,c]])
    history_cos = np.cos(h[:,target,2])
    history_sin = np.sin(h[:,target,2])
    history_rotation = np.stack([
        np.stack([history_cos,-history_sin],axis=-1),
        np.stack([history_sin,history_cos],axis=-1),
    ],axis=1)
    pos = np.einsum('tni,tij->tnj', h[:,:,:2]-h[:,target:target+1,:2], history_rotation)
    relvel = np.einsum('tni,tij->tnj', velocity-velocity[:,target:target+1], history_rotation)
    others = [j for j in range(h.shape[1]) if j != target]
    ranges = np.linalg.norm(pos[-1],axis=1)
    bearings = np.arctan2(pos[-1,:,1],pos[-1,:,0])
    # Treat roundoff-level range differences as ties, preserving the bearing
    # tie-break under a common rotation/translation of the world frame.
    others.sort(key=lambda j:(round(float(ranges[j]),12), round(float(bearings[j]),12), j))
    parts = [h[:,target,3], recent[target], (np.asarray(goals)[target]-ego[:2]) @ rotation]
    angle = h[:,others,2]-h[:,target:target+1,2]
    neighbor_features = np.concatenate([pos[:,others],relvel[:,others],
                                       np.sin(angle)[...,None],np.cos(angle)[...,None]],axis=-1)
    parts.append(neighbor_features.transpose(1,0,2).ravel())
    if routes is not None:
        route = routes[target]
        arc,lateral,tangent = route.project(ego[:2])
        tangent_local = tangent @ rotation
        samples = (route.sample(arc+2*np.arange(1,6))-ego[:2]) @ rotation
        parts += [np.array([lateral,*tangent_local,route.half_width]),samples.ravel()]
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite reference descriptor")
    return result


@dataclass
class DemonstrationLibrary:
    standardized: np.ndarray
    actions: np.ndarray
    source_groups: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray
    domain: str
    agents: int
    neighbors: int = 20
    bandwidth: float | None = None
    covariance_floor: tuple = (0.05**2, 0.10**2)

    @classmethod
    def fit(cls, contexts, actions, source_groups, *, domain, agents, allowed_groups, **kwargs):
        z, u, groups = np.asarray(contexts,float), np.asarray(actions,float), np.asarray(source_groups,str)
        if z.ndim != 2 or len(z) == 0 or u.shape != (len(z),2) or groups.shape != (len(z),):
            raise ValueError("invalid demonstration arrays")
        if not set(groups).issubset(set(allowed_groups)):
            raise ValueError("demonstration library contains non-training scene groups")
        if not np.isfinite(z).all() or not np.isfinite(u).all():
            raise ValueError("nonfinite demonstration data")
        mean,scale = z.mean(axis=0), z.std(axis=0)
        active = scale > 1e-12
        # The all-constant case is a valid unconditional action distribution.
        scale = np.where(active,scale,1.)
        result = cls((z[:,active]-mean[active])/scale[active],u,groups,mean,scale,active,domain,agents,**kwargs)
        if result.neighbors < 1 or min(result.covariance_floor) <= 0:
            raise ValueError("invalid neighbor count/covariance floor")
        if result.bandwidth is None:
            result.bandwidth = float(np.sqrt(max(1,active.sum())))
        if result.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        return result

    def query(self, context):
        context = np.asarray(context,float)
        if context.shape != self.mean.shape or not np.isfinite(context).all():
            raise ValueError("reference descriptor shape/value mismatch")
        q = (context[self.active]-self.mean[self.active])/self.scale[self.active]
        if not hasattr(self,"_squared_norms"):
            self._squared_norms = np.einsum('ij,ij->i',self.standardized,self.standardized)
        # Avoid allocating a library-sized [samples,descriptor_dim] temporary
        # on every horizon query. Search remains exact, not approximate.
        distance = np.maximum(self._squared_norms+q@q-2*(self.standardized@q),0.)
        count = min(self.neighbors,len(distance))
        cutoff = np.partition(distance,count-1)[count-1]
        closer = np.flatnonzero(distance < cutoff)
        tied = np.flatnonzero(distance == cutoff)[:count-len(closer)]
        indices = np.r_[closer,tied]
        indices = indices[np.lexsort((indices,distance[indices]))]
        logits = -distance[indices]/(2*self.bandwidth**2)
        weight = np.exp(logits-logits.max())
        weight /= weight.sum()
        samples = self.actions[indices]
        mean = weight @ samples
        deviations = samples-mean
        covariance = (deviations.T*weight) @ deviations + np.diag(self.covariance_floor)
        return mean,covariance

    def save(self, path):
        metadata = dict(descriptor_version=DESCRIPTOR_VERSION,
                        domain=self.domain,agents=self.agents,neighbors=self.neighbors,
                        bandwidth=self.bandwidth,covariance_floor=self.covariance_floor)
        np.savez_compressed(path,standardized=self.standardized,actions=self.actions,source_groups=self.source_groups,
                            mean=self.mean,scale=self.scale,active=self.active,metadata=json.dumps(metadata))

    @classmethod
    def load(cls,path):
        with np.load(Path(path),allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.pop("descriptor_version", None) != DESCRIPTOR_VERSION:
                raise ValueError("rebuild demonstration libraries with per-sample ego frames (Eq. 7); "
                                 "legacy descriptors and normalization statistics are incompatible")
            return cls(**{k:data[k] for k in ("standardized","actions","source_groups","mean","scale","active")},
                       **metadata)


class DemonstrationReference:
    def __init__(self, library, observed_history, goals, routes=None, dt=0.1, history_length=10):
        self.library, self.history, self.goals = library,np.asarray(observed_history,float),goals
        self.routes, self.dt, self.history_length = routes,dt,history_length
        if self.history.shape[1] != library.agents or len(self.history) < history_length:
            raise ValueError("library agent count/history length mismatch")

    def __call__(self, states, controls):
        nominal = states.reshape(len(states),self.library.agents,4)
        if not np.allclose(nominal[0], self.history[-1]):
            raise ValueError("nominal trajectory must start at the latest observation")
        result = []
        combined = np.concatenate([self.history,nominal[1:]])
        for t in range(len(controls)):
            # Append only nominal future states through t, never recorded futures.
            end = len(self.history)+t
            h = combined[end-self.history_length:end]
            row = []
            for i,context in enumerate(descriptors(h,self.goals,self.routes,self.dt)):
                mean,cov = self.library.query(context)
                row.append(GaussianReference(mean-controls[t,2*i:2*i+2],cov))
            result.append(row)
        return result
