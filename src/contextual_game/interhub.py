"""InterHub metadata + trajdata cache adapter and portable event format.

Official input is interactive_segments_index.csv and trajdata caches, NOT
an INTERACTION-format tracks.csv. See README.md for the data requirements.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import numpy as np
from .road import DiskRules, Route, rectangle, overlaps
from .splits import allocate_groups


PAPER_DATASETS = {"nuplan_train", "waymo_train", "lyft_train_full", "av2_motion_forecasting"}


def vectorized_map(routes, polygons):
    """Fixed-order geometry tokens [midpoint x,y,tangent x,y]."""
    tokens = []
    for route in routes:
        delta = np.diff(route.points,axis=0)
        tokens.append(np.column_stack([(route.points[:-1]+route.points[1:])/2,
                                       delta/np.linalg.norm(delta,axis=1,keepdims=True)]))
    for polygon in polygons:
        points = np.asarray(polygon)
        starts,ends = points,np.roll(points,-1,axis=0)
        delta = ends-starts
        keep = np.linalg.norm(delta,axis=1) > 0
        tokens.append(np.column_stack([(starts[keep]+ends[keep])/2,
                                       delta[keep]/np.linalg.norm(delta[keep],axis=1,keepdims=True)]))
    return np.concatenate(tokens) if tokens else np.empty((0,4),dtype=float)


@dataclass(frozen=True)
class EventIndex:
    event_id: str
    dataset: str
    folder: str
    scenario_idx: int
    track_ids: tuple
    start: float
    end: float
    group: str
    category: str
    metadata: dict

    def start_frame(self, dt, time_unit):
        if time_unit not in {"frames", "seconds"}:
            raise ValueError("time_unit must explicitly be frames or seconds")
        value = self.start if time_unit == "frames" else self.start/dt
        if value < 0 or not np.isclose(value, round(value), atol=1e-6):
            raise ValueError("start is not aligned to a nonnegative cache frame")
        return int(round(value))


def read_index(path, paper_filter=True):
    result = []
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset","folder","scenario_idx","track_id","start","end","original_scene_id","path_category"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"InterHub index is missing columns: {required-set(reader.fieldnames or [])}")
        for index,row in enumerate(reader):
            ids = tuple(x.strip() for x in row["track_id"].split(";") if x.strip())
            if paper_filter and (row["dataset"] not in PAPER_DATASETS or row["path_category"] not in {"MP","CP"} or len(ids) not in {2,3}):
                continue
            if not row["original_scene_id"].strip() or len(set(ids)) != len(ids) or not ids:
                raise ValueError(f"missing original scene ID or duplicate track ID at CSV row {index+2}")
            start,end = float(row["start"]),float(row["end"])
            if not np.isfinite([start,end]).all() or end < start:
                raise ValueError(f"invalid event interval at CSV row {index+2}")
            result.append(EventIndex(f"event_{index:07d}",row["dataset"],row["folder"],int(row["scenario_idx"]),
                                     ids,start,end,row["dataset"]+":"+row["original_scene_id"],row["path_category"],row))
    return result


@dataclass
class InteractionEvent:
    event_id: str
    group: str
    domain: str
    dt: float
    history_length: int
    states: np.ndarray  # [H+T,N,4]; history ends at H-1, future starts at H
    goals: np.ndarray
    sizes: np.ndarray  # [N,2], full length/width
    routes: tuple
    polygons: tuple
    track_ids: tuple
    metadata: dict
    true_weights: np.ndarray | None = None
    expert_controls: np.ndarray | None = None  # [T,N,2], exact synthetic expert controls

    def __post_init__(self):
        self.states,self.goals,self.sizes = map(lambda x:np.asarray(x,float),(self.states,self.goals,self.sizes))
        if self.states.ndim != 3 or self.states.shape[-1] != 4:
            raise ValueError("states must be [H+T,N,4]")
        n = self.states.shape[1]
        if not 5 <= self.history_length < len(self.states) or not np.isclose(self.dt,.1):
            raise ValueError("paper interface requires dt=0.1 and at least five history samples plus future")
        if self.goals.shape != (n,2) or self.sizes.shape != (n,2) or np.any(self.sizes <= 0):
            raise ValueError("goals/sizes mismatch")
        if self.domain not in {"camp", "rm", "interhub"}:
            raise ValueError("event domain must be camp, rm, or interhub")
        if len(self.track_ids) != n or len(set(self.track_ids)) != n:
            raise ValueError("event requires one unique ID per agent")
        if self.domain == "camp":
            if self.routes or self.polygons:
                raise ValueError("open-space CAMP uses empty routes and polygons")
        elif len(self.routes) != n or not self.polygons:
            raise ValueError("road events require one assigned route per agent and a drivable map")
        if not all(np.isfinite(x).all() for x in (self.states,self.goals,self.sizes)):
            raise ValueError("event has incomplete/nonfinite trajectories")
        if not self.group:
            raise ValueError("scene group required for leakage-free splitting")
        initial = self.states[self.history_length-1]
        if self.domain == "camp":
            if DiskRules(self.sizes).flags(initial)[0].any():
                raise ValueError("initial disk footprints overlap")
        else:
            for i in range(n):
                for j in range(i):
                    if overlaps(rectangle(initial[i],self.sizes[i]),rectangle(initial[j],self.sizes[j])):
                        raise ValueError("initial footprints overlap")
        if self.true_weights is not None:
            self.true_weights = np.asarray(self.true_weights, float)
            dimension = 3 if self.domain == "camp" else 4
            if (self.true_weights.shape != (n, dimension)
                    or not np.isfinite(self.true_weights).all()
                    or np.any(self.true_weights <= 0)
                    or not np.allclose(self.true_weights.sum(axis=1), 1)):
                raise ValueError("true_weights must contain positive normalized weights for this domain")
        if self.expert_controls is not None:
            self.expert_controls = np.asarray(self.expert_controls, float)
            if (self.expert_controls.shape != (len(self.states)-self.history_length, n, 2)
                    or not np.isfinite(self.expert_controls).all()):
                raise ValueError("expert_controls must have shape [evaluation steps,N,2]")

    @property
    def agents(self):
        return self.states.shape[1]

    @property
    def history(self):
        return self.states[:self.history_length].copy()

    @property
    def recorded(self):
        return self.states[self.history_length-1:].copy()

    def map_tokens(self):
        return vectorized_map(self.routes,self.polygons)

    def save(self,path):
        meta = dict(version=1,event_id=self.event_id,group=self.group,domain=self.domain,dt=self.dt,
                    history_length=self.history_length,track_ids=self.track_ids,metadata=self.metadata,
                    routes=[dict(points=r.points.tolist(),half_width=r.half_width) for r in self.routes],
                    polygons=[np.asarray(p).tolist() for p in self.polygons])
        arrays = dict(states=self.states, goals=self.goals, sizes=self.sizes, metadata=json.dumps(meta))
        if self.true_weights is not None:
            arrays["true_weights"] = self.true_weights
        if self.expert_controls is not None:
            arrays["expert_controls"] = self.expert_controls
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.pop("version") != 1:
                raise ValueError("unsupported event format")
            meta["routes"] = tuple(Route(**r) for r in meta["routes"])
            meta["polygons"] = tuple(np.asarray(p,float) for p in meta["polygons"])
            optional = {key: data[key] for key in ("true_weights", "expert_controls") if key in data}
            return cls(states=data["states"],goals=data["goals"],sizes=data["sizes"],**meta,**optional)


def split_events(events, seed=42, fractions=(.70,.15,.15)):
    """Split original-scene groups, never individual events. Persist the manifest."""
    groups = sorted({e.group for e in events})
    if len(groups) < 3:
        raise ValueError("at least three original scene groups required")
    if len({e.event_id for e in events}) != len(events):
        raise ValueError("duplicate event IDs")
    np.random.Generator(np.random.PCG64(seed)).shuffle(groups)
    n_train,n_val,_ = allocate_groups(len(groups),fractions)
    partitions = {"train":groups[:n_train],"validation":groups[n_train:n_train+n_val],"test":groups[n_train+n_val:]}
    manifest = {"version":1,"kind":"scene_group_split_manifest","seed":seed,
                "fractions":dict(zip(("train","validation","test"),map(float,fractions))),
                "shuffle":"numpy.PCG64","allocation":"largest_remainder_nonempty",
                "groups":partitions,"events":{}}
    for split, selected in partitions.items():
        chosen = set(selected)
        manifest["events"][split] = sorted(e.event_id for e in events if e.group in chosen)
    return manifest


def export_cache_event(row, cache_root, assignment, *, time_unit, history_length=10, future_steps=50):
    """Read raw cache states using InterHub's public trajdata access pattern.

    Assignment is an explicit map-derived route/goal/footprint sidecar; it is
    never generated from the recorded future. All geometry is in cache/world
    coordinates. `folder` selects the cache exactly as in InterHub metadata.
    """
    try:
        from trajdata import UnifiedDataset
        from trajdata.caching.df_cache import DataFrameCache
    except ImportError as exc:
        raise ImportError("install InterHub-compatible trajdata and its source-dataset extras; see README.md") from exc
    root = Path(cache_root).resolve()
    folder = (root / row.folder.replace("\\","/")).resolve()
    if not folder.is_relative_to(root):
        raise ValueError("cache folder must stay under the configured cache root")
    dataset = UnifiedDataset(desired_data=[row.dataset],standardize_data=False,rebuild_cache=False,
                             rebuild_maps=False,centric="scene",verbose=False,cache_location=str(folder),
                             num_workers=0,incl_vector_map=True,data_dirs={row.dataset:" "})
    matches = [i for i,s in enumerate(dataset.scenes()) if s.raw_data_idx == row.scenario_idx]
    if len(matches) != 1:
        raise ValueError("scenario_idx does not uniquely identify a cached scene")
    scene = dataset.get_scene(matches[0])
    if not np.isclose(scene.dt,.1):
        raise ValueError("resample cache to dt=0.1 before export; index time units must be preserved")
    frame = row.start_frame(scene.dt,time_unit)
    start,end = frame-history_length+1,frame+future_steps
    agents = {a.name:a for a in scene.agents}
    cache = DataFrameCache(cache_path=dataset.cache_path,scene=scene)
    columns = cache.column_dict
    states = np.empty((history_length+future_steps,len(row.track_ids),4))
    for j,agent_id in enumerate(row.track_ids):
        agent = agents[agent_id]
        if start < agent.first_timestep or end > agent.last_timestep:
            raise ValueError("incomplete history/future window")
        for t in range(start,end+1):
            value = cache.get_raw_state(agent_id=agent_id,scene_ts=t)
            states[t-start,j] = [value[columns['x']],value[columns['y']],value[columns['heading']],
                                  np.hypot(value[columns['vx']],value[columns['vy']])]
    if tuple(assignment["track_ids"]) != row.track_ids:
        raise ValueError("assignment track order does not match InterHub index")
    if assignment.get("source") != "map_and_observed_history":
        raise ValueError("route assignment source must be map_and_observed_history")
    origin = states[history_length-1,:,:2].mean(axis=0)
    first_route = Route(**assignment["routes"][0])
    tangent = first_route.project(states[history_length-1,0,:2])[2]
    angle = np.arctan2(tangent[1],tangent[0])
    c,s = np.cos(angle),np.sin(angle)
    rotation = np.array([[c,-s],[s,c]])
    transform = lambda points:(np.asarray(points)-origin) @ rotation
    states[:,:,:2] = transform(states[:,:,:2])
    states[:,:,2] = (states[:,:,2]-angle+np.pi)%(2*np.pi)-np.pi
    routes = tuple(Route(transform(r["points"]),r["half_width"]) for r in assignment["routes"])
    polygons = tuple(transform(p) for p in assignment["polygons"])
    return InteractionEvent(row.event_id,row.group,"interhub",float(scene.dt),history_length,states,
                            transform(assignment["goals"]),assignment["sizes"],routes,polygons,row.track_ids,
                            {"interhub":row.metadata,"time_unit":time_unit,"query_frame":frame,
                             "world_origin":origin.tolist(),"world_heading":float(angle),
                             "assignment_source":assignment["source"]})
