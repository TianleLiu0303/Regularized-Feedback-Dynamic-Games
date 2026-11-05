"""Load a manifest split and select a scene family, agent count, or event IDs."""
from pathlib import Path
import json
import numpy as np

from .interhub import InteractionEvent, vectorized_map
from .road import Route
from .splits import validate_manifest


def map_padding_capacity(directory, manifest, domain):
    """Fix MLP input capacity from known maps before training.

    Read only the geometry metadata of the declared cohort, including held-out
    maps. No trajectory, action, label, or normalization statistic is read or
    fitted. The resulting single capacity is saved in the trained checkpoint.
    """
    validate_manifest(manifest)
    capacity = 0
    for split,event_ids in manifest["events"].items():
        for event_id in event_ids:
            if Path(event_id).name != event_id or "/" in event_id or "\\" in event_id:
                raise ValueError("event IDs must be file basenames")
            path = Path(directory)/f"{event_id}.npz"
            if not path.exists():
                raise ValueError(f"map metadata missing for {event_id}; provide the declared cohort "
                                 "or set --map-capacity before KLGame training")
            with np.load(path,allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"]))
            if meta["event_id"] != event_id or meta["group"] not in manifest["groups"][split]:
                raise ValueError(f"map event ID/group does not match the manifest: {event_id}")
            if meta["domain"] == domain:
                routes = tuple(Route(**route) for route in meta["routes"])
                capacity = max(capacity,len(vectorized_map(routes,meta["polygons"])))
    return capacity


def select_events(directory, manifest, split, scene, *, agents=None, categories=None, event_ids=None):
    validate_manifest(manifest)
    if categories and scene != "interhub":
        raise ValueError("MP/CP filters apply only to InterHub")
    directory = Path(directory)
    expected = manifest["events"][split]
    groups = set(manifest["groups"][split])
    selected = []
    requested = set(event_ids or [])
    for event_id in expected:
        if requested and event_id not in requested:
            continue
        if Path(event_id).name != event_id or "/" in event_id or "\\" in event_id:
            raise ValueError("event IDs must be file basenames")
        event = InteractionEvent.load(directory / f"{event_id}.npz")
        if event.event_id != event_id or event.group not in groups:
            raise ValueError(f"event ID/group does not match the manifest: {event_id}")
        if event.domain != scene or (agents and event.agents not in agents):
            continue
        category = event.metadata.get("path_category", event.metadata.get("interhub", {}).get("path_category"))
        if categories and category not in categories:
            continue
        selected.append(event)
    if requested and requested != {event.event_id for event in selected}:
        raise ValueError("requested events are absent from this split or excluded by the scene filters")
    if not selected:
        raise ValueError("no events match the selected scene and filters")
    if len({event.history_length for event in selected}) != 1:
        raise ValueError("selected events must have a common history length")
    return selected
