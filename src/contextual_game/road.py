"""Polyline routes and footprint rules; coordinates and distances in metres."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Route:
    points: np.ndarray
    half_width: float

    def __post_init__(self):
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("route must have shape [L>=2,2]")
        if not np.isfinite(points).all() or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0):
            raise ValueError("route points must be finite and consecutive points distinct")
        if self.half_width <= 0:
            raise ValueError("route half width must be positive")
        object.__setattr__(self, "points", points)

    def project(self, position):
        delta = np.diff(self.points, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        fraction = np.clip(np.sum((position-self.points[:-1])*delta, axis=1)/lengths**2, 0, 1)
        closest = self.points[:-1] + fraction[:, None]*delta
        segment = int(np.argmin(np.linalg.norm(position-closest, axis=1)))
        tangent = delta[segment]/lengths[segment]
        lateral = np.dot(position-closest[segment], [-tangent[1], tangent[0]])
        arc = lengths[:segment].sum() + fraction[segment]*lengths[segment]
        return float(arc), float(lateral), tangent

    def sample(self, distances):
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        return np.column_stack([np.interp(distances, arc, self.points[:, d]) for d in range(2)])


def rectangle(state, size):
    """Counterclockwise footprint vertices; size=[length,width]."""
    x, y, heading = np.asarray(state)[:3]
    rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    corners = np.array([[-1,-1], [1,-1], [1,1], [-1,1]]) * np.asarray(size)/2
    return corners @ rotation.T + [x,y]


def overlaps(first, second):
    """Separating-axis test for convex rectangles (contact counts as collision)."""
    for polygon in (first, second):
        for edge in np.diff(np.vstack([polygon, polygon[0]]), axis=0):
            axis = np.array([-edge[1], edge[0]])
            a, b = first @ axis, second @ axis
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


class DiskRules:
    """Open-space CAMP disks; the placement square is not a road boundary."""
    def __init__(self, diameters):
        sizes = np.asarray(diameters, float)
        if sizes.ndim != 2 or sizes.shape[1] != 2 or not np.allclose(sizes[:, 0], sizes[:, 1]):
            raise ValueError("CAMP sizes must contain equal x/y diameters")
        if not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("disk diameters must be finite and positive")
        self.radii = sizes[:, 0] / 2

    def flags(self, state):
        positions = np.asarray(state).reshape(len(self.radii), 4)[:, :2]
        distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        collisions = distance < self.radii[:, None] + self.radii[None, :]
        np.fill_diagonal(collisions, False)
        return collisions.any(axis=1), np.zeros(len(self.radii), dtype=bool)

    def count(self, state, agent):
        return int(self.flags(state)[0][agent])


class RoadRules:
    """Polygon union coverage of complete footprints, not a centre/lane proxy."""
    def __init__(self, sizes, polygons):
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union
        except ImportError as exc:
            raise ImportError('road evaluation requires pip install -e ".[interhub]"') from exc
        self.Polygon = Polygon
        self.sizes = np.asarray(sizes, float)
        if not polygons:
            raise ValueError("road evaluation needs a drivable polygon map")
        shapes = [Polygon(p) for p in polygons]
        if any(not p.is_valid or p.area <= 0 for p in shapes):
            raise ValueError("invalid drivable polygon")
        self.drivable = unary_union(shapes)

    def flags(self, state):
        state = np.asarray(state).reshape(-1,4)
        footprints = [rectangle(x, size) for x,size in zip(state,self.sizes,strict=True)]
        collisions = np.zeros(len(state), bool)
        for i in range(len(state)):
            for j in range(i):
                if overlaps(footprints[i],footprints[j]):
                    collisions[i] = collisions[j] = True
        offroad = np.array([not self.drivable.covers(self.Polygon(p)) for p in footprints])
        return collisions, offroad

    def count(self, state, agent):
        coll, offroad = self.flags(state)
        return int(coll[agent]) + int(offroad[agent])
