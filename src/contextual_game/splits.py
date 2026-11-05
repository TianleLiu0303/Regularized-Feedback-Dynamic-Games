"""Scene-group split plans and persisted train/validation/test manifests."""
from pathlib import Path
import json
import numpy as np

SPLITS = ("train", "validation", "test")


def load_mapping(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            value = yaml.safe_load(stream)
        else:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def write_mapping(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def validate_plan(plan):
    if plan.get("version") != 1 or plan.get("kind") != "scene_group_split_plan":
        raise ValueError("expected a version-1 scene_group_split_plan")
    if plan.get("group_key") != ["dataset", "original_scene_id"]:
        raise ValueError("InterHub groups must use dataset and original_scene_id")
    if plan.get("shuffle") != "numpy.PCG64" or plan.get("allocation") != "largest_remainder_nonempty":
        raise ValueError("unsupported shuffle/allocation rule")
    if type(plan.get("seed")) is not int or plan["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    fractions = plan.get("fractions", {})
    if set(fractions) != set(SPLITS):
        raise ValueError("fractions must specify train, validation and test")
    values = np.array([fractions[s] for s in SPLITS], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0) or not np.isclose(values.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("split fractions must be positive and sum to one")
    return plan["seed"], values


def allocate_groups(count, fractions):
    """Largest remainder, with at least one group in each nonzero partition."""
    fractions = np.asarray(fractions, float)
    if count < 3 or fractions.shape != (3,) or np.any(fractions <= 0) or not np.isfinite(fractions).all() or not np.isclose(fractions.sum(),1.,atol=1e-12,rtol=0):
        raise ValueError("need at least three groups and three positive fractions summing to one")
    quota = count * fractions
    allocated = np.floor(quota).astype(int)
    order = np.argsort(-(quota-allocated), kind="stable")
    allocated[order[:count-int(allocated.sum())]] += 1
    for empty in np.flatnonzero(allocated == 0):
        donor = int(np.argmax(allocated))
        allocated[donor] -= 1
        allocated[empty] += 1
    return allocated.tolist()


def validate_manifest(manifest):
    if manifest.get("kind") == "scene_group_split_plan":
        raise ValueError("this YAML is a split plan, not a populated manifest; run the split command after supplying data")
    if manifest.get("version") != 1:
        raise ValueError("unsupported manifest version")
    if type(manifest.get("seed")) is not int or manifest["seed"] < 0:
        raise ValueError("manifest seed must be a nonnegative integer")
    for field in ("groups", "events"):
        partitions = manifest.get(field, {})
        if set(partitions) != set(SPLITS):
            raise ValueError(f"manifest {field} must contain all three splits")
        seen = set()
        for split in SPLITS:
            items = partitions[split]
            if not isinstance(items, list) or not items or not all(isinstance(x,str) and x for x in items):
                raise ValueError(f"manifest {field}.{split} must be a nonempty string list")
            if len(set(items)) != len(items) or seen.intersection(items):
                raise ValueError(f"duplicate/overlapping {field} in manifest")
            seen.update(items)
    return manifest
