"""Train contextual objective weights and prepare a training-only reference library."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from contextual_game.pipeline import build_libraries, training_examples
from contextual_game.selection import select_events, map_padding_capacity
from contextual_game.splits import load_mapping, write_mapping


# Edit these defaults or override them on the command line.
# Controls are [yaw rate (rad/s), acceleration (m/s²)].
SCENE_DEFAULTS = {
    "camp": {"lower": (-1.2, -2.), "upper": (1.2, 2.), "features": 3},
    "rm": {"lower": (-.3, -3.), "upper": (.3, 2.), "features": 4},
    "interhub": {"lower": (-.3, -3.), "upper": (.3, 2.), "features": 4},
}
TRAINING_DEFAULTS = {
    "learning_rate": 1e-3,
    "max_updates": 1000,
    "tolerance": 1e-5,
    "temperature": 1.,
    "gradient_chunk_size": 32,  # exact full-batch gradient accumulated in chunks
    "history_hidden": 128,
    "heads": 4,
    "key_dim": 64,
    "decoder_hidden": 64,
    "device": "cpu",
}
DATA_DEFAULTS = {
    "history_length": 10,
    "proposals": 30,
    "proposal_noise": (.2, 1.),
    "control_diagonal": (1., .25),  # own-agent R; all other-agent R blocks are zero
}
REFERENCE_DEFAULTS = {
    "neighbors": 20,
    "bandwidth": None,  # sqrt(number of active standardized descriptor dimensions)
    "covariance_std": (.05, .10),  # squared to form the covariance floor
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene", choices=SCENE_DEFAULTS, default="interhub")
    parser.add_argument("--method", choices=("main", "klgame"), default="main")
    parser.add_argument("--events", type=Path, help="Defaults to data/<scene>/events")
    parser.add_argument("--manifest", type=Path, help="Defaults to data/<scene>/split.yaml")
    parser.add_argument("--output", type=Path, help="Defaults to outputs/<scene>/<method>/model")
    parser.add_argument("--agents", type=int, nargs="+", help="Select agent counts, e.g. 2 4 20")
    parser.add_argument("--category", choices=("MP", "CP"), nargs="+", help="InterHub path types only")
    parser.add_argument("--event-ids", nargs="+", help="Select specific training event IDs")
    parser.add_argument("--seed", type=int, help="Training/proposal seed; defaults to the manifest seed")
    for name, value in TRAINING_DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value,
                            help=name.replace("_", " "))
    parser.add_argument("--map-capacity", type=int,
                        help="KLGame map capacity; default scans only map geometry in the declared cohort")
    parser.add_argument("--history-length", type=int, default=DATA_DEFAULTS["history_length"])
    parser.add_argument("--proposals", type=int, default=DATA_DEFAULTS["proposals"])
    parser.add_argument("--proposal-noise", type=float, nargs=2, default=DATA_DEFAULTS["proposal_noise"])
    parser.add_argument("--control-diagonal", type=float, nargs=2, default=DATA_DEFAULTS["control_diagonal"])
    parser.add_argument("--control-lower", type=float, nargs=2, help="Override scene lower control bounds")
    parser.add_argument("--control-upper", type=float, nargs=2, help="Override scene upper control bounds")
    parser.add_argument("--neighbors", type=int, default=REFERENCE_DEFAULTS["neighbors"])
    parser.add_argument("--bandwidth", type=float, default=REFERENCE_DEFAULTS["bandwidth"])
    parser.add_argument("--covariance-std", type=float, nargs=2, default=REFERENCE_DEFAULTS["covariance_std"])
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    scene = SCENE_DEFAULTS[args.scene]
    event_dir = args.events or Path("data") / args.scene / "events"
    manifest_path = args.manifest or Path("data") / args.scene / "split.yaml"
    output = args.output or Path("outputs") / args.scene / args.method / "model"
    if (output / "objective.pt").exists():
        raise FileExistsError("objective.pt already exists; choose a new --output to preserve the previous model")
    lower = np.asarray(args.control_lower if args.control_lower is not None else scene["lower"])
    upper = np.asarray(args.control_upper if args.control_upper is not None else scene["upper"])
    if (not np.isfinite([lower, upper]).all() or np.any(lower >= upper)
            or not np.isfinite(args.control_diagonal).all() or min(args.control_diagonal) <= 0
            or not np.isfinite(args.proposal_noise).all() or min(args.proposal_noise) < 0
            or args.proposals < 1 or args.log_every < 1 or args.neighbors < 1
            or not np.isfinite(args.covariance_std).all() or min(args.covariance_std) <= 0
            or (args.bandwidth is not None and (not np.isfinite(args.bandwidth) or args.bandwidth <= 0))):
        raise ValueError("invalid control bounds, proposal settings, or reference parameters")

    # 1. Read only the selected training split; never train on evaluation events.
    manifest = load_mapping(manifest_path)
    events = select_events(event_dir, manifest, "train", args.scene,
                           agents=args.agents, categories=args.category, event_ids=args.event_ids)
    if any(event.history_length != args.history_length for event in events):
        raise ValueError("--history-length must match the stored histories; this script does not resample data")
    seed = manifest["seed"] if args.seed is None else args.seed
    map_capacity = args.map_capacity
    if args.method == "klgame" and map_capacity is None:
        map_capacity = map_padding_capacity(event_dir,manifest,args.scene)
    output.mkdir(parents=True, exist_ok=True)
    selection = dict(scene=args.scene, method=args.method, event_ids=[event.event_id for event in events],
                     events=str(event_dir.resolve()), manifest=str(manifest_path.resolve()),
                     agents=args.agents, categories=args.category, history_length=args.history_length,
                     control_lower=lower.tolist(), control_upper=upper.tolist(),
                     control_diagonal=list(args.control_diagonal))

    # 2. Construct proposals once, then reuse their features throughout training.
    print(f"Preparing {len(events)} {args.scene.upper()} training events, {args.proposals} proposals each.", flush=True)
    examples = training_examples(events, manifest, args.proposals, seed,
                                  noise_std=args.proposal_noise, lower_control=lower, upper_control=upper,
                                  control_diagonal=args.control_diagonal)
    candidate_dir = output / "fixed_candidates"
    candidate_dir.mkdir(exist_ok=True)
    for example in examples:
        np.savez_compressed(candidate_dir / (example["event_id"] + ".npz"), **example)
    if examples[0]["features"].shape[-1] != scene["features"]:
        raise ValueError("scene cost dimension does not match the training data")

    # 3. Optimize the shared inverse model and retain its final iterate.
    from contextual_game.inverse.training import TrainingConfig, train
    config = TrainingConfig(**{name: getattr(args, name) for name in TRAINING_DEFAULTS},
                            seed=seed, map_capacity=map_capacity,
                            model_kind="contextual" if args.method == "main" else "mlp")

    def progress(row):
        if row["update"] == 1 or row["update"] % args.log_every == 0:
            print(f"Update {row['update']}: loss={row['loss']:.6g}, delta={row['parameter_delta']:.3g}", flush=True)

    logs = train(examples, config, output / "objective.pt", progress=progress, metadata=selection)
    (output / "training_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")

    # 4. Fit and save the main method's reference library using the same training scenes.
    reference = dict(neighbors=args.neighbors, bandwidth=args.bandwidth,
                     covariance_floor=np.square(args.covariance_std).tolist())
    reference_records = {}
    if args.method == "main":
        library_dir = output / "libraries"
        library_dir.mkdir(exist_ok=True)
        libraries = build_libraries(events, manifest, **reference)
        for (domain, agents), library in libraries.items():
            filename = f"{domain}_{agents}.npz"
            library.save(library_dir / filename)
            reference_records[filename] = dict(samples=len(library.actions), neighbors=library.neighbors,
                                               bandwidth=library.bandwidth,
                                               covariance_floor=list(library.covariance_floor))
        print(f"Saved {len(libraries)} demonstration libraries to {library_dir}.", flush=True)

    write_mapping(output / "manifest.yaml", manifest)
    write_mapping(output / "training_config.json", dict(
        selection=selection, training=asdict(config), proposals=args.proposals,
        proposal_noise=list(args.proposal_noise), reference=reference if args.method == "main" else None,
        reference_libraries=reference_records))
    print(f"Saved {output / 'objective.pt'} after {len(logs)} updates; final loss={logs[-1]['loss']:.6g}.")


if __name__ == "__main__":
    main()
