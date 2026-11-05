"""Prepare interaction data, train contextual costs, and evaluate feedback games."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from .config import PlannerConfig
from .controller import run_closed_loop
from .demonstration import DemonstrationLibrary
from .dynamics import UnicycleDynamics
from .interhub import InteractionEvent, read_index, split_events, export_cache_event
from .metrics import event_metrics, aggregate, evaluation_controls, expert_history_weight_trace
from .pipeline import build_libraries, training_examples, event_cost, control_bounds, event_envelope_radii
from .splits import load_mapping, write_mapping, validate_plan, validate_manifest
from .selection import map_padding_capacity
from .experiment import VARIANTS, configure_variant, variant_name, prepare_results

MODEL_KINDS = {"main": "contextual", "klgame": "mlp"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_events(directory):
    events = [InteractionEvent.load(path) for path in sorted(Path(directory).glob("*.npz"))]
    if not events:
        raise ValueError("no normalized events in directory")
    if len({event.event_id for event in events}) != len(events):
        raise ValueError("duplicate event IDs")
    return events


def select_events(events, manifest, split):
    validate_manifest(manifest)
    expected = set(manifest["events"][split])
    selected = [event for event in events if event.event_id in expected]
    if {event.event_id for event in selected} != expected:
        raise ValueError("event directory is incomplete for the requested split")
    if any(event.group not in manifest["groups"][split] for event in selected):
        raise ValueError("manifest event/group mismatch")
    return selected


def build_parser(subcommand=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="Convert official InterHub caches to event files")
    export.add_argument("--index", type=Path, required=True)
    export.add_argument("--cache-root", type=Path, required=True)
    export.add_argument("--assignments", type=Path, required=True)
    export.add_argument("--time-unit", choices=["frames", "seconds"], required=True)
    export.add_argument("--output", type=Path, required=True)

    split = commands.add_parser("split", help="Materialize a scene-disjoint YAML manifest")
    split.add_argument("--events", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, help="Override the recipe seed")
    split.add_argument("--config", type=Path, help="YAML split recipe")

    for name in ("library", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--events", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "library":
            command.add_argument("--neighbors", type=int, default=20)
            continue
        command.add_argument("--device", default="cpu")
        command.add_argument("--method", choices=MODEL_KINDS, default="main")
        if name == "train":
            command.add_argument("--max-updates", type=int, default=1000)
            command.add_argument("--proposals", type=int, default=30)
            command.add_argument("--map-capacity", type=int,
                                 help="KLGame capacity; default reads declared cohort map geometry only")
        else:
            command.add_argument("--variant", choices=VARIANTS,
                                 help="Matched ablation using the contextual checkpoint")
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--libraries", type=Path, help="Required for demonstration references")
            command.add_argument("--config", type=Path, help="Planner configuration; defaults to the selected method")
            command.add_argument("--split", choices=["validation", "test"], default="test")
    if subcommand is not None:
        parser = commands.choices[subcommand]
        parser.prog = Path(sys.argv[0]).name
        parser.set_defaults(command=subcommand)
    return parser


def planner_config(args, scenario="interhub"):
    """Resolve method defaults and matched reference/coefficient ablations."""
    if args.config is not None:
        config = PlannerConfig.from_json(args.config)
    else:
        config = PlannerConfig(scenario=scenario, horizon=60 if scenario == "camp" else 50)
        if args.method == "klgame":
            config = replace(config, reference_source="cost", coefficient_mode="fixed")
    if config.scenario != scenario:
        raise ValueError("planner configuration does not match the event scene family")
    return configure_variant(config, args.method, getattr(args, "variant", None))


def export_events(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("*.npz")):
        raise ValueError("export requires an empty event directory")
    assignments = load_mapping(args.assignments)
    exported, excluded = [], []
    for row in read_index(args.index):
        try:
            event = export_cache_event(
                row, args.cache_root, assignments[row.event_id], time_unit=args.time_unit
            )
            event.save(args.output / (row.event_id + ".npz"))
            exported.append(row.event_id)
        except (ValueError, KeyError, FileNotFoundError) as error:
            excluded.append(dict(event_id=row.event_id, reason=str(error)))
    write_json(args.output / "export_report.json", dict(exported=exported, excluded=excluded))
    print(f"Exported {len(exported)} events; excluded {len(excluded)}.")


def fit_objective(events, manifest, args):
    from .inverse.training import train, TrainingConfig

    if (args.output / "objective.pt").exists():
        raise FileExistsError("objective.pt already exists; choose a new --output to preserve the previous model")
    capacity = args.map_capacity
    if args.method == "klgame" and capacity is None:
        domains = {event.domain for event in events}
        if len(domains) != 1:
            raise ValueError("train one scene family at a time")
        capacity = map_padding_capacity(args.events,manifest,domains.pop())
    examples = training_examples(events, manifest, args.proposals, manifest["seed"])
    candidate_dir = args.output / "fixed_candidates"
    candidate_dir.mkdir(exist_ok=True)
    for example in examples:
        np.savez_compressed(candidate_dir / (example["event_id"] + ".npz"), **example)
    config = TrainingConfig(
        max_updates=args.max_updates, model_kind=MODEL_KINDS[args.method],
        device=args.device, seed=manifest["seed"], map_capacity=capacity,
    )
    logs = train(examples, config, args.output / "objective.pt")
    write_json(args.output / "training_log.json", logs)


def evaluate(events, manifest, args):
    from .inverse.training import LearnedObjective

    domains = {event.domain for event in events}
    if len(domains) != 1:
        raise ValueError("evaluate one scene family at a time")
    config = planner_config(args, domains.pop())
    if config.reference_source == "demonstration" and args.libraries is None:
        raise ValueError("--libraries is required for demonstration references")
    prepare_results(args.output)
    write_json(args.output / "run.json", dict(
        method=args.method, variant=variant_name(config) if args.method == "main" else "klgame",
        checkpoint=str(args.checkpoint.resolve()), split=args.split, settings=config.__dict__,
        event_ids=[event.event_id for event in events]))
    write_mapping(args.output / "manifest.yaml", manifest)
    training_groups = set(manifest["groups"]["train"])
    results, libraries = [], {}
    for event in events:
        objective = LearnedObjective(
            args.checkpoint, event.map_tokens(), device=args.device, evaluation_group=event.group
        )
        objective.validate_task(event.domain)
        if objective.kind != MODEL_KINDS[args.method]:
            raise ValueError("checkpoint model does not match --method")
        if not set(objective.training_groups).issubset(training_groups):
            raise ValueError("checkpoint includes scenes outside this training split")

        library = None
        if config.reference_source == "demonstration":
            key = (event.domain, event.agents)
            if key not in libraries:
                libraries[key] = DemonstrationLibrary.load(args.libraries / f"{key[0]}_{key[1]}.npz")
            library = libraries[key]
            if (library.domain, library.agents) != key:
                raise ValueError("library domain/agent-count mismatch")
            if not set(library.source_groups).issubset(training_groups):
                raise ValueError("library includes non-training scenes")

        if not np.isclose(config.dt, event.dt):
            raise ValueError("planner/event dt mismatch")
        radii = event_envelope_radii(event)
        result = run_closed_loop(
            UnicycleDynamics(event.agents, event.dt), event_cost(event), config,
            event.history, objective, len(event.recorded) - 1,
            *control_bounds(event.domain), library=library, envelope_radii=radii,
        )
        expert_trace = expert_history_weight_trace(event, objective) if event.domain != "interhub" else None
        final_expert_weights = expert_trace[-1] if expert_trace is not None else None
        arrays = dict(states=result.states,
            controls=result.controls, weights=result.weights, cycle_ms=result.cycle_ms,
            **result.solver_diagnostics(), event_id=event.event_id, dt=event.dt,
            track_ids=np.asarray(event.track_ids), envelope_radii=radii)
        if expert_trace is not None:
            arrays.update(expert_history_weights=final_expert_weights, expert_history_weight_trace=expert_trace)
        else:
            arrays["evaluation_controls"] = evaluation_controls(event, result.states)
        np.savez_compressed(args.output / (event.event_id + ".npz"), **arrays)
        results.append(event_metrics(event, result.states, result.controls, result.cycle_ms,
                                     estimated_weights=final_expert_weights))
        write_json(args.output / "events.json", results)
        metric = "jADE" if event.domain == "interhub" else "D_tra"
        print(event.event_id, metric+"=", results[-1][metric],
              f"tolerance reached={int(result.converged.sum())}/{len(result.converged)} cycles", flush=True)

    write_json(args.output / "summary.json", aggregate(results))


def main(argv=None, *, subcommand=None):
    args = build_parser(subcommand).parse_args(argv)
    if args.command == "export":
        export_events(args)
        return

    events = load_events(args.events)
    if args.command == "split":
        seed, fractions = ((42, (.7, .15, .15)) if args.config is None
                           else validate_plan(load_mapping(args.config)))
        if args.seed is not None:
            seed = args.seed
        write_mapping(args.output, split_events(events, seed, fractions))
        return

    manifest = load_mapping(args.manifest)
    split = args.split if args.command == "evaluate" else "train"
    selected = select_events(events, manifest, split)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "library":
        for (domain, count), library in build_libraries(selected, manifest, args.neighbors).items():
            library.save(args.output / f"{domain}_{count}.npz")
    elif args.command == "train":
        fit_objective(selected, manifest, args)
    else:
        evaluate(selected, manifest, args)
    write_mapping(args.output / "manifest.yaml", manifest)


if __name__ == "__main__":
    main()
