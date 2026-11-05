"""Select scenes, configure PA-KL, execute closed-loop planning, and save metrics."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from contextual_game.config import PlannerConfig
from contextual_game.controller import run_closed_loop
from contextual_game.demonstration import DemonstrationLibrary
from contextual_game.dynamics import UnicycleDynamics
from contextual_game.experiment import VARIANTS, configure_variant, variant_name, prepare_results
from contextual_game.metrics import aggregate, event_metrics, evaluation_controls, expert_history_weight_trace
from contextual_game.pipeline import event_cost, event_envelope_radii
from contextual_game.selection import select_events
from contextual_game.splits import load_mapping, write_mapping


# Edit defaults here. CLI overrides take precedence over --config and these values.
SCENE_DEFAULTS = {
    "camp": {"horizon": 60, "lower": (-1.2, -2.), "upper": (1.2, 2.), "features": 3},
    "rm": {"horizon": 50, "lower": (-.3, -3.), "upper": (.3, 2.), "features": 4},
    "interhub": {"horizon": 50, "lower": (-.3, -3.), "upper": (.3, 2.), "features": 4},
}
SOLVER_DEFAULTS = {
    "dt": .1,
    "outer_iterations": 30,
    "step_size": .5,
    "tolerance": 1e-3,  # L2 norm of concatenated state/control changes
    "lambda_min": .5,
    "lambda_max": 5.,
    "lambda_sigma": 1.2,
    "hessian_floor": 1e-3,
    "finite_difference_step": 1e-3,  # compatibility only; built-in costs use analytic derivatives
    "temperature": .99,  # KLGame cost-reference temperature
    "fixed_lambda": 2.,  # KLGame coefficient
    "seed": 42,
}
CONTROL_DIAGONAL = (1., .25)
EFFORT_DIAGONAL = (30., 1.)  # InterHub evaluation score, distinct from task cost
HISTORY_LENGTH = 10


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scene", choices=SCENE_DEFAULTS, default="interhub")
    parser.add_argument("--method", choices=("main", "klgame"), default="main")
    parser.add_argument("--variant", choices=VARIANTS,
                        help="Matched ablation using the main method's contextual checkpoint")
    parser.add_argument("--events", type=Path, help="Defaults to data/<scene>/events")
    parser.add_argument("--manifest", type=Path, help="Defaults to data/<scene>/split.yaml")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to outputs/<scene>/<method>/model/objective.pt")
    parser.add_argument("--libraries", type=Path, help="Defaults to the checkpoint directory's libraries/")
    parser.add_argument("--output", type=Path, help="Defaults to outputs/<scene>/<method>/results")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--agents", type=int, nargs="+", help="Select agent counts, e.g. 2 4 20")
    parser.add_argument("--category", choices=("MP", "CP"), nargs="+", help="InterHub path types only")
    parser.add_argument("--event-ids", nargs="+", help="Select specific evaluation event IDs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path, help="Optional planner JSON; --scene must agree with its scenario")
    parser.add_argument("--history-length", type=int, default=HISTORY_LENGTH)
    parser.add_argument("--horizon", type=int, help="Planning steps; defaults to CAMP 60, RM/InterHub 50")
    for name, value in SOLVER_DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value),
                            help=("Legacy option; built-in costs use analytic derivatives" if name == "finite_difference_step"
                                  else f"Override planner setting (built-in default: {value})"))
    parser.add_argument("--control-lower", type=float, nargs=2, help="Override scene yaw-rate/acceleration lower bounds")
    parser.add_argument("--control-upper", type=float, nargs=2, help="Override scene yaw-rate/acceleration upper bounds")
    parser.add_argument("--control-diagonal", type=float, nargs=2, default=CONTROL_DIAGONAL)
    parser.add_argument("--effort-diagonal", type=float, nargs=2, default=EFFORT_DIAGONAL)
    return parser


def configure(args):
    settings = dict(SOLVER_DEFAULTS, scenario=args.scene, horizon=SCENE_DEFAULTS[args.scene]["horizon"],
                    reference_source="demonstration", coefficient_mode="adaptive")
    if args.method == "klgame":
        settings.update(reference_source="cost", coefficient_mode="fixed")
    if args.config is not None:
        supplied = load_mapping(args.config)
        if supplied.get("scenario", args.scene) != args.scene:
            raise ValueError("--config scenario does not match --scene")
        settings.update(supplied)
    for name in (*SOLVER_DEFAULTS, "horizon"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    config = PlannerConfig(**settings)
    return configure_variant(config, args.method, args.variant)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = configure(args)
    scene = SCENE_DEFAULTS[args.scene]
    event_dir = args.events or Path("data") / args.scene / "events"
    manifest_path = args.manifest or Path("data") / args.scene / "split.yaml"
    model_dir = Path("outputs") / args.scene / args.method / "model"
    checkpoint = args.checkpoint or model_dir / "objective.pt"
    library_dir = args.libraries or checkpoint.parent / "libraries"
    variant = variant_name(config) if args.method == "main" else "klgame"
    result_name = "results" if variant in {"full", "klgame"} else f"results_{variant}"
    output = args.output or Path("outputs") / args.scene / args.method / result_name
    lower = np.asarray(args.control_lower if args.control_lower is not None else scene["lower"])
    upper = np.asarray(args.control_upper if args.control_upper is not None else scene["upper"])
    if (not np.isfinite([lower, upper]).all() or np.any(lower >= upper)
            or not np.isfinite(args.control_diagonal).all() or min(args.control_diagonal) <= 0
            or not np.isfinite(args.effort_diagonal).all() or min(args.effort_diagonal) <= 0):
        raise ValueError("invalid control bounds or cost/effort diagonal")

    # 1. Resolve scene filters against a fixed, scene-disjoint evaluation split.
    manifest = load_mapping(manifest_path)
    events = select_events(event_dir, manifest, args.split, args.scene,
                           agents=args.agents, categories=args.category, event_ids=args.event_ids)
    for event in events:
        if event.history_length != args.history_length or not np.isclose(event.dt, config.dt):
            raise ValueError("event history length or sampling interval differs from the configuration")
        if args.scene != "interhub" and (event.true_weights is None or event.expert_controls is None):
            raise ValueError("CAMP/RM evaluation needs saved true_weights and exact expert_controls")
    prepare_results(output)
    write_mapping(output / "run.json", dict(
        scene=args.scene, method=args.method, variant=variant, split=args.split, settings=asdict(config),
        checkpoint=str(checkpoint.resolve()), events=str(event_dir.resolve()),
        manifest=str(manifest_path.resolve()), libraries=str(library_dir.resolve()) if config.reference_source == "demonstration" else None,
        event_ids=[event.event_id for event in events], agents=args.agents, categories=args.category,
        history_length=args.history_length, control_lower=lower.tolist(), control_upper=upper.tolist(),
        control_diagonal=list(args.control_diagonal), effort_diagonal=list(args.effort_diagonal)))
    write_mapping(output / "manifest.yaml", manifest)
    print(f"{args.scene.upper()} / {args.method}: {len(events)} events, T={config.horizon}, "
          f"dt={config.dt}, iterations={config.outer_iterations}.", flush=True)

    # 2. Load the objective model and the matching domain/agent-count reference library.
    from contextual_game.inverse.training import LearnedObjective
    training_groups = set(manifest["groups"]["train"])
    expected_kind = "contextual" if args.method == "main" else "mlp"
    libraries, rows = {}, []
    for index, event in enumerate(events, start=1):
        objective = LearnedObjective(checkpoint, event.map_tokens(), device=args.device, evaluation_group=event.group)
        objective.validate_task(args.scene, args.control_diagonal)
        if objective.kind != expected_kind or objective.dimensions["features"] != scene["features"]:
            raise ValueError("checkpoint method or cost dimension does not match the selected scene")
        if objective.metadata.get("scene", args.scene) != args.scene:
            raise ValueError("checkpoint was trained for a different scene family")
        if not set(objective.training_groups).issubset(training_groups):
            raise ValueError("checkpoint contains scenes outside the training split")
        library = None
        if config.reference_source == "demonstration":
            key = (event.domain, event.agents)
            if key not in libraries:
                libraries[key] = DemonstrationLibrary.load(library_dir / f"{key[0]}_{key[1]}.npz")
            library = libraries[key]
            if (library.domain, library.agents) != key or not set(library.source_groups).issubset(training_groups):
                raise ValueError("library domain, agent count, or training scenes do not match this run")

        # 3. Execute the full evaluation window. Only planning horizon T is configurable;
        #    recorded future states are used later for scoring, never for online inference.
        radii = event_envelope_radii(event)
        print(f"[{index}/{len(events)}] Solving {event.event_id} ({event.agents} agents)...", flush=True)
        result = run_closed_loop(
            UnicycleDynamics(event.agents, event.dt),
            event_cost(event, control_diagonal=args.control_diagonal), config,
            event.history, objective, len(event.recorded)-1, lower, upper, library=library,
            envelope_radii=radii)
        # Separate score-only inference on common expert histories from the
        # method's own executed histories. Retain the final recorded window.
        expert_trace = expert_history_weight_trace(event, objective) if args.scene != "interhub" else None
        expert_history_weights = expert_trace[-1] if expert_trace is not None else None
        arrays = dict(states=result.states, controls=result.controls, weights=result.weights,
                      cycle_ms=result.cycle_ms, **result.solver_diagnostics(),
                      event_id=event.event_id, dt=event.dt, track_ids=np.asarray(event.track_ids),
                      envelope_radii=radii)
        if expert_history_weights is not None:
            arrays["expert_history_weights"] = expert_history_weights
            arrays["expert_history_weight_trace"] = expert_trace
        else:
            arrays["evaluation_controls"] = evaluation_controls(event, result.states)
        np.savez_compressed(output / f"{event.event_id}.npz", **arrays)

        # 4. Score executions using the scene's footprint model and metric definitions.
        row = event_metrics(event, result.states, result.controls, result.cycle_ms,
                            estimated_weights=expert_history_weights, control_diagonal=args.control_diagonal,
                            effort_diagonal=args.effort_diagonal)
        row["agents"] = event.agents
        rows.append(row)
        (output / "events.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        metric = "jADE" if args.scene == "interhub" else "D_tra"
        print(f"  {metric}={row[metric]:.4f} m, collision={row['collision']}, "
              f"tolerance reached={int(result.converged.sum())}/{len(result.converged)} cycles", flush=True)

    summary = aggregate(rows)
    summary["by_agent_count"] = {str(n): aggregate([row for row in rows if row["agents"] == n])
                                  for n in sorted({row["agents"] for row in rows})}
    write_mapping(output / "summary.json", summary)
    print(f"Saved trajectories and metrics to {output}.")


if __name__ == "__main__":
    main()
