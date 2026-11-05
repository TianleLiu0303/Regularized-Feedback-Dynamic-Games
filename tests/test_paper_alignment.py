"""Regression checks for manuscript/code alignment, not paper-result experiments."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from contextual_game import cli
from contextual_game.config import PlannerConfig
from contextual_game.demonstration import reconstruct_controls
from contextual_game.dynamics import UnicycleDynamics
from contextual_game.forward_solver import ForwardGameSolver
from contextual_game.geometry import adaptive_kl_coefficient, envelope_clearance
from contextual_game.interhub import InteractionEvent
from contextual_game.inverse.candidates import generate_candidate_set, precompute_features
from contextual_game.inverse.loss import finite_choice_nll
from contextual_game.inverse.training import LearnedObjective, TrainingConfig, train
from contextual_game.metrics import (aggregate, evaluation_controls, event_metrics,
                                     expert_history_weight_trace)
from contextual_game.pipeline import (build_libraries, event_cost, event_envelope_radii,
                                      training_examples)
from contextual_game.road import Route

CODE = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, CODE / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_event(domain="camp", event_id="unit", group="train", offset=0.):
    """Small deterministic fixtures exercise production functions only."""
    h, n = 10, 2
    history = np.zeros((h, n, 4))
    history[:, :, 0] = np.arange(h)[:, None] * .1 + np.array([-3., 3.]) + offset
    history[:, 1, 1] = 3.
    history[:, :, 3] = 1.
    controls = np.array([[[.2, 1.], [-.2, .5]],
                         [[-.2, -1.], [.2, -.5]],
                         [[.2, .5], [-.2, 1.]]])
    future = UnicycleDynamics(n).rollout(history[-1].ravel(), controls.reshape(3, -1))
    states = np.concatenate([history, future[1:].reshape(3, n, 4)])
    road = domain != "camp"
    routes = tuple(Route(np.array([[-10., y], [100., y]]), 1.75) for y in (0., 3.)) if road else ()
    polygons = (np.array([[-100., -100.], [100., -100.], [100., 100.], [-100., 100.]]),) if road else ()
    weights = np.tile(np.array([.1, .2, .3, .4]) if road else np.array([.2, .3, .5]), (n, 1))
    return InteractionEvent(event_id, group, domain, .1, h, states,
                            np.array([[10., 0.], [16., 3.]]),
                            np.tile([4.5, 1.8] if road else [.7, .7], (n, 1)),
                            routes, polygons, ("a", "b"), {},
                            true_weights=weights if domain != "interhub" else None,
                            expert_controls=controls if domain != "interhub" else None)


class LikelihoodTests(unittest.TestCase):
    def test_task_cost_only_scene_average_and_gradient(self):
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(2, 2, 4, 3, generator=generator, dtype=torch.float64)
        weights = torch.rand(2, 2, 3, generator=generator, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[True, True], [True, False]])
        tau = .7
        logits = -(features * weights.unsqueeze(2)).sum(-1) / tau
        logp = torch.log_softmax(logits, dim=-1)
        expected_loss = (-logp[0, :, 0].mean() - logp[1, 0, 0]) / 2
        loss = finite_choice_nll(weights, features, temperature=tau, agent_mask=mask)
        torch.testing.assert_close(loss, expected_loss)
        loss.backward()
        expected_gradient = (features[:, :, 0] -
                             (logp.exp().unsqueeze(-1) * features).sum(2)) / tau
        expected_gradient *= (mask / mask.sum(-1, keepdim=True) / 2).unsqueeze(-1)
        torch.testing.assert_close(weights.grad, expected_gradient)

    def test_large_costs_are_stable_and_no_rule_argument_remains(self):
        weights = torch.ones((1, 1, 1), dtype=torch.float64)
        features = torch.tensor([[[[10000.], [10001.]]]], dtype=torch.float64)
        self.assertAlmostEqual(finite_choice_nll(weights, features).item(), np.log1p(np.exp(-1)), places=10)
        self.assertNotIn("rule_penalty", asdict(TrainingConfig()))
        with self.assertRaises(TypeError):
            finite_choice_nll(weights, features, rule_penalty=.1)

    def test_candidates_and_features_have_no_geometric_filter_or_extra_counts(self):
        event = make_event()
        manifest = {"groups": {"train": [event.group]}}
        with patch("contextual_game.pipeline.event_rules", side_effect=AssertionError("not a training term")):
            examples = training_examples([event], manifest, proposals=30, seed=7)
        self.assertNotIn("violations", examples[0])
        self.assertEqual(examples[0]["features"].shape, (2, 31, 3))
        cost = event_cost(event)
        for i in range(event.agents):
            expected = cost.trajectory_features(event.recorded.reshape(4, -1),
                                                event.expert_controls.reshape(3, -1), i)
            np.testing.assert_allclose(examples[0]["features"][i, 0], expected)
        repeated = training_examples([event], manifest, proposals=30, seed=7)
        np.testing.assert_array_equal(examples[0]["features"], repeated[0]["features"])

    def test_proposals_are_bounded_rollouts_and_expert_is_preserved(self):
        event = make_event()
        dynamics = UnicycleDynamics(2)
        expert = event.expert_controls.reshape(3, -1)
        candidates = generate_candidate_set(dynamics, event.recorded[0].ravel(), expert,
                                             30, np.array([.2, 1.]), np.array([-.3, -3.]),
                                             np.array([.3, 2.]), np.random.default_rng(4),
                                             expert_states=event.recorded.reshape(4, -1))
        self.assertEqual(candidates.controls.shape, (31, 3, 4))
        np.testing.assert_array_equal(candidates.controls[0], expert)
        np.testing.assert_array_equal(candidates.states[0], event.recorded.reshape(4, -1))
        self.assertTrue(np.all(candidates.controls[1:] >= np.tile([-.3, -3.], 2)))
        self.assertTrue(np.all(candidates.controls[1:] <= np.tile([.3, 2.], 2)))
        for k in range(1, 31):
            np.testing.assert_allclose(candidates.states[k],
                                       dynamics.rollout(event.recorded[0].ravel(), candidates.controls[k]))
        self.assertEqual(precompute_features(candidates, event_cost(event)).shape, (2, 31, 3))


class ClearanceTests(unittest.TestCase):
    def test_nearest_envelope_is_not_necessarily_nearest_center(self):
        state = np.zeros((3, 4))
        state[:, 0] = [0., 3., 4.]
        radii = np.array([.5, .5, 2.5])
        self.assertEqual(envelope_clearance(state.ravel(), 0, radii), 1.)
        expected = .5 + 4.5 * np.exp(-1 / (2 * 1.2**2))
        self.assertAlmostEqual(adaptive_kl_coefficient(state.ravel(), 0, .5, 5., 1.2,
                                                       envelope_radii=radii), expected)

    def test_overlapping_envelopes_reach_upper_coefficient(self):
        state = np.zeros((2, 4))
        state[1, 0] = .5
        self.assertEqual(envelope_clearance(state, 0, [.35, .35]), 0.)
        self.assertEqual(adaptive_kl_coefficient(state, 0, .5, 5., 1.2,
                                                envelope_radii=[.35, .35]), 5.)

    def test_obstacle_distance_subtracts_own_radius_once(self):
        state = np.zeros((2, 4))
        state[1, 0] = 100.
        self.assertAlmostEqual(envelope_clearance(state, 0, [.35, .35], 1.), .65)
        self.assertEqual(envelope_clearance(state, 0, [.35, .35], .2), 0.)
        self.assertEqual(envelope_clearance(np.zeros(4), 0, [.35]), np.inf)

    def test_scene_radii_match_footprint_definitions(self):
        np.testing.assert_allclose(event_envelope_radii(make_event()), .35)
        for domain in ("rm", "interhub"):
            np.testing.assert_allclose(event_envelope_radii(make_event(domain)), np.hypot(4.5, 1.8) / 2)

    def test_solver_requires_radii_and_passes_them_to_schedule(self):
        event = make_event()
        cfg = PlannerConfig(horizon=1, reference_source="cost")
        with self.assertRaisesRegex(ValueError, "explicit circular-envelope"):
            ForwardGameSolver(UnicycleDynamics(2), event_cost(event), cfg, [-1., -2.], [1., 2.])
        solver = ForwardGameSolver(UnicycleDynamics(2), event_cost(event), cfg, [-1., -2.], [1., 2.],
                                   envelope_radii=event_envelope_radii(event))
        state = event.recorded.reshape(4, -1)
        expected = adaptive_kl_coefficient(state[0], 0, .5, 5., 1.2,
                                           envelope_radii=event_envelope_radii(event))
        self.assertAlmostEqual(solver._regularization(state)[0, 0], expected)
        for mode, value in (("fixed", 2.), ("none", 0.)):
            setting = replace(cfg, coefficient_mode="fixed") if mode == "fixed" else replace(cfg, reference_source="none")
            fixed = ForwardGameSolver(UnicycleDynamics(2), event_cost(event), setting, [-1., -2.], [1., 2.])
            np.testing.assert_array_equal(fixed._regularization(state), np.full((1, 2), value))


class EvaluationTests(unittest.TestCase):
    def test_expert_trace_ends_at_last_recorded_window(self):
        event = make_event()
        windows = []

        def infer(history):
            windows.append(history.copy())
            raw = np.tile([2. + history[-1, 0, 0], 3., 5.], (2, 1))
            return raw / raw.sum(axis=1, keepdims=True)

        trace = expert_history_weight_trace(event, infer)
        self.assertEqual(len(trace), len(event.recorded))
        np.testing.assert_array_equal(windows[0], event.history)
        np.testing.assert_array_equal(windows[-1], event.states[-event.history_length:])
        self.assertFalse(np.allclose(trace[0], trace[-1]))
        row = event_metrics(event, event.recorded, event.expert_controls, estimated_weights=trace[-1])
        expected = 1 - np.mean(np.sum(event.true_weights * trace[-1], axis=1) /
                               (np.linalg.norm(event.true_weights, axis=1) * np.linalg.norm(trace[-1], axis=1)))
        self.assertAlmostEqual(row["D_par"], expected)
        self.assertEqual(row["D_par_protocol"], "final_recorded_expert_history")

    def test_effort_uses_common_causal_reconstruction(self):
        event = make_event("interhub")
        scored = evaluation_controls(event, event.recorded)
        expected = reconstruct_controls(event.states, event.dt)[event.history_length-1:]
        np.testing.assert_allclose(scored, expected)
        raw = np.stack([np.diff(event.recorded[:, :, 2], axis=0),
                        np.diff(event.recorded[:, :, 3], axis=0)], axis=-1) / event.dt
        score = float(np.mean(np.sum(scored**2 * [30., 1.], axis=-1)))
        self.assertNotAlmostEqual(score, float(np.mean(np.sum(raw**2 * [30., 1.], axis=-1))))
        row = event_metrics(event, event.recorded, raw)
        self.assertAlmostEqual(row["E_u"], score)
        self.assertAlmostEqual(row["jADE"], 0.)
        self.assertAlmostEqual(row["P_rel"], 1.)

    def test_control_reconstruction_is_causal_and_unwraps_heading(self):
        states = np.zeros((15, 2, 4))
        angles = 3.1 + np.arange(15) * .03
        states[:, :, 2] = ((angles + np.pi) % (2*np.pi) - np.pi)[:, None]
        states[:, :, 3] = (1. + np.arange(15) * .02)[:, None]
        controls = reconstruct_controls(states)
        np.testing.assert_allclose(controls, np.broadcast_to([.3, .2], controls.shape), atol=1e-12)
        changed = states.copy()
        changed[-1, :, 2:] += .1
        np.testing.assert_allclose(reconstruct_controls(changed)[:-1], controls[:-1])

    def test_aggregate_uses_between_trial_sample_standard_deviation(self):
        rows = [dict(scene="camp", D_par=v, D_tra=0., D_cost=0., collision=False) for v in (.1, .3)]
        result = aggregate(rows)
        self.assertAlmostEqual(result["D_par"], .2)
        self.assertAlmostEqual(result["D_par_std"], np.std([.1, .3], ddof=1))


class ConfigurationTests(unittest.TestCase):
    def test_both_entry_points_have_matching_scene_defaults(self):
        solve = load_script("solve_pakl")
        self.assertEqual(PlannerConfig().lambda_sigma, 1.2)
        self.assertEqual(PlannerConfig.from_json(CODE / "configs/main.json").lambda_sigma, 1.2)
        for domain in ("camp", "rm", "interhub"):
            for method in ("main", "klgame"):
                script = solve.configure(solve.build_parser().parse_args(["--scene", domain, "--method", method]))
                command = cli.planner_config(SimpleNamespace(config=None, method=method), domain)
                self.assertEqual(script, command)
                self.assertEqual(script.horizon, 60 if domain == "camp" else 50)

    def test_training_cli_removes_old_penalty(self):
        script = load_script("train_inverse")
        self.assertNotIn("rule_penalty", vars(script.build_parser().parse_args([])))
        self.assertNotIn("--rule-penalty", script.build_parser().format_help())


class TrainingAndEntryPointTests(unittest.TestCase):
    def test_new_checkpoint_and_legacy_rule_penalty_handling(self):
        event = make_event()
        examples = training_examples([event], {"groups": {"train": [event.group]}}, proposals=2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "objective.pt"
            train(examples, TrainingConfig(max_updates=1), path)
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["version"], 2)
            self.assertEqual(checkpoint["likelihood"], "finite_choice_preference")
            self.assertNotIn("rule_penalty", checkpoint["config"])
            new = LearnedObjective(path, event.map_tokens())(event.history)
            np.testing.assert_allclose(new.sum(axis=1), 1., atol=1e-6)
            checkpoint["version"] = 1
            checkpoint["config"]["rule_penalty"] = .1
            legacy = Path(temporary) / "legacy.pt"
            torch.save(checkpoint, legacy)
            with self.assertRaisesRegex(ValueError, "legacy checkpoint"):
                LearnedObjective(legacy, event.map_tokens())
            checkpoint["config"]["rule_penalty"] = 0.
            torch.save(checkpoint, legacy)
            np.testing.assert_allclose(LearnedObjective(legacy, event.map_tokens())(event.history), new)

    def test_gradient_chunking_preserves_full_batch_update(self):
        events = [make_event(event_id=str(i), offset=i*.1) for i in range(2)]
        examples = training_examples(events, {"groups": {"train": ["train"]}}, proposals=2)
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / f"chunk{chunk}.pt" for chunk in (1, 2)]
            for chunk, path in zip((1, 2), paths):
                train(examples, TrainingConfig(max_updates=1, gradient_chunk_size=chunk), path)
            weights = [LearnedObjective(path, events[0].map_tokens())(events[0].history) for path in paths]
            np.testing.assert_allclose(weights[0], weights[1], rtol=1e-5, atol=1e-6)

    def test_main_and_klgame_through_both_evaluation_entry_points(self):
        solve_script = load_script("solve_pakl")
        for domain in ("camp", "rm", "interhub"):
            for method in ("main", "klgame"):
                with self.subTest(domain=domain, method=method), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    train_event = make_event(domain, "training", "train")
                    test_event = make_event(domain, "testing", "test", .1)
                    manifest = {"version": 1, "seed": 42,
                                "groups": {"train": ["train"], "validation": ["validation"], "test": ["test"]},
                                "events": {"train": ["training"], "validation": ["validation"], "test": ["testing"]}}
                    manifest_path = root / "manifest.json"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    events_dir = root / "events"
                    events_dir.mkdir()
                    test_event.save(events_dir / "testing.npz")
                    examples = training_examples([train_event], manifest, proposals=2)
                    checkpoint = root / "objective.pt"
                    train(examples, TrainingConfig(max_updates=1, model_kind="contextual" if method == "main" else "mlp",
                                                   map_capacity=len(train_event.map_tokens()) if method == "klgame" else None),
                          checkpoint, metadata={"scene": domain})
                    libraries = root / "libraries"
                    libraries.mkdir()
                    if method == "main":
                        for (scene, agents), library in build_libraries([train_event], manifest).items():
                            library.save(libraries / f"{scene}_{agents}.npz")
                    cfg = PlannerConfig(scenario=domain, horizon=1, outer_iterations=1,
                                        reference_source="demonstration" if method == "main" else "cost",
                                        coefficient_mode="adaptive" if method == "main" else "fixed")
                    config_path = root / "planner.json"
                    config_path.write_text(json.dumps(asdict(cfg)), encoding="utf-8")
                    outputs = [root / "script", root / "command"]
                    outputs[1].mkdir()
                    with redirect_stdout(io.StringIO()):
                        solve_script.main(["--scene", domain, "--method", method,
                                           "--events", str(events_dir), "--manifest", str(manifest_path),
                                           "--checkpoint", str(checkpoint), "--libraries", str(libraries),
                                           "--output", str(outputs[0]), "--config", str(config_path)])
                        cli.evaluate([test_event], manifest,
                                     SimpleNamespace(method=method, checkpoint=checkpoint, libraries=libraries,
                                                     device="cpu", config=config_path, output=outputs[1], split="test"))
                    rows = [json.loads((output / "events.json").read_text())[0] for output in outputs]
                    for output, row in zip(outputs, rows):
                        with np.load(output / "testing.npz", allow_pickle=False) as result:
                            np.testing.assert_allclose(result["envelope_radii"], event_envelope_radii(test_event))
                            if domain != "interhub":
                                objective = LearnedObjective(checkpoint, test_event.map_tokens())
                                np.testing.assert_allclose(result["expert_history_weights"],
                                                           objective(test_event.states[-test_event.history_length:]))
                                self.assertEqual(len(result["expert_history_weight_trace"]), len(test_event.recorded))
                                self.assertEqual(row["D_par_protocol"], "final_recorded_expert_history")
                            else:
                                expected = evaluation_controls(test_event, result["states"])
                                np.testing.assert_allclose(result["evaluation_controls"], expected)
                                self.assertAlmostEqual(row["E_u"], np.mean(np.sum(expected**2 * [30., 1.], axis=-1)))
                    for key in (("D_par", "D_cost", "D_tra") if domain != "interhub" else ("jADE", "jFDE", "E_u", "P_rel")):
                        self.assertAlmostEqual(rows[0][key], rows[1][key])


if __name__ == "__main__":
    unittest.main()
