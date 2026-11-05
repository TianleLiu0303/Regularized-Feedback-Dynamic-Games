"""Matched ablations, run isolation, and invalid-output scoring."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from contextual_game import cli
from contextual_game.config import PlannerConfig
from contextual_game.experiment import VARIANTS
from contextual_game.inverse.training import train, TrainingConfig, LearnedObjective
from contextual_game.metrics import event_metrics
from contextual_game.pipeline import training_examples, build_libraries
from test_paper_alignment import make_event, load_script


class SubmissionRunTests(unittest.TestCase):
    def test_matched_variants_through_both_entry_points_and_preserve_results(self):
        script = load_script("solve_pakl")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = make_event("interhub", "training", "train")
            testing = make_event("interhub", "testing", "test", .1)
            manifest = dict(version=1, seed=42,
                            groups=dict(train=["train"], validation=["validation"], test=["test"]),
                            events=dict(train=["training"], validation=["validation"], test=["testing"]))
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            events = root / "events"
            events.mkdir()
            testing.save(events / "testing.npz")
            checkpoint = root / "objective.pt"
            train(training_examples([training], manifest, proposals=2),
                  TrainingConfig(max_updates=1), checkpoint, metadata={"scene": "interhub"})
            libraries = root / "libraries"
            libraries.mkdir()
            for (domain, count), library in build_libraries([training], manifest).items():
                library.save(libraries / f"{domain}_{count}.npz")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(asdict(PlannerConfig(horizon=2, outer_iterations=2))),
                                   encoding="utf-8")
            for variant, (source, mode) in VARIANTS.items():
                with self.subTest(variant=variant):
                    outputs = [root / f"script_{variant}", root / f"command_{variant}"]
                    args = ["--events", str(events), "--manifest", str(manifest_path),
                            "--checkpoint", str(checkpoint), "--config", str(config_path),
                            "--variant", variant, "--output", str(outputs[0])]
                    library_path = libraries if source == "demonstration" else None
                    if library_path is not None:
                        args += ["--libraries", str(library_path)]
                    command = SimpleNamespace(method="main", variant=variant, checkpoint=checkpoint,
                                              libraries=library_path, device="cpu", config=config_path,
                                              output=outputs[1], split="test")
                    with redirect_stdout(io.StringIO()):
                        script.main(args)
                        cli.evaluate([testing], manifest, command)
                    for output in outputs:
                        run = json.loads((output / "run.json").read_text())
                        self.assertEqual(run["variant"], variant)
                        self.assertEqual(run["settings"]["reference_source"], source)
                        self.assertEqual(run["settings"]["coefficient_mode"], mode)
                        self.assertEqual(run["checkpoint"], str(checkpoint.resolve()))
                    with np.load(outputs[0] / "testing.npz") as first, np.load(outputs[1] / "testing.npz") as second:
                        for key in ("states", "controls", "weights"):
                            np.testing.assert_allclose(first[key], second[key])
                        self.assertTrue(np.isfinite(first["states"]).all())
                    before = [(output / "summary.json").read_bytes() for output in outputs]
                    with self.assertRaisesRegex(FileExistsError, "not empty"):
                        script.main(args)
                    with self.assertRaisesRegex(FileExistsError, "not empty"):
                        cli.evaluate([testing], manifest, command)
                    self.assertEqual(before, [(output / "summary.json").read_bytes() for output in outputs])

    def test_klgame_cannot_be_mislabeled_as_matched_reference_ablation(self):
        script = load_script("solve_pakl")
        args = script.build_parser().parse_args(["--method", "klgame", "--variant", "cost-reference"])
        with self.assertRaisesRegex(ValueError, "contextual checkpoint"):
            script.configure(args)

    def test_nonfinite_execution_is_not_scored_as_a_valid_event(self):
        event = make_event("camp")
        states = event.recorded.copy()
        states[1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            event_metrics(event, states, event.expert_controls, estimated_weights=event.true_weights)

    def test_cost_weights_cannot_be_reused_with_different_primitive_scaling(self):
        event = make_event("camp")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "objective.pt"
            examples = training_examples([event], {"groups": {"train": [event.group]}},
                                         proposals=2, control_diagonal=(2., .5))
            train(examples, TrainingConfig(max_updates=1), checkpoint,
                  metadata=dict(scene="camp", control_diagonal=[2., .5]))
            objective = LearnedObjective(checkpoint, event.map_tokens())
            objective.validate_task("camp", (2., .5))
            with self.assertRaisesRegex(ValueError, "training primitives"):
                objective.validate_task("camp", (1., .25))
            with self.assertRaisesRegex(ValueError, "scene"):
                objective.validate_task("rm", (2., .5))


if __name__ == "__main__":
    unittest.main()
