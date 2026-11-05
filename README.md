# Regularized Feedback Dynamic Games

Implementation of **Regularized Feedback Dynamic Games with Contextual Objective Inference for Multi-Agent Trajectory Planning**.

The planner combines contextual cost inference, Gaussian action references retrieved from demonstrations, and proximity-adaptive KL (PA-KL) regularization in a receding-horizon feedback game.

## Installation

Use Python 3.12. From the repository root:

```bash
python -m pip install -e ".[train,interhub,plot]"
```

## Usage

Prepare normalized interaction events and a scene-disjoint split manifest as described in [the data and usage guide](docs/usage.md).

```bash
# Train contextual cost inference and build demonstration libraries.
python scripts/train_inverse.py --scene interhub

# Run the full planner.
python scripts/solve_pakl.py --scene interhub

# Matched ablations using the same contextual checkpoint.
python scripts/solve_pakl.py --scene interhub --variant fixed
python scripts/solve_pakl.py --scene interhub --variant cost-reference
python scripts/solve_pakl.py --scene interhub --variant no-kl
```

Use `--scene camp` or `--scene rm` for synthetic scenarios. The task-adapted KLGame comparison uses `--method klgame` in both training and planning. Run either script with `--help` for options. Each evaluation requires a new or empty output directory.

## Repository

- `src/contextual_game/`: cost inference, demonstration retrieval, feedback-game solver, and evaluation.
- `scripts/`: training and planning entry points.
- `configs/`: planner settings and split recipe.
- `examples/`: trajectory and training plots.
- `tests/`: numerical and pipeline regression tests.

```bash
python -m unittest discover -s tests -v
```

## Reproduction scope

This repository includes the main method, matched reference/coefficient ablations, and a task-adapted KLGame baseline. Original datasets, experiment split manifests, trained checkpoints, hardware software, and paper-result logs are not included. Global-cost/no-map ablations and the complete baseline tables are outside this release. Reproducing the reported results requires the original experimental assets; software tests alone do not verify those results or hardware timing.

See [docs/usage.md](docs/usage.md) for data formats, configuration, and evaluation conventions.
