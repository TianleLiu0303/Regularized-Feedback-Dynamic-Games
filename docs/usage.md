# PA-KL trajectory planning

Contextual cost inference, retrieved demonstration references, and proximity-adaptive KL feedback planning for **Regularized Feedback Dynamic Games with Contextual Objective Inference for Multi-Agent Trajectory Planning**. This release includes the main method, matched reference/coefficient ablations, and a task-adapted KLGame comparison. Hardware software and the Global-cost/no-map ablations are not included.

## Directory

    configs/                  Optional InterHub planner presets and YAML data split
    scripts/train_inverse.py  Launch inverse objective training
    scripts/solve_pakl.py     Run closed-loop PA-KL planning and evaluation
    examples/                 Plot saved trajectories and training curves
    src/contextual_game/      Planner, cost learning, data interface, and metrics

The two scripts contain editable defaults, argument parsing, data selection, execution, and output saving. Core algorithms remain in controller.py, demonstration.py, inverse/training.py, and rakl_solver.py.

## Method and evaluation conventions

- Inverse training uses the finite-choice likelihood `softmax(-F @ omega / tau_R)` over the expert and all fixed proposals. There is no separate geometric rule-count penalty or geometric rejection of proposals. Separation and lane penalties remain in the task primitives. Candidate controls are clipped before dynamics rollout.
- Adaptive KL coefficients use `max(0, min(center_distance_ij - r_i - r_j, obstacle_distance_i - r_i))`, not centre distances. CAMP uses the stored disk diameter divided by two; RM and InterHub use half the footprint diagonal. The default decay length is `sigma=1.2 m`. Collision scoring still uses disks for CAMP and oriented rectangles for road scenes, not the circular scheduling envelopes.
- Event-based runners use an empty static-obstacle set unless an `obstacle_distance` callback is explicitly supplied to the controller. That callback returns centre-to-obstacle-set distance; the scheduler subtracts the agent radius. Drivable-map polygons remain part of off-road evaluation and route geometry, and are not silently converted into a static-obstacle set.
- Synthetic `D_par` uses the final complete rolling window of the common recorded expert trajectory, through its final observation. This score-only inference is separate from the weights inferred on each planner's executed histories. Means and sample standard deviations are taken across trials, not across inference iterations.
- InterHub `E_u` uses the same causal five-sample, second-order Savitzky–Golay control reconstruction for executed and recorded trajectories. The observed prefix supplies filter history at the initial transition. Raw executed commands are preserved separately and still drive the rollout.
- Reference histories use the ego position and heading at each historical sample, as in Eq. (7); goals and road geometry use the current ego frame. Library files record descriptor version 2. Earlier libraries must be rebuilt, including their normalization statistics, before rerunning planning; objective-network checkpoints are unaffected by this descriptor change.
- Local costs use analytic goal, pairwise separation, lane, and control derivatives. The position Hessian is eigenvalue-clipped and embedded into the joint state; the other state entries remain zero. Lane derivatives follow the selected polyline segment, with the inactive branch at the lane hinge. The legacy `finite_difference_step` setting is accepted but unused by these primitives.

## Installation

Use Python 3.12 in a virtual environment. From this directory:

    python -m pip install -e ".[train,interhub,plot]"

requirements.txt records the tested Windows CPU environment. For those versions, run `python -m pip install -r requirements.txt` before installing this project. The plot extra installs Matplotlib for visualization.

## Prepare data

Use `InteractionEvent.save/load` in `src/contextual_game/interhub.py` for normalized NPZ events under `data/<scene>/events`:

- `states`: `[H+T,N,4]` in `[x,y,heading,speed]` order, with metre/radian units and `dt=0.1`. By default `H=10`; planning starts at index `H-1`.
- `goals` and `sizes`: `[N,2]`; distinct `track_ids`, a unique `event_id`, a source scene `group`, and `domain` (`camp`, `rm`, or `interhub`). CAMP uses equal disk diameters in `sizes` and empty `routes`/`polygons`; road scenes use full vehicle length/width, one assigned `Route` per agent, and drivable polygons in the same coordinate frame.
- CAMP/RM evaluation also requires normalized positive `true_weights` (`[N,3]` / `[N,4]`) and exact `expert_controls` (`[T,N,2]`, yaw rate and acceleration), starting at transition `H-1` to `H`. True weights are used only for scoring; stored controls supply demonstration labels. InterHub requires neither array.

InterHub imports `interactive_segments_index.csv` and compatible trajdata caches. Install the matching trajdata version and source-dataset extras separately. The index fields are `dataset`, `folder`, `scenario_idx`, `track_id` (semicolon-separated IDs), `start`, `end`, `original_scene_id`, and `path_category` (MP/CP). An `assignments.json` supplies `source: "map_and_observed_history"`, `track_ids`, `goals`, `sizes`, `routes` (each with `points` and `half_width`), and `polygons`, keyed by `event_0000000` (the zero-based original CSV data-row index). Use cache/world coordinates and assign routes/goals from maps and observed history; the exporter transforms all geometry into the interaction frame.

    contextual-experiment export --index interactive_segments_index.csv --cache-root CACHE_PATH --assignments assignments.json --time-unit frames --output data/interhub/events

Choose `--time-unit frames` or `seconds` according to the index. `start` denotes the last observed frame; the default window contains ten observed and fifty future states. Export requires an empty output directory and reports excluded events in `export_report.json`. Create the scene-disjoint split with:

    contextual-experiment split --events data/interhub/events --config configs/interhub_split.yaml --output data/interhub/split.yaml

The YAML recipe uses seed 42 and a scene-disjoint 70/15/15 split. It randomly shuffles scene groups; it does not implement agent-count/path-category stratification. To evaluate the paper's reported cohort, supply its original scene/event manifest rather than regenerating a different split.

For CAMP/RM, change the directory to `data/camp` or `data/rm`, or supply your existing split. In synthetic data, each group must identify the nominal scene/seed family shared by its perturbations. Training automatically builds its demonstration libraries from the selected training events.

## Train the inverse model

    python scripts/train_inverse.py --scene camp --agents 4
    python scripts/train_inverse.py --scene rm --agents 5
    python scripts/train_inverse.py --scene interhub

These are alternative runs. Outputs default to `outputs/<scene>/main/model/`: the final `objective.pt`, `training_log.json`, `training_config.json`, fixed candidates, split manifest, and `libraries/`. `--events`, `--manifest`, and `--output` override paths. Match training and solve agent-count filters, or train on all counts before solving a subset.

Editable dictionaries at the top of train_inverse.py contain optimizer, network, proposal, control-cost, and retrieval parameters. CLI flags override them, for example:

    python scripts/train_inverse.py --scene interhub --max-updates 1000 --learning-rate 0.001 --proposals 30 --neighbors 20 --device cpu

The default seed comes from the manifest; `--seed` changes proposal/training randomness without changing the saved data split. Training retains the final iterate. KLGame's 5-by-256 MLP remains fixed; the GRU/attention dimension options apply to the main model.

Before KLGame training, both training entry points determine one fixed map-padding capacity from the known map geometry in the declared train/validation/test cohort and save it in the checkpoint. This scan reads only geometry metadata; training trajectories, loss, and reference normalization still use training scenes exclusively. Map tokens retain the same route-then-boundary order as the contextual model, with zero padding and validity masks. An explicit `--map-capacity` supports training-only data directories or a larger predeclared deployment capacity; it must cover every evaluated map. Maps are never truncated. Direct Python calls to `train` must supply `TrainingConfig.map_capacity` for road-domain MLPs. A checkpoint with insufficient capacity needs retraining.

New models use checkpoint format 2 and record the task-cost-only likelihood. A legacy format-1 model is accepted only when its saved `rule_penalty` is explicitly zero. Models trained with the removed penalty need retraining; changing an evaluation flag cannot undo that training objective. Training entry points refuse to overwrite an existing `objective.pt`; select a new `--output` directory to preserve previous runs.

## Solve with PA-KL

    python scripts/solve_pakl.py --scene camp --agents 4
    python scripts/solve_pakl.py --scene rm --agents 5
    python scripts/solve_pakl.py --scene interhub

The solver loads `outputs/<scene>/main/model/objective.pt` and its `libraries/`, and saves to `outputs/<scene>/main/results/`. Override these with `--checkpoint`, `--libraries`, and `--output`. It infers weights every cycle and executes receding-horizon joint controls on the test split; `--split validation` selects validation events.

SCENE_DEFAULTS and SOLVER_DEFAULTS at the top of solve_pakl.py define horizons, control bounds, iteration limits, step size, tolerances, and KL coefficients. Precedence is built-in defaults, then optional `--config`, then explicit CLI flags. The selected scene must agree with the config's scenario. A planning-horizon override does not shorten the evaluation window.

    python scripts/solve_pakl.py --scene interhub --category MP --agents 2 --horizon 50 --outer-iterations 30 --lambda-min 0.5 --lambda-max 5

Both scripts accept `--agents 2 4 20` and `--event-ids ID1 ID2`; InterHub additionally accepts `--category MP CP`. IDs must belong to the requested split. Use `--help` for all options. Training and planning require supplied event files; neither script generates substitute data.

Each event produces an NPZ with states, controls, weights, cycle times, convergence flags, and event/agent identifiers. `events.json` and `summary.json` report scene-appropriate metrics and summaries by agent count. `run.json` records resolved settings and selected IDs; `manifest.yaml` preserves the full split.

Evaluation requires an empty output directory. Existing or interrupted runs are preserved; choose a new `--output` for another run. Nonfinite trajectories or controls raise an error and are not scored as valid events.

The evaluation control-cost diagonal must match the checkpoint's training primitives. For custom training, pass the same `--control-diagonal` to `solve_pakl.py`; changing primitive scales without retraining changes the meaning of the inferred weights. Older checkpoints without this metadata use the original `(1, 0.25)` convention.

Solver diagnostics in each NPZ include `iterations`, `final_residual`, `stop_reason` (`tolerance` or `iteration_limit`), and `residual_history` (cycles by iteration budget, with NaN padding after early termination). The residual is the unnormalized Euclidean norm of concatenated state/control trajectory changes, with wrapped heading differences. Reaching the iteration limit is not convergence. The default remains 30 iterations, step size 0.5, and tolerance 1e-3; frozen-local equilibrium conditions do not guarantee convergence of the nonlinear outer iterations with refreshed references.

The implementation caches causal smoothing coefficients, shares history reconstruction across targets, vectorizes analytic primitive derivatives, and solves both coupled Nash right-hand sides with one factorization. Exactly-zero cross-player effort terms are skipped. Exact nearest-neighbor retrieval, curvature clipping, reference refreshes, coefficient scheduling, and damped/projected controls retain their original definitions. These changes reduce computation; they do not relax the stopping test or provide a convergence guarantee.

NPZ files also store the fixed `envelope_radii`. Synthetic results save `expert_history_weight_trace` and its final value, `expert_history_weights`; the `weights` array remains the actual closed-loop inference sequence. InterHub results save reconstructed `evaluation_controls` separately from the executed `controls`. Changes to the schedule require new rollouts; changes to score-only reconstruction or the selected expert window require recomputing the affected metrics from saved trajectories and models.

For the KLGame baseline, use `--method klgame` in both scripts. This trains a separate MLP, uses a cost-induced reference with temperature 0.99 and fixed coefficient 2, and saves under `outputs/<scene>/klgame/`. It needs no demonstration library. `configs/klgame.json` is an optional InterHub preset.

## Matched reference and coefficient ablations

Use the same main-method checkpoint and evaluation manifest for all four variants. Only the reference source and coefficient schedule change; each controller updates its own executed history.

| Variant | Cost model | Reference | KL coefficient |
| --- | --- | --- | --- |
| `full` | Contextual checkpoint | Retrieved demonstration | Adaptive |
| `fixed` | Same checkpoint | Same demonstration library | Fixed, default 2 |
| `cost-reference` | Same checkpoint | Auxiliary unregularized local-game Q | Adaptive |
| `no-kl` | Same checkpoint | None | Zero |

    python scripts/solve_pakl.py --scene interhub --variant full
    python scripts/solve_pakl.py --scene interhub --variant fixed
    python scripts/solve_pakl.py --scene interhub --variant cost-reference
    python scripts/solve_pakl.py --scene interhub --variant no-kl

The default checkpoint remains `outputs/interhub/main/model/objective.pt`. Results go to `results/`, `results_fixed/`, `results_cost-reference/`, and `results_no-kl/`, respectively, under `outputs/interhub/main/`. Neither `cost-reference` nor `no-kl` loads a demonstration library. Both public evaluation entry points accept `--variant`; an explicit variant overrides reference/coefficient fields in a supplied config. `--method klgame` cannot be combined with `--variant`, since that baseline changes the cost estimator as well.

## Visualize saved outputs

    python examples/plot_training.py --log outputs/interhub/main/model/training_log.json --output outputs/figures/training.png
    python examples/plot_trajectories.py --event data/interhub/events/event_0000000.npz --result outputs/interhub/main/results/event_0000000.npz --output outputs/figures/trajectories.png

Replace the example event ID with an evaluated event. The trajectory figure shows any road map, recorded and executed paths, and executed controls, with disks for CAMP and rectangles for road scenes. The training figure shows loss and parameter-update norms. Both scripts accept PNG, PDF, or SVG and optional `--show`; they only read saved files and plot them. Use `--label KLGame` for baseline trajectories.

## Scope

Real InterHub caches, the original experiment split manifest, trained research checkpoints, and paper-result logs are not bundled. The cache exporter has not been validated against the original research caches. Reproducing the reported numerical tables requires these original assets, including map/route assignments and synthetic scenario definitions. This source release provides training, planning, and evaluation code; its software tests do not verify the paper's reported numerical results or hardware timing.

## Regression checks

After installing this project with its training and InterHub dependencies, run from this directory:

    python -m unittest discover -s tests -v

These checks cover the task-cost-only loss and gradient, fixed proposals, per-sample reference coordinates, analytic derivatives against finite differences, envelope clearances, final expert-window scoring, common effort reconstruction, and checkpoint/library compatibility. End-to-end checks compare training and planning through both public entry points for CAMP (2/4/20 agents), RM (5), and InterHub (2/3), including held-out maps larger than the training maps. Their small temporary fixtures are software tests, not replacements for the paper's datasets or experimental results.

Numerical checks additionally verify the frozen game's action stationarity, covariance precision, and value derivatives, including nonzero cross-player effort and zero/mixed KL coefficients. They also check Boltzmann references against the auxiliary conditional Q and distinguish iteration-budget termination from tolerance termination.

Submission checks exercise all four matched variants through both evaluation entry points, reject accidental reuse of a result directory, and reject nonfinite execution data.
