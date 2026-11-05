"""Plot saved planner trajectories and controls against the matching event."""
import argparse
from pathlib import Path

import numpy as np

from contextual_game.interhub import InteractionEvent
from contextual_game.road import rectangle


def load_result(path, event):
    with np.load(path, allow_pickle=False) as result:
        event_id = str(result["event_id"]) if "event_id" in result else path.stem
        if event_id != event.event_id:
            raise ValueError("result and event IDs differ")
        if "dt" in result and not np.isclose(result["dt"], event.dt):
            raise ValueError("result and event sampling intervals differ")
        if "track_ids" in result and tuple(result["track_ids"].tolist()) != tuple(event.track_ids):
            raise ValueError("result and event agent order differs")
        states = np.asarray(result["states"], dtype=float)
        controls = np.asarray(result["controls"], dtype=float)
    steps = len(event.recorded) - 1
    if states.shape != (steps + 1, event.agents * 4):
        raise ValueError("result states must cover the complete event window")
    if controls.shape != (steps, event.agents * 2):
        raise ValueError("result controls must match the event horizon and agent count")
    states = states.reshape(-1, event.agents, 4)
    controls = controls.reshape(-1, event.agents, 2)
    if not np.isfinite(states).all() or not np.isfinite(controls).all():
        raise ValueError("result contains nonfinite values")
    if not np.allclose(states[0], event.recorded[0], atol=1e-6, rtol=0):
        raise ValueError("result and event initial states differ")
    return states, controls


def plot(event, states, controls, label):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.5, 1))
    trajectory = fig.add_subplot(grid[:, 0])
    yaw = fig.add_subplot(grid[0, 1])
    acceleration = fig.add_subplot(grid[1, 1], sharex=yaw)
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(event.agents)]

    for polygon in event.polygons:
        trajectory.add_patch(Polygon(polygon, facecolor="#eeeeee", edgecolor="#bbbbbb", zorder=0))
    for route in event.routes:
        trajectory.plot(*route.points.T, color="#aaaaaa", linewidth=0.8, linestyle=":", zorder=1)

    recorded = event.recorded
    history = event.history
    time = np.arange(len(controls) + 1) * event.dt
    for i, color in enumerate(colors):
        trajectory.plot(*history[:, i, :2].T, color=color, linestyle=":", linewidth=1)
        trajectory.plot(*recorded[:, i, :2].T, color=color, linestyle="--", linewidth=1.4)
        trajectory.plot(*states[:, i, :2].T, color=color, linewidth=1.8)
        footprint = (Circle(states[0, i, :2], event.sizes[i, 0]/2) if event.domain == "camp" else
                     Polygon(rectangle(states[0, i], event.sizes[i])))
        footprint.set(facecolor=color, edgecolor=color, alpha=0.25)
        trajectory.add_patch(footprint)
        trajectory.plot(*states[-1, i, :2], marker="o", color=color, markersize=4)
        trajectory.plot(*event.goals[i], marker="*", color=color, markersize=8)
        yaw.stairs(controls[:, i, 0], time, color=color, linewidth=1.3, baseline=None)
        acceleration.stairs(controls[:, i, 1], time, color=color, linewidth=1.3, baseline=None)

    points = np.concatenate([history[:, :, :2], recorded[:, :, :2], states[:, :, :2]]).reshape(-1, 2)
    margin = max(2.0, float(np.max(event.sizes)))
    lower, upper = points.min(axis=0) - margin, points.max(axis=0) + margin
    trajectory.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                   xlabel="x (m)", ylabel="y (m)", title=event.event_id)
    trajectory.set_aspect("equal", adjustable="box")
    handles = [Line2D([], [], color=c, label=f"Agent {agent}")
               for agent, c in zip(event.track_ids, colors)]
    handles += [Line2D([], [], color="black", label=label),
                Line2D([], [], color="black", linestyle="--", label="Recorded"),
                Line2D([], [], color="black", linestyle=":", label="History"),
                Line2D([], [], color="black", marker="*", linestyle="none", label="Goal")]
    trajectory.legend(handles=handles, fontsize=8, ncol=2)
    yaw.set(ylabel="Yaw rate (rad/s)", title="Executed controls")
    acceleration.set(xlabel="Time (s)", ylabel="Acceleration (m/s²)")
    for axis in (yaw, acceleration):
        axis.set_xlim(time[0], time[-1])
        axis.grid(alpha=0.2)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, required=True, help="Normalized CAMP, RM, or InterHub event NPZ")
    parser.add_argument("--result", type=Path, required=True, help="Matching solver result NPZ")
    parser.add_argument("--output", type=Path, required=True, help="Figure path: PNG, PDF, or SVG")
    parser.add_argument("--label", default="PA-KL", help="Planner name in the legend")
    parser.add_argument("--show", action="store_true", help="Also open an interactive figure")
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    event = InteractionEvent.load(args.event)
    states, controls = load_result(args.result, event)
    figure = plot(event, states, controls, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
