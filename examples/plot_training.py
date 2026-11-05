"""Plot the saved inverse-training loss and parameter-update norm."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="training_log.json from inverse training")
    parser.add_argument("--output", type=Path, required=True, help="Figure path: PNG, PDF, or SVG")
    parser.add_argument("--show", action="store_true", help="Also open an interactive figure")
    args = parser.parse_args()

    rows = json.loads(args.log.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("training log is empty")
    values = np.array([[r["update"], r["loss"], r["parameter_delta"]] for r in rows], dtype=float)
    if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("training log must contain finite, ordered updates")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
    for axis, column, ylabel in zip(axes, (1, 2), ("Training loss", "Parameter update L2 norm")):
        axis.plot(values[:, 0], values[:, column], color="#2463a0", linewidth=1.5)
        axis.set(xlabel="Update", ylabel=ylabel)
        axis.grid(alpha=0.2)
    if np.all(values[:, 2] > 0):
        axes[1].set_yscale("log")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
