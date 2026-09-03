#!/usr/bin/env python3
"""Export scalar metrics from the newest TensorBoard run as PNG or SVG plots."""

import argparse
import os
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PHASES = ("train", "val", "test")
DISPLAY_NAMES = {"avg_loss": "Loss", "avg_acc": "Accuracy", "learning_rate": "Learning rate"}
FILE_NAMES = {"avg_loss": "loss", "avg_acc": "accuracy", "learning_rate": "learning_rate"}
PLOT_FORMATS = ("png", "svg")


def find_newest_run(logdir: Path) -> Path:
    """Return the directory containing the most recently modified event file."""
    event_files = [path for path in logdir.rglob("events.out.tfevents.*") if path.is_file()]
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found below {logdir}")
    newest = max(event_files, key=lambda path: path.stat().st_mtime_ns)
    return newest.parent


def metric_groups(tags: list[str]) -> dict[str, dict[str, str]]:
    """Group phase metrics and Lightning learning-rate scalar tags."""
    groups: dict[str, dict[str, str]] = {}
    for tag in tags:
        # Lightning commonly prefixes a tag with a namespace, such as "metrics/".
        leaf = tag.rsplit("/", 1)[-1]
        match = re.fullmatch(r"(train|val|validation|test)_(.+)", leaf)
        if match:
            phase, metric = match.groups()
            phase = "val" if phase == "validation" else phase
            groups.setdefault(metric, {})[phase] = tag
            continue
        learning_rate = re.search(r"(?:^|/)(lr-[^/]+)(?:/(.+))?$", tag)
        if learning_rate:
            optimizer, parameter_group = learning_rate.groups()
            label = optimizer if parameter_group is None else f"{optimizer}/{parameter_group}"
            groups.setdefault("learning_rate", {})[label] = tag
    return groups


def _plot_name(metric: str) -> str:
    return FILE_NAMES.get(metric, re.sub(r"[^A-Za-z0-9_.-]+", "_", metric).strip("_"))


def export_plots(logdir: Path, output_dir: Path, plot_format: str = "png") -> list[Path]:
    """Reload the newest run and atomically export every phase-grouped scalar plot."""
    if plot_format not in PLOT_FORMATS:
        raise ValueError(f"Plot format must be one of: {', '.join(PLOT_FORMATS)}")
    run_dir = find_newest_run(logdir)
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    groups = metric_groups(accumulator.Tags().get("scalars", []))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for metric, phase_tags in sorted(groups.items()):
        fig, axis = plt.subplots(figsize=(8, 5))
        plotted = False
        ordered_labels = [phase for phase in PHASES if phase in phase_tags]
        ordered_labels.extend(sorted(label for label in phase_tags if label not in PHASES))
        for label in ordered_labels:
            tag = phase_tags[label]
            if not tag:
                continue
            events = accumulator.Scalars(tag)
            if not events:  # A tag may exist before its first complete event is readable.
                continue
            axis.plot([event.step for event in events], [event.value for event in events], label=label)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue

        title = DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())
        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.set_ylabel(title)
        axis.legend(title="Optimizer" if metric == "learning_rate" else "Phase")
        axis.grid(alpha=0.25)
        if metric == "avg_acc" or "iou" in metric.lower():
            axis.set_ylim(0, 1)
        fig.tight_layout()

        destination = output_dir / f"{_plot_name(metric)}.{plot_format}"
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-", suffix=f".{plot_format}", dir=output_dir
        )
        os.close(fd)
        try:
            metadata = {"Date": None} if plot_format == "svg" else None
            with matplotlib.rc_context({"svg.hashsalt": "nuxnet-training"}):
                fig.savefig(temporary_name, dpi=150, format=plot_format, metadata=metadata)
            os.replace(temporary_name, destination)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
            plt.close(fig)
        written.append(destination)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True, help="Directory containing TensorBoard runs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported plot files")
    parser.add_argument(
        "--format", choices=PLOT_FORMATS, default="png", dest="plot_format", help="Plot file format (default: png)"
    )
    args = parser.parse_args()
    try:
        files = export_plots(args.logdir, args.output_dir, args.plot_format)
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Exported {len(files)} plot(s) from {find_newest_run(args.logdir)} to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
