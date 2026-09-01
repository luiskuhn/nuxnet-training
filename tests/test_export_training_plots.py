import importlib.util
import time
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


SCRIPT = Path(__file__).parents[1] / "tools" / "export_training_plots.py"
SPEC = importlib.util.spec_from_file_location("export_training_plots", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write_scalars(directory, scalars):
    writer = EventFileWriter(str(directory))
    for tag, value in scalars.items():
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        writer.add_event(Event(wall_time=time.time(), step=0, summary=summary))
    writer.close()


def test_exports_grouped_plots_from_newest_run(tmp_path):
    older = tmp_path / "logs" / "version_0"
    newer = tmp_path / "logs" / "version_1"
    write_scalars(older, {"train_avg_loss": 2.0})
    write_scalars(
        newer,
        {
            "train_avg_loss": 1.0,
            "val_avg_loss": 1.5,
            "train_avg_acc": 0.75,
            "test_iou_1": 0.5,
            "unrelated_scalar": 12,
        },
    )

    # Ensure selection is deterministic even on filesystems with coarse timestamps.
    for event_file in older.iterdir():
        event_file.touch()
        event_file.chmod(0o600)
    newest_event = next(newer.iterdir())
    newest_event.touch()
    stat = newest_event.stat()
    import os
    os.utime(newest_event, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    output = tmp_path / "nested" / "plots"
    written = exporter.export_plots(tmp_path / "logs", output)

    assert {path.name for path in written} == {"loss.png", "accuracy.png", "iou_1.png"}
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in written)
    exporter.export_plots(tmp_path / "logs", output)  # Existing files are safely replaced.


def test_no_event_files_is_clear_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="No TensorBoard event files"):
        exporter.find_newest_run(tmp_path)
