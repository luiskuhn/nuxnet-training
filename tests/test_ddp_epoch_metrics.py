import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from numorph_nuclei_segmentation.metrics.metrics import metrics_from_confusion


@pytest.mark.parametrize("batch_size", [1, 2])
def test_two_process_ddp_completes_and_aggregates_exact_counts(tmp_path, batch_size):
    output = tmp_path / f"batch-{batch_size}"
    job = Path(__file__).with_name("ddp_confusion_job.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(job),
            "--batch-size",
            str(batch_size),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    reports = [
        json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(2)
    ]
    assert [report["rank"] for report in reports] == [0, 1]
    counts = torch.tensor(reports[0]["counts"])
    iou, _, _ = metrics_from_confusion(counts)
    for report in reports:
        assert report["events"]["train_start"] == [0, 1]
        assert report["events"]["train_end"] == [0, 1]
        assert any(event["sanity"] for event in report["events"]["validation"])
        assert sum(not event["sanity"] for event in report["events"]["validation"]) == 2
        assert report["events"]["test"] == [2]
        assert report["val_iou_1"] == pytest.approx(iou[1].item())
        assert report["test_iou_1"] == pytest.approx(iou[1].item())
        assert report["checkpoint_exists"]
        assert report["scheduler_monitor"] == "val_iou_1"
