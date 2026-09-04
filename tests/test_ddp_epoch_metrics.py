import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.ddp_confusion_job import PREDICTIONS, TARGETS


ACCELERATORS = ["cpu"] + (["gpu"] if torch.cuda.device_count() >= 2 else [])


@pytest.mark.parametrize("accelerator", ACCELERATORS)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_two_process_ddp_completes_and_aggregates_metrics(
    tmp_path, batch_size, accelerator
):
    output = tmp_path / f"{accelerator}-batch-{batch_size}"
    job = Path(__file__).with_name("ddp_confusion_job.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(job),
            "--batch-size",
            str(batch_size),
            "--output",
            str(output),
            "--accelerator",
            accelerator,
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
    true_positive = ((PREDICTIONS == 1) & (TARGETS == 1)).sum().item()
    union = ((PREDICTIONS == 1) | (TARGETS == 1)).sum().item()
    expected_iou = true_positive / union
    for report in reports:
        assert report["events"]["train_start"] == [0]
        assert report["events"]["train_end"] == [0]
        assert any(event["sanity"] for event in report["events"]["validation"])
        assert sum(not event["sanity"] for event in report["events"]["validation"]) == 1
        assert report["events"]["test"] == [1]
        assert report["val_iou_1"] == pytest.approx(expected_iou)
        assert report["test_iou_1"] == pytest.approx(expected_iou)
        assert report["checkpoint_exists"]
        assert report["scheduler_monitor"] == "val_iou_1"
