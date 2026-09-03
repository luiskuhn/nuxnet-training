import pytest
import torch

from numorph_nuclei_segmentation.numorph_nuclei_segmentation import build_parser
from numorph_nuclei_segmentation.model.model import NumorphSegmentator


def test_plateau_scheduler_uses_configurable_training_loss_strategy():
    args = vars(build_parser().parse_args([]))
    model = NumorphSegmentator(**args)

    configuration = model.configure_optimizers()
    scheduler_config = configuration["lr_scheduler"]
    scheduler = scheduler_config["scheduler"]

    assert isinstance(configuration["optimizer"], torch.optim.Adam)
    assert scheduler_config["monitor"] == "train_avg_loss"
    assert scheduler.mode == "min"
    assert scheduler.factor == pytest.approx(0.5)
    assert scheduler.patience == 5
    assert scheduler.threshold == pytest.approx(1e-5)
    assert scheduler.threshold_mode == "abs"
    assert scheduler.cooldown == 2
    assert scheduler.min_lrs == pytest.approx([1e-6])
