import pytest
import torch
import json
import hashlib

from numorph_nuclei_segmentation.numorph_nuclei_segmentation import (
    build_parser,
    validate_parent_initialization,
)
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


def test_parent_weights_initialize_a_new_training_cycle(tmp_path):
    args = vars(build_parser().parse_args([]))
    source = NumorphSegmentator(**args)
    with torch.no_grad():
        source.model.outc.conv_1.weight.fill_(2.5)
    weights = tmp_path / "parent.pt"
    torch.save(source.model.state_dict(), weights)

    args["initial_weights"] = str(weights)
    child = NumorphSegmentator(**args)

    assert torch.equal(child.model.outc.conv_1.weight, source.model.outc.conv_1.weight)


def test_parent_weights_and_metadata_are_cryptographically_paired(tmp_path):
    weights = tmp_path / "parent.pt"
    weights.write_bytes(b"parent weights")
    metadata = tmp_path / "parent.json"
    metadata.write_text(
        json.dumps({"weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )

    validate_parent_initialization(str(weights), str(metadata))
    weights.write_bytes(b"different weights")

    with pytest.raises(ValueError, match="do not match"):
        validate_parent_initialization(str(weights), str(metadata))
