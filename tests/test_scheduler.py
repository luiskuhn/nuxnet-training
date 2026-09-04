import pytest
import torch
import json
import hashlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from numorph_nuclei_segmentation.numorph_nuclei_segmentation import (
    build_parser,
    validate_parent_initialization,
)
from numorph_nuclei_segmentation.model.model import NumorphSegmentator


def test_plateau_scheduler_uses_configurable_validation_iou_strategy():
    args = vars(build_parser().parse_args([]))
    model = NumorphSegmentator(**args)

    configuration = model.configure_optimizers()
    scheduler_config = configuration["lr_scheduler"]
    scheduler = scheduler_config["scheduler"]

    assert isinstance(configuration["optimizer"], torch.optim.Adam)
    assert scheduler_config["monitor"] == "val_iou_1"
    assert scheduler_config["interval"] == "epoch"
    assert scheduler_config["frequency"] == args["test_epochs"]
    assert scheduler.mode == "max"
    assert scheduler.factor == pytest.approx(0.5)
    assert scheduler.patience == 5
    assert scheduler.threshold == pytest.approx(1e-5)
    assert scheduler.threshold_mode == "abs"
    assert scheduler.cooldown == 2
    assert scheduler.min_lrs == pytest.approx([1e-6])


@pytest.mark.parametrize("value", ["0", "-1"])
def test_test_epochs_must_be_positive_on_cli(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--test-epochs", value])


@pytest.mark.parametrize("value", [0, -1])
def test_test_epochs_must_be_positive_in_configuration(value):
    args = vars(build_parser().parse_args([]))
    args["test_epochs"] = value
    with pytest.raises(ValueError, match="test_epochs must be a positive integer"):
        NumorphSegmentator(**args)


class _TinyEpochAwareNetwork(torch.nn.Module):
    """Cheap training network with distinct predictions for each validation run."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, kernel_size=1)
        self.validation_class = 1

    def forward(self, image):
        if self.training:
            return self.conv(image)
        logits = self.conv(image) * 0
        logits[:, self.validation_class] += 1
        return logits


class _RecordingSegmentator(NumorphSegmentator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = _TinyEpochAwareNetwork()
        self.scheduler_steps = []
        self.validation_observations = []

    def on_validation_epoch_start(self):
        # The first and second real validations intentionally yield different IoUs.
        self.model.validation_class = 1 if self.current_epoch == 2 else 0

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        metric = float(self.trainer.callback_metrics["val_iou_1"])
        self.validation_observations.append((self.current_epoch, metric))

    def configure_optimizers(self):
        configuration = super().configure_optimizers()
        scheduler = configuration["lr_scheduler"]["scheduler"]
        original_step = scheduler.step

        def record_step(metric, *args, **kwargs):
            self.scheduler_steps.append((self.current_epoch, float(metric)))
            return original_step(metric, *args, **kwargs)

        scheduler.step = record_step
        return configuration


def test_trainer_steps_plateau_scheduler_only_after_real_validation():
    args = vars(build_parser().parse_args([]))
    args.update(
        max_epochs=6,
        test_epochs=3,
        lr_scheduler_patience=10,
        patch_size="2,2,2",
    )
    model = _RecordingSegmentator(**args)
    images = torch.ones(1, 1, 2, 2, 2)
    targets = torch.ones(1, 2, 2, 2, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, targets), batch_size=1)
    trainer = pl.Trainer(
        max_epochs=args["max_epochs"],
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=args["test_epochs"],
    )

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

    assert model.validation_observations == [(2, 1.0), (5, 0.0)]
    assert model.scheduler_steps == model.validation_observations


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
        json.dumps(
            {"weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
        ),
        encoding="utf-8",
    )

    validate_parent_initialization(str(weights), str(metadata))
    weights.write_bytes(b"different weights")

    with pytest.raises(ValueError, match="do not match"):
        validate_parent_initialization(str(weights), str(metadata))
