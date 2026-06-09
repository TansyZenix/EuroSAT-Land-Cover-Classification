from pathlib import Path

from PIL import Image

import train_eurosat as te


def _make_tiny_dataset(root: Path) -> None:
    for class_name in ("AnnualCrop", "Forest"):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for idx in range(5):
            image = Image.new("RGB", (64, 64), color=(idx * 20, 40, 80))
            image.save(class_dir / f"{class_name}_{idx}.jpg")


def test_discover_samples_reads_image_folder_classes(tmp_path: Path) -> None:
    _make_tiny_dataset(tmp_path)

    samples, classes = te.discover_samples(tmp_path)

    assert classes == ["AnnualCrop", "Forest"]
    assert len(samples) == 10
    assert samples[0].label in (0, 1)


def test_stratified_split_preserves_each_class(tmp_path: Path) -> None:
    _make_tiny_dataset(tmp_path)
    samples, _ = te.discover_samples(tmp_path)

    train, val, test = te.stratified_split(samples, val_ratio=0.2, test_ratio=0.2, seed=7)

    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert {sample.label for sample in val} == {0, 1}
    assert {sample.label for sample in test} == {0, 1}


def test_resolution_transform_keeps_output_size() -> None:
    transform = te.build_transform(image_size=64, train=False, simulated_resolution=30)
    image = Image.new("RGB", (64, 64), color=(120, 90, 60))

    tensor = transform(image)

    assert tuple(tensor.shape) == (3, 64, 64)


def test_hrnet_model_outputs_class_logits() -> None:
    model = te.build_model(num_classes=10, model_name="hrnet")
    output = model(te.torch.zeros(2, 3, 64, 64))

    assert output.shape == (2, 10)
    assert model.__class__.__name__ == "HRNetClassifier"


def test_hrnet_no_fusion_model_outputs_class_logits() -> None:
    model = te.build_model(num_classes=10, model_name="hrnet_no_fusion")
    output = model(te.torch.zeros(2, 3, 64, 64))

    assert output.shape == (2, 10)
    assert model.__class__.__name__ == "HRNetNoFusionClassifier"


def test_training_transform_can_disable_augmentation() -> None:
    transform = te.build_transform(image_size=64, train=True, use_augmentation=False)
    op_names = [op.__class__.__name__ for op in transform.transforms]

    assert "RandomHorizontalFlip" not in op_names
    assert "RandomRotation" not in op_names


def test_default_epochs_are_twenty() -> None:
    args = te.parse_args([])

    assert args.epochs == 20
    assert args.model == "hrnet"
