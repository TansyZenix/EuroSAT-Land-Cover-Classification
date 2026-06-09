import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


EUROSAT_MEAN = (0.3444, 0.3802, 0.4074)
EUROSAT_STD = (0.2036, 0.1366, 0.1148)


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int


class EuroSATDataset(Dataset):
    def __init__(self, samples, class_names, transform):
        self.samples = list(samples)
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            return self.transform(image), sample.label


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def discover_samples(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]
    if not class_names:
        raise ValueError(f"No class folders found in {data_dir}")

    samples = []
    for label, class_dir in enumerate(class_dirs):
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not image_paths:
            raise ValueError(f"No images found in class folder: {class_dir}")
        samples.extend(ImageSample(path=path, label=label) for path in image_paths)
    return samples, class_names


def stratified_split(samples, val_ratio=0.15, test_ratio=0.15, seed=42):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for label_samples in grouped.values():
        label_samples = list(label_samples)
        rng.shuffle(label_samples)
        total = len(label_samples)
        test_count = max(1, int(round(total * test_ratio)))
        val_count = max(1, int(round(total * val_ratio)))
        test.extend(label_samples[:test_count])
        val.extend(label_samples[test_count : test_count + val_count])
        train.extend(label_samples[test_count + val_count :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_transform(image_size=64, train=False, simulated_resolution=10, use_augmentation=True):
    ops = []
    if simulated_resolution == 30:
        low_size = max(8, image_size // 3)
        ops.extend(
            [
                transforms.Resize((low_size, low_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            ]
        )
    else:
        ops.append(transforms.Resize((image_size, image_size)))

    if train and use_augmentation:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD)])
    return transforms.Compose(ops)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class HRNetClassifier(nn.Module):
    """Small HRNet-style classifier for 64x64 EuroSAT image-level labels."""

    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(3, 32), ConvBNReLU(32, 32))
        self.branch_high = nn.Sequential(BasicBlock(32), BasicBlock(32))
        self.to_medium = ConvBNReLU(32, 48, stride=2)
        self.branch_medium = nn.Sequential(BasicBlock(48), BasicBlock(48))
        self.to_low = ConvBNReLU(48, 64, stride=2)
        self.branch_low = nn.Sequential(BasicBlock(64), BasicBlock(64))
        self.fuse = nn.Sequential(
            ConvBNReLU(32 + 48 + 64, 96),
            BasicBlock(96),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x):
        high = self.branch_high(self.stem(x))
        medium = self.branch_medium(self.to_medium(high))
        low = self.branch_low(self.to_low(medium))
        target_size = high.shape[-2:]
        medium_up = F.interpolate(medium, size=target_size, mode="bilinear", align_corners=False)
        low_up = F.interpolate(low, size=target_size, mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([high, medium_up, low_up], dim=1))
        return self.classifier(torch.flatten(fused, 1))


class HRNetNoFusionClassifier(nn.Module):
    """Ablation model that keeps only the high-resolution branch."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32),
            ConvBNReLU(32, 32),
            BasicBlock(32),
            BasicBlock(32),
            BasicBlock(32),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(torch.flatten(features, 1))


def build_model(num_classes, model_name="hrnet"):
    if model_name == "hrnet":
        return HRNetClassifier(num_classes)
    if model_name == "hrnet_no_fusion":
        return HRNetNoFusionClassifier(num_classes)
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def run_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total += batch_size
        for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
            confusion[true_label, pred_label] += 1

    precision, recall, f1 = macro_scores(confusion.numpy())
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": confusion.numpy(),
    }


def macro_scores(confusion):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for index in range(confusion.shape[0]):
        tp = confusion[index, index]
        fp = confusion[:, index].sum() - tp
        fn = confusion[index, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    return float(np.mean(precision_scores)), float(np.mean(recall_scores)), float(np.mean(f1_scores))


def make_loaders(train_samples, val_samples, test_samples, class_names, args, simulated_resolution=10):
    train_dataset = EuroSATDataset(
        train_samples,
        class_names,
        build_transform(
            args.image_size,
            train=True,
            simulated_resolution=simulated_resolution,
            use_augmentation=args.use_augmentation,
        ),
    )
    eval_transform = build_transform(args.image_size, train=False, simulated_resolution=simulated_resolution)
    val_dataset = EuroSATDataset(val_samples, class_names, eval_transform)
    test_dataset = EuroSATDataset(test_samples, class_names, eval_transform)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_args),
        DataLoader(val_dataset, shuffle=False, **loader_args),
        DataLoader(test_dataset, shuffle=False, **loader_args),
    )


def plot_class_distribution(samples, class_names, output_path):
    counts = Counter(sample.label for sample in samples)
    values = [counts[index] for index in range(len(class_names))]
    plt.figure(figsize=(11, 5))
    plt.bar(class_names, values, color="#3f7f93")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Image count")
    plt.title("EuroSAT class distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_history(history, output_path):
    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train")
    plt.plot(epochs, [item["val_loss"] for item in history], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [item["train_accuracy"] for item in history], label="Train")
    plt.plot(epochs, [item["val_accuracy"] for item in history], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion_matrix(confusion, class_names, output_path):
    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(9, 8))
    plt.imshow(normalized, cmap="YlGnBu", vmin=0, vmax=1)
    plt.colorbar(label="Recall ratio")
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Normalized confusion matrix")
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            value = normalized[row, col]
            color = "white" if value > 0.55 else "black"
            plt.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_resolution_comparison(metrics_10m, metrics_30m, output_path):
    labels = ["10m original", "30m simulated"]
    accuracy = [metrics_10m["accuracy"], metrics_30m["accuracy"]]
    macro_f1 = [metrics_10m["macro_f1"], metrics_30m["macro_f1"]]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, accuracy, width, label="Accuracy", color="#3f7f93")
    plt.bar(x + width / 2, macro_f1, width, label="Macro F1", color="#c57b57")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Resolution robustness comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


@torch.no_grad()
def plot_sample_predictions(model, samples, class_names, transform, device, output_path, max_images=16):
    model.eval()
    chosen = samples[:max_images]
    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    plt.figure(figsize=(12, 3 * rows))
    for idx, sample in enumerate(chosen):
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            pred = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0, pred].item()
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(image)
            color = "green" if pred == sample.label else "red"
            plt.title(
                f"T: {class_names[sample.label]}\nP: {class_names[pred]} ({confidence:.2f})",
                color=color,
                fontsize=9,
            )
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_report(output_dir, args, class_names, history, test_10m, test_30m, figure_paths):
    report_path = output_dir / "work_summary.md"
    best_epoch = max(history, key=lambda item: item["val_accuracy"])
    lines = [
        "# EuroSAT遥感图像地物分类代码与实验说明",
        "",
        "## 工作概述",
        "",
        “本次工作完成了题目12”高分辨率遥感图像地物分类”的代码实现、模型训练、评估与结果图保存。数据集使用 `data/eurosat/2750` 下的 EuroSAT RGB 图像，共10类地物，任务形式为整幅遥感图像分类。”,
        "",
        "## 方法说明",
        "",
        "- 数据划分：按类别分层划分训练集、验证集、测试集，默认比例为70%/15%/15%。",
        f"- 模型：{args.model}分类网络。当前数据集为整图类别标签，因此采用HRNet风格的多分辨率分类结构；DeepLabV3+更适合像素级分割标注。",
        f"- 训练策略：交叉熵损失、AdamW优化器、余弦退火学习率调度，数据增强状态为{'开启' if args.use_augmentation else '关闭'}。",
        "- 分辨率对比：保留原始64x64图像作为10m设置；将图像下采样至约1/3后再放大回64x64，模拟30m低分辨率输入。",
        "",
        "## 关键超参数",
        "",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Learning rate: {args.learning_rate}",
        f"- Image size: {args.image_size}",
        f"- Random seed: {args.seed}",
        f"- Best validation epoch: {best_epoch['epoch']}",
        "",
        "## 实验结果",
        "",
        f"- 10m测试集 Accuracy: {test_10m['accuracy']:.4f}",
        f"- 10m测试集 Macro F1: {test_10m['macro_f1']:.4f}",
        f"- 30m模拟测试集 Accuracy: {test_30m['accuracy']:.4f}",
        f"- 30m模拟测试集 Macro F1: {test_30m['macro_f1']:.4f}",
        "",
        "## 图像与报告描述",
        "",
        f"![类别分布]({figure_paths['class_distribution'].name})",
        "",
        "图1 类别分布图：展示EuroSAT 10类地物样本数量。可以用于报告的数据准备部分，说明数据集类别较均衡但各类数量并不完全一致。",
        "",
        f"![训练曲线]({figure_paths['training_curves'].name})",
        "",
        "图2 训练曲线：展示训练集与验证集的损失、准确率随epoch变化。可以用于说明模型收敛情况和是否存在过拟合。",
        "",
        f"![混淆矩阵]({figure_paths['confusion_matrix'].name})",
        "",
        "图3 归一化混淆矩阵：横轴为预测类别，纵轴为真实类别。对角线越亮表示该类别识别越准确，非对角线可用于分析易混淆地物类型。",
        "",
        f"![样例预测]({figure_paths['sample_predictions'].name})",
        "",
        "图4 样例预测图：展示测试集中若干遥感图像的真实类别、预测类别和置信度。绿色标题表示预测正确，红色标题表示预测错误。",
        "",
        f"![分辨率对比]({figure_paths['resolution_comparison'].name})",
        "",
        "图5 分辨率对比图：比较原始10m输入与模拟30m输入下的Accuracy和Macro F1，用于报告扩展部分讨论空间分辨率降低对分类性能的影响。",
        "",
        "## 输出文件",
        "",
        "- `best_model.pth`：验证集准确率最高的模型权重。",
        "- `metrics.json`：类别名称、训练历史、10m/30m测试指标。",
        "- `class_distribution.png`、`training_curves.png`、`confusion_matrix.png`、`sample_predictions.png`、`resolution_comparison.png`：可直接放入报告的实验图。",
        "",
        "## 运行方式",
        "",
        "```powershell",
        "conda activate eurosat_env",
        "python train_eurosat.py --epochs 20 --batch-size 128 --model hrnet",
        "```",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_args(argv=None):
    default_data = Path(__file__).resolve().parent / "data" / "eurosat" / "2750"
    default_output = Path(__file__).resolve().parent / "results" / "eurosat"
    parser = argparse.ArgumentParser(description="Train and evaluate EuroSAT land-cover classifier.")
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=("hrnet", "hrnet_no_fusion", "resnet18"), default="hrnet")
    parser.add_argument("--no-augmentation", action="store_false", dest="use_augmentation")
    parser.set_defaults(use_augmentation=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    samples, class_names = discover_samples(args.data_dir)
    train_samples, val_samples, test_samples = stratified_split(samples, seed=args.seed)

    train_loader, val_loader, test_loader_10m = make_loaders(
        train_samples, val_samples, test_samples, class_names, args, simulated_resolution=10
    )
    _, _, test_loader_30m = make_loaders(
        train_samples, val_samples, test_samples, class_names, args, simulated_resolution=30
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(class_names), args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    figure_paths = {
        "class_distribution": run_dir / "class_distribution.png",
        "training_curves": run_dir / "training_curves.png",
        "confusion_matrix": run_dir / "confusion_matrix.png",
        "sample_predictions": run_dir / "sample_predictions.png",
        "resolution_comparison": run_dir / "resolution_comparison.png",
    }
    plot_class_distribution(samples, class_names, figure_paths["class_distribution"])

    history = []
    best_val_accuracy = -1.0
    best_path = run_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device, len(class_names))
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "args": vars(args),
                    "best_val_accuracy": best_val_accuracy,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_10m = evaluate(model, test_loader_10m, criterion, device, len(class_names))
    test_30m = evaluate(model, test_loader_30m, criterion, device, len(class_names))

    plot_history(history, figure_paths["training_curves"])
    plot_confusion_matrix(test_10m["confusion_matrix"], class_names, figure_paths["confusion_matrix"])
    plot_sample_predictions(
        model,
        test_samples,
        class_names,
        build_transform(args.image_size, train=False, simulated_resolution=10),
        device,
        figure_paths["sample_predictions"],
    )
    plot_resolution_comparison(test_10m, test_30m, figure_paths["resolution_comparison"])

    metrics = {
        "class_names": class_names,
        "splits": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
        "history": history,
        "test_10m": {k: v for k, v in test_10m.items() if k != "confusion_matrix"},
        "test_30m": {k: v for k, v in test_30m.items() if k != "confusion_matrix"},
        "confusion_matrix": test_10m["confusion_matrix"].tolist(),
        "figures": {key: display_path(path) for key, path in figure_paths.items()},
        "best_model": display_path(best_path),
        "model": args.model,
        "use_augmentation": args.use_augmentation,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(run_dir, args, class_names, history, test_10m, test_30m, figure_paths)

    print(f"Device: {device}")
    print(f"Run directory: {display_path(run_dir)}")
    print(f"Best model: {display_path(best_path)}")
    print(f"Report: {display_path(report_path)}")
    print(f"10m test accuracy: {test_10m['accuracy']:.4f}, macro F1: {test_10m['macro_f1']:.4f}")
    print(f"30m test accuracy: {test_30m['accuracy']:.4f}, macro F1: {test_30m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
