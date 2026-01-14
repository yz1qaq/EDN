import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABEL_NAMES = ["好", "乐", "怒", "哀", "惧", "恶", "惊", "无情绪"]


def compute_class_weights_from_loader(train_loader, n_classes: int, device) -> torch.Tensor:
    labels = []
    for batch in train_loader:
        y = batch["label"].detach().cpu().numpy().tolist()
        labels.extend(y)

    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=n_classes)
    counts = np.maximum(counts, 1)  # avoid div0
    N = int(counts.sum())
    C = int(n_classes)
    weights = N / (C * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        alpha = self.alpha.to(logits.device) if self.alpha is not None else None
        ce = F.cross_entropy(logits, targets, reduction="none", weight=alpha)
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0, lambda_ce: float = 0.7):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.focal = FocalLoss(alpha=class_weights, gamma=gamma)
        self.lambda_ce = float(lambda_ce)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce_loss = self.ce(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.lambda_ce * ce_loss + (1.0 - self.lambda_ce) * focal_loss


@torch.no_grad()
def evaluate(model, dataloader, device) -> Tuple[float, float, str]:
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds, target_names=LABEL_NAMES, digits=3, zero_division=0
    )
    return float(acc), float(f1), report


def train(
    model,
    train_loader,
    valid_loader,
    device,
    save_path: str = "saved/best_model.pt",
    epochs: int = 30,
    lr: float = 2e-5,
    early_stopping_patience: int = 5,
    best_f1: float = 0.0,
    loss_strategy: str = "combined",
    focal_gamma: float = 2.0,
    lambda_ce: float = 0.7,
):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    n_classes = len(LABEL_NAMES)
    class_weights = compute_class_weights_from_loader(train_loader, n_classes, device)
    print(f"类别权重: {class_weights.detach().cpu().numpy().round(4).tolist()}")

    if loss_strategy == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_strategy == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    else:
        criterion = CombinedLoss(class_weights=class_weights, gamma=focal_gamma, lambda_ce=lambda_ce)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    history = {"train_loss": [], "valid_acc": [], "valid_f1": [], "learning_rate": []}
    best_acc = 0.0
    es_counter = 0

    print(f"开始训练: epochs={epochs}, lr={lr}, loss={loss_strategy}, save={save_path}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = datetime.now()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)
        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{running/step:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        train_loss = running / max(len(train_loader), 1)
        lr_now = float(optimizer.param_groups[0]["lr"])

        acc, f1, report = evaluate(model, valid_loader, device)
        dt = (datetime.now() - t0).total_seconds()

        print(f"\nEpoch {epoch}/{epochs} | {dt:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Valid Acc: {acc:.4f} | Valid Macro-F1: {f1:.4f} | LR: {lr_now:.2e}")
        print(report)

        history["train_loss"].append(train_loss)
        history["valid_acc"].append(acc)
        history["valid_f1"].append(f1)
        history["learning_rate"].append(lr_now)

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            es_counter = 0

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "best_acc": best_acc,
                "history": history,
            }
            torch.save(ckpt, save_path)
            print(f"保存新最佳模型: {save_path} (Macro-F1={best_f1:.4f}, Acc={best_acc:.4f})")
        else:
            es_counter += 1
            print(f"验证集 Macro-F1 未提升: {es_counter}/{early_stopping_patience}")

        if es_counter >= early_stopping_patience:
            print(f"早停触发: 连续 {early_stopping_patience} 次未提升")
            break

        print("-" * 60)

    hist_path = save_path.replace(".pt", "_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"训练结束 | best Macro-F1={best_f1:.4f}, best Acc={best_acc:.4f}")
    print(f"History saved: {hist_path}")

    return history


def plot_history(history: Dict, save_fig_path: str = "saved/train_plot.png"):
    os.makedirs(os.path.dirname(save_fig_path) if os.path.dirname(save_fig_path) else ".", exist_ok=True)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], marker="o")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history["valid_acc"], marker="s")
    plt.title("Valid Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(history["valid_f1"], marker="^")
    plt.title("Valid Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存: {save_fig_path}")


@torch.no_grad()
def test_model(model, test_loader, device, save_results: bool = True, results_path: str = "saved/test_results.json"):
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(test_loader, desc="Testing", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_f1 = f1_score(all_labels, all_preds, average=None)
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, digits=3, zero_division=0)

    print(f"\nTest Acc: {acc:.4f} | Test Macro-F1: {macro_f1:.4f}")
    print(report)

    if save_results:
        os.makedirs(os.path.dirname(results_path) if os.path.dirname(results_path) else ".", exist_ok=True)
        payload = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "f1_per_class": {LABEL_NAMES[i]: float(per_f1[i]) for i in range(len(LABEL_NAMES))},
            "classification_report": report,
            "predictions": [int(p) for p in all_preds],
            "true_labels": [int(y) for y in all_labels],
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"测试结果已保存: {results_path}")

    return all_preds, all_labels


def load_checkpoint(model, checkpoint_path: str, device):
    if not os.path.exists(checkpoint_path):
        print(f"未找到 checkpoint: {checkpoint_path}")
        return None

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_f1 = ckpt.get("best_f1", None)
    best_acc = ckpt.get("best_acc", None)

    if best_f1 is not None and best_acc is not None:
        print(f"已加载 checkpoint: {checkpoint_path} | best_f1={best_f1:.4f}, best_acc={best_acc:.4f}")
    else:
        print(f"已加载 checkpoint: {checkpoint_path}")

    return ckpt
