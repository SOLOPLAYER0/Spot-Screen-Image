import os
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np

REAL_DIR = "real"
SCREEN_DIR = "screen"
IMG_SIZE = 320           
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────

class ScreenDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def gather_samples():
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    samples = []
    for label, folder in [(0, REAL_DIR), (1, SCREEN_DIR)]:  # 0=real, 1=screen
        for p in Path(folder).iterdir():
            if p.suffix in exts:
                samples.append((str(p), label))
    return samples



train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────

def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.features[:-3].parameters():
        param.requires_grad = False

    # Slightly higher dropout for the same reason.
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 1)
    )
    return model.to(device)


# ──────────────────────────────────────────────
# TRAIN / EVAL LOOP
# ──────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


def train_one_fold(train_samples, val_samples, fold_num, verbose=True):
    train_ds = ScreenDataset(train_samples, train_transform)
    val_ds = ScreenDataset(val_samples, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.BCEWithLogitsLoss()
    # Weight decay (L2 regularization) added to further fight overfitting
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 5, 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step()

        # Early stopping on val_loss (more stable signal than val_acc with n=20)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            marker = "  <- best" if improved else ""
            print(f"  Fold {fold_num} Epoch {epoch:2d}/{EPOCHS}  "
                  f"train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch} (no val_loss improvement for {patience} epochs)")
            break

    return best_val_acc, best_val_loss, best_state


def main():
    print("Gathering dataset...")
    samples = gather_samples()
    n_real = sum(1 for _, l in samples if l == 0)
    n_screen = sum(1 for _, l in samples if l == 1)
    print(f"Total: {len(samples)} images ({n_real} real, {n_screen} screen)")

    rng = np.random.RandomState(SEED)
    real_samples = [s for s in samples if s[1] == 0]
    screen_samples = [s for s in samples if s[1] == 1]
    rng.shuffle(real_samples)
    rng.shuffle(screen_samples)

    N_FOLDS = 5

    def make_folds(lst, n_folds):
        """Split a list of (path, label) tuples into n_folds roughly-equal chunks."""
        folds = [[] for _ in range(n_folds)]
        for i, item in enumerate(lst):
            folds[i % n_folds].append(item)
        return folds

    real_folds = make_folds(real_samples, N_FOLDS)
    screen_folds = make_folds(screen_samples, N_FOLDS)

    print("\n" + "="*50)
    print(f"{N_FOLDS}-FOLD CROSS-VALIDATION")
    print("="*50)

    fold_accs = []
    best_overall_acc = 0.0
    best_overall_state = None

    for fold_idx in range(N_FOLDS):
        val_samples = real_folds[fold_idx] + screen_folds[fold_idx]
        train_samples = []
        for i in range(N_FOLDS):
            if i != fold_idx:
                train_samples += real_folds[i] + screen_folds[i]

        print(f"\n--- Fold {fold_idx+1}/{N_FOLDS} (train={len(train_samples)}, val={len(val_samples)}) ---")
        val_acc, val_loss, state = train_one_fold(train_samples, val_samples, fold_idx+1, verbose=True)
        fold_accs.append(val_acc)
        print(f"  Fold {fold_idx+1} best: val_acc={val_acc:.4f} val_loss={val_loss:.4f}")

        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_overall_state = state

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print("\n" + "="*50)
    print("CROSS-VALIDATION SUMMARY")
    print("="*50)
    print(f"Per-fold accuracy: {[f'{a:.3f}' for a in fold_accs]}")
    print(f"Mean CV accuracy:  {mean_acc:.4f} (+/- {std_acc:.4f})")
    print("\nThis mean is your honest, reportable accuracy estimate.")

    # ── Final model: retrain on ALL data using the best fold's hyperparameters/epoch count ──
    # (Using all 100 images for the deployed model, since more data = better,
    #  while the CV mean above remains the honest accuracy estimate to report.)
    print("\n" + "="*50)
    print("Training FINAL model on full dataset (for deployment)")
    print("="*50)
    full_train_ds = ScreenDataset(samples, train_transform)
    full_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    final_model = build_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, final_model.parameters()),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    FINAL_EPOCHS = 8
    for epoch in range(1, FINAL_EPOCHS + 1):
        train_loss, train_acc = run_epoch(final_model, full_loader, criterion, optimizer)
        scheduler.step()
        print(f"  Final Epoch {epoch}/{FINAL_EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

    final_state = final_model.state_dict()

    # Latency benchmark (CPU)
    print("\n⏱  Benchmarking latency (CPU)...")
    model_cpu = build_model()
    model_cpu.load_state_dict(final_state)
    model_cpu.to("cpu")
    model_cpu.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    n_runs = 30
    for _ in range(3):
        with torch.no_grad():
            _ = model_cpu(dummy)
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model_cpu(dummy)
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    print(f"   Average latency: {elapsed:.2f} ms/image (CPU)")

    torch.save({
        "model_state_dict": final_state,
        "img_size": IMG_SIZE,
        "cv_mean_accuracy": float(mean_acc),
        "cv_std_accuracy": float(std_acc),
        "cv_fold_accuracies": [float(a) for a in fold_accs],
        "architecture": "mobilenet_v2",
    }, "model.pth")

    n_params = sum(p.numel() for p in final_model.parameters())
    print(f"\n✅ Saved model.pth")
    print(f"   CV accuracy (honest estimate): {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"   Latency: ~{elapsed:.1f} ms/image on CPU")
    print(f"   Model size: {n_params:,} params (~{n_params*4/1e6:.1f} MB as float32)")


if __name__ == "__main__":
    main()
