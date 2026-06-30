"""
Train classifier for Screen vs Real photo detection.
Uses the 4 validated features: color_std, noise_level, laplacian_variance, fft_peak_ratio
Run: python train.py
Expects: real/ and screen/ folders in same directory
Outputs: model.pkl (classifier + scaler + feature stats)
"""

import os
import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

REAL_DIR   = "real"
SCREEN_DIR = "screen"
IMG_SIZE   = (256, 256)


# ──────────────────────────────────────────────
# FEATURE EXTRACTION (same as validation script, 4 features only)
# ──────────────────────────────────────────────

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.resize(img, IMG_SIZE)


def feature_fft_peak_ratio(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    h, w = magnitude.shape
    cx, cy = h // 2, w // 2
    magnitude[cx-5:cx+5, cy-5:cy+5] = 0
    peak = np.percentile(magnitude, 99)
    mean = magnitude.mean()
    return float(peak / (mean + 1e-8))


def feature_laplacian_variance(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def feature_color_std(img):
    stds = [img[:,:,c].std() for c in range(3)]
    return float(np.mean(stds))


def feature_noise_level(gray):
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    noise   = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    return float(noise.mean())


def feature_radial_fft_energy(gray):
    """
    Proper moire/screen-grid detector: radial profile of the 2D FFT.
    Screens produce energy concentrated in a specific high-frequency
    annulus (the pixel grid period), which shows up as a bump in the
    radial spectrum that real photos don't have.
    Returns ratio of energy in the high-frequency annulus vs low-mid.
    Robust to lighting because FFT magnitude here is taken on a
    locally-normalized (contrast-equalized) image.
    """
    g = cv2.equalizeHist(gray)  # normalize contrast/lighting first
    f = np.fft.fft2(g.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r = min(cx, cy)

    # radial bins
    low_mid_mask = (r > max_r * 0.05) & (r < max_r * 0.4)
    high_mask    = (r >= max_r * 0.4) & (r < max_r * 0.85)

    low_mid_energy = magnitude[low_mid_mask].mean()
    high_energy    = magnitude[high_mask].mean()

    return float(high_energy / (low_mid_energy + 1e-8))


def feature_lbp_texture_entropy(gray):
    """
    Local Binary Pattern entropy - captures micro-texture independent
    of overall brightness/contrast (lighting invariant by construction,
    since LBP only encodes local relative pixel comparisons).
    Screen recaptures have a more regular/repetitive micro-texture
    (lower entropy) due to the underlying pixel grid; real-world
    surfaces have more varied texture (higher entropy) -- though the
    direction isn't assumed, we just let the classifier learn it.
    """
    # Simple 8-neighbor LBP
    h, w = gray.shape
    g = gray.astype(np.int16)
    center = g[1:-1, 1:-1]

    lbp = np.zeros_like(center, dtype=np.uint8)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for i, (dy, dx) in enumerate(offsets):
        neighbor = g[1+dy:h-1+dy, 1+dx:w-1+dx]
        lbp |= ((neighbor >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(lbp, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def feature_edge_density(gray):
    """
    Canny edge density. Screen photos often show a faint secondary
    edge pattern from pixel/subpixel boundaries layered on top of the
    photographed content, increasing edge density vs a real photo of
    similar content. Auto-thresholded via median to stay lighting-robust.
    """
    median = np.median(gray)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    return float(np.mean(edges > 0))


def feature_local_contrast_std(gray):
    """
    Std of local contrast (block-wise std of pixel values), normalized
    by global contrast. Screens tend to have more UNIFORM local contrast
    across blocks (panel backlighting evens things out) vs real scenes
    which have more variable local contrast (depth, shadows, materials).
    This ratio form cancels out absolute lighting level.
    """
    block = 16
    h, w = gray.shape
    h2, w2 = h - h % block, w - w % block
    g = gray[:h2, :w2].astype(np.float32)
    blocks = g.reshape(h2//block, block, w2//block, block).swapaxes(1,2).reshape(-1, block, block)
    local_stds = blocks.std(axis=(1,2))
    global_std = g.std() + 1e-8
    return float(local_stds.std() / global_std)


def extract_features(img):
    """Returns feature vector in FIXED order - must match predict.py"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return [
        feature_fft_peak_ratio(gray),
        feature_laplacian_variance(gray),
        feature_color_std(img),
        feature_noise_level(gray),
        feature_radial_fft_energy(gray),
        feature_lbp_texture_entropy(gray),
        feature_edge_density(gray),
        feature_local_contrast_std(gray),
    ]

FEATURE_NAMES = ["fft_peak_ratio", "laplacian_variance", "color_std", "noise_level",
                  "radial_fft_energy", "lbp_texture_entropy", "edge_density", "local_contrast_std"]


# ──────────────────────────────────────────────
# DATASET LOADING
# ──────────────────────────────────────────────

def load_dataset():
    X, y, paths = [], [], []
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for label, folder in [(0, REAL_DIR), (1, SCREEN_DIR)]:  # 0=real, 1=screen
        files = [p for p in Path(folder).iterdir() if p.suffix in exts]
        print(f"Loading {len(files)} images from {folder}/")
        for p in files:
            img = load_image(p)
            if img is not None:
                X.append(extract_features(img))
                y.append(label)
                paths.append(str(p))

    return np.array(X), np.array(y), paths


# ──────────────────────────────────────────────
# TRAIN + EVALUATE
# ──────────────────────────────────────────────

def main():
    print("🔍 Loading dataset and extracting features...")
    X, y, paths = load_dataset()
    print(f"\nTotal: {len(X)} images ({(y==0).sum()} real, {(y==1).sum()} screen)")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation to get honest accuracy estimate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*50)
    print("MODEL COMPARISON (5-fold cross-validation)")
    print("="*50)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "SVM (RBF)":            SVC(kernel="rbf", C=1.0),
        "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
    }

    best_model_name = None
    best_score = 0

    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        print(f"{name:<22} acc = {mean_acc:.4f}  (folds: {np.round(scores,3)})")
        if mean_acc > best_score:
            best_score = mean_acc
            best_model_name = name

    print(f"\n🏆 Best model: {best_model_name} ({best_score:.4f} accuracy)")

    # Train final model on FULL dataset
    # Wrap in calibration so predict_proba is available regardless of model type
    from sklearn.calibration import CalibratedClassifierCV
    base_model = models[best_model_name]
    final_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    final_model.fit(X_scaled, y)

    # ── Honest misclassification analysis via cross-validated predictions ──
    from sklearn.model_selection import cross_val_predict
    cv_preds = cross_val_predict(models[best_model_name], X_scaled, y, cv=skf)
    print("\n" + "="*50)
    print("MISCLASSIFIED IMAGES (held-out CV predictions)")
    print("="*50)
    n_wrong = 0
    for i in range(len(y)):
        if cv_preds[i] != y[i]:
            n_wrong += 1
            true_label = "real" if y[i] == 0 else "screen"
            pred_label = "real" if cv_preds[i] == 0 else "screen"
            print(f"  {paths[i]:<40} true={true_label:<7} predicted={pred_label}")
    print(f"\nTotal misclassified: {n_wrong}/{len(y)}")

    # ── Feature importance (only meaningful for tree models) ──
    if hasattr(base_model, "feature_importances_") or best_model_name in ("Random Forest", "Gradient Boosting"):
        rf_check = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        rf_check.fit(X_scaled, y)
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE (Random Forest, for diagnostic purposes)")
        print("="*50)
        importances = rf_check.feature_importances_
        order = np.argsort(importances)[::-1]
        for idx in order:
            print(f"  {FEATURE_NAMES[idx]:<22} {importances[idx]:.4f}")

    # In-sample predictions for confusion matrix / report (informational only,
    # the CV score above is the honest estimate)
    y_pred = final_model.predict(X_scaled)
    print("\n--- Full-data fit (NOT held-out, for reference only) ---")
    print(classification_report(y, y_pred, target_names=["real", "screen"]))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))

    # Latency benchmark
    print("\n⏱  Benchmarking latency...")
    sample_img = load_image(Path(paths[0]))
    n_runs = 50
    start = time.perf_counter()
    for _ in range(n_runs):
        feats = extract_features(sample_img)
        feats_scaled = scaler.transform([feats])
        _ = final_model.predict_proba(feats_scaled)
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    print(f"   Average latency: {elapsed:.2f} ms/image (CPU)")

    # Save everything needed for predict.py
    bundle = {
        "model": final_model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "img_size": IMG_SIZE,
        "cv_accuracy": best_score,
        "model_name": best_model_name,
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n✅ Saved model.pkl ({best_model_name}, CV accuracy: {best_score:.4f})")
    print(f"   Latency: ~{elapsed:.1f} ms/image on CPU")


if __name__ == "__main__":
    main()
