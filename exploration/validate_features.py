

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


REAL_DIR   = "real"
SCREEN_DIR = "screen"
IMG_SIZE   = (256, 256)   # resize for consistent feature extraction



def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    return img


def feature_fft_peak_ratio(gray):

    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    # Mask out DC component (center)
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


def feature_moire_score(gray):

    # Horizontal projection
    row_means = gray.mean(axis=1).astype(np.float32)
    fft_rows  = np.abs(np.fft.rfft(row_means))

    # Mid-freq band energy (skip DC and very high freq noise)
    n = len(fft_rows)
    mid_lo, mid_hi = n // 8, n // 2
    mid_energy = fft_rows[mid_lo:mid_hi].mean()
    total_energy = fft_rows[1:].mean() + 1e-8
    return float(mid_energy / total_energy)


def feature_hsv_saturation_mean(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[:,:,1].mean())


def feature_noise_level(gray):

    # Difference from a slightly blurred version
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    noise   = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    return float(noise.mean())


def extract_all_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "fft_peak_ratio":       feature_fft_peak_ratio(gray),
        "laplacian_variance":   feature_laplacian_variance(gray),
        "color_std":            feature_color_std(img),
        "moire_score":          feature_moire_score(gray),
        "hsv_saturation_mean":  feature_hsv_saturation_mean(img),
        "noise_level":          feature_noise_level(gray),
    }



def load_dataset():
    data = {"real": [], "screen": []}
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for label, folder in [("real", REAL_DIR), ("screen", SCREEN_DIR)]:
        paths = [p for p in Path(folder).iterdir() if p.suffix in exts]
        print(f"Loading {len(paths)} images from {folder}/")
        for p in paths:
            img = load_image(p)
            if img is not None:
                feats = extract_all_features(img)
                data[label].append(feats)
            else:
                print(f"  ⚠ Could not load: {p}")

    return data



def visualize(data):
    feature_names = list(data["real"][0].keys())
    real_arr   = {f: [d[f] for d in data["real"]]   for f in feature_names}
    screen_arr = {f: [d[f] for d in data["screen"]] for f in feature_names}

    n_feats = len(feature_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Feature Distribution: Real vs Screen", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        ax = axes[i]
        r = real_arr[feat]
        s = screen_arr[feat]

        # KDE / histogram overlay
        ax.hist(r, bins=20, alpha=0.6, color="steelblue", label="real",   density=True)
        ax.hist(s, bins=20, alpha=0.6, color="tomato",    label="screen", density=True)

        # Mann-Whitney U p-value (non-parametric separability test)
        _, p = stats.mannwhitneyu(r, s, alternative="two-sided")
        sep = "separable" if p < 0.05 else "❌ overlapping"

        ax.set_title(f"{feat}\n(p={p:.4f}) {sep}", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlabel("value", fontsize=8)

    plt.tight_layout()
    plt.savefig("feature_distributions.png", dpi=150)
    print("\nSaved: feature_distributions.png")
    plt.show()


def print_summary(data):
    feature_names = list(data["real"][0].keys())
    print("\n" + "="*60)
    print(f"{'Feature':<25} {'Real mean':>12} {'Screen mean':>12} {'Ratio':>8}")
    print("="*60)
    for feat in feature_names:
        r_mean = np.mean([d[feat] for d in data["real"]])
        s_mean = np.mean([d[feat] for d in data["screen"]])
        ratio  = s_mean / (r_mean + 1e-8)
        print(f"{feat:<25} {r_mean:>12.4f} {s_mean:>12.4f} {ratio:>8.2f}x")
    print("="*60)
    print(f"\nDataset: {len(data['real'])} real, {len(data['screen'])} screen images")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("🔍 Loading dataset...")
    data = load_dataset()

    if not data["real"] or not data["screen"]:
        print(" Could not load images. Check that real/ and screen/ folders exist.")
        exit(1)

    print_summary(data)
    print("\n Generating feature distribution plots...")
    visualize(data)

    print("\n Done! Check feature_distributions.png")
    print("   Features marked are useful for your classifier.")
    print("   Features marked may not help — consider dropping them.")
