# Spot the Fake Photo 🔍
 
A lightweight screen recapture detector — given a photo, it tells you whether it's a **real photo** or a **photo of a screen** (a "recapture").
 
Built with a fine-tuned MobileNetV2 CNN. Runs in ~45ms on CPU, ~9MB model, deployable on-device.
 
---
 
## The Problem
 
Mobile apps can be cheated: instead of taking a real photo of an object, a user photographs a screen (phone/laptop) displaying a picture of that object. This detector flags such attempts.
 
```
python predict.py some_image.jpg
→ 0.04   # real photo (close to 0)
→ 0.93   # photo of a screen (close to 1)
```
 
---
 
## How It Works
 
### Approach
I first tried hand-crafted CV features (FFT frequency peaks, Laplacian variance, LBP texture entropy, edge density) with an SVM/Random Forest. This capped at ~84% accuracy — the features were too sensitive to lighting and angle variation to reliably detect the subtle moire/pixel-grid artifacts of a recaptured screen.
 
I switched to **fine-tuning MobileNetV2** (pretrained on ImageNet):
- Froze most of the network, trained only the **last 3 feature blocks + a new binary classification head**
- Used **320×320 input** (above MobileNetV2's default 224) — screen recapture artifacts are fine-grained and get lost at lower resolution; this single change was the biggest accuracy jump
- Heavy **data augmentation** (brightness/contrast jitter, rotation, random crop, random erasing) to handle the lighting and angle variation in the dataset
### Results
| Metric | Value |
|--------|-------|
| Accuracy | 94.0% ± 5.8% (5-fold CV) |
| Latency | ~45ms / image (CPU) |
| Model size | ~9MB |
| Cost at scale | ~$0 on-device / ~$1 per million images cloud |
 
---
 
## Project Structure
 
```
├── predict.py          # One-line predictor: python predict.py image.jpg
├── train_cnn.py        # Full training pipeline (5-fold CV + final model)
├── model.pth           # Trained MobileNetV2 weights
├── server.py           # Flask backend for live demo
├── demo.html           # Live webcam demo page
├── REPORT.md           # Approach note (accuracy, latency, cost, bonus Qs)
└── exploration/        # Earlier hand-crafted feature approach (SVM/RF, ~84%)
    ├── validate_features.py
    ├── train.py
    └── feature_distributions.png
```
 
---
 
## Quickstart
 
**Install dependencies:**
```bash
pip install torch torchvision pillow
```
 
**Run prediction on a single image:**
```bash
python predict.py some_image.jpg
```
Outputs a single float: `0` = real photo, `1` = photo of a screen.
 
**Retrain from scratch** (optional — needs `real/` and `screen/` image folders):
```bash
pip install torch torchvision pillow numpy scikit-learn
python train_cnn.py
```
Runs 5-fold cross-validation across all models, picks the best, trains a final model on the full dataset, and saves `model.pth`.
 
---
 
## Live Demo
 
A webcam page that shows **REAL 🟢 / SCREEN 🔴** predictions updating every ~1 second.
 
**1. Start the backend:**
```bash
pip install flask flask-cors
python server.py
```
 
**2. Open `demo.html`** directly in your browser (double-click it — no need to serve it separately).
 
Point the camera at something real → should show **REAL**. Hold up a phone/laptop screen displaying a photo → should flip to **SCREEN** within ~1 second.
 
> Note: both `server.py` and `demo.html` need to be running/open at the same time. Camera access requires the browser to allow permissions on first load.
 
---
 
## What I'd Improve With More Time
 
- **More training data** — 100 images is small for a CNN; doubling or tripling it (more devices, more lighting, more objects) would likely push past 95% and tighten the CV variance
- **INT8 quantization** — converting to TFLite or ONNX Runtime mobile would cut model size and latency further for real on-device deployment
- **Hard-example mining** — retrain periodically on failure cases as cheaters adapt (higher-quality screens, anti-moire filters, etc.)
---
 
## Tech Stack
 
`Python` · `PyTorch` · `MobileNetV2` · `torchvision` · `Flask` · `OpenCV` · `scikit-learn`
 
