# Spot the Fake Photo

## Files

- `predict.py` — the one-line predictor (`python predict.py some_image.jpg`)
- `train_cnn.py` — full training pipeline (5-fold CV + final model training)
- `model.pth` — trained model weights (MobileNetV2, fine-tuned)
- `REPORT.md` — approach, accuracy, latency, cost-per-image, and bonus answers
- `server.py` + `demo.html` — optional live camera demo

## Running predict.py

```bash
pip install torch torchvision pillow
python predict.py some_image.jpg
```

Prints a single number 0–1 (0 = real, 1 = screen recapture).

## Running the live demo (optional)

This is a small local web demo: a Flask server runs the model, and a webpage
uses your camera to show live REAL / SCREEN predictions updating roughly once
per second.

**1. Install dependencies (in addition to the ones above):**
```bash
pip install flask flask-cors
```

**2. Start the server** (keep this terminal running):
```bash
python server.py
```
You should see `Server starting at http://localhost:5000`.

**3. Open `demo.html`** directly in a browser (just double-click it, no need
to serve it separately). Allow camera access when prompted.

**4. Try it out:** point the camera at a normal object/scene — it should show
**REAL**. Then hold up a phone or laptop screen displaying a photo — it
should switch to **SCREEN** within about a second.

Note: this demo needs to run locally (it calls `localhost:5000`), so it's
meant to be run on your machine rather than hosted publicly — browser camera
access also requires HTTPS once off `localhost`, which a basic local Flask
server doesn't provide. If you'd rather not set it up, a short screen
recording of it running is available on request / attached.

## Retraining (optional)

If you want to retrain from scratch:
```bash
pip install torch torchvision pillow numpy
python train_cnn.py
```
Expects `real/` and `screen/` folders of training images in the same
directory. Takes a few minutes on a GPU (much longer on CPU). Outputs a new
`model.pth`.
