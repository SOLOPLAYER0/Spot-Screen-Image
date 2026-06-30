import io
import base64

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  


def build_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 1)
    )
    return model


print("Loading model...")
checkpoint = torch.load("model.pth", map_location="cpu", weights_only=False)
model = build_model()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
IMG_SIZE = checkpoint["img_size"]
print(f"Model loaded. Input size: {IMG_SIZE}x{IMG_SIZE}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_b64 = data["image"].split(",")[-1]  
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logit = model(img_tensor)
        prob_screen = torch.sigmoid(logit).item()

    return jsonify({"score": prob_screen})


@app.route("/")
def index():
    return "Spot the Fake Photo - backend running. Open demo.html separately."


if __name__ == "__main__":
    print("\nServer starting at http://localhost:5000")
    print("Now open demo.html in your browser.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
