import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def build_model():
    """Architecture must match train_cnn.py exactly."""
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 1)
    )
    return model


def load_model(model_path="model.pth"):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["img_size"]


def predict(image_path, model_path="model.pth"):
    model, img_size = load_model(model_path)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logit = model(img_tensor)
        prob_screen = torch.sigmoid(logit).item()

    return prob_screen


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    score = predict(image_path)
    print(f"{score:.4f}")
