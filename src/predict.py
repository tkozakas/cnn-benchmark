import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_model(model_path, device='cpu'):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(input_path):
    image = Image.open(input_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict(model, input_path, device):
    input_path = Path(input_path)
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        image_tensor = preprocess_image(input_path)
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            preds = outputs.argmax(dim=1)
            return f"Predicted text: {preds.item()}"
    else:
        raise ValueError("Unsupported file format. Please provide an image file (JPEG, PNG, BMP, TIFF).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict text using a trained model.")
    parser.add_argument("--model-path", type=str, default="trained/model.pth", help="Path to the trained .pth model.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file (image, PDF, or .txt).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for prediction (default: 'cpu').")
    args = parser.parse_args()

    model = load_model(args.model_path, device=args.device)
    result = predict(model, args.input, args.device)

    print(result)
