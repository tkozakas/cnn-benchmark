"""
Predict the classes of segmented characters from an input image using a trained PyTorch model,
showing each segmented character alongside its predicted label, and drawing bounding boxes on
the original image to indicate where each character was found.

Usage:
  predict.py <model_path> <image_path> [--device=<device>]

Options:
  <model_path>       Path to the trained PyTorch model file.
  <image_path>       Path to the input image.
  --device=<device>  Device to use for prediction (default: "cuda" if available, else "cpu").
"""
import math
import random

import cv2
import numpy as np
import pytesseract
import torch
import torchvision.transforms.functional as F
from PIL import Image
from docopt import docopt
from matplotlib import pyplot as plt
from torchvision import transforms, datasets

from src.config import train_config
from src.model import get_model

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model(model_path, device):
    model = get_model("EmnistCNN").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image):
    return transform(image)


def segment_characters(image_np, padding=10):
    _, binary = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 10 and w > 5:
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(binary.shape[1], x + w + padding)
            y_end = min(binary.shape[0], y + h + padding)

            char_crop = binary[y_start:y_end, x_start:x_end]
            char_crop = cv2.copyMakeBorder(char_crop, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0])
            char_resized = cv2.resize(char_crop, (28, 28), interpolation=cv2.INTER_AREA)
            char_images.append(char_resized)

    return char_images


def predict_character(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
    return predicted_class, probabilities[0][predicted_class].item()


def process_and_predict_image(model, image, device):
    char_images = segment_characters(np.array(image))
    predictions = []

    for char_img in char_images:
        char_pil = Image.fromarray(char_img)
        image_tensor = preprocess_image(char_pil).unsqueeze(0).to(device)

        predicted_class, confidence = predict_character(model, image_tensor)

        _, tesseract_ready = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tesseract_prediction = pytesseract.image_to_string(
            Image.fromarray(tesseract_ready),
            config="--psm 10 -c tessedit_char_whitelist=0123456789"
        ).strip()

        predictions.append({
            "char_img": char_img,
            "pytorch": {"class": predicted_class, "confidence": confidence},
            "tesseract": tesseract_prediction
        })

    return predictions

def load_dataset(emnist_type):
    return datasets.EMNIST(root="../data", split=emnist_type, train=True, download=True, transform=transform)


def predict_from_dataset(model, dataset, device, num_images=1, cols=1):
    rows = math.ceil(num_images / cols)
    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(num_images):
        image, label = dataset[random.randint(0, len(dataset) - 1)]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_class, confidence = predict_character(model, image_tensor)

        image_corrected = F.rotate(image, angle=-90)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_corrected.squeeze(0), cmap="gray")
        plt.title(
            f"Predicted: {predicted_class}, Conf: {confidence:.2f}\nActual: {label}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def display_predictions(predictions, num_segments=3, cols=1):
    segment_predictions = predictions[:num_segments]
    num_images = len(segment_predictions)
    rows = math.ceil(num_images / cols)
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, pred in enumerate(segment_predictions):
        char_img = pred["char_img"]
        pytorch_pred = pred["pytorch"]
        tesseract_pred = pred["tesseract"]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(char_img, cmap="gray")
        plt.title(
            f"PyTorch: {pytorch_pred['class']}, Conf: {pytorch_pred['confidence']:.2f}\n"
            f"Tesseract: {tesseract_pred}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = docopt(__doc__)
    model_path = args["<model_path>"]
    image_path = args["<image_path>"]
    device = args["--device"] or ("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)

    dataset = load_dataset(train_config["emnist_type"])
    predict_from_dataset(model, dataset, device)

    image = Image.open(image_path).convert("L")
    predictions = process_and_predict_image(model, image, device)
    display_predictions(predictions)
