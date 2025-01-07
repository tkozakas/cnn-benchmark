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
import torch
import torchvision.transforms.functional as F
from PIL import Image
from docopt import docopt
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
import pytesseract
from src.config import train_config
from src.model import get_model

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model(model_path, device):
    model = get_model("EmnistCNN").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image, device):
    if isinstance(image, torch.Tensor):
        return image.unsqueeze(0).to(device)
    return transform(image).unsqueeze(0).to(device)


def preprocess_segmented_characters(image_path, padding=5):
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)

    _, binary = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 10:  # Filter out noise (e.g., very small segments)
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(binary.shape[1], x + w + padding)
            y_end = min(binary.shape[0], y + h + padding)

            char_crop = binary[y_start:y_end, x_start:x_end]
            char_resized = cv2.resize(char_crop, (28, 28), interpolation=cv2.INTER_AREA)
            char_images.append(char_resized)

    return char_images


def predict_character(model, image, device):
    image_tensor = preprocess_image(image, device)
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
    return predicted_class


def load_dataset(emnist_type):
    return datasets.EMNIST(root="../data", split=emnist_type, train=True, download=True, transform=transform)


def display_random_predicted_images_from_dataset(model, dataset, device, num_images=20, cols=5):
    rows = math.ceil(num_images / cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(num_images):
        image, label = dataset[random.randint(0, len(dataset) - 1)]
        predicted_class = predict_character(model, image, device)

        image_corrected = F.rotate(image, angle=-90)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_corrected.squeeze(0), cmap='gray')
        plt.title(f"Predicted: {predicted_class}\nActual: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_segmented_characters_with_predictions(model, char_images, device, cols=5):
    num_images = len(char_images)
    rows = math.ceil(num_images / cols)
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, char_img in enumerate(char_images):
        char_pil = Image.fromarray(char_img)
        char_tensor = transform(char_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(char_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()

        tesseract_prediction = pytesseract.image_to_string(char_pil, config="--psm 10 -c tessedit_char_whitelist=0123456789")
        tesseract_prediction = tesseract_prediction.strip()

        plt.subplot(rows, cols, i + 1)
        plt.imshow(char_img, cmap="gray")
        plt.title(f"PyTorch: {predicted_class}, Prob: {probabilities[0][predicted_class]:.4f}\nTesseract: {tesseract_prediction}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = docopt(__doc__)
    model_path = args["<model_path>"]
    image_path = args["<image_path>"]
    device = args["--device"] or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(train_config["emnist_type"])
    model = load_model(model_path, device)

    # Shows random images from the dataset with their predicted labels
    display_random_predicted_images_from_dataset(model, dataset, device)

    # Preprocess the input image
    char_images = preprocess_segmented_characters(image_path)

    # Show segmented characters with predictions
    display_segmented_characters_with_predictions(model, char_images, device)
