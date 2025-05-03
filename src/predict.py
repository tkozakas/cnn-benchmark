"""
Predict the classes of segmented characters from an input image using a trained PyTorch model,
showing each segmented character alongside its predicted label, and drawing bounding boxes on
the original image to indicate where each character was found.

Usage:
  predict.py <model_path> <image_path>
             [--device=<device>]
             [--architecture=<arch>]
             [--emnist-type=<type>]

Options:
  <model_path>              Path to the trained PyTorch model file.
  <image_path>              Path to the input image.
  --device=<device>         Device to use for prediction [default: cpu].
  --architecture=<arch>     Model architecture name
                            [default: EmnistCNN_32_128_256].
  --emnist-type=<type>      EMNIST split to use (letters/digits/balanced)
                            [default: balanced].
"""
import random

import cv2
import math
import numpy as np
import pytesseract
import torch
from PIL import Image
from docopt import docopt
from matplotlib import pyplot as plt
from torchvision import transforms, datasets

from model import get_model
from utility import get_transforms

transform = get_transforms()


def load_trained_model(arch, model_path, device):
    model = get_model(arch).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def segment_characters(image_np, padding=10):
    _, binary = cv2.threshold(
        image_np, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    crops = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 5 or h < 10:
            continue
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(binary.shape[1], x + w + padding)
        y1 = min(binary.shape[0], y + h + padding)

        crop = binary[y0:y1, x0:x1]
        crop = cv2.copyMakeBorder(crop, 5, 5, 5, 5,
                                  cv2.BORDER_CONSTANT, value=[0])
        crop = cv2.resize(crop, (28, 28),
                          interpolation=cv2.INTER_AREA)
        crops.append(crop)
    return crops


def predict_character(model, tensor, device):
    with torch.no_grad():
        out = model(tensor.to(device))
        probs = torch.nn.functional.softmax(out, dim=1)
        cls = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, cls].item())
    return cls, conf


def process_and_predict(model, pil_img, device):
    gray = np.array(pil_img)
    char_imgs = segment_characters(gray)
    preds = []

    for img in char_imgs:
        pil_c = Image.fromarray(img)
        t = transform(pil_c).unsqueeze(0)
        cls, conf = predict_character(model, t, device)

        _, thresh = cv2.threshold(
            img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        txt = pytesseract.image_to_string(
            Image.fromarray(thresh),
            config="--psm 10"
        ).strip()

        preds.append({
            "char_img": img,
            "pytorch": {"class": cls, "confidence": conf},
            "tesseract": txt
        })
    return preds


def display_predictions(preds, cols=5):
    n = len(preds)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, p in enumerate(preds):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(p["char_img"], cmap="gray")
        plt.title(
            f"PT: {p['pytorch']['class']} ({p['pytorch']['confidence']:.2f})\n"
            f"TESS: {p['tesseract']}"
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def preview_emnist(split, n=10):
    # load raw EMNIST just to preview examples side by side
    ds = datasets.EMNIST(
        root="../data",
        split=split,
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )
    samples = random.sample(range(len(ds)), n)
    plt.figure(figsize=(15, 6))
    for idx, i in enumerate(samples):
        img, lbl = ds[i]
        plt.subplot(2, n, idx + 1)
        plt.imshow(img.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.title(f"Label: {lbl}")
        plt.axis("off")

        plt.subplot(2, n, n + idx + 1)
        pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        p = transform(pil)
        plt.imshow(p.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.axis("off")
    plt.suptitle("EMNIST Raw vs. Transformed")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = docopt(__doc__)
    MODEL_PATH = args["<model_path>"]
    IMAGE_PATH = args["<image_path>"]
    DEVICE = args["--device"]
    ARCHITECTURE = args["--architecture"]
    EMNIST_TYPE = args["--emnist-type"]

    device = torch.device(DEVICE)

    # 1) Preview a few raw vs. preprocessed EMNIST examples
    preview_emnist(EMNIST_TYPE, n=10)

    # 2) Process a random batch from EMNIST and show PyTorch predictions
    model = load_trained_model(ARCHITECTURE, MODEL_PATH, device)
    ds = datasets.EMNIST(
        root="../data",
        split=EMNIST_TYPE,
        train=True, download=True,
        transform=transform
    )
    process_and_predict_ds = lambda count=20: [
        # pick random idx and predict
        predict_character(model, ds[idx][0].unsqueeze(0), device)
        for idx in random.sample(range(len(ds)), count)
    ]

    # 3) Finally segment & predict your custom image
    img = Image.open(IMAGE_PATH).convert("L")
    predictions = process_and_predict(model, img, device)
    display_predictions(predictions, cols=5)
