"""
Predict the classes of segmented characters from an input image using a trained PyTorch model,
showing each segmented character alongside its predicted label, and drawing bounding boxes on
the original image to indicate where each character was found.

Usage:
  predict.py <model_path> <image_path> [--device=<device>] [--architecture=<arch>] [--emnist-type=<type>]

Options:
  <model_path>              Path to the trained PyTorch model file.
  <image_path>              Path to the input image.
  --device=<device>         Device to use for prediction [default: cpu].
  --architecture=<arch>     Model architecture name [default: EmnistCNN_32_128_256].
  --emnist-type=<type>      EMNIST split to use (letters/digits/balanced) [default: balanced].
"""
import random
import math
import cv2
import numpy as np
import torch
from PIL import Image
from docopt import docopt
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

from model import get_model
from utility import get_transforms, get_emnist_class_num

transform = get_transforms()
raw_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

def load_trained_model(arch, emnist_type, model_path, device):
    model = get_model(arch, num_classes=get_emnist_class_num(emnist_type)).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    return model

def segment_characters(image_np, padding=5, min_area=100, max_area=5000):
    blur = cv2.GaussianBlur(image_np, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, 0.4 * dist.max(), 255, cv2.THRESH_BINARY)
    fg = np.uint8(fg)
    bg = cv2.dilate(clean, kernel, iterations=2)
    unknown = cv2.subtract(bg, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR), markers)
    crops = []
    for lbl in range(2, markers.max() + 1):
        mask = (markers == lbl).astype('uint8') * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        x, y, w, h = cv2.boundingRect(cnts[0])
        area = w * h
        if area < min_area or area > max_area:
            continue
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1 = min(clean.shape[1], x + w + padding)
        y1 = min(clean.shape[0], y + h + padding)
        crop = clean[y0:y1, x0:x1]
        crop = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
        crops.append(crop)
    if not crops:
        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_area or area > max_area:
                continue
            x0, y0 = max(0, x - padding), max(0, y - padding)
            x1 = min(clean.shape[1], x + w + padding)
            y1 = min(clean.shape[0], y + h + padding)
            crop = clean[y0:y1, x0:x1]
            crop = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
            crops.append(crop)
    return crops

def predict_character(model, tensor, device):
    with torch.no_grad():
        out = model(tensor.to(device))
        probs = torch.nn.functional.softmax(out, dim=1)
        cls = int(probs.argmax(dim=1).item())
        conf = float(probs[0, cls].item())
    return cls, conf

def process_and_predict(model, pil_img, device):
    gray = np.array(pil_img)
    char_imgs = segment_characters(gray)
    preds = []
    for raw in char_imgs:
        t = transform(Image.fromarray(raw)).unsqueeze(0)
        cls, conf = predict_character(model, t, device)
        preds.append({
            "char_img": raw,
            "pytorch": {"class": cls, "confidence": conf},
        })
    return preds

def display_predictions(preds, classes, rows=2, cols=5, flip=False, rotate_k=0):
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for i, p in enumerate(preds):
        arr = p["char_img"]
        if flip:
            arr = np.fliplr(arr)
        if rotate_k:
            arr = np.rot90(arr, k=rotate_k)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(arr, cmap="gray")
        lbl = p["pytorch"]["class"]
        title_lines = [
            f"True:  {p.get('sample_true', '?')} ({lbl})",
            f"PyT:   {classes[lbl]} ({p['pytorch']['confidence']:.2f})"
        ]
        ax.set_title("\n".join(title_lines), fontsize=10)
        ax.axis("off")
    for j in range(len(preds), rows * cols):
        fig.add_subplot(rows, cols, j + 1).axis("off")
    plt.tight_layout()
    plt.show()

def preview_emnist(split, rows=2, cols=5):
    ds = datasets.EMNIST(
        root="../data", split=split,
        train=True, download=False,
        transform=transforms.ToTensor()
    )
    classes = ds.classes
    n = rows * cols
    samples = random.sample(range(len(ds)), n)
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(samples):
        img, lbl = ds[idx]
        img = np.rot90(np.fliplr(img.squeeze().numpy()), k=1)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{classes[lbl]} ({lbl})", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = docopt(__doc__)
    MODEL_PATH  = args["<model_path>"]
    IMAGE_PATH  = args["<image_path>"]
    DEVICE      = args["--device"]
    ARCH        = args["--architecture"]
    EMNIST_TYPE = args["--emnist-type"]
    device = torch.device(DEVICE)

    preview_emnist(EMNIST_TYPE, rows=2, cols=5)

    model = load_trained_model(ARCH, EMNIST_TYPE, MODEL_PATH, device)

    raw_ds = datasets.EMNIST(
        root="../data", split=EMNIST_TYPE,
        train=True, download=False,
        transform=raw_transform
    )
    aug_ds = datasets.EMNIST(
        root="../data", split=EMNIST_TYPE,
        train=True, download=False,
        transform=transform
    )
    classes = raw_ds.classes
    samples = random.sample(range(len(raw_ds)), 10)
    preds = []
    for idx in samples:
        raw_t, lbl = raw_ds[idx]
        aug_t, _ = aug_ds[idx]
        cls, conf = predict_character(model, aug_t.unsqueeze(0), device)
        arr = (raw_t.squeeze().numpy() * 255).astype(np.uint8)
        preds.append({
            "char_img": arr,
            "sample_true": classes[lbl],
            "pytorch": {"class": cls, "confidence": conf},
        })
    display_predictions(preds, classes, rows=2, cols=5, flip=True, rotate_k=1)

    img = Image.open(IMAGE_PATH).convert("L")
    predictions = process_and_predict(model, img, device)
    wrapped = [{
        "char_img":    p["char_img"],
        "sample_true": "?",
        "pytorch":     p["pytorch"],
    } for p in predictions]
    cols = 5
    rows = math.ceil(len(wrapped) / cols)
    display_predictions(wrapped, classes, rows=rows, cols=cols)
