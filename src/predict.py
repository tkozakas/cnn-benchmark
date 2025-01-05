import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from src.model import SimpleCNN

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAPPING_FILE = os.path.join(PROJECT_ROOT, "data/emnist-balanced-mapping.txt")
DEMO_FOLDER = os.path.join(PROJECT_ROOT, "demo")
PREDICTION_FOLDER = os.path.join(PROJECT_ROOT, "demo/predictions")

def load_label_map(mapping_file_path):
    label_map = []
    with open(mapping_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            idx, ascii_val = int(parts[0]), int(parts[1])
            c = chr(ascii_val)
            if len(label_map) <= idx:
                label_map.extend(["?"] * (idx - len(label_map) + 1))
            label_map[idx] = c
    return label_map

def load_model(model_path, num_classes, device="cpu"):
    model = SimpleCNN(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model

def segment_characters(image_path, debug=False):
    to_gray = T.Compose([T.Grayscale(num_output_channels=1), T.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    gray = to_gray(image).squeeze(0).numpy()
    binary = gray > threshold_otsu(gray)
    labeled = label(binary)
    props = regionprops(labeled)
    segments = []

    if debug:
        os.makedirs(PREDICTION_FOLDER, exist_ok=True)
        labeled_rgb = label2rgb(labeled, image=gray, bg_label=0)
        plt.figure(figsize=(10, 6))
        plt.imshow(labeled_rgb, cmap="gray")
        for region in props:
            minr, minc, maxr, maxc = region.bbox
            plt.gca().add_patch(
                plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor="red", fill=False, linewidth=2)
            )
        plt.title(f"Bounding Boxes for {os.path.basename(image_path)}")
        plt.axis("off")
        plt.savefig(os.path.join(PREDICTION_FOLDER, f"bounding_boxes_{os.path.basename(image_path)}.png"))
        plt.close()

    for i, region in enumerate(props):
        r0, c0, r1, c1 = region.bbox
        crop = gray[r0:r1, c0:c1]
        crop_img = Image.fromarray((crop * 255).astype("uint8"))
        if debug:
            crop_img.save(os.path.join(PREDICTION_FOLDER, f"segment_{i}_{os.path.basename(image_path)}"))
        segments.append(crop_img)

    return segments

def predict_characters(model, char_images, label_map, device="cpu"):
    tform = T.Compose([T.Resize((28, 28)), T.ToTensor()])
    result = []
    with torch.no_grad():
        for i, img in enumerate(char_images):
            x = tform(img).unsqueeze(0).to(device)
            logits = model(x)
            _, idx = logits.max(dim=1)
            idx = idx.item()
            result.append(label_map[idx])
    return "".join(result)

def main(model_path, device="cpu"):
    label_map = load_label_map(MAPPING_FILE)
    model = load_model(model_path, len(label_map), device=device)
    for f in os.listdir(DEMO_FOLDER):
        path = os.path.join(DEMO_FOLDER, f)
        if os.path.isfile(path):
            chars = segment_characters(path, debug=True)
            text = predict_characters(model, chars, label_map, device=device)
            print(f"{f}: {text}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <device>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
