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

import cv2
import torch
from docopt import docopt
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from src.model import get_model


def load_model(model_path, device):
    """
    Load the trained PyTorch model.
    """
    model = get_model("EmnistCNN").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image, device):
    """
    Preprocess a single image for the model.
    """
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Ensure size is 28x28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0).to(device)


def segment_characters(image_path, debug=False):
    """
    Segment characters from the input image using skimage methods.

    Args:
        image_path (str): Path to the input image.
        debug (bool): Whether to save debugging visualizations.

    Returns:
        segments (list): List of segmented character images.
        bounding_boxes (list): List of bounding box coordinates (minr, minc, maxr, maxc).
    """
    import os

    to_gray = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    gray = to_gray(image).squeeze(0).numpy()
    binary = gray > threshold_otsu(gray)

    # Ensure binary is converted to an integer array for labeling
    binary = np.asarray(binary, dtype=np.uint8)

    # Create labeled regions
    labeled = label(binary)

    # Validate labeled output
    if not isinstance(labeled, np.ndarray):
        raise ValueError("Unexpected output from 'label': expected a numpy array.")

    props = regionprops(labeled)
    segments = []

    if debug:
        os.makedirs("../predictions", exist_ok=True)
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
        plt.savefig(os.path.join("../predictions", f"bounding_boxes_{os.path.basename(image_path)}.png"))
        plt.close()

    bounding_boxes = []
    for i, region in enumerate(props):
        minr, minc, maxr, maxc = region.bbox
        crop = gray[minr:maxr, minc:maxc]
        crop_img = Image.fromarray((crop * 255).astype("uint8")).resize((28, 28), Image.Resampling.LANCZOS)
        if debug:
            crop_img.save(os.path.join("../predictions", f"segment_{i}_{os.path.basename(image_path)}"))
        segments.append(crop_img)
        bounding_boxes.append((minr, minc, maxr, maxc))

    return segments, bounding_boxes


def predict(model, image_tensor):
    """
    Predict the class of a single segmented character image.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs.argmax(dim=1).item()


if __name__ == "__main__":
    # Parse command-line arguments
    args = docopt(__doc__)
    model_path = args["<model_path>"]
    image_path = args["<image_path>"]
    device = args["--device"] or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = load_model(model_path, device)

    # Segment characters and get bounding boxes
    segmented_images, bounding_boxes = segment_characters(image_path)

    # Predict each segmented character
    predictions = []
    for char_image in segmented_images:
        # Preprocess the segmented character
        image_tensor = preprocess_image(char_image, device)
        # Predict the class
        prediction = predict(model, image_tensor)
        predictions.append(prediction)

    # Print predicted classes
    print(f"Predicted Classes: {predictions}")

    # Visualize results
    original_image = cv2.imread(image_path)
    for (minr, minc, maxr, maxc), pred in zip(bounding_boxes, predictions):
        # Draw bounding boxes and predictions
        cv2.rectangle(original_image, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
        cv2.putText(original_image, str(pred), (minc, minr - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the original image with predictions
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Predictions with Bounding Boxes")
    plt.axis("off")
    plt.show()

    # Show segmented characters with predictions
    plt.figure(figsize=(len(segmented_images) * 2, 4))
    for i, (char_image, pred) in enumerate(zip(segmented_images, predictions)):
        plt.subplot(1, len(segmented_images), i + 1)
        plt.imshow(char_image, cmap="gray")
        plt.title(f"Pred: {pred}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
