"""
predict.py - Inference logic for the Screw Defect Detector.
Kept separate from the UI (app.py) as required by the brief.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import gdown

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CLASS_NAMES = ["defective", "good"]  # must match training order
MODEL_PATH = "screw_defect_model_final.pth"
DRIVE_FILE_ID = "1PkH22I-UOEfAMGcz6XWGSHaNWhmi4PIw"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def download_model_if_needed():
    """Downloads the trained model from Google Drive on first run."""
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)


def load_model():
    """Loads the trained model once. Call this at app startup."""
    download_model_if_needed()
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_image(model, pil_image, threshold=0.5):
    """
    Takes a PIL image and a decision threshold, returns (label, confidence_percent).
    Lower threshold = more sensitive to catching defects (more false alarms too).
    """
    input_tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        defective_prob = probs[0].item()  # class 0 = defective

    if defective_prob >= threshold:
        return "defective", defective_prob * 100
    else:
        return "good", (1 - defective_prob) * 100


def get_gradcam_overlay(model, pil_image):
    """
    Returns a Grad-CAM heatmap overlay (numpy image array) showing which
    regions of the image most influenced the 'defective' prediction.

    NOTE ON LAYER CHOICE: layer4[-1] on a 224x224 input only gives a 7x7
    feature map, which produces coarse, blocky boxes that can only ever
    land on one of a small grid of regions. layer3[-1] gives a 14x14 map,
    which tracks smaller, more localized defects (scratches, thread damage)
    noticeably better in practice.

    NOTE ON CAM VARIANT: using GradCAM++ instead of vanilla GradCAM here.
    Both read the same frozen, unmodified model - GradCAM++ just weights
    the gradients differently (per-pixel importance weighting rather than
    a single global weight per channel), which sometimes gives tighter,
    less diffuse attention on the same underlying features. It cannot fix
    attention that is genuinely on the wrong feature.

    IMPORTANT CAVEAT (still applies regardless of layer/CAM choice): if
    boxes land on background/head/shaft rather than the actual visible
    defect, the most likely root cause is the model learning a
    background/lighting shortcut rather than the defect itself - not a
    localization-algorithm problem. Verify by running this function on
    correctly-classified "good" images with ClassifierOutputTarget(0)
    forced (already done automatically below) - if it lights up the same
    generic regions there too, that confirms a shortcut, and no layer or
    CAM-variant swap will fix it; only rebalancing/cropping the training
    data will.
    """
    img_resized = pil_image.convert("RGB").resize((224, 224))
    input_tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)

    target_layers = [model.layer3[-1]]  # finer 14x14 grid vs layer4's 7x7
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = np.array(img_resized) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam


def get_defect_bounding_box(grayscale_cam, threshold=0.6, top_percentile=88, max_area_fraction=0.85):
    """
    Derives an approximate bounding box around the most-attended region
    of the Grad-CAM heatmap - a weakly-supervised localization technique.

    Uses an ADAPTIVE threshold: the effective cutoff is the higher of a
    fixed floor (`threshold`) and a percentile-based cutoff computed from
    this specific heatmap's own value distribution. This means:
      - Sharp, concentrated heatmaps get a tight, accurate box.
      - Diffuse heatmaps (common with small training sets / subtle defects)
        don't fall back to an oversized box just because they never cross
        a fixed cutoff cleanly.

    top_percentile nudged from 85 -> 88 to go with the finer layer3 map:
    a finer grid means a tighter box is achievable, so the cutoff can
    afford to be a bit stricter before it starts throwing away real signal.

    As a safety check, if the resulting box still covers most of the image
    (> max_area_fraction), localization is treated as unreliable and None
    is returned rather than drawing a box that would mislead the viewer
    into thinking the model pinpointed a specific defect location.

    Returns (x_min, y_min, x_max, y_max) in the 224x224 image space, or None
    if no reliable region could be determined.
    """
    if grayscale_cam is None or grayscale_cam.size == 0:
        return None

    percentile_cutoff = np.percentile(grayscale_cam, top_percentile)
    effective_threshold = max(threshold, percentile_cutoff)

    mask = grayscale_cam >= effective_threshold
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    x_min, y_min, x_max, y_max = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    image_area = grayscale_cam.shape[0] * grayscale_cam.shape[1]
    box_area = max(0, (x_max - x_min)) * max(0, (y_max - y_min))
    if image_area > 0 and (box_area / image_area) > max_area_fraction:
        return None

    return x_min, y_min, x_max, y_max
