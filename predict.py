"""
OPTIVISION — Real-Time Prediction Pipeline
Load model → Preprocess image → Predict DR stage + blindness risk
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union

from utils.dataset import get_inference_transform, DR_CLASSES
from models.resnet_model import load_model

# ── Blindness risk mapping ────────────────────────────────────────────────────
BLINDNESS_RISK = {
    "No DR":           ("Low",      "#27ae60"),
    "Mild DR":         ("Moderate", "#f39c12"),
    "Moderate DR":     ("High",     "#e67e22"),
    "Severe DR":       ("Very High","#c0392b"),
    "Proliferative DR":("Critical", "#8e1a0e"),
}

# ── Singleton model cache ─────────────────────────────────────────────────────
_model  = None
_device = None


def _get_model(checkpoint_path: str, num_classes: int = 5):
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model  = load_model(checkpoint_path, num_classes=num_classes,
                             device=_device)
    return _model, _device


# ── Core prediction function ──────────────────────────────────────────────────
def predict(image_input: Union[str, Image.Image],
            checkpoint_path: str = "outputs/best_model.pth",
            num_classes: int = 5) -> dict:
    """
    Run DR classification on a single retinal fundus image.

    Args:
        image_input:      File path (str) or PIL Image.
        checkpoint_path:  Path to best_model.pth.
        num_classes:      Must match training config.

    Returns:
        dict with keys:
            predicted_class  – e.g. "Moderate DR"
            confidence       – float 0–1
            blindness_risk   – e.g. "High"
            risk_color       – hex color string
            probabilities    – {class_name: probability} for all classes
    """
    # Load & preprocess image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    transform = get_inference_transform()
    tensor    = transform(image).unsqueeze(0)   # shape: (1, 3, 224, 224)

    # Load model
    model, device = _get_model(checkpoint_path, num_classes)
    tensor = tensor.to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)                  # (1, num_classes)
        probs  = F.softmax(logits, dim=1)[0]    # (num_classes,)

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()
    pred_class = DR_CLASSES[pred_idx]
    risk, color = BLINDNESS_RISK[pred_class]

    probabilities = {
        DR_CLASSES[i]: round(probs[i].item(), 4)
        for i in range(len(DR_CLASSES))
    }

    return {
        "predicted_class": pred_class,
        "confidence":      round(confidence, 4),
        "blindness_risk":  risk,
        "risk_color":      color,
        "probabilities":   probabilities,
    }


def reset_model():
    """Force model reload (e.g. after retraining)."""
    global _model
    _model = None
