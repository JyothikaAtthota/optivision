"""
OPTIVISION — ResNet50 Model
Pretrained backbone with custom DR classification head.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 5, freeze_backbone: bool = False,
                pretrained: bool = True) -> nn.Module:
    """
    Load ResNet50 and replace the final FC layer for DR classification.

    Args:
        num_classes:      Number of DR stages (default 5).
        freeze_backbone:  If True, freeze all layers except the new FC head.
        pretrained:       Download ImageNet weights if True (requires internet).
                          Falls back to random init if download fails.

    Returns:
        model (nn.Module)
    """
    if pretrained:
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            print("[OPTIVISION] WARNING: Could not download pretrained weights "
                  "(no internet?). Training from random initialisation.")
            model = models.resnet50(weights=None)
    else:
        model = models.resnet50(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer
    in_features = model.fc.in_features          # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model


def load_model(checkpoint_path: str, num_classes: int = 5,
               device: torch.device | None = None) -> nn.Module:
    """
    Load a saved model checkpoint.

    Args:
        checkpoint_path: Path to best_model.pth
        num_classes:     Must match the checkpoint.
        device:          Target device (auto-detected if None).

    Returns:
        model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
