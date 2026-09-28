"""
OPTIVISION — Dataset & Preprocessing Pipeline
Resize → Normalize (ImageNet) → Augment → Tensor
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── DR stage labels ──────────────────────────────────────────────────────────
DR_CLASSES = [
    "No DR",          # 0
    "Mild DR",        # 1
    "Moderate DR",    # 2
    "Severe DR",      # 3
    "Proliferative DR"# 4
]

NUM_CLASSES = len(DR_CLASSES)

# ── Transforms ───────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_inference_transform():
    """Single-image transform for real-time prediction."""
    return get_val_transforms()


# ── Dataset ──────────────────────────────────────────────────────────────────
class DRDataset(Dataset):
    """
    Expects folder layout:
        root/
          No DR/          *.jpg / *.png
          Mild DR/
          Moderate DR/
          Severe DR/
          Proliferative DR/
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.samples   = []          # list of (image_path, label_index)

        for label_idx, class_name in enumerate(DR_CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label_idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── DataLoader helpers ────────────────────────────────────────────────────────
def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    train_ds = DRDataset(os.path.join(data_dir, "train"), get_train_transforms())
    val_ds   = DRDataset(os.path.join(data_dir, "val"),   get_val_transforms())

    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training images found in '{os.path.join(data_dir, 'train')}'.\n"
            "  -> Run:  python utils/make_dummy_data.py   (synthetic test data)\n"
            "  -> Or populate data/train/<ClassName>/ with real fundus images."
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            f"No validation images found in '{os.path.join(data_dir, 'val')}'.\n"
            "  -> Run:  python utils/make_dummy_data.py   (synthetic test data)\n"
            "  -> Or populate data/val/<ClassName>/ with real fundus images."
        )

    # num_workers=0 on Windows to avoid multiprocessing spawn issues
    nw = 0 if os.name == "nt" else num_workers

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)
    return train_loader, val_loader
