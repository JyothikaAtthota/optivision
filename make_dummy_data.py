"""
OPTIVISION — Synthetic Data Generator
Creates dummy retinal-like images for smoke-testing the training pipeline.
Run: python utils/make_dummy_data.py
"""

import os
import random
from PIL import Image, ImageDraw, ImageFilter
import argparse


DR_CLASSES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]


def make_fundus_image(width: int = 256, height: int = 256,
                      stage_idx: int = 0) -> Image.Image:
    """
    Generate a synthetic fundus-like image with increasing visual complexity
    per DR stage (purely for pipeline smoke-testing, not clinical accuracy).
    """
    # Dark reddish background — mimics fundus colour
    r = random.randint(60, 100)
    img = Image.new("RGB", (width, height), color=(r, r // 3, r // 4))
    draw = ImageDraw.Draw(img)

    # Optic disc — bright circle near centre-right
    cx, cy = int(width * 0.65), height // 2
    disc_r = width // 10
    draw.ellipse(
        [cx - disc_r, cy - disc_r, cx + disc_r, cy + disc_r],
        fill=(220, 200, 140),
    )

    # Blood vessels — a few random lines
    for _ in range(random.randint(6, 12)):
        x0, y0 = random.randint(0, width), random.randint(0, height)
        x1, y1 = random.randint(0, width), random.randint(0, height)
        draw.line([(x0, y0), (x1, y1)], fill=(180, 60, 60), width=1)

    # Microaneurysms — small red dots; more with higher stage
    n_micro = stage_idx * random.randint(3, 8)
    for _ in range(n_micro):
        x = random.randint(10, width - 10)
        y = random.randint(10, height - 10)
        r2 = random.randint(2, 5)
        draw.ellipse([x - r2, y - r2, x + r2, y + r2],
                     fill=(200, 30, 30))

    # Haemorrhages — larger blobs at severe/proliferative stages
    if stage_idx >= 3:
        for _ in range(random.randint(2, 5)):
            x = random.randint(20, width - 20)
            y = random.randint(20, height - 20)
            r3 = random.randint(6, 14)
            draw.ellipse([x - r3, y - r3, x + r3, y + r3],
                         fill=(140, 10, 10))

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def generate_split(split: str, data_dir: str,
                   n_per_class: int, img_size: int):
    for idx, cls in enumerate(DR_CLASSES):
        cls_dir = os.path.join(data_dir, split, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            img = make_fundus_image(img_size, img_size, stage_idx=idx)
            img.save(os.path.join(cls_dir, f"{cls.replace(' ', '_')}_{i:04d}.png"))
    print(f"  [{split:>5}] {n_per_class} images x {len(DR_CLASSES)} classes "
          f"= {n_per_class * len(DR_CLASSES)} images  ->  {os.path.join(data_dir, split)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic DR dummy data for pipeline testing."
    )
    parser.add_argument("--data_dir",     default="data")
    parser.add_argument("--train_n",      type=int, default=50,
                        help="Images per class for train split")
    parser.add_argument("--val_n",        type=int, default=15,
                        help="Images per class for val split")
    parser.add_argument("--test_n",       type=int, default=10,
                        help="Images per class for test split")
    parser.add_argument("--img_size",     type=int, default=256)
    args = parser.parse_args()

    print("[OPTIVISION] Generating synthetic fundus images ...")
    generate_split("train", args.data_dir, args.train_n, args.img_size)
    generate_split("val",   args.data_dir, args.val_n,   args.img_size)
    generate_split("test",  args.data_dir, args.test_n,  args.img_size)
    print("[OPTIVISION] Done. Replace with real fundus images before actual training.")
