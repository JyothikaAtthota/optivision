# OPTIVISION — AI-Powered Diabetic Retinopathy Detection

> Real-time DR detection and blindness prediction using Deep Learning (ResNet50 + PyTorch)

---

## Project Structure

```
OPTIVISION/
├── data/
│   ├── train/          # Training images (one subfolder per DR class)
│   ├── val/            # Validation images
│   └── test/           # Test images
├── models/
│   └── resnet_model.py # ResNet50 + custom DR head
├── utils/
│   └── dataset.py      # DRDataset, transforms, DataLoaders
├── outputs/
│   └── best_model.pth  # Saved after training
├── train.py            # Training script
├── predict.py          # Real-time prediction pipeline
├── app.py              # Gradio web interface
└── requirements.txt
```

---

## DR Classes

| Index | Stage            | Blindness Risk |
|-------|------------------|----------------|
| 0     | No DR            | Low            |
| 1     | Mild DR          | Moderate       |
| 2     | Moderate DR      | High           |
| 3     | Severe DR        | Very High      |
| 4     | Proliferative DR | Critical       |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data Layout

Organize retinal fundus images into class-named folders:

```
data/train/No DR/          image1.jpg ...
data/train/Mild DR/        image2.jpg ...
data/train/Moderate DR/    ...
data/train/Severe DR/      ...
data/train/Proliferative DR/  ...
data/val/   (same structure)
```

---

## Training

```bash
python train.py \
  --data_dir data \
  --epochs 25 \
  --batch_size 32 \
  --lr 1e-4 \
  --save_path outputs/best_model.pth
```

- **Model:** ResNet50 (ImageNet pretrained)
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (weight_decay=1e-5)
- **Scheduler:** ReduceLROnPlateau (patience=3)
- **Augmentation:** Random flip, rotation ±15°, color jitter
- **Input:** 224×224, normalized with ImageNet mean/std

---

## Real-Time Prediction (Python API)

```python
from predict import predict

result = predict("path/to/fundus_image.jpg")
print(result["predicted_class"])   # e.g. "Moderate DR"
print(result["confidence"])        # e.g. 0.8712
print(result["blindness_risk"])    # e.g. "High"
print(result["probabilities"])     # {class: prob} for all 5 stages
```

---

## Web Interface (Gradio)

```bash
python app.py
# Opens at http://localhost:7860
```

- Upload or webcam-capture a retinal fundus image
- Instant DR stage classification and blindness risk rating
- Probability bar chart for all 5 stages
- Severity reference table

---

## Model Architecture

```
ResNet50 (pretrained, ImageNet)
  └── FC head replaced:
        Dropout(0.4)
        Linear(2048 → 512)
        ReLU
        Dropout(0.3)
        Linear(512 → 5)   ← DR stages
```

---

## Disclaimer
OPTIVISION is a **screening aid only**. It is not a substitute for professional ophthalmological diagnosis. Always consult a qualified eye care specialist.
