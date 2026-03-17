import tensorflow_datasets as tfds
import os
from pathlib import Path
from PIL import Image

# Load first 5000 training images
ds, info = tfds.load(
    "cifar10",
    split="train[:5000]",
    with_info=True
)

# Get class names
class_names = info.features["label"].names

# Desktop path
desktop_path = Path.home() / "Desktop" / "cifar_images"

# Create main folder
os.makedirs(desktop_path, exist_ok=True)

# Create class folders
for name in class_names:
    os.makedirs(desktop_path / name, exist_ok=True)

# Save images into respective class folders
for i, example in enumerate(ds):
    img = example["image"].numpy()
    label = example["label"].numpy()
    class_name = class_names[label]

    Image.fromarray(img).save(
        desktop_path / class_name / f"img_{i}.png"
    )

print("5000 images saved successfully on Desktop")