import os
import shutil
import random

# SOURCE DATA
safe_source = r"C:\Users\Lenovo\OneDrive\Desktop\cifar_images"
suspicious_source = r"C:\Users\Lenovo\OneDrive\Desktop\cifar_suspicious"

# DESTINATION (inside Hackathon project)
dataset_root = r"C:\Users\Lenovo\OneDrive\Desktop\Hackathon\dataset"

splits = {"train":0.7, "val":0.15, "test":0.15}


def collect_images(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                images.append(os.path.join(root,file))
    return images


def split_and_copy(images, label):

    random.shuffle(images)

    total = len(images)
    train_end = int(total * splits["train"])
    val_end = train_end + int(total * splits["val"])

    split_sets = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in split_sets.items():

        dest_folder = os.path.join(dataset_root, split_name, label)
        os.makedirs(dest_folder, exist_ok=True)

        for img in split_images:
            filename = os.path.basename(img)
            shutil.copy(img, os.path.join(dest_folder, filename))


print("Collecting safe images...")
safe_images = collect_images(safe_source)
print("Total safe images:", len(safe_images))

print("Collecting suspicious images...")
suspicious_images = collect_images(suspicious_source)
print("Total suspicious images:", len(suspicious_images))

split_and_copy(safe_images, "safe")
split_and_copy(suspicious_images, "suspicious")

print("✅ Dataset split completed successfully!")