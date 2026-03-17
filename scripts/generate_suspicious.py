import os
import cv2
import numpy as np
import random

# 🔹 INPUT & OUTPUT PATHS
input_root = r"C:\Users\Lenovo\OneDrive\Desktop\cifar_images"
output_root = r"C:\Users\Lenovo\OneDrive\Desktop\cifar_suspicious"

os.makedirs(output_root, exist_ok=True)

# 🔹 Transformations

def gaussian_noise(image):
    noise = np.random.normal(0, 25, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def salt_pepper_noise(image):
    noisy = image.copy()
    prob = 0.02
    rnd = np.random.rand(*image.shape[:2])
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy

def blur_image(image):
    return cv2.GaussianBlur(image, (9, 9), 0)

def compress_image(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)

def color_distortion(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = hsv[:,:,1] * 0.5
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

transformations = [
    gaussian_noise,
    salt_pepper_noise,
    blur_image,
    compress_image,
    color_distortion
]

# 🔹 Loop Through Category Folders
for category in os.listdir(input_root):

    category_path = os.path.join(input_root, category)

    if not os.path.isdir(category_path):
        continue

    output_category_path = os.path.join(output_root, category)
    os.makedirs(output_category_path, exist_ok=True)

    for filename in os.listdir(category_path):

        img_path = os.path.join(category_path, filename)
        image = cv2.imread(img_path)

        if image is None:
            continue

        transform = random.choice(transformations)
        modified = transform(image)

        save_path = os.path.join(output_category_path, filename)
        cv2.imwrite(save_path, modified)

print("✅ Suspicious images generated successfully!")