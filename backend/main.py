from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# BASE PATH
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
weights_path = os.path.join(BASE_DIR, "models", "model.weights.h5")

# 🔥 BUILD MODEL (same architecture as training)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32,32,3)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(2, activation='softmax')
])

# ✅ LOAD ONLY WEIGHTS (no version issue)
model.load_weights(weights_path)

@app.get("/")
def home():
    return {"message": "AI Image Security API running"}

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((32,32))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]

    safe_score = float(prediction[0])
    suspicious_score = float(prediction[1])

    result = "suspicious" if suspicious_score > safe_score else "safe"

    return {
        "status": result,
        "safe_score": safe_score,
        "suspicious_score": suspicious_score
    }