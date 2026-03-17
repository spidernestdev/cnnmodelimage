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
    allow_origins=["*"],  # allow all for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "clean_model.keras")  # ✅ updated

model = tf.keras.models.load_model(model_path, compile=False)

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