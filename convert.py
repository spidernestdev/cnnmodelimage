import tensorflow as tf

model = tf.keras.models.load_model("models/security_model.h5")
model.save("models/security_model.keras")

print("✅ Model converted")