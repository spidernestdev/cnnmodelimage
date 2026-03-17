import tensorflow as tf

model = tf.keras.models.load_model("models/security_model.keras", compile=False)
model.save_weights("models/model.weights.h5")
print("✅ weights saved")