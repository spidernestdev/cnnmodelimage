import tensorflow as tf

# DATASET PATHS
train_dir = r"C:\Users\anshu\Desktop\Hackathon\dataset\train"
val_dir = r"C:\Users\anshu\Desktop\Hackathon\dataset\val"
test_dir = r"C:\Users\anshu\Desktop\Hackathon\dataset\test"

# SETTINGS
IMG_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 10

# LOAD DATASETS
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# PRINT CLASS NAMES
print("Classes:", train_dataset.class_names)

# NORMALIZATION (0–255 → 0–1)
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# DATA PIPELINE OPTIMIZATION
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# PRINT DATASET INFO
print("Train batches:", len(train_dataset))
print("Validation batches:", len(val_dataset))
print("Test batches:", len(test_dataset))

# BUILD CNN MODEL
# BUILD MODEL
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

# COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# SHOW MODEL STRUCTURE
model.summary()

# TRAIN MODEL
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# SAVE MODEL
model.save("models/security_model.h5")
print("Model saved to models/security_model.h5")

# TEST MODEL
test_loss, test_acc = model.evaluate(test_dataset)

print("Test Accuracy:", test_acc)