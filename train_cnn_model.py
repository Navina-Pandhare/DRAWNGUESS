import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------
# 1. Load Dataset
# -----------------------
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(64, 64),        # resize all images to 64x64
    color_mode="grayscale",     # doodles are black & white
    batch_size=32
)

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(64, 64),
    color_mode="grayscale",
    batch_size=32
)

# Save class names before mapping
class_names = train_ds_raw.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Normalize pixel values (0–255 -> 0–1)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds_raw.map(lambda x, y: (normalization_layer(x), y))

# -----------------------
# 2. Build CNN Model
# -----------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# 3. Train Model (with EarlyStopping & ReduceLROnPlateau)
# -----------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

history = model.fit(
    train_ds,
    epochs=30,                  # higher limit; ES will stop earlier
    validation_data=test_ds,
    callbacks=[early_stop, reduce_lr]
)

# -----------------------
# 4. Evaluate
# -----------------------
test_loss, test_acc = model.evaluate(test_ds)
print("✅ Test Accuracy:", test_acc)

# -----------------------
# 5. Save Model
# -----------------------
model.save("models/doodle_model.h5")
print("🎉 Model saved as doodle_model.h5")

# -----------------------
# 6. Plot Training History
# -----------------------
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
