import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

# ---------------- Parameters ----------------
img_size = (28, 28)   # image resolution
batch_size = 32
color_mode = "grayscale"   # change to "rgb" if you want 3 channels
epochs = 30

dataset_dir = "dataset"   # path to your dataset folder

# ---------------- Load Datasets ----------------
train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "train"),
    image_size=img_size,
    batch_size=batch_size,
    color_mode=color_mode
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "val"),
    image_size=img_size,
    batch_size=batch_size,
    color_mode=color_mode
)

test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(dataset_dir, "test"),
    image_size=img_size,
    batch_size=batch_size,
    color_mode=color_mode
)

# Class names
class_names = train_ds.class_names
print("Classes:", class_names)

# Save labels to labels.txt
with open("labels.txt", "w") as f:
    for i, name in enumerate(class_names):
        f.write(f"{i} {name}\n")

# ---------------- Dataset Performance Tweaks ----------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------- Data Augmentation ----------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
])

# ---------------- CNN Model ----------------
num_classes = len(class_names)
input_shape = img_size + ((1,) if color_mode=="grayscale" else (3,))

model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=input_shape),
    data_augmentation,

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

# ---------------- Compile Model ----------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- Train Model ----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ---------------- Evaluate on Test ----------------
test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)

# ---------------- Save Model ----------------
model.save("doodle_model.h5")
print("Model saved as doodle_model.h5")
