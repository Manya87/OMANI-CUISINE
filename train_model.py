
import tensorflow as tf
from pathlib import Path
import json
import numpy as np

# Configuration
DATA_DIR = Path("dataset_split")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

tf.random.set_seed(SEED)

# Discover all 50 classes from training directory
train_root = DATA_DIR / "train"

class_names = sorted([
    f"{category.name}/{dish.name}"
    for category in train_root.iterdir()
    if category.is_dir()
    for dish in category.iterdir()
    if dish.is_dir()
])

assert len(class_names) == 50, f"Expected 50 classes, found {len(class_names)}"

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

# Save class names
Path("models").mkdir(exist_ok=True)

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

# Load images explicitly from nested folders
def load_split(split):
    images = []
    labels = []

    for class_name in class_names:
        folder = DATA_DIR / split / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for file in folder.iterdir():
            if file.suffix.lower() not in [
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            ]:
                continue

            try:
                image = tf.keras.utils.load_img(
                    file,
                    target_size=IMG_SIZE
                )

                image = tf.keras.utils.img_to_array(image)

                images.append(image)
                labels.append(class_to_index[class_name])

            except Exception as e:
                print(f"Skipping {file}: {e}")

    if not images:
        raise ValueError(f"No valid images found in {split}")

    print(f"{split}: {len(images)} images, {len(set(labels))} classes")

    return np.array(images, dtype=np.float32), np.array(labels)

# Load all splits
X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")
X_test, y_test = load_split("test")

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
).shuffle(len(X_train), seed=SEED).batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1)
])

# MobileNetV2 pretrained model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# Classification model
inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save model
model.save("models/omani_cuisine_model.keras")

print("\nTRAINING COMPLETED!")
print("Total classes:", len(class_names))
print("Best validation accuracy:",
      max(history.history["val_accuracy"]))
