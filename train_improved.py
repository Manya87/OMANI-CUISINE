
import tensorflow as tf
import numpy as np
import json
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = Path("dataset_split")
MODEL_DIR = Path("models")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 10
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

MODEL_DIR.mkdir(exist_ok=True)

# ==========================================
# 2. IDENTIFY 50 DISH CLASSES
# ==========================================

train_root = DATA_DIR / "train"

class_names = sorted([
    f"{category.name}/{dish.name}"
    for category in train_root.iterdir()
    if category.is_dir()
    for dish in category.iterdir()
    if dish.is_dir()
])

assert len(class_names) == 50, (
    f"Expected 50 classes, found {len(class_names)}"
)

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

with open(MODEL_DIR / "class_names_improved.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("Total dish classes:", len(class_names))

# ==========================================
# 3. LOAD DATASET
# ==========================================

def load_split(split):

    images = []
    labels = []

    for class_name in class_names:

        folder = DATA_DIR / split / class_name

        if not folder.exists():
            raise FileNotFoundError(folder)

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

                image = tf.keras.utils.img_to_array(
                    image,
                    dtype="uint8"
                )

                images.append(image)
                labels.append(class_to_index[class_name])

            except Exception as e:
                print("Skipped:", file, e)

    if not images:
        raise ValueError(f"No images found in {split}")

    X = np.asarray(images, dtype=np.uint8)
    y = np.asarray(labels, dtype=np.int32)

    print(
        split.upper(),
        "Images:", len(X),
        "Classes:", len(np.unique(y))
    )

    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

# ==========================================
# 4. MILD CLASS WEIGHTING
# ==========================================

counts = np.bincount(y_train, minlength=len(class_names))

raw_weights = np.sqrt(
    len(y_train) / (len(class_names) * counts)
)

raw_weights = raw_weights / np.mean(raw_weights)

class_weights = {
    i: float(raw_weights[i])
    for i in range(len(class_names))
}

print("\nCLASS WEIGHTS")
print("Minimum:", min(class_weights.values()))
print("Maximum:", max(class_weights.values()))

# ==========================================
# 5. TENSORFLOW DATA PIPELINE
# ==========================================

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(len(X_train), seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ==========================================
# 6. DATA AUGMENTATION
# ==========================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomContrast(0.10)
], name="food_augmentation")

# ==========================================
# 7. LOAD MOBILENETV2
# ==========================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ==========================================
# 8. BUILD CLASSIFICATION MODEL
# ==========================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.4)(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# ==========================================
# 9. INITIAL TRAINING
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "phase1_best.keras"),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.00001
    )
]

print("\n========== INITIAL TRAINING ==========")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# 10. FINE-TUNING
# ==========================================

print("\n========== FINE-TUNING ==========")

base_model.trainable = True

# Freeze earlier layers and fine-tune last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "phase2_best.keras"),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.000001
    )
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=fine_callbacks
)

# ==========================================
# 11. SELECT BEST VALIDATION MODEL
# ==========================================

print("\n========== MODEL COMPARISON ==========")

phase1 = tf.keras.models.load_model(
    MODEL_DIR / "phase1_best.keras"
)

phase2 = tf.keras.models.load_model(
    MODEL_DIR / "phase2_best.keras"
)

loss1, acc1 = phase1.evaluate(val_ds, verbose=0)
loss2, acc2 = phase2.evaluate(val_ds, verbose=0)

print("Phase 1 validation accuracy:", acc1)
print("Phase 2 validation accuracy:", acc2)

if loss2 < loss1:
    best_model = phase2
    selected = "Fine-tuned model"
else:
    best_model = phase1
    selected = "Initial transfer-learning model"

best_model.save(
    MODEL_DIR / "omani_cuisine_model_improved.keras"
)

print("\n========== COMPLETED ==========")
print("Selected model:", selected)
print("Best validation accuracy:", max(
    acc1 if loss1 <= loss2 else acc2,
    0
))
print("Saved: models/omani_cuisine_model_improved.keras")
print("Classes:", len(class_names))
