import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report

IMG_SIZE = 128
DATASET = "dataset"

classes = ["clean", "gaussian", "salt_pepper", "speckle", "motion_blur"]
label_map = {c: i for i, c in enumerate(classes)}

X = []
y = []

# ---- LOAD DATA ----
for label in classes:
    folder = os.path.join(DATASET, label)

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Missing '{folder}/'. Run generate_dataset.py first to build the dataset."
        )

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(label_map[label])

if not X:
    raise RuntimeError(
        "No training images found in dataset/. "
        "Run generate_dataset.py first, and make sure clean_images/ has photos in it."
    )

X = np.array(X)
y_int = np.array(y)
y = to_categorical(y_int, num_classes=len(classes))

print("Dataset loaded:", X.shape)
for i, c in enumerate(classes):
    print(f"  {c}: {(y_int == i).sum()} images")

# ---- SHUFFLE ----
# Images were appended class-by-class, so without shuffling, Keras'
# validation_split (which takes the LAST 20% of the array) could end up
# validating on only one or two classes instead of a representative mix.
rng = np.random.default_rng(42)
perm = rng.permutation(len(X))
X, y, y_int = X[perm], y[perm], y_int[perm]

# ---- CNN MODEL ----
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---- CALLBACKS ----
# Keep the best-validation-accuracy weights instead of whatever the last
# epoch happens to produce, and stop early if it stops improving.
callbacks = [
    ModelCheckpoint("cnn_noise_model.h5", monitor="val_accuracy",
                     save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=4,
                   restore_best_weights=True, verbose=1),
]

# ---- TRAIN ----
history = model.fit(
    X, y,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
)

# ---- SAVE MODEL ----
# ModelCheckpoint already saved the best version to cnn_noise_model.h5,
# but save again here in case EarlyStopping restored better weights.
model.save("cnn_noise_model.h5")
print("✅ CNN model trained and saved!")

# ---- QUICK SANITY CHECK ----
# Per-class report on the held-out validation split (Keras' validation_split
# takes the LAST 20% of the arrays passed to fit(), so this matches it),
# so you can see if any single class (e.g. speckle vs gaussian, which look
# similar) is being confused with another before trusting the model.
val_split_idx = int(len(X) * 0.8)
X_val, y_val_int = X[val_split_idx:], y_int[val_split_idx:]

y_pred = np.argmax(model.predict(X_val), axis=1)
print("\nValidation classification report:")
print(classification_report(y_val_int, y_pred, target_names=classes))
