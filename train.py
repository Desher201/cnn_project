import json
import os
from idlelib import history
from pathlib import Path

import tensorflow as tf
from keras.src.callbacks import csv_logger
from tensorflow.keras import layers, models
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

from image_preparation import clean_images

# Основні налаштування
DATA_DIR = Path('cats_and_dogs_filtered')
CHECKPOINT_DIR = Path('checkpoints')
LOG_DIR = Path('logs')
MODEL_DIR = Path('model')
IMAGE_SIZE = (500, 500)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

clean_images(DATA_DIR / "train/cats")
clean_images(DATA_DIR / "train/dogs")

# Завантаження набору даних
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)
print(type(train_ds))
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "validation",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

# Отримання порядків класів для розпізнавання
class_names = train_ds.class_names
print("Класи: ", class_names)
print(type(class_names))

# Збереження імен класів у файлі
with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8-sig") as file:
    json.dump(class_names, file, ensure_ascii=False, indent=4)

# Оптимізація завантаження даних
AUTOTUNE = tf.data.AUTOTUNE
print(type(AUTOTUNE))
print(AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Збільшення різноманітності навчальних даних (агментація)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# Створення CNN

model = models.Sequential([
    layers.Input(shape=(*IMAGE_SIZE, 3)),
    data_augmentation,
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),

    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),

])

# Компіляція моделі
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Збереження останнього стану після кожної епохи
last_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    # file_path = str(CHECKPOINT_DIR / "latest.ckpt"),

    filepath=str(CHECKPOINT_DIR / "latest.keras"),
    save_best_only=True,
    save_freq="epoch",
)
# Збереження найкращої моделі за точністю на валідації
best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(CHECKPOINT_DIR / "best.keras"),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

# Збереження показників кожної епохи в CSV
csv_log = tf.keras.callbacks.CSVLogger(
    filename=str(LOG_DIR / "training.csv"),
    append=True,
)

# Зупинка якщо модель перестала покращуватись
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
)
# Запис графів та метрик для інформації яка буде відображатись через TensorBoard
tensor_board = tf.keras.callbacks.TensorBoard(
    log_dir=str(LOG_DIR / "tensorboard"),

)

# Навчання моделі
history_1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        last_checkpoint,
        best_checkpoint,
        csv_log,
        early_stop,
        tensor_board
    ],
)

# Збереження готової моделі після навчання
final_model_path = Path(MODEL_DIR / "final_model.keras")
model.save(final_model_path)