import os

# Removes TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from model import (
    build_model,
    miniyolo_load_example,
    miniyolo_optimizer,
    miniyolo_model_callback,
    miniyolo_weights_callback,
    MiniyoloLoss,
)

# ------------------------------

# WORKFLOW:
# CONFIGURATION
# 1. DATASET IMPORT
# 2. DATASET PREPROCESSING
# 3. MODEL INITIALIZATION
# 4. MODEL COMPILATION
# 5. MODEL TRAINING AND SAVING
# 6. EXPORT FINAL MODEL: for STM pipeline (needs a purely functional model)

# ------------------------------

# --- CONFIGURATION START ---

# Generic configs
DATA_DIR_IMAGES = "./data/images"
DATA_DIR_ANNOTATIONS = "./data/annotations"
BASE_DIR = "./trained"
MODEL_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Dataset configs
TRAIN_VAL_RATIO = 0.8

# Model configs
SELECTED_CLASSES = ["chair", "car", "person"]
B = 2
S = 2
C = len(SELECTED_CLASSES)
MAX_OBJECTS = 3
IMG_SIZE = (88, 88)

# Optimizer configs
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Training configs
EPOCH_NUM = 1
BATCH_SIZE = 64

# Loss function configs
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

# --- CONFIGURATION END ---


def run_training():
    # 1. DATASET IMPORT
    image_files = sorted(
        os.path.join(DATA_DIR_IMAGES, f)
        for f in os.listdir(DATA_DIR_IMAGES)
        if f.lower().endswith(".jpg")
    )

    xml_files = [
        f.replace(DATA_DIR_IMAGES, DATA_DIR_ANNOTATIONS).replace(".jpg", ".xml")
        for f in image_files
    ]

    dataset = tf.data.Dataset.from_tensor_slices((image_files, xml_files))

    # 2. DATASET PREPROCESSING
    dataset = dataset.map(
        lambda image_path, xml_path: miniyolo_load_example(
            image_path,
            xml_path,
            MAX_OBJECTS,
            SELECTED_CLASSES,
            S,
            B,
            C,
            IMG_SIZE[0],
            IMG_SIZE[1],
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_size = int(len(dataset) * TRAIN_VAL_RATIO)
    train_ds = dataset.take(train_size)
    validation_ds = dataset.skip(train_size)

    print(f"TOTAL IMAGES count: {len(train_ds)+len(validation_ds)}")
    print(f"TRAINING dataset image count: {len(train_ds)}")
    print(f"VALIDATION dataset image count: {len(validation_ds)}")

    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"Epochs: {EPOCH_NUM}")
    print(f"Batches per epoch: {len(train_ds)}")

    # 3. MODEL INITIALIZATION
    model = build_model(S, B, C, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    # 4. MODEL COMPILATION
    opt = miniyolo_optimizer(LEARNING_RATE, MOMENTUM, WEIGHT_DECAY)
    loss_fn = MiniyoloLoss(S, B, C, LAMBDA_COORD, LAMBDA_NOOBJ)

    model.compile(
        optimizer=opt,
        loss=loss_fn,
    )

    # 5. MODEL TRAINING AND SAVING
    model_callback = miniyolo_model_callback(MODEL_DIR)
    weights_callback = miniyolo_weights_callback(WEIGHTS_DIR)

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCH_NUM,
        callbacks=[model_callback, weights_callback],
    )

    # 6. EXPORT FINAL MODEL: for STM pipeline
    export_model = build_model(S, B, C, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    export_model.load_weights(os.path.join(WEIGHTS_DIR, "final.weights.h5"))
    export_model.save(os.path.join(MODEL_DIR, "final-model.keras"))


if __name__ == "__main__":
    print("Starting training process.")
    run_training()
    print("Training ended successfully.")
