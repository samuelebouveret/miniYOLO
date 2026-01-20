import os

# Removes tf logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import tensorflow_datasets as tfds
import tensorflow as tf

from keras.layers import Input
from model import (
    MiniYOLOModel,
    miniYOLO_load_example,
    miniYOLO_optimizer,
    miniYOLO_saving_callback,
)
from loss import MiniYOLO_loss

# TODO -- NEXT STEPS:
# LOSS AND METRICS FUNCTIONS OR JUST LOSS PROBABLY
# ADD IMAGE AUGMENTATION LAYERS
# TOO MANY RESIZES: PREPROCESSING+MODEL MAYBE SEE WHICH ONE TO KEEP CONSIDERING MICROC CAMERA OR ASSUME PERFECT INPUT IMAGE SIZE
# See if you want to add: shuffle, config, play with % ds sizes | validation set is left for inference see %s
# CHANGE DOCSTRING SBC , S IS NOT NECESSARILY PIXEL WISE BUT JUST VIRTUAL DIVISION AND B DESCRIPTION IS DEFINITELY NOT ACCURATE

# CONSIDERATIONS FOR INFERENCE: IMAGES IN DATASET ARE RESIZE TO IMG_SIZE DURING PREPROCESSING, SO INFERENCE
# MIGHT BE WEIRD DEPENDING ON IMAGE SIZE TODO LATER WHEN TESTING INFERENCE


# WORKFLOW:
# CONFIGURATION
# 1. DATASET IMPORT
# 2. DATASET PREPROCESSING
# 3. MODEL INITIALIZATION
# 4. MODEL COMPILATION
# 5. MODEL TRAINING AND SAVING

# ------------------------------------------------------------------------------

# --- CONFIGURATION START ---

# TODO -- DEBUG MODE TO RUN EAGERLY -- TESTING ONLY
tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True)

# Generic configs
DATA_DIR_IMAGES = "./data/images"
DATA_DIR_ANNOTATIONS = "./data/annotations"
SAVE_DIR = "./model-saves"

# TODO -- Maybe download and move to "images" "annotations" directories but yeah idk maybe not useful for this project
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset configs
TRAIN_VAL_RATIO = 0.9

# Model configs
ALL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

SELECTED_CLASSES = ["chair", "car", "person"]
C = len(SELECTED_CLASSES)
B = 2
S = 2
MAX_OBJECTS = 3
IMG_SIZE = (224, 224)

# Optimizer configs
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Training configs
EPOCH_NUM = 1
BATCH_SIZE = 32

# Loss function configs
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

# --- CONFIGUARTION END ---


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
        lambda image_path, xml_path: miniYOLO_load_example(
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

    # # TODO -- Debugging only
    # for image, target in dataset:
    #     print(f"LABEL -> {type(target)} -- TARGET -> {target}")

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
    input_layer = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model = MiniYOLOModel(S, B, C)
    output = model(input_layer)
    model.summary()

    # 4. MODEL COMPILATION
    opt = miniYOLO_optimizer(LEARNING_RATE, MOMENTUM, WEIGHT_DECAY)
    loss_fn = MiniYOLO_loss(S, B, C, LAMBDA_COORD, LAMBDA_NOOBJ)

    model.compile(
        optimizer=opt,
        loss=loss_fn,
    )

    # 5. MODEL TRAINING AND SAVING
    callback = miniYOLO_saving_callback(SAVE_DIR)

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCH_NUM,
        callbacks=callback,
    )


if __name__ == "__main__":
    print("Starting training process.")
    run_training()
    print("Training ended succesfully.")
