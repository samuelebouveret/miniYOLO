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
# ONLY CHAIR[9] PERSON[15] CAR[7] FROM DATASET -- DONE
# IUO FUNCTIONS
# ADD IMAGE AUGMENTATION LAYERS
# TOO MANY RESIZES: PREPROCESSING+MODEL MAYBE SEE WHICH ONE TO KEEP CONSIDERING MICROC CAMERA OR ASSUME PERFECT INPUT IMAGE SIZE
# See if you want to add: shuffle, config, play with % ds sizes | validation set is left for inference see %s

# CONSIDERATIONS FOR INFERENCE: IMAGES IN DATASET ARE RESIZE TO IMG_SIZE DURING PREPROCESSING, SO INFERENCE
# MIGHT BE WEIRD DEPENDING ON IMAGE SIZE TODO LATER WHEN TESTING INFERENCE


# WORKFLOW:
# 1. DATASET IMPORT:
# 2. DATASET PREPROCESSING
# 3. MODEL INITIALIZATION
# 4. MODEL COMPILATION
# 5. MODEL TRAINING AND SAVING

# ------------------------------------------------------------------------------

# SETTINGS

# TODO -- DEBUG MODE TO RUN EAGERLY -- TESTING ONLY
tf.data.experimental.enable_debug_mode()

# Generics
DATA_DIR_IMAGES = "./data-temp/images"
DATA_DIR_ANNOTATIONS = "./data-temp/annotations"
SAVE_DIR = "./model-saves"

# TODO -- Maybe download and move to "images" "annotations" directories but yeah idk maybe not useful for this project
os.makedirs(SAVE_DIR, exist_ok=True)

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
B = 1
S = 2
MAX_OBJECTS = 3
IMG_SIZE = (224, 224)


# Optimizer configs
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Training configs
EPOCH_NUM = 1
BATCH_SIZE = 32

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

# TODO -- Debugging only
for image, target in dataset:
    print(f"LABEL -> {image.shape} -- BBOX -> {target.shape}")

ds_size = dataset.cardinality().numpy()
train_size = int(ds_size * TRAIN_VAL_RATIO)

train_ds = dataset.take(train_size)
validation_ds = dataset.skip(train_size)

print(f"TOTAL IMAGES count: {len(train_ds)+len(validation_ds)}")
print(f"TRAINING dataset image count: {len(train_ds)}")
print(f"VALIDATION dataset image count: {len(validation_ds)}")

# TODO -- Maybe keep shuffle for training - Maybe keep forced batch here (model.fit also batches with a defaul but can keep here to make it explicit)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Calculate batches before training
n_batches = tf.data.experimental.cardinality(train_ds).numpy()
print(f"Epochs: {EPOCH_NUM}")
print(f"Batches per epoch: {n_batches}")

# 3. MODEL INITIALIZATION
input_layer = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
model = MiniYOLOModel(S, B, C)
output = model(input_layer)
model.summary()

# 4. MODEL COMPILATION
model.compile(
    optimizer=miniYOLO_optimizer(LEARNING_RATE, MOMENTUM, WEIGHT_DECAY),
    # loss="binary_crossentropy", TODO -- Obviously change was just for testing
)

# 5. MODEL TRAINING AND SAVING
model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCH_NUM,
    callbacks=miniYOLO_saving_callback(SAVE_DIR),
)
