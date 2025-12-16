import os
import glob

# Removes tf logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import tensorflow_datasets as tfds
import tensorflow as tf

from keras.layers import Input
from model import (
    MiniYOLO,
    miniYOLO_optimizer,
    miniYOLO_load_example,
    miniYOLO_saving_callback,
)

# NEXT STEPS:
# LOSS AND METRICS FUNCTIONS
# ONLY CHAIR[9] PERSON[15] CAR[7] FROM DATASET
# IUO FUNCTIONS
# ADD IMAGE AUGMENTATION LAYERS

# WORKFLOW:
# 1. DATASET IMPORT: See if you want to add: shuffle, config, play with % ds sizes | validation set is left for inference see %s
# 2. DATASET PREPROCESSING: keeps useful tensors from tf dataset
# 3. MODEL INITIALIZATION: configurate inside MiniYOLO class
# 4. MODEL TRAIN the model: TODO
# 5. MODEL SAVE the model: TODO

# ------------------------------------------------------------------------------

# SETTINGS

# DEBUG MODE TO RUN EAGERLY
tf.data.experimental.enable_debug_mode()

# Generics
# tfds dataset -- not useful anymore?
DATA_DIR = "./data"
DATA_DIR_IMAGES = "./data-temp/images"
DATA_DIR_ANNOTATIONS = "./data-temp/annotations"
SAVE_DIR = "./model-saves"

os.makedirs(DATA_DIR, exist_ok=True)
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
B = 2
S = 4
MAX_OBJECTS = 7
IMG_SIZE = (88, 88)


# Optimizer configs
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Training configs
EPOCH_NUM = 1
BATCH_SIZE = 32

# 1. DATASET IMPORT
image_files = sorted(glob.glob(os.path.join(DATA_DIR_IMAGES, "*.jpg")))
xml_files = [
    f.replace(DATA_DIR_IMAGES, DATA_DIR_ANNOTATIONS).replace(".jpg", ".xml")
    for f in image_files
]

dataset = tf.data.Dataset.from_tensor_slices((image_files, xml_files))

# 2. DATASET PREPROCESSING
dataset = dataset.map(
    lambda image_path, xml_path: miniYOLO_load_example(
        image_path, xml_path, MAX_OBJECTS, SELECTED_CLASSES
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# TODO -- Remove its for debugging only
for image, label, bbox in dataset:
    print(f"LABEL -> {label} -- BBOX -> {bbox}")

ds_size = dataset.cardinality().numpy()
train_size = int(ds_size * TRAIN_VAL_RATIO)

train_ds = dataset.take(train_size)
validation_ds = dataset.skip(train_size)

print(f"TOTAL IMAGES count: {len(train_ds)+len(validation_ds)}")
print(f"TRAINING dataset image count: {len(train_ds)}")
print(f"VALIDATION dataset image count: {len(validation_ds)}")

# TODO -- Overwrite batch size since model.fit batches before training -- Either remove or change batch/shuffle size here
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Calculate batches before training
n_batches = tf.data.experimental.cardinality(train_ds).numpy()
print(f"Epochs: {EPOCH_NUM}")
print(f"Batches per epoch: {n_batches}")

# 3. MODEL INITIALIZATION
input_layer = Input(shape=(None, None, 3))
model = MiniYOLO(IMG_SIZE[0], IMG_SIZE[1], S, B, C)
output = model(input_layer)
model.summary()

# 4. MODEL COMPILATION
model.compile(
    optimizer=miniYOLO_optimizer(LEARNING_RATE, MOMENTUM, WEIGHT_DECAY),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 5. MODEL TRAINING AND SAVING
model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCH_NUM,
    callbacks=miniYOLO_saving_callback(SAVE_DIR),
)
