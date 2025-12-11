import os

# Removes tf logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow_datasets as tfds
import tensorflow as tf

from keras.layers import Input
from model import MiniYOLO, miniYOLO_optimizer, prepare_input, miniYOLO_saving_callback

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

# Generics
DATA_DIR = "./data"
SAVE_DIR = "./model-saves"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Model configs
IMG_SIZE = (244, 244)
BATCH_SIZE = 32
B = 2
S = 4
C = 20

# Optimizer configs
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Training configs
EPOCH_NUM = 10

# 1. DATASET IMPORT

# VOC2007 Dowload and assignes data
train_ds, validation_ds = tfds.load(
    "voc",
    split=[
        "train[:90%]+test[:90%]+validation[:90%]",
        "train[90%:]+test[90%:]+validation[90%:]",
    ],
    data_dir=DATA_DIR,
)

# TODO -- Remove (needed for Pylance only)
train_ds: tf.data.Dataset
validation_ds: tf.data.Dataset

# 2. DATASET PREPROCESSING
train_ds = train_ds.map(prepare_input, num_parallel_calls=tf.data.AUTOTUNE)
validation_ds = validation_ds.map(prepare_input, num_parallel_calls=tf.data.AUTOTUNE)

print(f"TOTAL IMAGES count: {len(train_ds)+len(validation_ds)}")
print(f"TRAINING dataset image count: {len(train_ds)}")
print(f"VALIDATION dataset image count: {len(validation_ds)}")

# TODO -- REMOVE
# it = iter(train_ds)
# print(next(it))

# TODO -- Overwrite batch size since model.fit batches before training -- Either remove or change batch/shuffle size here
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Calculate batches before training
n_batches = tf.data.experimental.cardinality(train_ds).numpy()
print(f"Epochs: {EPOCH_NUM}")
print(f"Batches per epoch: {n_batches}")

# 3. MODEL INITIALIZATION
input_layer = Input(shape=(None, None, 3))
model = MiniYOLO(IMG_SIZE[0], IMG_SIZE[1])
output = model(input_layer)
model.summary(expand_nested=True)

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
