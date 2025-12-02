import os

# Removes tf logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow_datasets as tfds
import tensorflow as tf

from keras.layers import Input
from model import MiniYOLO, miniYOLO_optimizer


# WORKFLOW:
# 1. DATASET IMPORT: See if you want to add: shuffle, config, play with % ds sizes | validation set is left for inference see %s
# 2. DATASET PREPROCESSING: keeps useful tensors from tf dataset
# 3. MODEL INITIALIZATION: configurate inside MiniYOLO class
# 4. MODEL TRAIN the model: TODO
# 5. MODEL SAVE the model: TODO

# ------------------------------------------------------------------------------

# SETTINGS

# Generics
# TODO Implement directory initialization (if not exist etc.) and implement model versioning, training saves backups etc
DATA_DIR = "./data"
MODEL_DIR = "./model-saves"

# Model configs
IMG_SIZE = (244, 244)
BATCH_SIZE = 32
B = 2
S = 4
C = 20

# Optimizer configs
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.005

# Training configs
EPOCH_NUM = 10


# FUNCTIONS
def preprocess(example):
    image = example["image"]
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    # TODO -- Pick the first label if multiple objects exist -- change this based on how many class we want
    label = example["objects"]["label"][0]
    return image, label


# 1. DATASET IMPORT

# VOC2007 Dowload and assignes data
train_ds, validation_ds = tfds.load(
    "voc",
    split=[
        "train[:80%]+test[:80%]+validation[:80%]",
        "train[80%:90%]+test[80%:90%]+validation[80%:90%]",
    ],
    data_dir=DATA_DIR,
)

# TODO -- Remove (needed for Pylance only)
train_ds: tf.data.Dataset
validation_ds: tf.data.Dataset

# 2. DATASET PREPROCESSING

train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
validation_ds = validation_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

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
input_layer = Input(shape=(244, 244, 3))
model = MiniYOLO(IMG_SIZE[0], IMG_SIZE[1])
output = model(input_layer)
model.summary()

# 4. MODEL TRAIN

model.compile(
    optimizer=miniYOLO_optimizer(LEARNING_RATE, MOMENTUM),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=validation_ds, epochs=EPOCH_NUM)

# 5. MODEL SAVE
model_path = os.path.join(MODEL_DIR, "yolov2_test_with_reshaperesize.keras")
model.save(model_path)
