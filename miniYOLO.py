import os

# Removes tf logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow_datasets as tfds
import tensorflow as tf

from keras.layers import Input, Resizing, Rescaling
from model import MiniYOLO


# WORKFLOW:
# 1. DATASET IMPORT: See if you want to add: shuffle, config, play with % ds sizes | validation set is left for inference see %s
# 2. PREPROCESS: tf.resize vs Keras model Resizing layer
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
PREPROCESS_IN_MODEL = False
IMG_SIZE = (244, 244)
BATCH_SIZE = 32
B = 2
S = 4
C = 20

# Training configs
EPOCH_NUM = 1


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

print(f"TOTAL IMAGES count: {len(train_ds)+len(validation_ds)}")
print(f"TRAINING dataset image count: {len(train_ds)}")
print(f"VALIDATION dataset image count: {len(validation_ds)}")

# 2. PREPROCESSING:
# PREPROCESS_IN_MODEL is True ->  preprocessing is part of the model with a resize and rescale layer
# PREPROCESS_IN_MODEL is False -> preprocessing is done on the dataset

# INPUT layer
input_ly = Input(shape=(244, 244, 3))

# if PREPROCESS_IN_MODEL:
#     print("Creating Keras Reisizing and Rescaling layer for preprocessing any inputs.")
#     x = Resizing(height=IMG_SIZE[0], width=IMG_SIZE[1])(input_ly)
#     x = Rescaling(1.0 / 255)(x)
# else:
#     print("Tensorflow preprocessing the training and validation ds. No Keras layers")
#     train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
#     validation_ds = validation_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# TODO -- Overwrite batch size since model.fit batches before training -- Either remove or change batch/shuffle size here
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_ds = validation_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Calculate batches before training
n_batches = tf.data.experimental.cardinality(train_ds).numpy()
print(f"Epochs: {EPOCH_NUM}")
print(f"Batches per epoch: {n_batches}")

# 3. Model initialization
model = MiniYOLO(IMG_SIZE[0], IMG_SIZE[1])
output = model(input_ly)
model.summary()

# 4. MODEL TRAIN
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(train_ds, validation_data=validation_ds, epochs=EPOCH_NUM)

# 5. MODEL SAVE
model_path = os.path.join(MODEL_DIR, "yolov1_test_with_reshaperesize.keras")
model.save(model_path)
