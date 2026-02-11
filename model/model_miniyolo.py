"""This file contains the model definition and related functions for preprocessing and training."""

import os
from keras import Model, Input
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    LeakyReLU,
    Reshape,
    Concatenate,
    Activation,
    Resizing,
    Softmax,
)

import matplotlib.pyplot as plt

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET


def build_model(S, B, C, input_shape):
    """Builds the MiniYOLO model using the Functional API.

    Args:
        S (int): Number of division of the image (S² is the total number of cells). Ex. S=2 we divide the image in cells pixel wise ([0,0],[0,1],[1,0],[1,1]), 4 cells total.
        B (int): Maximum number of boxes recognizable by the model for each cell. Only one is actually filled for each image in the dataset.
        C (int): Number of classes recognizable by the model.
        input_shape (tuple): Input image shape as (height, width, channels).

    Returns:
        keras.Model: Functional Miniyolo model.
    """

    output_channels = B * 5 + C
    leaky_layer = LeakyReLU(0.1)

    inputs = Input(shape=input_shape, name="image")

    x = Conv2D(16, (3, 3), activation=leaky_layer, padding="same")(inputs)
    x = Conv2D(16, (3, 3), activation=leaky_layer, padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(16, (3, 3), activation=leaky_layer, padding="same")(x)
    x = Conv2D(32, (3, 3), activation=leaky_layer, padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(32, (3, 3), activation=leaky_layer, padding="same")(x)
    x = Conv2D(64, (3, 3), activation=leaky_layer, padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(64, (3, 3), activation=leaky_layer, padding="same")(x)
    x = Conv2D(64, (3, 3), activation=leaky_layer, padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(128, (3, 3), activation=leaky_layer, padding="same")(x)
    x = Conv2D(128, (3, 3), activation=leaky_layer, padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(256, (3, 3), activation=leaky_layer, padding="same")(x)
    x = Conv2D(output_channels, (1, 1), padding="same")(x)
    x = Resizing(S, S, interpolation="bilinear")(x)

    y_pred = x

    pred_boxes = y_pred[..., : B * 5]
    pred_boxes = Reshape((S, S, B, 5))(pred_boxes)

    pred_xy = Activation("sigmoid")(pred_boxes[..., 0:2])
    pred_wh = pred_boxes[..., 2:4]
    pred_conf = Activation("sigmoid")(pred_boxes[..., 4:5])
    pred_boxes = Concatenate(axis=-1)([pred_xy, pred_wh, pred_conf])

    pred_classes = Softmax(axis=-1)(y_pred[..., B * 5 :])

    pred_boxes_flat = Reshape((S, S, B * 5))(pred_boxes)
    y_pred_safe = Concatenate(axis=-1)([pred_boxes_flat, pred_classes])

    return Model(inputs=inputs, outputs=y_pred_safe, name="Miniyolo")


def miniyolo_load_example(
    image_path,
    xml_path,
    max_objects,
    selected_classes,
    S,
    B,
    C,
    image_width,
    image_height,
):
    """Preprocesses each image and prepares the YOLOv1 (S,S,(B*5+C)) target tensor and image for model training.

    Args:
        image_path (str): Path to the image.
        xml_path (str): Path to the xml annotation file related to the image.
        max_objects (int): Maximum number of objects recognizable by the model inside the whole image.
        selected_classes (list(str)): List of classes that the model is trained for recognition.
        S (int): Number of division of the image (S² is the total number of cells). Ex. S=2 we divide the image in cells ([0,0],[0,1],[1,0],[1,1]), 4 cells total.
        B (int): Maximum number of boxes recognizable by the model for each cell. Only one is actually filled for each image in the dataset.
        C (int): Number of classes recognizable by the model.
        image_width (int): Number of pixels to resize the input image width.
        image_height (int): Number of pixels to resize the input image height.

    Returns:
        Resized image Tensor: Model ready image tensor.
        Target Tensor: Target tensor as in YOLOv1 definition.
    """

    image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    image = tf.image.resize(image, (image_height, image_width))
    image = image / 255.0

    labels, bboxes = tf.py_function(
        func=lambda x, y, z: _parse_dataset_xml(x.numpy(), y, z.numpy()),
        inp=[xml_path, max_objects, selected_classes],
        Tout=[tf.int32, tf.float32],
    )

    labels.set_shape([max_objects])
    bboxes.set_shape([max_objects, 4])

    target = tf.py_function(
        func=lambda x, y: _set_target(x.numpy(), y.numpy(), S, B, C),
        inp=[labels, bboxes],
        Tout=tf.float32,
    )

    target.set_shape([S, S, B * 5 + C])
    image.set_shape([image_height, image_width, 3])

    return image, target


def _parse_dataset_xml(xml_path, max_objects, selected_classes):
    """Retrieves and prepares label and bbox tensor from xml files. Function is YOLOv1 agnostic at this point and is separated from _set_target for logic purpose only.

    Args:
        xml_path (str): File path to look into.
        max_objects (int): Maximum number of objects detectable per image.
        selected_classes (list(str)): List containing all the classes to be detected.

    Returns:
        Label Tensor: Tensor containing the classes matches, 0 if none is found. Not yet 1hot encoded.
        Bboxes Tensor: Bboxes tensor containing image relative information for the bounding boxes.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []
    bboxes = []

    selected_classes = [x.decode() for x in selected_classes]

    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    for obj in root.findall("object"):
        if len(labels) == max_objects:
            break
        name = obj.find("name").text

        class_id = {classes: i + 1 for i, classes in enumerate(selected_classes)}.get(
            name, -1
        )

        if class_id != -1:
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) / image_width
            ymin = float(bbox.find("ymin").text) / image_height
            xmax = float(bbox.find("xmax").text) / image_width
            ymax = float(bbox.find("ymax").text) / image_height

            labels.append(class_id)
            bboxes.append([xmin, ymin, xmax, ymax])

    labels = np.array(labels, dtype=np.int32)
    bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)

    num_objs = len(labels)
    if num_objs < max_objects:
        pad_labels = np.zeros((max_objects - num_objs,), dtype=np.int32)
        pad_bboxes = np.zeros((max_objects - num_objs, 4), dtype=np.float32)

        labels = np.concatenate([labels, pad_labels], axis=0)
        bboxes = np.concatenate([bboxes, pad_bboxes], axis=0)

    return labels, bboxes


def _set_target(labels, bboxes, S, B, C):
    """Prepares the true YOLOv1 targets, using previously parsed data from the xml files and S,B,C parameters. The bbox center offset is calculated here and correct
    shaping is applied. Labels are also 1hot encoded here.

    Args:
        labels (Tensor): Labels tensor from _parse_dataset_xml function
        bboxes (Tensor): Bboxes tensor from _parse_dataset_xml
        S (int): Number of division of the image (S² is the total number of cells). Ex. S=2 we divide the image in cells ([0,0],[0,1],[1,0],[1,1]), 4 cells total.
        B (int): Maximum number of boxes recognizable by the model for each cell. Only one is actually filled for each image in the dataset.
        C (int): Number of classes recognizable by the model.

    Returns:
        Target tensor: Final SxSx(B*5+C) YOLOv1 target tensor.
    """

    target = np.zeros((S, S, B * 5 + C), dtype=np.float32)

    if not np.all(labels == 0):
        for label, bbox in zip(labels, bboxes):
            if label == 0:
                continue

            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            cell_x = int(x_center * S)
            cell_y = int(y_center * S)
            cell_x = min(cell_x, S - 1)
            cell_y = min(cell_y, S - 1)

            x_center_offset = x_center * S - cell_x
            y_center_offset = y_center * S - cell_y

            if target[cell_y, cell_x, 4] == 1:
                curr_w = target[cell_y, cell_x, 2]
                curr_h = target[cell_y, cell_x, 3]
                if (w * h) <= (curr_w * curr_h):
                    continue

            target[cell_y, cell_x, 0:5] = [x_center_offset, y_center_offset, w, h, 1.0]
            target[cell_y, cell_x, B * 5 + (label - 1)] = 1.0

    return target


def miniyolo_optimizer(lr, mo, wd):
    """Creates a Stochastic Gradient Descend optimizer for the learning process.

    Args:
        lr (float): Learning rate at which the parameters are changed during learning.
        mo (float): Momentum keeps the model learning towards improvements.
        wd (float): Weight decay reduces large weights to prevent overfitting.

    Returns:
        keras.optimizers.SGD: SGD object used for training.
    """

    return SGD(learning_rate=lr, momentum=mo, weight_decay=wd)


def miniyolo_model_callback(dir_path):
    """Callback that saves the model model and configuration on improvements after each epoch.

    Args:
        dir_path (str): Path to the saving directory.

    Returns:
        keras.callbacks.ModelCheckpoint: Returns the callback.
    """

    path = os.path.join(
        dir_path,
        "trained-model-epoch:{epoch:03d}-val_loss{val_loss:.3f}-loss{loss:.3f}.keras",
    )
    return ModelCheckpoint(
        filepath=path, monitor="val_loss", save_best_only=False, verbose=1
    )


def miniyolo_weights_callback(dir_path):
    """Callback that saves the model weights only on model improvements after each epoch.

    Args:
        dir_path (str): Path to the saving directory.

    Returns:
        keras.callbacks.ModelCheckpointype: Returns the callback.
    """

    path = os.path.join(
        dir_path,
        "final-weights-epoch:{epoch:03d}-val_loss{val_loss:.3f}-loss{loss:.3f}.weights.h5",
    )
    return ModelCheckpoint(
        filepath=path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1,
    )


def plot_training_history(history, out_dir):
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    if not loss:
        print("No loss history to plot.")
        return

    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss, label="loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to: {out_path}")
