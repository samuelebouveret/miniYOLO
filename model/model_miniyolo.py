"""This file contains the model definition and related functions for preprocessing and training."""

import os
from keras import Model
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    LeakyReLU,
)

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET


# TODO -- CHANGE HEAD LATER FOR BETTER MEMORY (TOO MANY PARAMETERS NOW) OR REDUCE DENSE FILTER
# TODO -- Maybe add param leaky
class MiniyoloModel(Model):
    """Defines the MiniYOLO model overriding __init__ and call functions as in Keras documentation."""

    def __init__(self, S, B, C):
        """Creates the model structure.

        Args:
            S (int): Number of division of the image (S² is the total number of cells). Ex. S=2 we divide the image in cells pixel wise ([0,0],[0,1],[1,0],[1,1]), 4 cells total.
            B (int): Maximum number of boxes recognizable by the model for each cell. Only one is actually filled for each image in the dataset.
            C (int): Number of classes recognizable by the model.
        """

        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.output_channels = B * 5 + C

        # FOR DENSE HEAD
        self.output_size = S * S * (B * 5 + C)

        leaky_layer = LeakyReLU(0.1)
        self.conv1 = Conv2D(16, (3, 3), activation=leaky_layer, padding="same")
        self.conv2 = Conv2D(16, (3, 3), activation=leaky_layer, padding="same")
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv3 = Conv2D(16, (3, 3), activation=leaky_layer, padding="same")
        self.conv4 = Conv2D(32, (3, 3), activation=leaky_layer, padding="same")
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv5 = Conv2D(32, (3, 3), activation=leaky_layer, padding="same")
        self.conv6 = Conv2D(64, (3, 3), activation=leaky_layer, padding="same")
        self.pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv7 = Conv2D(64, (3, 3), activation=leaky_layer, padding="same")
        self.conv8 = Conv2D(64, (3, 3), activation=leaky_layer, padding="same")
        self.pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv9 = Conv2D(128, (3, 3), activation=leaky_layer, padding="same")
        self.conv10 = Conv2D(128, (3, 3), activation=leaky_layer, padding="same")
        self.pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)

        # DENSE HEAD
        # self.flatten = Flatten()
        # self.fc1 = Dense(256, activation=leaky_layer)
        # self.fc2 = Dense(self.output_size)

        # CONVVOLUTIONAL HEAD
        self.conv11 = Conv2D(256, (3, 3), activation=leaky_layer, padding="same")
        self.conv_out = Conv2D(self.output_channels, (1, 1), padding="same")

    def call(self, inputs):
        """Define the forward propagation of the model.

        Args:
            inputs (keras.layers.Input): Input layer corresponding to the image input.

        Returns:
            Tensor: Reshaped output YOLOv1 tensor -> (S,S,(B*5+C)).
        """

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool4(x)

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool5(x)

        # DENSE HEAD
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)

        # CONVOLUTIONAL HEAD
        x = self.conv11(x)
        x = self.conv_out(x)
        x = tf.image.resize(x, (self.S, self.S), method="bilinear")

        y_pred = tf.reshape(x, (-1, self.S, self.S, self.B * 5 + self.C))

        pred_boxes = y_pred[..., : self.B * 5]
        pred_boxes = tf.reshape(pred_boxes, (-1, self.S, self.S, self.B, 5))

        pred_boxes = tf.concat(
            [
                tf.sigmoid(pred_boxes[..., 0:2]),
                tf.sigmoid(pred_boxes[..., 2:4]),
                tf.sigmoid(pred_boxes[..., 4:5]),
            ],
            axis=-1,
        )

        pred_classes = y_pred[..., self.B * 5 :]

        y_pred_safe = tf.concat(
            [tf.reshape(pred_boxes, (-1, self.S, self.S, self.B * 5)), pred_classes],
            axis=-1,
        )

        return y_pred_safe


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
    training=False,
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


def miniyolo_saving_callback(dir_path):
    """Callback that saves the model weights and configuration on model improvements after each epoch.

    Args:
        dir_path (str): Path to the saving directory.

    Returns:
        keras.callbacks.ModelCheckpointype: Returns the model.
    """

    path = os.path.join(dir_path, "trained-model-{epoch:02d}-{val_loss:.3f}.keras")
    return ModelCheckpoint(
        filepath=path, monitor="loss", save_best_only=True, mode="min", verbose=1
    )
