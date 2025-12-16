from keras import Model
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Resizing,
    Rescaling,
    LeakyReLU,
    Reshape,
)

from os.path import join
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from tensorflow import cast, float32
from tensorflow.python.ops.image_ops_impl import resize_images_v2 as resize

import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np


class MiniYOLO(Model):
    def __init__(self, h=88, w=88, S=2, B=2, C=3):
        super().__init__()
        # TODO -- Maybe add param leaky
        output_filters = B * 5 + C
        leaky_layer = LeakyReLU(0.1)
        self.resize = Resizing(height=h, width=w)
        self.rescale = Rescaling(1.0 / 255)
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

        self.convoutput = Conv2D(output_filters, 1, padding="same")
        # self.reshape = Reshape((S, S, output_filters))

    def call(self, inputs):
        x = self.resize(inputs)
        x = self.rescale(x)
        x = self.conv1(x)
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

        x = self.convoutput(x)
        return x


def miniYOLO_optimizer(lr, mo, wd):
    return SGD(learning_rate=lr, momentum=mo, weight_decay=wd)


def parse_dataset_xml(xml_path, max_objects, selected_classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    labels = []
    bboxes = []

    selected_classes = [x.decode() for x in selected_classes]

    for obj in root.findall("object"):
        if len(labels) == max_objects:
            break
        name = obj.find("name").text

        class_id = {classes: i + 1 for i, classes in enumerate(selected_classes)}.get(
            name, -1
        )

        # TODO NEED TO PAD CUZ RATIOS ARE NOT THE SAME IN INPUT AND AFTER MODEL RESIZES!!!
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


def miniYOLO_load_example(image_path, xml_path, max_objects, selected_classes):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0

    labels, bboxes = tf.py_function(
        func=lambda x, y, z: parse_dataset_xml(x.numpy(), y, z.numpy()),
        inp=[xml_path, max_objects, selected_classes],
        Tout=[tf.int32, tf.float32],
    )

    return image, labels, bboxes


def miniYOLO_saving_callback(dir_path):
    path = join(dir_path, "trained_model-{epoch:02d}-{loss:.3f}.keras")
    return ModelCheckpoint(filepath=path, monitor="loss", save_best_only=True)
