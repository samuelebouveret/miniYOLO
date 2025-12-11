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
        self.reshape = Reshape((S, S, output_filters))

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
        return self.reshape(x)


def miniYOLO_optimizer(lr, mo, wd):
    return SGD(learning_rate=lr, momentum=mo, weight_decay=wd)


def parse_dataset_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/width").text)

    labels = []
    bboxes = []

    for obj in root.findall("object"):
        name = obj.find("name").text

        # convert class names to integer ids (example)
        class_id = {"car": 0, "person": 1, "chair": 2}.get(name, -1)

        if class_id == -1:
            continue

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text) / image_width
        ymin = float(bbox.find("ymin").text) / image_height
        xmax = float(bbox.find("xmax").text) / image_width
        ymax = float(bbox.find("ymax").text) / image_height

        labels.append(class_id)
        bboxes.append([xmin, ymin, xmax, ymax])

    return labels, bboxes


def load_example(image_path, xml_path):
    # --- load & preprocess image ---
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (244, 244))
    img = tf.cast(img, tf.float32) / 255.0

    # --- parse XML through py_function ---
    labels, bboxes = tf.py_function(
        lambda x: parse_dataset_xml(x.numpy().decode()),
        inp=[xml_path],
        Tout=[tf.int32, tf.float32],
    )

    # shapes unknown → fix them
    labels.set_shape([None])
    bboxes.set_shape([None, 4])

    return img, labels, bboxes


def miniYOLO_saving_callback(dir_path):
    path = join(dir_path, "trained_model-{epoch:02d}-{loss:.3f}.keras")
    return ModelCheckpoint(filepath=path, monitor="loss", save_best_only=True)
