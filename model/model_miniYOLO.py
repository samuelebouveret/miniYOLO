from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Resizing, Rescaling
from keras.optimizers import SGD


class MiniYOLO(Model):
    def __init__(self, h=88, w=88):
        super().__init__()
        self.resize = Resizing(height=h, width=w)
        self.rescale = Rescaling(1.0 / 255)
        self.conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")
        self.conv2 = Conv2D(16, (3, 3), activation="relu", padding="same")
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv3 = Conv2D(16, (3, 3), activation="relu", padding="same")
        self.conv4 = Conv2D(32, (3, 3), activation="relu", padding="same")
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv5 = Conv2D(32, (3, 3), activation="relu", padding="same")
        self.conv6 = Conv2D(64, (3, 3), activation="relu", padding="same")
        self.pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv7 = Conv2D(64, (3, 3), activation="relu", padding="same")
        self.conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")
        self.pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv9 = Conv2D(128, (3, 3), activation="relu", padding="same")
        self.conv10 = Conv2D(128, (3, 3), activation="relu", padding="same")
        self.pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()
        self.dense = Dense(256, activation="relu")

        # TODO -- CONSIDER A 1x1 CONV2D instead of this (uses less resources, see YOLOv2+ implementations)
        # x = keras.layers.Conv2D(filters=256, kernel_size=1)(x)

        # TODO --IMPLEMENT correct output configuration for YOLO model
        # output_ly = keras.layers.Dense((S * S * (B * 5 + C)), activation="linear")(x)
        # output_ly = keras.layers.Conv2D(filters=(B * 5 + C), kernel_size=1)(x)
        # output_ly = keras.layers.Reshape(target_shape=(4, 4, 11))(output_ly)

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

        x = self.flatten(x)
        return self.dense(x)


def miniYOLO_optimizer(lr=0.01, mo=0.9):
    return SGD(learning_rate=lr, momentum=mo)
