from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# x = keras.layers.Flatten()(x)
# x = keras.layers.Dense(256, activation="relu")(x)
# print(x.count_params())


# output_ly = keras.layers.Dense((S * S * (B * 5 + C)), activation="linear")(x)

# NAH
# output_ly = keras.layers.Conv2D(filters=(B * 5 + C), kernel_size=1)(x)
# output_ly = keras.layers.Reshape(target_shape=(4, 4, 11))(output_ly)


class MiniYOLO(Model):
    def __init__(self):
        super().__init__()

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

        # CONSIDER A 1x1 CONV2D instead of this (uses less resources, see YOLOv2+ implementations) -- TODO
        # x = keras.layers.Conv2D(filters=256, kernel_size=1)(x)

    def call(self, inputs):
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

        x = self.flatten(x)
        return self.dense(x)
