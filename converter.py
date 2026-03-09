"""
This script converts a MiniYOLO model to TFLite format with quantization. Either weights or a full Keras model can be provided as input.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import tensorflow as tf
import numpy as np

from model import build_model


def representative_data_gen():
    for _ in range(100):
        yield [np.random.rand(1, 88, 88, 3).astype(np.float32)]


S = 2
B = 2
C = 3
IMG_SIZE = (88, 88)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MiniYOLO model.")
    parser.add_argument(
        "--path-weights",
        help="Path to .weights.h5 file.",
        required=False,
    )
    parser.add_argument(
        "--path-model",
        help="Path to .keras file.",
        required=False,
    )

    args = parser.parse_args()

    if args.path_weights is None and args.path_model is None:
        print("Please provide either --path-weights or --path-model arguments.")
        exit(1)

    if args.path_weights is not None:
        model = build_model(S, B, C, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        model.build((None, IMG_SIZE[0], IMG_SIZE[1], 3))
        model.load_weights(args.path_weights)
    else:
        model = tf.keras.models.load_model(args.path_model, compile=False)

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_data_gen

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    int8_tflite_model = converter.convert()

    with open("model-int81.tflite", "wb") as f:
        f.write(int8_tflite_model)
