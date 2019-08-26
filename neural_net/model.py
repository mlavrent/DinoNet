import tensorflow as tf
from PIL import Image
import numpy as np
from datetime import datetime
from random import randint
import argparse
from typing import Tuple, List
from data_processing.data_processing import DataLoader, DataType, loadImage


class Model(tf.keras.Model):
    def __init__(self, loadFile: str = None, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        self.conv1 = tf.keras.layers.Conv2D(filters=10,
                                            kernel_size=(10, 30),
                                            strides=(1, 30),
                                            padding="same",
                                            data_format="channels_last",
                                            activation=tf.keras.activations.softmax,
                                            use_bias=True,
                                            kernel_initializer=tf.keras.initializers.glorot_normal,
                                            bias_initializer=tf.keras.initializers.zeros)

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 1),
                                               strides=(2, 1),
                                               padding="same",
                                               data_format="channels_last")

        self.conv2 = tf.keras.layers.Conv2D(filters=10,
                                            kernel_size=(4, 1),
                                            strides=(2, 1),
                                            padding="same",
                                            data_format="channels_last",
                                            activation=tf.keras.activations.softmax,
                                            use_bias=True,
                                            kernel_initializer=tf.keras.initializers.glorot_normal,
                                            bias_initializer=tf.keras.initializers.zeros)

        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 1),
                                               strides=(2, 1),
                                               padding="same",
                                               data_format="channels_last")

        self.flattened = tf.keras.layers.Flatten(data_format="channels_last")

        self.fcl1 = tf.keras.layers.Dense(units=500,
                                          activation=tf.keras.activations.softmax,
                                          use_bias=True,
                                          kernel_initializer=tf.keras.initializers.glorot_normal,
                                          bias_initializer=tf.keras.initializers.zeros)

        self.drop1 = tf.keras.layers.Dropout(rate=0.2)

        self.fcl2 = tf.keras.layers.Dense(units=3,
                                          activation=tf.keras.activations.softmax,
                                          use_bias=True,
                                          kernel_initializer=tf.keras.initializers.glorot_normal,
                                          bias_initializer=tf.keras.initializers.zeros)

        if loadFile is not None:
            self.load_weights(loadFile)

    def call(self, inputs, training: bool = False):
        conv1 = self.conv1(inputs)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        flattened = self.flattened(pool2)

        fcl1 = self.fcl1(flattened)

        if training:
            fcl2 = self.fcl2(self.drop1(fcl1))
        else:
            fcl2 = self.fcl2(fcl1)
        return fcl2

# returns list of (action, value) tuples where action is one of:
#   - "train": int value, number of epochs to train for
#   - "evaluate": string value, path to image to evaluate model for
def parse_args():
    parser = argparse.ArgumentParser(description="Control training, running, and saving the tf model.")

    # Group for determining action to take
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--train", required=False, type=int, help="Number of epochs to train for.")
    action_group.add_argument("--eval", required=False, help="Image file to evaluate model on.")
    action_group.add_argument("--test", action="store_true")

    # Args for loading/saving model
    parser.add_argument("--load", required=False, help="File to load stored model from.")
    parser.add_argument("--save", required=False)

    args = vars(parser.parse_args())
    returnables = []

    # Add load arg to returnable
    if args["load"] is not None:
        returnables.append(("load", args["load"]))
    if args["save"] is not None:
        returnables.append(("save", args["save"].replace("/", "\\")))

    # Add actions to returnable
    if args["train"] is not None:
        returnables.append(("train", args["train"]))
    if args["eval"] is not None:
        returnables.append(("eval", args["eval"]))
    if args["test"]:
        returnables.append(("test", True))

    return returnables


if __name__ == "__main__":
    # Create the model
    model = Model()
    # model._run_eagerly = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # Create Data Loader
    datasets = ["game1", "game2", "game3", "game4", "game5"]
    trainData = DataLoader(datasets, DataType.TRAINING, batchSize=100, resampleCategories=True)
    valData = DataLoader(datasets, DataType.VALIDATION, batchSize=500)

    # Actions, in order they should be performed
    actions = parse_args()

    for action, value in actions:
        if action == "load":
            print("Loading model from {}".format(value))
            model.load_weights(value)

        elif action == "train":
            print("Training model for {} epochs".format(value))

            # Set up tensorboard for logging
            logDir = "logs\\dinoModel\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tbCallback = tf.keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=0, profile_batch=0)

            # Set up early stopping to prevent overfitting
            earlyStopping = tf.keras.callbacks.EarlyStopping(monitor="val_acc",
                                                             min_delta=0.001,
                                                             patience=5,
                                                             mode="auto",
                                                             restore_best_weights=True)

            model.fit_generator(generator=trainData,
                                epochs=value,
                                verbose=2,
                                callbacks=[tbCallback],
                                validation_data=valData,
                                validation_steps=1,
                                validation_freq=1,
                                workers=2,
                                use_multiprocessing=True,
                                shuffle=True,)

        elif action == "eval":
            print("Evaluating on image at {}.".format(value))

            pilImg, _ = loadImage(value)
            imgArr = np.array(pilImg).reshape((1, pilImg.size[1], pilImg.size[0], 2)) / 255
            output = model.predict(imgArr)

            print("Output: {}".format(value, output))

        elif action == "test":
            testData = DataLoader(datasets, DataType.TESTING, batchSize=None)

            print("Metrics for test dataset:")
            for v, n in zip(model.evaluate_generator(testData), model.metrics_names):
                print("{}: {:.3f}".format(n, v))

        elif action == "save":
            print("Saving model to {}".format(value))
            model.save_weights(value, overwrite=True, save_format="tf")

        else:
            raise LookupError("Unknown action requested")
