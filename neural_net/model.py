import tensorflow as tf
import functools
from time import time
import argparse
from typing import Tuple, List
from data_processing.data_processing import DataLoader


def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def conv_layer(input, filter_size: Tuple[int, int], step: Tuple[int, int], channels_in: int, channels_out: int):

    W = tf.Variable(tf.truncated_normal([filter_size[0], filter_size[1], channels_in, channels_out], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[channels_out]))
    conv = tf.nn.conv2d(input, W, strides=[step[0], step[1], 1, 1], padding="SAME")
    activation = tf.nn.relu(conv + b)

    tf.summary.histogram("weights", W)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("act", activation)
    return activation


def fc_layer(input, channels_in, channels_out):
    W = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
    ff = tf.matmul(input, W) + b
    activation = tf.nn.relu(ff)

    return activation


class Model:
    def __init__(self, dataLoader: DataLoader, inShape: List[int], outShape: List[int], saveDir: str):
        self.dataLoader = dataLoader
        self.x = tf.placeholder(tf.float32, inShape, name="x")
        self.y = tf.placeholder(tf.float32, outShape, name="y")
        self.prediction
        self.optimizer
        self.error

        self.saveDir = saveDir

    @define_scope
    def prediction(self):
        #x = tf.placeholder(tf.float32, shape=(120, 30, 1), name="x")  # size: (120, 30, 1) x1

        conv1 = conv_layer(self.x, (10, 30), (1, 30), 1, 10)  # size: (120, 1, 1) x10
        pool1 = tf.nn.max_pool(conv1, (2, 1, 1, 1), (2, 1, 1, 1), padding="SAME")  # size: (60, 1, 1) x10

        conv2 = conv_layer(pool1, (4, 1), (2, 1), 10, 10)  # size: (30, 1, 1) x10
        pool2 = tf.nn.max_pool(conv2, (2, 1, 1, 1), (2, 1, 1, 1), padding="SAME")  # size: (15, 1, 1) x10

        flattened = tf.reshape(pool2, (-1))

        fcl1 = fc_layer(flattened, 150, 500)
        fcl2 = fc_layer(fcl1, 500, 4)

        return fcl2

    @define_scope
    def optimize(self):
        xent = - tf.reduce_sum(self.y, tf.log(self.prediction))
        optimizer = tf.train.AdamOptimizer(0.03)
        return optimizer.minimize(xent)

    @define_scope
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.y, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def train(self, numBatches: int, batchSize: int, sess,
              modelNum=1, tbLogDir: str = None):

        # Set up tensorboard
        if tbLogDir:
            writer = tf.summary.FileWriter(tbLogDir + str(modelNum))
            writer.add_graph(sess.graph)
            merged_summary = tf.summary.merge_all()

        # Create saver to checkpoint our training
        saver = tf.train.Saver(max_to_keep=5)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Helper variables for logging
        startTime = lastLog = time()

        for i in range(numBatches):

            # Report accuracy every 120 seconds
            if time() - lastLog > 120:
                print("Step {}: Training accuracy: {}".format(i, 1 - self.error))

            sess.run(self.optimize(), feed_dict={'x': ..., 'y': ...})


# returns list of (action, value) tuples where action is one of:
#   - "train": int value, number of epochs to train for
#   - "evaluate": string value, path to image to evaluate model for
def parse_args():
    parser = argparse.ArgumentParser(description="Run or train the model")

    # Group for determining action to take
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--train", required=False, type=int, help="Number of epochs to train for.")
    action_group.add_argument("--eval", required=False, help="Image file to evaluate model on.")

    # Args for loading/saving model
    parser.add_argument("--load", required=False, help="File to load stored model from.")
    parser.add_argument("--save", required=False, default="saved_models")

    args = vars(parser.parse_args())
    returnables = []

    # Add load arg to returnable
    if args["load"] is not None:
        returnables.append(("load", args["load"]))

    # Add actions to returnable
    if args["train"] is not None:
        returnables.append(("train", args["train"]))
    if args["eval"] is not None:
        returnables.append(("eval", args["eval"]))

    # Add save arg to returnable
    returnables.append(("save", args["save"]))

    return returnables


if __name__ == "__main__":
    # Actions, in order they should be performed
    actions = parse_args()

    for action in actions:
        if action == "load":
            ...
        elif action == "train":
            ...
        elif action == "eval":
            ...
        elif action == "save":
            ...
        else:
            raise LookupError("Unknown action requested")

