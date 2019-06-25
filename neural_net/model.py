import tensorflow as tf
import functools
from time import time
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

    @lazy_property
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


    @lazy_property
    def optimize(self):
        xent = - tf.reduce_sum(self.y, tf.log(self.prediction))
        optimizer = tf.train.AdamOptimizer(0.03)
        return optimizer.minimize(xent)

    @lazy_property
    def error(self):
        # TODO: rework how cost is calculated
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def train(self, numBatches: int, batchSize: int, sess,
              modelNum=1, tbLogDir: str = None):

        # Set up tensorboard
        if tbLogDir:
            writer = tf.summary.FileWriter(tbLogDir + str(modelNum))
            writer.add_graph(sess.graph)
            merged_summary = tf.summary.merge_all()

        for i in range(numBatches):
            ...


def main(argv):
    sess = tf.Session()

    dataLoader = DataLoader(["game4", "game5"])
    model = Model(dataLoader, [120, 30, 1], [4], saveDir="models/")

    model.train(200, 100, sess, modelNum=1, tbLogDir="tensorboard/")

    sess.close()


if __name__ == "__main__":
    tf.app.run(main=main)
