import tensorflow as tf
from tensorflow_addons.optimizers import MovingAverage
from functools import partial
from base import RandAugmentTrainer, TFDSTrainer
from utils import RequiredLength, Conv2D, cli_builder
from border import Conv2D as BorderConv2D

class SimpleModel(tf.keras.Model):
    def __init__(self, outputs, data_format, conv):
        super().__init__()
        self.conv = partial(conv, padding='same', data_format=data_format,
                            activation=tf.nn.relu)
        self.conv0 = self.conv(128, 4, 2)
        self.conv1 = self.conv(192, 1, 2)
        self.conv2 = self.conv(256, 4, 2)
        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format)
        self.dense = tf.keras.layers.Dense(outputs, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.dense(self.pool(x))

class SimpleTrainer(RandAugmentTrainer, TFDSTrainer):
    opt = lambda _, lr: MovingAverage(
        tf.keras.optimizers.RMSprop(lr, 0.9, 0.9, 0.001))

    @cli_builder
    def build(self, border_conv=False, size=64, **kwargs):
        self.mapper = lambda f: lambda x, y: (
            f(x), tf.one_hot(y, self.outputs))
        super().build(size=size, **kwargs)

        conv = BorderConv2D if border_conv else Conv2D
        self.model = SimpleModel(self.outputs, self.data_format, conv)
        self.compile(tf.keras.losses.CategoricalCrossentropy(True, 0.1),
                     ["categorical_accuracy"])

    @classmethod
    def cli(cls, parser):
        parser.add_argument("--border-conv", action="store_true")
        super().cli(parser)
