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
        self.conv1 = self.conv(192, 4, 2)
        self.conv2 = self.conv(256, 4, 2)
        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format)
        self.dense = tf.keras.layers.Dense(outputs, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.dense(self.pool(x))

class SimpleTrainer(RandAugmentTrainer, TFDSTrainer):
    @classmethod
    def cli(cls, parser):
        parser.add_argument("--border-conv", action="store_true")
        super().cli(parser)

    @cli_builder
    def __init__(self, learning_rate=1e-6, decay=False, augment=False,
                 dataset="mnist", **kw):
        super().__init__(learning_rate=learning_rate, decay=decay,
                         augment=augment, dataset=dataset, **kw)

    def opt(self, lr):
        return tf.keras.optimizers.Adam(lr)

    @cli_builder
    def build(self, border_conv=False, size=32, **kw):
        super().build(size=size, **kw)

        conv = BorderConv2D if border_conv else Conv2D
        self.model = SimpleModel(self.outputs, self.data_format, conv)
        self.compile(tf.keras.losses.CategoricalCrossentropy(True, 0.1),
                     ["categorical_accuracy"])
