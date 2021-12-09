import tensorflow as tf
from functools import partial
from base import RandAugmentTrainer, TFDSTrainer
from utils import RequiredLength, Conv2D, cli_builder
from border import Conv2D as BorderConv2D

class SimpleModel(tf.keras.Model):
    def __init__(self, outputs, data_format, conv):
        super().__init__()
        self.conv = partial(conv, padding='same', data_format=data_format)
        self.conv0 = self.conv(128, 2, 2)
        self.conv1 = self.conv(192, 3, 2)
        self.conv2 = self.conv(256, 4, 2)
        self.conv3 = self.conv(320, 5, 2)
        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format)
        self.dense = tf.keras.layers.Dense(outputs, activation=tf.nn.softmax)

    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.dense(self.pool(x))

class SimpleTrainer(RandAugmentTrainer, TFDSTrainer):
    @classmethod
    def cli(cls, parser):
        parser.add_argument("--border-conv", action="store_true", help=(
            "use border aware convolutions"))
        super().cli(parser)

    def opt(self, lr):
        return tf.keras.optimizers.Adam(lr)

    @cli_builder
    def __init__(self, learning_rate=1e-6, decay=False, augment=False,
                 dataset="mnist", border_conv=False, size=32, **kw):
        super().__init__(learning_rate=learning_rate, decay=decay,
                         augment=augment, dataset=dataset, size=size, **kw)

        conv = BorderConv2D if border_conv else Conv2D
        self.model = SimpleModel(self.outputs, self.data_format, conv)
        self.compile(tf.keras.losses.CategoricalCrossentropy(True, 0.1),
                     ["categorical_accuracy"])
